// PS5 platform memory with demand-backed 64 KiB direct-memory units.
// Guest commits record RW page state in the 2-level chunk directory without
// allocating physical DMEM; first access (SIGSEGV/SIGBUS) or explicit host
// kernel-write / GPU-import backing (plat_mem_back) allocates, maps, zeroes,
// and applies the unit's mixed native protections.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "platform.h"
#include "prot_watch.h"
#include "ps5_guest_memory.h"
#include "ps5_mixed.h"

#include <ps5platform/kernel.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

#define ZSTD_STATIC_LINKING_ONLY   // ZSTD_initStaticDCtx: the fault handler decompresses without malloc
#include <zstd.h>

enum {
    PAGE = PS5_GUEST_PAGE,
    CHUNK_SHIFT = 24, // 16 MiB per chunk = 256 x 64 KiB Ps5MixedBlock units
    CHUNK_SIZE = 1u << CHUNK_SHIFT,
    CHUNK_UNITS = CHUNK_SIZE / PS5_MEM_UNIT,
    ALT_STACK_BYTES = 65536,
    LOG_EVERY_UNITS = 4096, // Log backed totals every 256 MiB of physical growth.
};

_Static_assert(PS5_GUEST_NONE + PLAT_MEM_RESERVED == PS5_GUEST_RESERVED &&
               PS5_GUEST_NONE + PLAT_PROT_READ == PS5_GUEST_READ &&
               PS5_GUEST_NONE + PLAT_PROT_RW == PS5_GUEST_RW, "guest/platform page states differ");
static const uint64_t address_limit = 1ull << 39;

typedef enum { RANGE_OTHER = 0, RANGE_HEAP = 1, RANGE_ALIAS = 2 } RangeKind;

typedef struct VmRange {
    uint64_t base, end;
    // Fixed-size directory of 16 MiB chunks, allocated once on reserve and
    // populated lazily on commit. Never reallocated or freed before shutdown,
    // so fault_handler and plat_mem_back can walk it without taking vm_lock.
    Ps5MixedBlock *volatile *chunks;
    size_t chunk_count;
    bool host; // Native anonymous allocation, with no guest subpages.
    RangeKind kind;
    struct VmRange *next;
} VmRange;

static VmRange *ranges;
static VmRange *cleanup_pending;
static bool stopped;
static pthread_mutex_t vm_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t backed_units[3];
static size_t backed_by_cause[PLAT_BACK_N];
static size_t next_log_units = LOG_EVERY_UNITS;
static bool oom_logged;
static void (*g_vk_census)(char *out, size_t cap, int in_signal);

int sceKernelGetPageTableStats(int *cpu_total, int *cpu_free, int *gpu_total, int *gpu_free);

void ps5_mem_set_vk_census(void (*fn)(char *out, size_t cap, int in_signal)) {
    __atomic_store_n(&g_vk_census, fn, __ATOMIC_RELEASE);
}

static int (*g_fault_hook)(uint64_t addr);
static int (*g_watch_hook)(uint64_t addr, uint64_t rip);
static void (*g_crash_hook)(const PlatCrash *c);
static void (*g_early_hold_hook)(uint64_t addr, size_t n);
static int (*g_frozen_hook)(uint64_t addr);

_Thread_local int plat_fault_write;
_Thread_local uint64_t plat_fault_pc, plat_fault_ret;

static struct sigaction prev_sigsegv, prev_sigbus, prev_sigill, prev_sigfpe;
static pthread_once_t fault_once = PTHREAD_ONCE_INIT;

static inline void unit_lock(Ps5MixedBlock *u) {
    while (__atomic_test_and_set(&u->lock, __ATOMIC_ACQUIRE)) {}
}

static inline void unit_unlock(Ps5MixedBlock *u) {
    __atomic_clear(&u->lock, __ATOMIC_RELEASE);
}

static bool sdk_ok(const char *operation, uint64_t address, uint64_t bytes, int result) {
    if (result)
        fprintf(stderr, "PS5 memory %s %#llx+%#llx failed (%#x)\n", operation, (unsigned long long)address,
                (unsigned long long)bytes, (unsigned)result);
    return !result;
}

static void log_backed_totals(const char *call, uint64_t address, int err, int in_signal) {
    if (err && __atomic_exchange_n(&oom_logged, true, __ATOMIC_ACQ_REL)) return;
    int64_t phys = 0;
    size_t avail = 0, flex_avail = 0, flex_cfg = 0;
    int64_t dmem_total = sceKernelGetDirectMemorySize();
    int avail_rc = sceKernelAvailableDirectMemorySize(0, INT64_MAX, PS5_MEM_UNIT, &phys, &avail);
    int flex_a_rc = sceKernelAvailableFlexibleMemorySize(&flex_avail);
    int flex_c_rc = sceKernelConfiguredFlexibleMemorySize(&flex_cfg);
    size_t heap_mib = (__atomic_load_n(&backed_units[RANGE_HEAP], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t alias_mib = (__atomic_load_n(&backed_units[RANGE_ALIAS], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t other_mib = (__atomic_load_n(&backed_units[RANGE_OTHER], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t total_mib = heap_mib + alias_mib + other_mib;
    size_t cw_mib = (__atomic_load_n(&backed_by_cause[PLAT_BACK_WRITE_FAULT], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t cr_mib = (__atomic_load_n(&backed_by_cause[PLAT_BACK_READ_FAULT], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t crr_mib = (__atomic_load_n(&backed_by_cause[PLAT_BACK_RT_READ], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    size_t ci_mib = (__atomic_load_n(&backed_by_cause[PLAT_BACK_IMPORT], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    char vk_buf[256] = "vk=none";
    void (*vk_fn)(char *, size_t, int) = __atomic_load_n(&g_vk_census, __ATOMIC_ACQUIRE);
    if (vk_fn) vk_fn(vk_buf, sizeof vk_buf, in_signal);
    int pt_cpu_tot = 0, pt_cpu_free = 0, pt_gpu_tot = 0, pt_gpu_free = 0;
    int pt_rc = sceKernelGetPageTableStats(&pt_cpu_tot, &pt_cpu_free, &pt_gpu_tot, &pt_gpu_free);
    char pt_buf[64];
    if (pt_rc == 0) snprintf(pt_buf, sizeof pt_buf, "pt=cpu:%d/%d gpu:%d/%d", pt_cpu_free, pt_cpu_tot, pt_gpu_free, pt_gpu_tot);
    else snprintf(pt_buf, sizeof pt_buf, "pt_rc=%#x", (unsigned)pt_rc);
    if (err) {
        fprintf(stderr,
                "PS5 memory back %#llx failed in %s (%#x): backed=%zu MiB (heap=%zu, alias=%zu, other=%zu), "
                "causes(w=%zu,rf=%zu,rr=%zu,imp=%zu MiB), "
                "dmem_avail=%zu/%lld MiB (rc=%#x), flex=%zu/%zu MiB (rc=%#x/%#x), %s, %s\n",
                (unsigned long long)address, call, (unsigned)err, total_mib, heap_mib, alias_mib, other_mib,
                cw_mib, cr_mib, crr_mib, ci_mib,
                avail_rc == 0 ? (avail >> 20) : 0, dmem_total > 0 ? (long long)(dmem_total >> 20) : 0ll,
                (unsigned)avail_rc, flex_a_rc == 0 ? (flex_avail >> 20) : 0,
                flex_c_rc == 0 ? (flex_cfg >> 20) : 0, (unsigned)flex_a_rc, (unsigned)flex_c_rc, pt_buf, vk_buf);
    } else {
        fprintf(stderr,
                "PS5 memory %s: backed=%zu MiB (heap=%zu, alias=%zu, other=%zu), causes(w=%zu,rf=%zu,rr=%zu,imp=%zu MiB), "
                "dmem_avail=%zu/%lld MiB, flex=%zu/%zu MiB, %s, %s\n",
                call, total_mib, heap_mib, alias_mib, other_mib, cw_mib, cr_mib, crr_mib, ci_mib,
                avail_rc == 0 ? (avail >> 20) : 0,
                dmem_total > 0 ? (long long)(dmem_total >> 20) : 0ll,
                flex_a_rc == 0 ? (flex_avail >> 20) : 0, flex_c_rc == 0 ? (flex_cfg >> 20) : 0, pt_buf, vk_buf);
    }
}

// The guest's backed memory by range kind and cause, and free direct memory, as one line for mkw.log (the census in
// log_backed_totals goes to stderr, the console's klog, so a run's log could not say where direct memory went: run
// nocache, 16/11/2026, free memory 1104 -> 105 MiB over three races with the VMA heap flat).
static void pk_backed_note(char *out, size_t cap);   // MKW_PACK, below

// sceKernelAvailableDirectMemorySize answers with one free area of the range, apparently the largest (840 MiB unchanged
// for 2,877 governor passes that were freeing memory, with ~2 GiB unaccounted for, fp-base7-20261119). This adds up the
// areas by asking again on each side of every one found, the widest range first; at most DMEM_SCAN_CALLS calls, so the
// total is a lower bound when *cut is set. For the log only (a few lines a run): nothing decides on it yet.
#define DMEM_SCAN_CALLS 1024
typedef struct {
    struct { int64_t lo, hi; } todo[DMEM_SCAN_CALLS + 2];
    int busy;
} DmemScan;
// scan_bench: plat_mem_pressure's own (MKW_BENCH's 1 Hz sampler), so a sample never finds the governor's scan busy
// and the governor never finds the sampler's busy (it would read that as no answer and act on the largest area alone).
static DmemScan scan_main, scan_bench;
static uint64_t dmem_free_scan_with(DmemScan *sc, int64_t dmem_total, int *areas, int *cut) {
    int n = 1, calls = 0;
    uint64_t total = 0;
    *areas = 0, *cut = 1;
    if (__atomic_exchange_n(&sc->busy, 1, __ATOMIC_ACQUIRE)) return 0;
    __typeof__(sc->todo[0]) *todo = sc->todo;
    todo[0].lo = 0, todo[0].hi = dmem_total;
    *cut = 0;
    while (n) {
        if (calls++ == DMEM_SCAN_CALLS) { *cut = 1; break; }
        int w = 0;
        for (int i = 1; i < n; i++)
            if (todo[i].hi - todo[i].lo > todo[w].hi - todo[w].lo) w = i;
        int64_t lo = todo[w].lo, hi = todo[w].hi, phys = 0;
        todo[w] = todo[--n];
        size_t size = 0;
        if (sceKernelAvailableDirectMemorySize(lo, hi, PS5_MEM_UNIT, &phys, &size) != 0 || !size) continue;
        if (phys < lo || (uint64_t)phys + size > (uint64_t)hi) continue;   // (not an answer about this range)
        total += size, ++*areas;
        if (phys - lo >= (int64_t)PS5_MEM_UNIT) todo[n].lo = lo, todo[n++].hi = phys;
        if (hi - phys - (int64_t)size >= (int64_t)PS5_MEM_UNIT) todo[n].lo = phys + (int64_t)size, todo[n++].hi = hi;
    }
    __atomic_store_n(&sc->busy, 0, __ATOMIC_RELEASE);
    return total;
}

static uint64_t dmem_free_scan(int64_t dmem_total, int *areas, int *cut) {
    return dmem_free_scan_with(&scan_main, dmem_total, areas, cut);
}

void plat_mem_backed_summary(char *out, size_t cap) {
    int64_t phys = 0;
    size_t avail = 0;
    int64_t dmem_total = sceKernelGetDirectMemorySize();
    int rc = sceKernelAvailableDirectMemorySize(0, INT64_MAX, PS5_MEM_UNIT, &phys, &avail);
    size_t u[3], c[4];
    for (int i = 0; i < 3; i++) u[i] = (__atomic_load_n(&backed_units[i], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    static const int causes[4] = {PLAT_BACK_WRITE_FAULT, PLAT_BACK_READ_FAULT, PLAT_BACK_RT_READ, PLAT_BACK_IMPORT};
    for (int i = 0; i < 4; i++) c[i] = (__atomic_load_n(&backed_by_cause[causes[i]], __ATOMIC_RELAXED) * PS5_MEM_UNIT) >> 20;
    snprintf(out, cap, "backed %zu MiB (heap %zu, alias %zu, other %zu; by write fault %zu, read fault %zu, runtime read "
             "%zu, import %zu), dmem free %zu of %lld MiB",
             u[RANGE_HEAP] + u[RANGE_ALIAS] + u[RANGE_OTHER], u[RANGE_HEAP], u[RANGE_ALIAS], u[RANGE_OTHER], c[0], c[1],
             c[2], c[3], rc == 0 ? avail >> 20 : 0, dmem_total > 0 ? (long long)(dmem_total >> 20) : 0ll);
    size_t at = strlen(out);
    if (dmem_total > 0 && at < cap) {
        int areas, cut;
        uint64_t t0 = plat_now_ns(), total = dmem_free_scan(dmem_total, &areas, &cut);
        snprintf(out + at, cap - at, " (all free areas: %s%llu MiB in %d, scan %llu us)", cut ? "at least " : "",
                 (unsigned long long)(total >> 20), areas, (unsigned long long)((plat_now_ns() - t0) / 1000));
        at = strlen(out);
    }
    if (at < cap) pk_backed_note(out + at, cap - at);
}

void ps5_mem_log_census(const char *tag) { log_backed_totals(tag ? tag : "census", 0, 0, 0); }

void plat_mem_census_log(const char *tag) { ps5_mem_log_census(tag); }

uint64_t plat_mem_dmem_free_total(void) {
    int64_t dmem_total = sceKernelGetDirectMemorySize();
    if (dmem_total <= 0) return UINT64_MAX;
    int areas, cut;
    uint64_t total = dmem_free_scan(dmem_total, &areas, &cut);
    return cut && !areas ? UINT64_MAX : total;   // (no area and cut: another thread's scan was running)
}

// MKW_BENCH's memory report (core/diag/bench_mem.c), once a second on its own thread: free direct memory (all areas,
// on scan_bench, and the largest), the guest's backed memory, free flexible memory (the C heap's) and the page-table
// pools (the CPU pool of 16,384 units ran out on sparse address space before: docs/ps5).
void plat_mem_pressure(PlatMemPressure *p) {
    *p = (PlatMemPressure){-1, -1, -1, -1, -1, -1, -1, -1};
    int64_t dmem_total = sceKernelGetDirectMemorySize(), phys = 0;
    size_t largest = 0, flex = 0;
    if (dmem_total > 0) {
        int areas, cut;
        uint64_t t = dmem_free_scan_with(&scan_bench, dmem_total, &areas, &cut);
        if (areas) p->dmem_free = (int64_t)t;
    }
    if (sceKernelAvailableDirectMemorySize(0, INT64_MAX, PS5_MEM_UNIT, &phys, &largest) == 0) p->dmem_largest = (int64_t)largest;
    size_t units = 0;
    for (int i = 0; i < 3; i++) units += __atomic_load_n(&backed_units[i], __ATOMIC_RELAXED);
    p->guest_backed = (int64_t)(units * PS5_MEM_UNIT);
    if (sceKernelAvailableFlexibleMemorySize(&flex) == 0) p->flex_free = (int64_t)flex;
    int ct = 0, cf = 0, gt = 0, gf = 0;
    if (sceKernelGetPageTableStats(&ct, &cf, &gt, &gf) == 0) p->pt_cpu_free = cf, p->pt_cpu_total = ct, p->pt_gpu_free = gf, p->pt_gpu_total = gt;
}

uint64_t plat_mem_dmem_avail(void) {
    int64_t phys = 0;
    size_t avail = 0;
    if (sceKernelAvailableDirectMemorySize(0, INT64_MAX, PS5_MEM_UNIT, &phys, &avail) == 0)
        return (uint64_t)avail;
    return UINT64_MAX;
}

// ---- MKW_PACK: cold units compressed, their direct memory given back (plat_mem_pack, task 37)
// Run memgrow2 (16/11/2026): ~1.2 GiB of one pool is earlier courses' data the game freed inside its own heap and never
// gave back; it is not zero, so the zero reclaim keeps it, and releasing it is not safe (the game sampled the character
// cards again after their pool's finalize, task 32). EglantinePS5 (GPL-3.0, the idea only) packed units unwritten for a
// while with LZ4 and unpacked them on the next fault: 1.0-2.4 GB into 0.2-0.5 GB a session, 1,627 unpacks checked
// against a kept copy, 0 differed. Here: zstd level 1 (64 KiB units of the romfs's models 0.46 of their size,
// textures 0.74, ~130 us to compress and ~50 us to decompress one on this machine's Zen 2, task 37), a unit packed only
// when frozen (read-only, nothing wrote it since the freeze) for MKW_PACK_COLD_S and compressed to 75 % or less.
// A packed unit's guest pages keep their states and its backing is released; back_unit_locked (a fault, or
// plat_mem_back* before a system call or an import) decompresses it before the protections apply and the access retries.
typedef struct PackBlob {
    struct PackBlob *next;   // pk_free once unpacked: the fault handler does not call free()
    uint8_t *raw;            // MKW_PACK_VERIFY: the unit's bytes when packed; NULL otherwise
    uint32_t csize;
    uint32_t bytes;          // what the store holds for it (header, blob and raw)
    uint32_t zmask;          // guest pages decommitted since the pack: committed again, they read zero
    uint8_t data[];
} PackBlob;

enum { PK_DCTX = 4, PK_THREADS = 16, PK_MAX_PCT = 75 };
// A unit the game reads again after its pack waits twice as long, cold, before the next one, up to 2^PK_BOUNCE_MAX
// times MKW_PACK_COLD_S. A race read ~2,300-3,300 packed units back every 30 s at ~260 us each (max 1.5 ms), on the
// game's threads and ~40 % on the replay's, while ~5,000 were packed: memory a lap reads once, colder than 30 s and
// not cold for good (ab-capreuse, 25/11/2026). MKW_PACK_BACKOFF=0: the same cold time every pack, as before.
enum { PK_BOUNCE_MAX = 3 };

// The store: blobs and verify copies in direct memory, not the C heap. The C heap is the payload's flexible memory,
// 640 MiB that the rest of the runtime fills (24 MiB free all through a 1.8.0 Grand Prix, cold-census-20261124), and
// the census found ~1.3 GiB to give back by packing (~3.4 GiB unwritten for 30 s, ~45 % of it as stored bytes), so the
// store takes its own memory from the pool it frees. 1 MiB slabs mapped at 1 MiB, one size class per 4 KiB; the slab's
// first slot holds its header, so a slot finds its slab by rounding its address down. The governor's thread only: blobs
// are made by plat_mem_pack and freed by plat_mem_pack_collect (the fault handler and decommits hand them over by
// pk_drop).
enum { PKS_SLAB = 1 << 20, PKS_STEP = 4096, PKS_CLASSES = 2 * PS5_MEM_UNIT / PKS_STEP + 1 };
typedef struct PksSlab {
    struct PksSlab *next, *prev;   // slabs of the class with a free slot
    int64_t physical;
    uint32_t cls, used, cap, bump;   // slot size cls * PKS_STEP; slots 1..cap; bump: the next never used
    void *free;                      // freed slots, linked through their first word
} PksSlab;
_Static_assert(sizeof(PksSlab) <= PKS_STEP, "a slab's header fits its first slot");
static PksSlab *pks_open[PKS_CLASSES];
static uint64_t pks_slabs, pks_slot_bytes;   // slabs mapped; bytes of the slots in use (the store's size)

static void pks_unlink(PksSlab *sl) {
    if (sl->prev) sl->prev->next = sl->next;
    else pks_open[sl->cls] = sl->next;
    if (sl->next) sl->next->prev = sl->prev;
    sl->next = sl->prev = NULL;
}

static void pks_link(PksSlab *sl) {
    sl->prev = NULL, sl->next = pks_open[sl->cls];
    if (sl->next) sl->next->prev = sl;
    pks_open[sl->cls] = sl;
}

static void *pks_alloc(size_t bytes, uint32_t *slot_bytes) {
    uint32_t cls = (uint32_t)((bytes + PKS_STEP - 1) / PKS_STEP);
    if (!cls || cls >= PKS_CLASSES) return NULL;
    PksSlab *sl = pks_open[cls];
    if (!sl) {
        int64_t phys = 0;
        sl = ps5_mem_map_host(PKS_SLAB, PKS_SLAB, &phys);
        if (!sl) return NULL;
        *sl = (PksSlab){.physical = phys, .cls = cls, .cap = PKS_SLAB / (cls * PKS_STEP) - 1, .bump = 1};
        pks_link(sl);
        pks_slabs++;
    }
    void *p = sl->free;
    if (p) sl->free = *(void **)p;
    else p = (uint8_t *)sl + (size_t)sl->bump++ * cls * PKS_STEP;
    if (++sl->used == sl->cap) pks_unlink(sl);
    *slot_bytes = cls * PKS_STEP;
    pks_slot_bytes += *slot_bytes;
    return p;
}

static void pks_free(void *p) {
    if (!p) return;
    PksSlab *sl = (PksSlab *)((uintptr_t)p & ~(uintptr_t)(PKS_SLAB - 1));
    pks_slot_bytes -= sl->cls * PKS_STEP;
    if (sl->used-- == sl->cap) pks_link(sl);   // it was full: open again
    *(void **)p = sl->free, sl->free = p;
    if (sl->used) return;
    pks_unlink(sl);
    if (ps5_mem_unmap_host(sl, PKS_SLAB, sl->physical) == 0) pks_slabs--;
}
static struct { volatile int lock; ZSTD_DCtx *d; } pk_dctx[PK_DCTX];   // shared by the fault handlers, made up front
static PackBlob *volatile pk_free;                                     // unpacked blobs, freed by plat_mem_pack_collect
static size_t packed_units[3];
static uint64_t pk_packed_bytes, pk_store_bytes, pk_ever;
// Since the last plat_mem_pack_summary (exchanged there).
static uint64_t pk_unpacks, pk_unpack_ns, pk_unpack_max_ns, pk_unpack_cause[PLAT_BACK_N];
static uint64_t pk_unpack_total_ns;   // never reset: plat_mem_cost_ns
static uint64_t pk_verify_checked, pk_verify_bad, pk_verify_addr, pk_fill_bad;
static struct { uint32_t tid; uint64_t n; char name[24]; } pk_thr[PK_THREADS];   // unpacks by thread
static uint32_t (*g_frozen_ms_hook)(uint64_t addr);

// The fault handler's half: signal-safe, no malloc, no lock but a free decompression context's.
static int unpack_fill(void *ctx, void *dst) {
    PackBlob *b = ctx;
    size_t got = 0;
    for (int done = 0; !done;)
        for (int i = 0; i < PK_DCTX && !done; i++) {
            if (__atomic_test_and_set(&pk_dctx[i].lock, __ATOMIC_ACQUIRE)) continue;
            got = ZSTD_decompressDCtx(pk_dctx[i].d, dst, PS5_MEM_UNIT, b->data, b->csize);
            __atomic_clear(&pk_dctx[i].lock, __ATOMIC_RELEASE);
            done = 1;
        }
    int bad = ZSTD_isError(got) || got != PS5_MEM_UNIT;
    if (bad) __atomic_add_fetch(&pk_fill_bad, 1, __ATOMIC_RELAXED);
    if (b->raw) {   // MKW_PACK_VERIFY: the kept bytes win, and a difference is counted
        __atomic_add_fetch(&pk_verify_checked, 1, __ATOMIC_RELAXED);
        if (bad || memcmp(dst, b->raw, PS5_MEM_UNIT)) {
            uint64_t z = 0;
            __atomic_add_fetch(&pk_verify_bad, 1, __ATOMIC_RELAXED);
            __atomic_compare_exchange_n(&pk_verify_addr, &z, (uint64_t)(uintptr_t)dst, 0, __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED);
            memcpy(dst, b->raw, PS5_MEM_UNIT);
            bad = 0;
        }
    }
    if (bad) return PS5_MEM_BAD_REQUEST;
    for (unsigned g = 0; g < PS5_MEM_UNIT / PAGE; g++)
        if (b->zmask >> g & 1) memset((uint8_t *)dst + (size_t)g * PAGE, 0, PAGE);
    return 0;
}

static void pk_drop(PackBlob *b) {   // any thread: to the free list
    PackBlob *head = __atomic_load_n(&pk_free, __ATOMIC_RELAXED);
    do b->next = head;
    while (!__atomic_compare_exchange_n(&pk_free, &head, b, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED));
}

static void pk_unpacked(PackBlob *b, RangeKind kind, int cause, uint64_t ns) {
    __atomic_sub_fetch(&packed_units[kind], 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&pk_packed_bytes, b->csize, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pk_unpacks, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pk_unpack_ns, ns, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pk_unpack_total_ns, ns, __ATOMIC_RELAXED);
    uint64_t m = __atomic_load_n(&pk_unpack_max_ns, __ATOMIC_RELAXED);
    while (ns > m && !__atomic_compare_exchange_n(&pk_unpack_max_ns, &m, ns, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    if ((unsigned)cause < PLAT_BACK_N) __atomic_add_fetch(&pk_unpack_cause[cause], 1, __ATOMIC_RELAXED);
    uint32_t tid = plat_thread_id();
    for (int i = 0; i < PK_THREADS; i++) {
        uint32_t t = __atomic_load_n(&pk_thr[i].tid, __ATOMIC_ACQUIRE);
        if (!t) {
            uint32_t z = 0;
            if (__atomic_compare_exchange_n(&pk_thr[i].tid, &z, tid, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                char nm[24];
                if (!plat_sample_self_name(nm, sizeof nm) || !nm[0]) memcpy(nm, "?", 2);
                memcpy(pk_thr[i].name, nm, sizeof nm);
                t = tid;
            } else t = z;
        }
        if (t == tid) {
            __atomic_add_fetch(&pk_thr[i].n, 1, __ATOMIC_RELAXED);
            break;
        }
    }
    pk_drop(b);
}

// Whether the guest may write some page of the native page at off in unit (ps5_mixed_back_fill maps it read-write).
static bool native_rw(const Ps5MixedBlock *unit, uint64_t off) {
    for (size_t i = off / PAGE; i < (off + PS5_MEM_PAGE) / PAGE; ++i)
        if (unit->pages[i] == PS5_GUEST_RW) return true;
    return false;
}

static bool back_unit_locked(Ps5MixedBlock *unit, RangeKind kind, int in_signal, int cause) {
    if (unit->backing.committed) return true;
    PackBlob *pk = unit->pack;
    uint64_t t0 = pk ? plat_now_ns() : 0;
    Ps5MemCommitReport rep = {0};
    int rc = pk ? ps5_mixed_back_fill(unit, &rep, unpack_fill, pk) : ps5_mixed_back_report(unit, &rep);
    if (rc) {
        if (pk && !rep.allocation_status && !rep.private_status && !rep.private_unmap_status && !rep.mapping_status)
            fprintf(stderr, "PS5 memory unpack %#llx failed: its packed bytes do not decompress (%u bytes)\n",
                    (unsigned long long)unit->backing.address, pk->csize);
        const char *call = rep.allocation_status ? "sceKernelAllocateDirectMemory"
                           : rep.private_status ? "sceKernelMapDirectMemory (private)"
                           : rep.private_unmap_status ? "sceKernelMunmap (private)"
                           : (rep.fixed_attempted && rep.mapping_status) ? "sceKernelMapDirectMemory"
                           : "ps5_mixed_back";
        int err = rep.allocation_status ? rep.allocation_status
                  : rep.private_status ? rep.private_status
                  : rep.private_unmap_status ? rep.private_unmap_status
                  : (rep.fixed_attempted && rep.mapping_status) ? rep.mapping_status : rc;
        log_backed_totals(call, unit->backing.address, err, in_signal);
        return false;
    }
    __atomic_add_fetch(&backed_units[kind], 1, __ATOMIC_RELAXED);
    if (pk) {   // the same bytes again, not growth: counted as an unpack, not by cause
        unit->pack = NULL;
        unit->pk_unpack_ms = (uint32_t)plat_now_ms() | 1;
        if (unit->pk_bounces < PK_BOUNCE_MAX) unit->pk_bounces++;
        pk_unpacked(pk, kind, cause, plat_now_ns() - t0);
    } else if ((unsigned)cause < PLAT_BACK_N)
        __atomic_add_fetch(&backed_by_cause[cause], 1, __ATOMIC_RELAXED);
    for (uint64_t off = 0; off < PS5_MEM_UNIT; off += PS5_MEM_PAGE) {
        uint64_t p = unit->backing.address + off;
        if (prot_watch_is_early(p)) {
            int prot = prot_watch_has_gpu(p, PS5_MEM_PAGE)
                           ? (PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE)
                           : 0;
            sceKernelMprotect((void *)(uintptr_t)p, PS5_MEM_PAGE, prot);
        } else if (prot_watch_is_armed(p) && native_rw(unit, off)) {
            // A unit given back and backed again (MKW_PACK's unpack, a zero unit's release) is mapped by the guest's
            // modes alone, read-write where the guest may write. A page the write watch armed before the release still
            // reads armed, so the game's writes to it would not fault and the textures there would never see them
            // (issue 11, 0.3.0: glyphs drawn into the font cache after a Grand Prix showed the ones they replaced).
            // Read-only again, as the watch expects; if that fails, counted as written instead.
            if (ps5_mem_protect_native(&unit->backing, off, PS5_MEM_PAGE, PS5_MEM_READ))
                prot_watch_mark_written(p, PS5_MEM_PAGE);
        }
    }
    prot_watch_touch(unit->backing.address, PS5_MEM_UNIT);   // the pages not armed: written
    size_t total = __atomic_load_n(&backed_units[RANGE_HEAP], __ATOMIC_RELAXED) +
                   __atomic_load_n(&backed_units[RANGE_ALIAS], __ATOMIC_RELAXED) +
                   __atomic_load_n(&backed_units[RANGE_OTHER], __ATOMIC_RELAXED);
    size_t target = __atomic_load_n(&next_log_units, __ATOMIC_RELAXED);
    if (total >= target &&
        __atomic_compare_exchange_n(&next_log_units, &target, target + LOG_EVERY_UNITS, false,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        log_backed_totals("summary", unit->backing.address, 0, in_signal);
    }
    return true;
}

static VmRange **range_slot(uint64_t address) {
    VmRange **slot = &ranges;
    while (*slot && (*slot)->end <= address) slot = &(*slot)->next;
    return slot;
}

static VmRange *find_range_lockfree(uint64_t address) {
    for (VmRange *r = __atomic_load_n(&ranges, __ATOMIC_ACQUIRE); r;
         r = __atomic_load_n(&r->next, __ATOMIC_ACQUIRE)) {
        if (address < r->base) return NULL;
        if (address < r->end) return r;
    }
    return NULL;
}

static Ps5MixedBlock *find_unit_lockfree(const VmRange *range, uint64_t address) {
    if (!range || range->host || !range->chunks || address < range->base || address >= range->end) return NULL;
    size_t cidx = (size_t)((address - range->base) >> CHUNK_SHIFT);
    Ps5MixedBlock *chunk = __atomic_load_n(&range->chunks[cidx], __ATOMIC_ACQUIRE);
    if (!chunk) return NULL;
    size_t uidx = (size_t)(((address - range->base) & (CHUNK_SIZE - 1)) / PS5_MEM_UNIT);
    return &chunk[uidx];
}

// Page-protection changes by caller (diagnosis, 60 fps work 15/11/2026: the game's threads spent ~10 ms a frame of CPU in
// change_locked, ps5_pw_protect and ps5_mixed_protect in a race). plat_mem_protect and plat_mem_early_protect count
// by their caller's return address; the write watch's own calls by kind (1 a fault's unprotect, 2 an arm).
// perf.c logs the table every 600 frames (plat_mem_protect_stats).
#define PSTAT_N 64
static struct { uintptr_t ra; uint64_t n, ns; } pstat[PSTAT_N];
static uint64_t pstat_kind_ns[PLAT_MEM_COST_N];   // never reset: plat_mem_cost_ns (by PLAT_MEM_COST_PROT_* kind)

static void pstat_add(uintptr_t ra, uint64_t ns, int kind) {
    __atomic_add_fetch(&pstat_kind_ns[kind], ns, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pstat_kind_ns[PLAT_MEM_COST_PROT_CALLS], 1, __ATOMIC_RELAXED);
    for (uint32_t i = (uint32_t)((ra >> 4) * 0x9e3779b1u) % PSTAT_N, k = 0; k < PSTAT_N; k++, i = (i + 1) % PSTAT_N) {
        uintptr_t cur = __atomic_load_n(&pstat[i].ra, __ATOMIC_ACQUIRE);
        if (!cur) {
            uintptr_t z = 0;
            if (!__atomic_compare_exchange_n(&pstat[i].ra, &z, ra, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) cur = z;
            else cur = ra;
        }
        if (cur == ra) {
            __atomic_add_fetch(&pstat[i].n, 1, __ATOMIC_RELAXED);
            __atomic_add_fetch(&pstat[i].ns, ns, __ATOMIC_RELAXED);
            return;
        }
    }
}

void plat_mem_cost_ns(uint64_t out[PLAT_MEM_COST_N]) {
    for (int i = 0; i < PLAT_MEM_COST_N; i++) out[i] = __atomic_load_n(&pstat_kind_ns[i], __ATOMIC_RELAXED);
    out[PLAT_MEM_COST_UNPACK] = __atomic_load_n(&pk_unpack_total_ns, __ATOMIC_RELAXED);
}

void plat_mem_protect_stats(void (*emit)(uintptr_t ra, uint64_t n, uint64_t ns, void *ctx), void *ctx) {
    for (int i = 0; i < PSTAT_N; i++) {
        uintptr_t ra = __atomic_load_n(&pstat[i].ra, __ATOMIC_ACQUIRE);
        if (!ra) continue;
        uint64_t n = __atomic_exchange_n(&pstat[i].n, 0, __ATOMIC_RELAXED);
        uint64_t ns = __atomic_exchange_n(&pstat[i].ns, 0, __ATOMIC_RELAXED);
        if (n) emit(ra, n, ns, ctx);
    }
}

static int ps5_pw_protect(uint64_t addr, size_t size, int rw) {
    int prot = rw ? (PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE) : PS5_KERNEL_PROT_CPU_READ;
    if (prot_watch_has_gpu(addr, size)) prot |= PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
    uint64_t t0 = plat_now_ns();
    int ok = sceKernelMprotect((void *)(uintptr_t)addr, size, prot) == 0;
    pstat_add(rw == 3 ? 3 : rw ? 1 : 2, plat_now_ns() - t0,   // 3: a hold's unprotect (prot_watch_hold)
              rw == 3 ? PLAT_MEM_COST_PROT_HOLD : rw ? PLAT_MEM_COST_PROT_FAULT : PLAT_MEM_COST_PROT_ARM);
    return ok;
}

static int ps5_pw_state(uint64_t addr) {
    VmRange *r = find_range_lockfree(addr);
    Ps5MixedBlock *u = find_unit_lockfree(r, addr);
    if (!u || !u->backing.reserved || !u->backing.committed || !u->live_pages) return PLAT_MEM_RESERVED;
    size_t first = ((addr - u->backing.address) / PS5_MEM_PAGE) * (PS5_MEM_PAGE / PAGE);
    bool any_rw = false, any_read = false;
    for (size_t i = first; i < first + (PS5_MEM_PAGE / PAGE); ++i) {
        Ps5GuestPage p = (Ps5GuestPage)__atomic_load_n(&u->pages[i], __ATOMIC_RELAXED);
        if (p == PS5_GUEST_RW) { any_rw = true; break; }
        if (p == PS5_GUEST_READ) any_read = true;
    }
    return any_rw ? PLAT_PROT_RW : any_read ? PLAT_PROT_READ : PLAT_PROT_NONE;
}

static struct sigaction *prev_action_for(int sig) {
    switch (sig) {
    case SIGSEGV: return &prev_sigsegv;
    case SIGBUS:  return &prev_sigbus;
    case SIGILL:  return &prev_sigill;
    case SIGFPE:  return &prev_sigfpe;
    default:      return NULL;
    }
}

// Fault storms (MKW_TRACE=perf, perf.c): per frame, the 2 MiB regions the game's writes faulted in, write-watch
// unprotects (w) and unit backings (b: a packed unit's unpack, or a unit's first commit), how many hit the page right
// after the region's previous one (seq), and the instructions that wrote. The protsplit bench (28/11/2026) found the
// 1 % low's protection frames to be 130-374 fault unprotects in one frame (~20 normally) at the usual cost each, with
// unpacks in the same frames: whether they are sequential runs and the same regions every lap decides the fix.
// Lock-free and async-signal-safe (claimed slots, relaxed counts); plat_mem_storm_take clears the tables while other
// threads may still add (a diagnostic: a few counts can land in the wrong frame).
enum { STORM_REGS = 64, STORM_PCS = 32 };
static struct { uint64_t key; uint32_t n[2], seq; uint64_t last; } storm_reg[STORM_REGS];
static struct { uint64_t key; uint32_t n; } storm_pc[STORM_PCS];
static uint32_t storm_total;

static void storm_note(uint64_t addr, uint64_t pc, int kind) {
    uint64_t key = (addr >> 21) + 1, page = addr & ~(uint64_t)(PS5_MEM_PAGE - 1);
    __atomic_add_fetch(&storm_total, 1, __ATOMIC_RELAXED);
    for (uint32_t i = (uint32_t)(key * 0x9e3779b1u) % STORM_REGS, k = 0; k < STORM_REGS; k++, i = (i + 1) % STORM_REGS) {
        uint64_t cur = __atomic_load_n(&storm_reg[i].key, __ATOMIC_ACQUIRE), z = 0;
        if (!cur && __atomic_compare_exchange_n(&storm_reg[i].key, &z, key, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            cur = key;
        else if (!cur)
            cur = z;
        if (cur != key) continue;
        __atomic_add_fetch(&storm_reg[i].n[kind], 1, __ATOMIC_RELAXED);
        if (__atomic_exchange_n(&storm_reg[i].last, page, __ATOMIC_RELAXED) + PS5_MEM_PAGE == page)
            __atomic_add_fetch(&storm_reg[i].seq, 1, __ATOMIC_RELAXED);
        break;
    }
    if (!pc) return;
    for (uint32_t i = (uint32_t)((pc >> 2) * 0x9e3779b1u) % STORM_PCS, k = 0; k < STORM_PCS; k++, i = (i + 1) % STORM_PCS) {
        uint64_t cur = __atomic_load_n(&storm_pc[i].key, __ATOMIC_ACQUIRE), z = 0;
        if (!cur && __atomic_compare_exchange_n(&storm_pc[i].key, &z, pc, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            cur = pc;
        else if (!cur)
            cur = z;
        if (cur != pc) continue;
        __atomic_add_fetch(&storm_pc[i].n, 1, __ATOMIC_RELAXED);
        break;
    }
}

int plat_mem_storm_take(char *out, size_t cap, uint32_t min_faults) {
    uint32_t total = __atomic_exchange_n(&storm_total, 0, __ATOMIC_RELAXED);
    int want = total >= min_faults && cap;
    size_t o = 0;
    if (want) o += (size_t)snprintf(out, cap, "%u faults;", total);
    // the 6 busiest regions, then the 4 busiest writers, each emptied as it is read
    for (int r = 0; r < 6; r++) {
        int best = -1;
        uint32_t bn = 0;
        for (int i = 0; i < STORM_REGS; i++) {
            uint32_t n = storm_reg[i].n[0] + storm_reg[i].n[1];
            if (storm_reg[i].key && n > bn) best = i, bn = n;
        }
        if (best < 0) break;
        if (want && o < cap)
            o += (size_t)snprintf(out + o, cap - o, " %#llx w %u b %u seq %u;",
                                  (unsigned long long)((storm_reg[best].key - 1) << 21), storm_reg[best].n[0],
                                  storm_reg[best].n[1], storm_reg[best].seq);
        storm_reg[best].n[0] = storm_reg[best].n[1] = 0;
    }
    if (want && o < cap) o += (size_t)snprintf(out + o, cap - o, " writers:");
    uintptr_t base = plat_exe_base();
    for (int r = 0; r < 4; r++) {
        int best = -1;
        uint32_t bn = 0;
        for (int i = 0; i < STORM_PCS; i++)
            if (storm_pc[i].key && storm_pc[i].n > bn) best = i, bn = storm_pc[i].n;
        if (best < 0) break;
        if (want && o < cap)
            o += (size_t)snprintf(out + o, cap - o, " exe+%#llx %u", (unsigned long long)(storm_pc[best].key - base), bn);
        storm_pc[best].n = 0;
    }
    for (int i = 0; i < STORM_REGS; i++) storm_reg[i].key = 0, storm_reg[i].n[0] = storm_reg[i].n[1] = 0, storm_reg[i].seq = 0, storm_reg[i].last = 0;
    for (int i = 0; i < STORM_PCS; i++) storm_pc[i].key = 0, storm_pc[i].n = 0;
    if (o >= cap && cap) out[cap - 1] = 0;
    return want;
}

static void fault_handler(int sig, siginfo_t *si, void *uc) {
    ucontext_t *u = (ucontext_t *)uc;
#ifdef __PROSPERO__
    uint64_t rip = (uint64_t)u->uc_mcontext.mc_rip;
    uint64_t rsp = (uint64_t)u->uc_mcontext.mc_rsp;
    uint64_t err = (uint64_t)u->uc_mcontext.mc_err;
#elif defined(__linux__)
    uint64_t rip = (uint64_t)u->uc_mcontext.gregs[REG_RIP];
    uint64_t rsp = (uint64_t)u->uc_mcontext.gregs[REG_RSP];
    uint64_t err = (uint64_t)u->uc_mcontext.gregs[REG_ERR];
#else
    (void)u;
    uint64_t rip = 0, rsp = 0, err = 0;
#endif
    uint64_t addr = si ? (uint64_t)(uintptr_t)si->si_addr : 0;
    bool is_write = (err & 2) != 0;
    bool is_exec = (err & 16) != 0;

    bool back_failed = false;
    // A protect batch open on this thread (change) must not swallow the protections this handler makes.
    Ps5ProtBatch *batch = ps5_mem_batch_swap(NULL);
    if (sig == SIGSEGV || sig == SIGBUS) {
        VmRange *r = find_range_lockfree(addr);
        Ps5MixedBlock *unit = find_unit_lockfree(r, addr);
        if (unit) {
            unit_lock(unit);
            bool committed = unit->backing.reserved && unit->live_pages != 0;
            Ps5GuestPage st = ps5_mixed_state(unit, addr);
            bool perm_ok = is_write ? (st == PS5_GUEST_RW) : (st >= PS5_GUEST_READ);
            if (committed && perm_ok && !unit->backing.committed) {
                bool backed = back_unit_locked(unit, r->kind, 1, is_write ? PLAT_BACK_WRITE_FAULT : PLAT_BACK_READ_FAULT);
                storm_note(addr, rip, 1);
                // An armed page is read-only again (back_unit_locked): the retry faults once more and prot_watch_fault
                // takes it as any watched write. Marked here, it would stay read-only with nothing to unprotect it.
                uint64_t pg = addr & ~(uint64_t)(PS5_MEM_PAGE - 1);
                if (backed && is_write && !prot_watch_is_armed(pg)) prot_watch_mark_written(pg, PS5_MEM_PAGE);
                unit_unlock(unit);
                if (backed) { ps5_mem_batch_swap(batch); return; } // Retry the faulting access now that the unit is backed.
                // Backing a committed guest unit failed (OOM / VM map limit). Never fall through to
                // thaw_fault (which sees an RW guest page and returns 1, causing an infinite fault loop).
                back_failed = true;
            } else if (committed && perm_ok && unit->backing.committed) {
                if (prot_watch_is_early(addr)) {
                    unit_unlock(unit);
                } else if (is_write && prot_watch_fault(addr)) {
                    unit_unlock(unit);
                    storm_note(addr, rip, 0);
                    ps5_mem_batch_swap(batch);
                    return;
                } else {
                    unit_unlock(unit);
                }
            } else {
                unit_unlock(unit);
            }
        }
    }

    if (sig == SIGSEGV || sig == SIGBUS) plat_fault_guarded();   // a guarded copy (fault_guard.c): no hook, back to it
    plat_fault_write = is_write ? 1 : 0;
    plat_fault_pc = rip;
    plat_fault_ret = (rsp && !(rsp & 7)) ? *(const uint64_t *)(uintptr_t)rsp : 0;

    if (!back_failed && (sig == SIGSEGV || sig == SIGBUS)) {
        int (*fault_fn)(uint64_t) = __atomic_load_n(&g_fault_hook, __ATOMIC_ACQUIRE);
        plat_fault_uc = uc;   // (plat_fault_emulate_load)
        int retry = fault_fn && fault_fn(addr);
        plat_fault_uc = NULL;
        if (retry) {
#ifdef __PROSPERO__
            plat_fault_step_arm((uint64_t *)&u->uc_mcontext.mc_rflags);   // (the hook asked for one instruction)
#elif defined(__linux__)
            plat_fault_step_arm((uint64_t *)&u->uc_mcontext.gregs[REG_EFL]);
#endif
            ps5_mem_batch_swap(batch);
            return;
        }
        int (*watch_fn)(uint64_t, uint64_t) = __atomic_load_n(&g_watch_hook, __ATOMIC_ACQUIRE);
        if (is_write && watch_fn && watch_fn(addr, rip)) { ps5_mem_batch_swap(batch); return; }
    }

    void (*crash_fn)(const PlatCrash *) = __atomic_load_n(&g_crash_hook, __ATOMIC_ACQUIRE);
    if (crash_fn) {
        PlatCrash c = {
            .what = back_failed ? "direct memory backing failed" :
                    sig == SIGSEGV ? "access violation" :
                    sig == SIGBUS  ? "bus error" :
                    sig == SIGILL  ? "illegal instruction" : "arithmetic exception",
            .code = (uint32_t)sig,
            .rip = rip,
            .offset = rip,   // the eboot's load base is not known here: report the absolute rip
            .access = -1,
            .ret = plat_fault_ret,
            .module = "no module",
        };
        if (sig == SIGSEGV || sig == SIGBUS) {
            c.access = is_exec ? 8 : is_write ? 1 : 0;
            c.addr = addr;
        }
        crash_fn(&c);
    }
    if (back_failed) {
        fflush(NULL);
        _exit(1);
    }

    struct sigaction *prev = prev_action_for(sig);
    if (prev) sigaction(sig, prev, NULL);
    else signal(sig, SIG_DFL);
}

// The debug trap after an instruction the fault hook let through (plat_fault_step); any other SIGTRAP as by default.
static void trap_handler(int sig, siginfo_t *si, void *uc) {
    (void)si;
    ucontext_t *u = (ucontext_t *)uc;
#ifdef __PROSPERO__
    if (plat_fault_step_trap((uint64_t *)&u->uc_mcontext.mc_rflags)) return;
#elif defined(__linux__)
    if (plat_fault_step_trap((uint64_t *)&u->uc_mcontext.gregs[REG_EFL])) return;
#else
    (void)u;
#endif
    signal(sig, SIG_DFL);
    raise(sig);
}

static void ensure_alt_stack(void) {
    stack_t cur = {0};
    if (sigaltstack(NULL, &cur) == 0 && !(cur.ss_flags & SS_DISABLE) && cur.ss_size >= ALT_STACK_BYTES) return;
    void *mem = malloc(ALT_STACK_BYTES);
    if (!mem) return;
    stack_t ss = {.ss_sp = mem, .ss_size = ALT_STACK_BYTES};
    if (sigaltstack(&ss, NULL) != 0) free(mem);
}

static void install_signal_handlers(void) {
    prot_watch_init(PS5_MEM_PAGE, ps5_pw_protect, ps5_pw_state);
    struct sigaction sa = {0};
    sa.sa_sigaction = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigaction(SIGSEGV, &sa, &prev_sigsegv);
    sigaction(SIGBUS, &sa, &prev_sigbus);
    sigaction(SIGILL, &sa, &prev_sigill);
    sigaction(SIGFPE, &sa, &prev_sigfpe);
    struct sigaction st = {0};
    st.sa_sigaction = trap_handler;
    sigemptyset(&st.sa_mask);
    st.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    if (!sigaction(SIGTRAP, &st, NULL)) plat_fault_step_init();
}

static void fault_install(void) {
    ensure_alt_stack();
    pthread_once(&fault_once, install_signal_handlers);
}

static bool pending_overlap(uint64_t address, uint64_t end) {
    for (const VmRange *range = cleanup_pending; range; range = range->next)
        if (address < range->end && range->base < end) return true;
    return false;
}

static void discard_unexpected(VmRange *range, void *address, size_t bytes) {
    if (address && !sdk_ok("release unexpected mapping", (uintptr_t)address, bytes,
                           sceKernelMunmap(address, bytes))) {
        *range = (VmRange){.base = (uintptr_t)address, .end = (uintptr_t)address + bytes,
                           .next = cleanup_pending};
        cleanup_pending = range;
    } else free(range);
}

int plat_mem_reserve(uint64_t address, uint64_t bytes, int watch) {
    fault_install();
    if (!address || !bytes || (address | bytes) % PS5_MEM_UNIT ||
        address >= address_limit || bytes > address_limit - address) return 0;
    size_t chunk_count = (size_t)((bytes + CHUNK_SIZE - 1) >> CHUNK_SHIFT);
    VmRange *range = calloc(1, sizeof *range);
    Ps5MixedBlock *volatile *chunks = calloc(chunk_count, sizeof *chunks);
    if (!range || !chunks) {
        free((void *)chunks);
        free(range);
        return 0;
    }
    RangeKind kind = RANGE_OTHER;
    if (bytes == (1ull << 36)) kind = RANGE_ALIAS;
    else if (watch && bytes >= (4ull << 30)) kind = RANGE_HEAP;

    pthread_mutex_lock(&vm_lock);
    VmRange **slot = range_slot(address);
    bool exact = false;
    void *unexpected = NULL;
    if (!stopped && (!*slot || (*slot)->base >= address + bytes) && !pending_overlap(address, address + bytes)) {
        void *placed = (void *)(uintptr_t)address;
        int result = sceKernelReserveVirtualRange(&placed, bytes, 0, PS5_MEM_UNIT);
        if (sdk_ok("reserve", address, bytes, result)) {
            exact = placed == (void *)(uintptr_t)address;
            if (!exact) {
                fprintf(stderr, "PS5 memory reserve %#llx relocated to %p\n", (unsigned long long)address, placed);
                unexpected = placed;
            }
        }
    }
    if (exact) {
        *range = (VmRange){
            .base = address,
            .end = address + bytes,
            .chunks = chunks,
            .chunk_count = chunk_count,
            .kind = kind,
            .next = *slot,
        };
        __atomic_store_n(slot, range, __ATOMIC_RELEASE);
        if (watch) prot_watch_reserve(address, bytes);
    } else {
        free((void *)chunks);
        discard_unexpected(range, unexpected, bytes);
    }
    pthread_mutex_unlock(&vm_lock);
    return exact;
}

void *plat_mem_alloc(size_t bytes) {
    if (!bytes || bytes > SIZE_MAX - (PS5_MEM_PAGE - 1)) return NULL;
    bytes = (bytes + PS5_MEM_PAGE - 1) & ~(size_t)(PS5_MEM_PAGE - 1);
    VmRange *range = calloc(1, sizeof *range);
    if (!range) return NULL;
    pthread_mutex_lock(&vm_lock);
    void *mapped = stopped ? MAP_FAILED :
                   mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
        uint64_t base = (uintptr_t)mapped;
        VmRange **slot = range_slot(base);
        // A hint-free allocation could land in a hole left by a lost guest
        // slot. Keep that address unavailable to either allocator.
        if ((*slot && (*slot)->base < base + bytes) || pending_overlap(base, base + bytes)) {
            discard_unexpected(range, mapped, bytes);
            mapped = MAP_FAILED;
        } else {
            *range = (VmRange){.base = base, .end = base + bytes, .host = true, .next = *slot};
            __atomic_store_n(slot, range, __ATOMIC_RELEASE);
        }
    } else free(range);
    pthread_mutex_unlock(&vm_lock);
    return mapped == MAP_FAILED ? NULL : mapped;
}

// Caller must hold vm_lock.
static Ps5MixedBlock *find_unit(VmRange *range, uint64_t address, bool create) {
    uint64_t base = address & ~(uint64_t)(PS5_MEM_UNIT - 1);
    size_t cidx = (size_t)((base - range->base) >> CHUNK_SHIFT);
    Ps5MixedBlock *chunk = __atomic_load_n(&range->chunks[cidx], __ATOMIC_ACQUIRE);
    if (!chunk) {
        if (!create) return NULL;
        chunk = calloc(CHUNK_UNITS, sizeof *chunk);
        if (!chunk) return NULL;
        uint64_t chunk_base = range->base + ((uint64_t)cidx << CHUNK_SHIFT);
        for (size_t i = 0; i < CHUNK_UNITS; ++i) {
            uint64_t uaddr = chunk_base + (uint64_t)i * PS5_MEM_UNIT;
            if (uaddr >= range->end) break;
            // The full window owns this unit. There is no separate reservation call.
            chunk[i].backing = (Ps5MemBlock){.address = uaddr, .reserved = true};
        }
        __atomic_store_n(&range->chunks[cidx], chunk, __ATOMIC_RELEASE);
    }
    size_t uidx = (size_t)(((base - range->base) & (CHUNK_SIZE - 1)) / PS5_MEM_UNIT);
    return &chunk[uidx];
}

typedef enum { COMMIT, DECOMMIT, PROTECT } Change;
static bool change(uint64_t address, uint64_t bytes, Change action, int protection) {
    if (!bytes || (address | bytes) % PAGE || bytes > UINT64_MAX - address) return false;
    uint64_t end = address + bytes;
    VmRange *range = *range_slot(address);
    if (!range || address < range->base || end > range->end) return false;
    if (range->host) {
        // The dispatch table is native-page aligned. A partial native page
        // may contain other live host data.
        if (action != PROTECT || (address | bytes) % PS5_MEM_PAGE) return false;
        const int flags[] = {0, PS5_KERNEL_PROT_CPU_READ, PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE};
        return sdk_ok("protect host", address, bytes,
                      sceKernelMprotect((void *)(uintptr_t)address, bytes, flags[protection]));
    }

    // Reject invalid spans before making any changes.
    if ((action == DECOMMIT || (action == PROTECT && protection == PLAT_PROT_NONE)) &&
        prot_watch_has_gpu(address, bytes))
        return false;
    // Each page's state under its unit's lock: a release of the unit's backing (plat_mem_reclaim_zeros, of frozen
    // units only, or plat_mem_release_*) unmaps it and reserves it again under that lock, and in between the page
    // reads FREE. Read without it, a frozen region's thaw found such a page and was refused ("memory: thawing ...
    // failed", 15/11/2026, the governor's zero reclaim running).
    for (uint64_t at = address; at < end; at += PAGE) {
        Ps5MixedBlock *unit = find_unit(range, at, false);
        if (unit) unit_lock(unit);
        Ps5GuestPage state = unit ? ps5_mixed_state(unit, at) : PS5_GUEST_RESERVED;
        if (unit) unit_unlock(unit);
        if (state == PS5_GUEST_FREE || (action == PROTECT && state == PS5_GUEST_RESERVED)) {
            fprintf(stderr, "PS5 memory %s %#llx+%#llx refused: page %#llx is %s\n",
                    action == COMMIT ? "commit" : action == DECOMMIT ? "decommit" : "protect",
                    (unsigned long long)address, (unsigned long long)bytes, (unsigned long long)at,
                    state == PS5_GUEST_FREE ? "free (no reservation)" : "reserved (not committed)");
            return false;
        }
    }

    // Commits stay lazy per 64 KiB unit regardless of span size: MKW's
    // nninitStartup commits the entire 8 GiB heap in one 2 MiB-aligned call,
    // so eager backing of >= 2 MiB spans would exhaust direct memory immediately.
    // PROTECT: one sceKernelMprotect per run of native pages left in one mode (ps5_mem_batch_begin), the units held
    // locked, in address order, until the flush. Nothing else holds two unit locks.
    Ps5ProtBatch batch;
    Ps5MixedBlock *held[64];
    int nheld = 0, batching = action == PROTECT;
    if (batching) ps5_mem_batch_begin(&batch);
#define CHANGE_RELEASE()                                                                                               \
    do {                                                                                                               \
        int berr_ = batching ? ps5_mem_batch_flush() : 0;                                                              \
        for (int k_ = 0; k_ < nheld; k_++) unit_unlock(held[k_]);                                                      \
        nheld = 0;                                                                                                     \
        if (berr_) {                                                                                                   \
            sdk_ok("protect", address, bytes, berr_);                                                                  \
            ok_all = false;                                                                                            \
        }                                                                                                              \
    } while (0)
    bool ok_all = true;
    for (uint64_t at = address; at < end; at += PAGE) {
        Ps5MixedBlock *unit = find_unit(range, at, action == COMMIT);
        if (!unit) {
            if (action == DECOMMIT) continue;
            if (batching) { CHANGE_RELEASE(); ps5_mem_batch_end(); }
            return false;
        }
        if (!batching) {
            unit_lock(unit);
        } else if (!nheld || held[nheld - 1] != unit) {
            if (nheld == (int)(sizeof held / sizeof *held)) CHANGE_RELEASE();
            if (!ok_all) { ps5_mem_batch_end(); return false; }
            unit_lock(unit);
            held[nheld++] = unit;
        }
        const size_t offset = at - unit->backing.address;
        Ps5GuestPage state = ps5_mixed_state(unit, at);
        bool was_committed = unit->backing.committed;
        int result;
        if (action == DECOMMIT)
            result = state == PS5_GUEST_RESERVED ? 0 : ps5_mixed_decommit(unit, offset);
        else if (state == PS5_GUEST_RESERVED)
            result = ps5_mixed_commit_lazy(unit, offset);
        else
            result = ps5_mixed_protect(unit, offset, (Ps5GuestPage)(PS5_GUEST_NONE + protection));
        if (!result) {
            if (action == DECOMMIT && state != PS5_GUEST_RESERVED)
                prot_watch_mark_written(at & ~(uint64_t)(PS5_MEM_PAGE - 1), PS5_MEM_PAGE);
            else if (action == PROTECT && protection == PLAT_PROT_RW && unit->backing.committed &&
                     prot_watch_is_armed(at))
                result = ps5_mem_protect_native(&unit->backing,
                                                (at - unit->backing.address) & ~(uint64_t)(PS5_MEM_PAGE - 1),
                                                PS5_MEM_PAGE, PS5_MEM_READ);
        }
        if (action == DECOMMIT && was_committed && !unit->backing.committed)
            __atomic_sub_fetch(&backed_units[range->kind], 1, __ATOMIC_RELAXED);
        // A packed unit (MKW_PACK): a page decommitted now reads zero when committed again, so its packed bytes must
        // not come back; and a unit with no page left has nothing to unpack.
        if (!result && unit->pack && state != PS5_GUEST_RESERVED && action == DECOMMIT) {
            PackBlob *b = unit->pack;
            b->zmask |= 1u << (offset / PAGE);
            if (!unit->live_pages) {
                unit->pack = NULL;
                __atomic_sub_fetch(&packed_units[range->kind], 1, __ATOMIC_RELAXED);
                __atomic_sub_fetch(&pk_packed_bytes, b->csize, __ATOMIC_RELAXED);
                pk_drop(b);
            }
        }
        if (!batching) unit_unlock(unit);
        if (!sdk_ok(action == COMMIT ? "commit" : action == DECOMMIT ? "decommit" : "protect", at, PAGE, result)) {
            if (batching) { CHANGE_RELEASE(); ps5_mem_batch_end(); }
            return false;
        }
    }
    if (batching) { CHANGE_RELEASE(); ps5_mem_batch_end(); }
#undef CHANGE_RELEASE
    return ok_all;
}

static int change_locked(uint64_t address, uint64_t bytes, Change action, int protection) {
    pthread_mutex_lock(&vm_lock);
    bool ok = change(address, bytes, action, protection);
    pthread_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_commit(uint64_t address, size_t bytes) { return change_locked(address, bytes, COMMIT, PLAT_PROT_RW); }
int plat_mem_decommit(uint64_t address, uint64_t bytes) { return change_locked(address, bytes, DECOMMIT, 0); }
__attribute__((noinline)) int plat_mem_protect(uint64_t address, uint64_t bytes, int protection) {
    uint64_t t0 = plat_now_ns();
    int ok = protection >= PLAT_PROT_NONE && protection <= PLAT_PROT_RW &&
             change_locked(address, bytes, PROTECT, protection);
    pstat_add((uintptr_t)__builtin_return_address(0), plat_now_ns() - t0, PLAT_MEM_COST_PROT_CHANGE);
    return ok;
}

size_t plat_mem_early_page_size(void) { return PS5_MEM_PAGE; }
size_t plat_mem_watch_page(void) { return PS5_MEM_PAGE; }   // prot_watch_init(PS5_MEM_PAGE, ...)

static PlatEarlyResult early_protect_failed(uint64_t lo, uint64_t restore_hi, int on) {
    if (!on) return PLAT_EARLY_UNRESTORED;
    if (restore_hi > lo &&
        plat_mem_early_protect(lo, (size_t)(restore_hi - lo), 0, 0) != PLAT_EARLY_OK)
        return PLAT_EARLY_UNRESTORED;
    return PLAT_EARLY_FAILED;
}

// Mark a run before arming to keep write-watch from changing the protection. Keep it marked until a restore succeeds.
static int early_protect_run(uint64_t lo, uint64_t hi, int prot, int on) {
    if (on)
        for (uint64_t p = lo; p < hi; p += PS5_MEM_PAGE) prot_watch_set_early(p, 1);
    if (sceKernelMprotect((void *)(uintptr_t)lo, (size_t)(hi - lo), prot) != 0) return 0;
    if (!on)
        for (uint64_t p = lo; p < hi; p += PS5_MEM_PAGE) prot_watch_set_early(p, 0);
    return 1;
}

static PlatEarlyResult early_protect(uint64_t addr, size_t n, int on, int restore_prot) {
    (void)restore_prot;
    if (!n || n > UINT64_MAX - (PS5_MEM_PAGE - 1) || addr > UINT64_MAX - n - (PS5_MEM_PAGE - 1))
        return on ? PLAT_EARLY_FAILED : PLAT_EARLY_UNRESTORED;
    uint64_t lo = addr & ~(uint64_t)(PS5_MEM_PAGE - 1);
    uint64_t hi = (addr + n + PS5_MEM_PAGE - 1) & ~(uint64_t)(PS5_MEM_PAGE - 1);
    // A run of equal protection goes across 64 KiB units in one sceKernelMprotect (~44 us each on the PS5): the
    // early-read watch's arms (memory_early.c flush_watch_run) were one call a unit, 85 us each, ~1.2 ms a frame in a
    // race (15/11/2026). The run's units stay locked, in address order, until it is made; a failure unlocks them all
    // before early_protect_failed, which locks units itself.
    Ps5MixedBlock *held[64];
    int nheld = 0;
    uint64_t run_lo = 0, run_hi = 0;
    int run_prot = -1;
#define EP_UNLOCK_ALL() do { for (int k_ = 0; k_ < nheld; k_++) unit_unlock(held[k_]); nheld = 0; } while (0)
#define EP_FLUSH() (run_lo < run_hi ? early_protect_run(run_lo, run_hi, run_prot, on) : 1)
    for (uint64_t p = lo; p < hi;) {
        VmRange *r = find_range_lockfree(p);
        Ps5MixedBlock *u = find_unit_lockfree(r, p);
        uint64_t u_end = (p & ~(uint64_t)(PS5_MEM_UNIT - 1)) + PS5_MEM_UNIT;
        if (u_end > hi) u_end = hi;
        if (!u || nheld == (int)(sizeof held / sizeof *held)) {   // the run ends here: made, then its units unlocked
            int ok = EP_FLUSH();
            uint64_t made_hi = run_hi;
            run_lo = run_hi = 0;
            EP_UNLOCK_ALL();
            if (!ok) return early_protect_failed(lo, made_hi, on);
            if (!u) {
                if (on) return early_protect_failed(lo, p, on);
                p = u_end;
                continue;
            }
        }
        unit_lock(u);
        held[nheld++] = u;
        if (!u->backing.reserved || !u->live_pages) {
            int ok = EP_FLUSH();
            uint64_t made_hi = run_hi;
            run_lo = run_hi = 0;
            if (!on)
                for (uint64_t q = p; q < u_end; q += PS5_MEM_PAGE) prot_watch_set_early(q, 0);
            EP_UNLOCK_ALL();
            if (!ok) return early_protect_failed(lo, made_hi, on);
            if (on) return early_protect_failed(lo, p, on);
            p = u_end;
            continue;
        }
        for (; p < u_end; p += PS5_MEM_PAGE) {
            size_t first = ((p - u->backing.address) / PS5_MEM_PAGE) * (PS5_MEM_PAGE / PAGE);
            bool any_rw = false, any_read = false;
            for (size_t i = first; i < first + (PS5_MEM_PAGE / PAGE); ++i) {
                Ps5GuestPage st = (Ps5GuestPage)__atomic_load_n(&u->pages[i], __ATOMIC_RELAXED);
                if (st == PS5_GUEST_RW) { any_rw = true; break; }
                if (st == PS5_GUEST_READ) any_read = true;
            }
            if (on && !any_rw && !any_read) {
                int ok = EP_FLUSH();
                uint64_t made_hi = run_lo < run_hi ? run_hi : p;
                EP_UNLOCK_ALL();
                (void)ok;
                return early_protect_failed(lo, made_hi, on);
            }
            if (!u->backing.committed) {   // no mapping to protect: the run stops before it
                if (!EP_FLUSH()) {
                    uint64_t made_hi = run_hi;
                    EP_UNLOCK_ALL();
                    return early_protect_failed(lo, made_hi, on);
                }
                run_lo = run_hi = 0;
                prot_watch_set_early(p, on != 0);
                continue;
            }
            int prot;
            if (on) {
                prot = prot_watch_has_gpu(p, PS5_MEM_PAGE)
                           ? (PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE)
                           : 0;
            } else {
                int rw = any_rw && !prot_watch_is_armed(p);
                prot = rw ? (PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE)
                          : (any_read || any_rw) ? PS5_KERNEL_PROT_CPU_READ
                                                 : 0;
                if ((any_read || any_rw) && prot_watch_has_gpu(p, PS5_MEM_PAGE))
                    prot |= PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
            }
            if (run_lo < run_hi && (prot != run_prot || run_hi != p)) {
                if (!early_protect_run(run_lo, run_hi, run_prot, on)) {
                    uint64_t made_hi = run_hi;
                    EP_UNLOCK_ALL();
                    return early_protect_failed(lo, made_hi, on);
                }
                run_lo = run_hi = 0;
                // The units before u held only that run: they can go (u may hold the next one).
                for (int k = 0; k < nheld - 1; k++) unit_unlock(held[k]);
                held[0] = held[nheld - 1], nheld = 1;
            }
            if (!run_hi) {
                run_lo = p;
                run_prot = prot;
            }
            run_hi = p + PS5_MEM_PAGE;
        }
    }
    {
        int ok = EP_FLUSH();
        uint64_t made_hi = run_hi;
        EP_UNLOCK_ALL();
        if (!ok) return early_protect_failed(lo, made_hi, on);
    }
#undef EP_FLUSH
#undef EP_UNLOCK_ALL
    return PLAT_EARLY_OK;
}

// Counted per native page, hashed: memory_early.c's page snapshots are good only while their page was not opened.
#define EARLY_GEN_N 64
static uint32_t early_gen[EARLY_GEN_N];

uint32_t plat_mem_early_gen(uint64_t addr) {
    return __atomic_load_n(&early_gen[(addr / PS5_MEM_PAGE) % EARLY_GEN_N], __ATOMIC_ACQUIRE);
}

__attribute__((noinline)) PlatEarlyResult plat_mem_early_protect(uint64_t addr, size_t n, int on, int restore_prot) {
    uint64_t t0 = plat_now_ns();
    PlatEarlyResult r = early_protect(addr, n, on, restore_prot);
    uint64_t first = addr / PS5_MEM_PAGE, count = n ? (addr + n - 1) / PS5_MEM_PAGE - first + 1 : 0;
    if (count > EARLY_GEN_N) first = 0, count = EARLY_GEN_N;
    for (uint64_t i = 0; i < count; i++)
        __atomic_add_fetch(&early_gen[(first + i) % EARLY_GEN_N], 1, __ATOMIC_RELEASE);
    pstat_add((uintptr_t)__builtin_return_address(0), plat_now_ns() - t0, PLAT_MEM_COST_PROT_EARLY);
    return r;
}

__attribute__((noinline)) int plat_mem_early_open_read(uint64_t addr, size_t n) {
    if (n != PS5_MEM_PAGE || addr & (PS5_MEM_PAGE - 1)) return 0;
    uint64_t t0 = plat_now_ns();
    Ps5MixedBlock *u = find_unit_lockfree(find_range_lockfree(addr), addr);
    if (!u) return 0;
    unit_lock(u);
    int ok = u->backing.reserved && u->live_pages && u->backing.committed;
    if (ok) {
        int prot = PS5_KERNEL_PROT_CPU_READ;
        if (prot_watch_has_gpu(addr, PS5_MEM_PAGE)) prot |= PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
        ok = sceKernelMprotect((void *)(uintptr_t)addr, n, prot) == 0;
    }
    unit_unlock(u);
    pstat_add((uintptr_t)__builtin_return_address(0), plat_now_ns() - t0, PLAT_MEM_COST_PROT_EARLY);
    return ok;
}

int plat_mem_back_hold_as(uint64_t addr, size_t n, int cause) {
    if (!n) return 1;
    if (n > UINT64_MAX - addr) return 0;
    void (*eh)(uint64_t, size_t) = __atomic_load_n(&g_early_hold_hook, __ATOMIC_ACQUIRE);
    if (eh) eh(addr, n);
    uint64_t end = addr + n;
    uint64_t at = addr & ~(uint64_t)(PS5_MEM_UNIT - 1);
    while (at < end) {
        VmRange *r = find_range_lockfree(at);
        if (!r || r->host) {
            if (r && r->end > at) {
                at = (r->end + PS5_MEM_UNIT - 1) & ~(uint64_t)(PS5_MEM_UNIT - 1);
                continue;
            }
            at += PS5_MEM_UNIT;
            continue;
        }
        Ps5MixedBlock *u = find_unit_lockfree(r, at);
        if (u) {
            unit_lock(u);
            bool ok = true;
            if (u->backing.reserved && u->live_pages && !u->backing.committed)
                ok = back_unit_locked(u, r->kind, 0, cause);
            unit_unlock(u);
            if (!ok) return 0;
            at += PS5_MEM_UNIT;
        } else {
            // Skip the entire unallocated 16 MiB chunk.
            uint64_t next_chunk = r->base + (((at - r->base) >> CHUNK_SHIFT) + 1) * (uint64_t)CHUNK_SIZE;
            at = next_chunk < r->end ? next_chunk : r->end;
        }
    }
    prot_watch_hold(addr, n);
    return 1;
}

static void (*thaw_fn)(uint64_t addr, size_t n);
void plat_mem_thaw_hook(void (*fn)(uint64_t addr, size_t n)) { __atomic_store_n(&thaw_fn, fn, __ATOMIC_RELEASE); }

void plat_mem_early_resolve(uint64_t addr, size_t n) {
    if (!n) return;
    void (*eh)(uint64_t, size_t) = __atomic_load_n(&g_early_hold_hook, __ATOMIC_ACQUIRE);
    if (eh) eh(addr, n);
    void (*th)(uint64_t, size_t) = __atomic_load_n(&thaw_fn, __ATOMIC_ACQUIRE);
    if (th) th(addr, n);
}

int plat_mem_back_hold(uint64_t addr, size_t n) {
    return plat_mem_back_hold_as(addr, n, PLAT_BACK_RT_READ);
}

void plat_mem_back_release(uint64_t addr, size_t n) {
    if (n) prot_watch_release(addr, n);
}

int plat_mem_watch_info(uint64_t addr) { return prot_watch_page_info(addr); }
void plat_mem_gpu_import(uint64_t addr, size_t n, int on) { prot_watch_gpu_import(addr, n, on); }

int plat_mem_back(uint64_t addr, size_t n) {
    if (!plat_mem_back_hold(addr, n)) return 0;
    plat_mem_back_release(addr, n);
    return 1;
}

static int page_state(VmRange *range, uint64_t address, uint64_t *end) {
    if (!range || address < range->base) {
        *end = range ? range->base : UINT64_MAX;
        return PLAT_MEM_FREE;
    }
    if (range->host) {
        void *start = NULL, *finish = NULL;
        uint32_t flags = 0;
        if (sceKernelQueryMemoryProtection((void *)(uintptr_t)address, &start, &finish, &flags) ||
            (uintptr_t)start > address || (uintptr_t)finish <= address) {
            *end = range->end;
            return PLAT_MEM_FREE;
        }
        *end = (uintptr_t)finish < range->end ? (uintptr_t)finish : range->end;
        if (!(flags & PS5_KERNEL_PROT_CPU_READ)) return PLAT_PROT_NONE;
        return flags & PS5_KERNEL_PROT_CPU_WRITE ? PLAT_PROT_RW : PLAT_PROT_READ;
    }
    const uint64_t base = address & ~(uint64_t)(PS5_MEM_UNIT - 1);
    size_t cidx = (size_t)((base - range->base) >> CHUNK_SHIFT);
    Ps5MixedBlock *chunk = range->chunks[cidx];
    if (!chunk) {
        size_t next_c = cidx + 1;
        while (next_c < range->chunk_count && !range->chunks[next_c]) ++next_c;
        uint64_t next_addr = range->base + ((uint64_t)next_c << CHUNK_SHIFT);
        *end = next_addr < range->end ? next_addr : range->end;
        return PLAT_MEM_RESERVED;
    }
    const Ps5MixedBlock *unit = &chunk[((base - range->base) & (CHUNK_SIZE - 1)) / PS5_MEM_UNIT];
    if (!unit->backing.reserved || !unit->live_pages) {
        *end = base + PS5_MEM_UNIT;
        return unit->backing.reserved ? PLAT_MEM_RESERVED : PLAT_MEM_FREE;
    }
    *end = (address | (PAGE - 1)) + 1;
    return ps5_mixed_state(unit, address) - PS5_GUEST_NONE;
}

static int state_locked(uint64_t address, uint64_t limit, uint64_t *end) {
    VmRange *range = *range_slot(address);
    int state = page_state(range, address, end);
    // Adjacent records or windows can have the same logical state.
    while (*end < limit) {
        if (range && *end >= range->end) range = range->next;
        uint64_t next;
        if (page_state(range, *end, &next) != state) break;
        *end = next;
    }
    if (*end > limit) *end = limit;
    return state;
}

int plat_mem_state(uint64_t address, uint64_t limit, uint64_t *end) {
    pthread_mutex_lock(&vm_lock);
    int state = state_locked(address, limit, end);
    pthread_mutex_unlock(&vm_lock);
    return state;
}

static bool readable_locked(uint64_t address, uint64_t bytes) {
    if (bytes > UINT64_MAX - address) return false;
    for (uint64_t at = address, end = address + bytes, next; at < end; at = next)
        if (state_locked(at, end, &next) < PLAT_PROT_READ) return false;
    return true;
}

int plat_mem_readable(uint64_t address, uint64_t bytes) {
    pthread_mutex_lock(&vm_lock);
    bool ok = readable_locked(address, bytes ? bytes : 1);
    pthread_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_read(uint64_t address, void *out, size_t bytes) {
    if (!bytes) return 1;
    if (!out) return 0;
    pthread_mutex_lock(&vm_lock);
    bool ok = readable_locked(address, bytes);
    pthread_mutex_unlock(&vm_lock);
    if (!ok || !plat_mem_back(address, bytes)) return 0;
    pthread_mutex_lock(&vm_lock);
    ok = readable_locked(address, bytes);
    // Guarded: a page the early-read watch protects faults here with vm_lock held, and the fault hook's frz_lock is
    // held by mem_early_watch waiting for vm_lock (run 25's hang, fault_guard.c). The copy fails instead; mem_copy_out
    // then reads it the plain way, holding no platform lock.
    if (ok) ok = plat_mem_copy_guarded(out, (const void *)(uintptr_t)address, bytes);
    pthread_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_peek(void *out, uint64_t address, size_t bytes) { return plat_mem_read(address, out, bytes); }
void *plat_aligned_alloc(size_t size, size_t alignment) {
    void *p = NULL;
    return posix_memalign(&p, alignment, size) ? NULL : p;
}
void plat_aligned_free(void *p) { free(p); }

int plat_mem_poke(uint64_t address, const void *src, size_t bytes) {
    (void)address; (void)src; (void)bytes;
    return 0; // Protected write-through is unavailable on this platform.
}
int plat_mem_replace_anon(uint64_t address, uint64_t bytes) {
    (void)address; (void)bytes;
    return 0; // The anonymous replacement path is Linux-only.
}
uint64_t plat_mem_written(uint64_t address, uint64_t bytes, uint64_t *pages, uint64_t max, int reset) {
    static int pt_reset_logged;
    int did_protect = 0;
    uint64_t got = prot_watch_written(address, bytes, pages, max, reset, &did_protect);
    if (did_protect && !__atomic_exchange_n(&pt_reset_logged, 1, __ATOMIC_ACQ_REL)) {
        int cpu_tot = 0, cpu_free = 0, gpu_tot = 0, gpu_free = 0;
        if (sceKernelGetPageTableStats(&cpu_tot, &cpu_free, &gpu_tot, &gpu_free) == 0)
            fprintf(stderr, "PS5 memory write-watch first reset: pt=cpu:%d/%d gpu:%d/%d\n",
                    cpu_free, cpu_tot, gpu_free, gpu_tot);
    }
    return got;
}

// MKW_SNAP_SCOPED (memory.c mem_written_any): a peek of the write watch, nothing protected (prot_watch_dirty).
int plat_mem_written_any(uint64_t address, uint64_t bytes) { return prot_watch_dirty(address, bytes); }
uint64_t plat_mem_write_watch_faults(int reset) { return prot_watch_faults(reset); }
uint64_t plat_mem_write_watch_repeats(int reset) { return prot_watch_repeat_faults(reset); }
uint64_t plat_mem_write_watch_hot_skips(int reset) { return prot_watch_hot_skips(reset); }

uint64_t plat_mem_populated(uint64_t address, uint64_t bytes, uint64_t (*runs)[2], uint64_t max) {
    if (!max || !bytes) return 0;
    if (bytes > UINT64_MAX - address) return UINT64_MAX;
    uint64_t lo = address & ~(uint64_t)(PAGE - 1);
    uint64_t hi = (address + bytes + PAGE - 1) & ~(uint64_t)(PAGE - 1);
    uint64_t got = 0;
    uint64_t at = lo;
    while (at < hi && got < max) {
        VmRange *r = find_range_lockfree(at);
        if (!r || r->host) return UINT64_MAX;
        uint64_t ubase = at & ~(uint64_t)(PS5_MEM_UNIT - 1);
        Ps5MixedBlock *u = find_unit_lockfree(r, ubase);
        if (!u) {
            uint64_t next_chunk = r->base + (((ubase - r->base) >> CHUNK_SHIFT) + 1) * (uint64_t)CHUNK_SIZE;
            at = next_chunk < hi ? next_chunk : hi;
            continue;
        }
        uint64_t uend = ubase + PS5_MEM_UNIT;
        uint64_t seg_lo = at > ubase ? at : ubase;
        uint64_t seg_hi = uend < hi ? uend : hi;
        // A packed unit holds data too (MKW_PACK): reading it unpacks it.
        if (__atomic_load_n(&u->backing.committed, __ATOMIC_ACQUIRE) || __atomic_load_n(&u->pack, __ATOMIC_ACQUIRE)) {
            if (got && runs[got - 1][1] == seg_lo) runs[got - 1][1] = seg_hi;
            else runs[got][0] = seg_lo, runs[got][1] = seg_hi, ++got;
        }
        at = seg_hi;
    }
    return got;
}

void plat_mem_early_hold_hook(void (*fn)(uint64_t addr, size_t n)) {
    __atomic_store_n(&g_early_hold_hook, fn, __ATOMIC_RELEASE);
}
void plat_mem_frozen_hook(int (*fn)(uint64_t addr)) {
    __atomic_store_n(&g_frozen_hook, fn, __ATOMIC_RELEASE);
}
void plat_mem_frozen_ms_hook(uint32_t (*fn)(uint64_t addr)) {
    __atomic_store_n(&g_frozen_ms_hook, fn, __ATOMIC_RELEASE);
}

uint64_t plat_mem_reclaim_zeros(uint64_t target_bytes, uint64_t *scanned_bytes, int cache) {
    int (*frozen_fn)(uint64_t) = __atomic_load_n(&g_frozen_hook, __ATOMIC_ACQUIRE);
    if (!frozen_fn) {
        if (scanned_bytes) *scanned_bytes = 0;
        return 0;
    }
    uint64_t scanned = 0, reclaimed = 0;
    for (VmRange *r = __atomic_load_n(&ranges, __ATOMIC_ACQUIRE); r && (!target_bytes || reclaimed < target_bytes);
         r = __atomic_load_n(&r->next, __ATOMIC_ACQUIRE)) {
        if (r->host || !r->chunks || (r->kind != RANGE_HEAP && r->kind != RANGE_ALIAS)) continue;
        for (size_t c = 0; c < r->chunk_count && (!target_bytes || reclaimed < target_bytes); ++c) {
            Ps5MixedBlock *chunk = __atomic_load_n(&r->chunks[c], __ATOMIC_ACQUIRE);
            if (!chunk) continue;
            for (size_t i = 0; i < CHUNK_UNITS && (!target_bytes || reclaimed < target_bytes); ++i) {
                Ps5MixedBlock *u = &chunk[i];
                if (!__atomic_load_n(&u->backing.committed, __ATOMIC_ACQUIRE)) continue;
                uint64_t uaddr = u->backing.address;
                uint32_t fz = (uint32_t)frozen_fn(uaddr);
                // A unit found not all zero in this same freeze is still so: frozen, nothing wrote it (a write thaws
                // first, and a new freeze has a new number). A stale or racy number only skips a unit, never frees one.
                if (!fz || (cache && __atomic_load_n(&u->nz_freeze, __ATOMIC_RELAXED) == fz) ||
                    prot_watch_has_gpu(uaddr, PS5_MEM_UNIT))
                    continue;
                unit_lock(u);
                bool can_scan = u->backing.committed && u->live_pages &&
                                !prot_watch_has_gpu(uaddr, PS5_MEM_UNIT) && frozen_fn(uaddr);
                for (size_t p = 0; can_scan && p < PS5_MEM_UNIT / PAGE; ++p)
                    if (u->pages[p] < PS5_GUEST_READ || prot_watch_is_early(uaddr + p * PAGE)) can_scan = false;
                if (can_scan) {
                    scanned += PS5_MEM_UNIT;
                    const uint64_t *q = (const uint64_t *)(uintptr_t)uaddr;
                    bool zero = true;
                    for (size_t k = 0; k < PS5_MEM_UNIT / sizeof(uint64_t); ++k) {
                        if (q[k]) { zero = false; break; }
                    }
                    // The number read after the scan: a thaw and refreeze during it leaves a different number, and
                    // the unit is scanned again next time.
                    if (!zero && (uint32_t)frozen_fn(uaddr) == fz) __atomic_store_n(&u->nz_freeze, fz, __ATOMIC_RELAXED);
                    if (zero && ps5_mem_decommit(&u->backing) == 0) {
                        __atomic_sub_fetch(&backed_units[r->kind], 1, __ATOMIC_RELAXED);
                        reclaimed += PS5_MEM_UNIT;
                    }
                }
                unit_unlock(u);
            }
        }
    }
    if (scanned_bytes) *scanned_bytes = scanned;
    return reclaimed;
}

// MKW_PACK: plat_mem_pack and its census. The governor's thread only, but for the summary (the census's thread).
enum { SK_GPU, SK_EARLY, SK_YOUNG, SK_RATIO, SK_RATIO_KNOWN, SK_STORE, SK_OTHER, PK_SKIPS };
static ZSTD_CCtx *pk_cctx;
static uint8_t *pk_copy, *pk_cbuf;
static size_t pk_cbound;
static uint64_t pk_cursor;                        // where the last pass stopped: the next starts there
static uint64_t pk_pass_skip[PK_SKIPS], pk_pass_frozen;   // the pass running
static uint64_t pk_last_skip[PK_SKIPS], pk_last_frozen;   // the last pass's, for the summary
// Since the last summary (exchanged there).
static uint64_t pk_packs, pk_zeros, pk_raw_in, pk_comp_out, pk_pack_ns, pk_passes;

static bool pk_init(void) {
    static int state;   // 1 ready, -1 failed
    if (state) return state > 0;
    state = -1;
    size_t dsz = ZSTD_estimateDCtxSize();
    for (int i = 0; i < PK_DCTX; i++) {
        void *m = malloc(dsz);
        if (!m || !(pk_dctx[i].d = ZSTD_initStaticDCtx(m, dsz))) return false;
    }
    pk_cbound = ZSTD_compressBound(PS5_MEM_UNIT);
    pk_cctx = ZSTD_createCCtx();
    pk_copy = malloc(PS5_MEM_UNIT);
    pk_cbuf = malloc(pk_cbound);
    if (!pk_cctx || !pk_copy || !pk_cbuf) return false;
    // The decompression contexts (~94 KiB each) and the packer's buffers are the store's too.
    __atomic_add_fetch(&pk_store_bytes, PK_DCTX * dsz + PS5_MEM_UNIT + pk_cbound, __ATOMIC_RELAXED);
    state = 1;
    return true;
}

// Under the unit's lock: -1 when the unit can be packed in freeze fz, else why not. Every page read-only and the region
// still in freeze fz, at the copy and again at the release, means nothing wrote the unit in between: a thaw makes the
// pages read-write under this lock before any write, and freezing them again takes a new number (memory.c).
static int pk_check_locked(const Ps5MixedBlock *u, int (*frozen_fn)(uint64_t), uint32_t fz) {
    uint64_t a = u->backing.address;
    if (!u->backing.committed || !u->live_pages || u->pack || (uint32_t)frozen_fn(a) != fz) return SK_OTHER;
    if (prot_watch_has_gpu(a, PS5_MEM_UNIT)) return SK_GPU;
    for (size_t p = 0; p < PS5_MEM_UNIT / PAGE; ++p)
        if (u->pages[p] != PS5_GUEST_READ) return SK_OTHER;
    for (uint64_t off = 0; off < PS5_MEM_UNIT; off += PS5_MEM_PAGE)
        if (prot_watch_is_early(a + off)) return SK_EARLY;
    return -1;
}

static void pk_skip(int why) { pk_pass_skip[why]++; }

// One unit; 1 stops the pass (the store is full).
static int pk_unit(Ps5MixedBlock *u, RangeKind kind, const PlatPackParams *pp, int (*frozen_fn)(uint64_t),
                   uint32_t (*since_fn)(uint64_t), uint32_t now_ms, uint64_t *freed) {
    if (!__atomic_load_n(&u->backing.committed, __ATOMIC_ACQUIRE) || __atomic_load_n(&u->pack, __ATOMIC_RELAXED))
        return 0;
    uint64_t a = u->backing.address;
    uint32_t fz = (uint32_t)frozen_fn(a);
    if (!fz) return 0;
    pk_pass_frozen++;
    // Compressed too little in this freeze: the same bytes now (a write thaws, and a new freeze has a new number).
    if (__atomic_load_n(&u->pk_fail_freeze, __ATOMIC_RELAXED) == fz) return pk_skip(SK_RATIO_KNOWN), 0;
    if (prot_watch_has_gpu(a, PS5_MEM_UNIT)) return pk_skip(SK_GPU), 0;
    // Signed: a time made odd (| 1) may be 1 ms ahead of now_ms.
    uint32_t since = since_fn(a), up = __atomic_load_n(&u->pk_unpack_ms, __ATOMIC_RELAXED);
    // MKW_PACK_BACKOFF=2: a unit read back once stays unpacked. Moving the pack's work to other threads or into
    // slices changed nothing (28/11/2026: PACK_SLICE and an unpack-ahead thread, in-race A/Bs), so the lever left is
    // fewer pack and unpack cycles: memory a lap reads again is not cold, whatever its cold time.
    uint32_t bounces = __atomic_load_n(&u->pk_bounces, __ATOMIC_RELAXED);
    if (pp->backoff == 2 && bounces) return pk_skip(SK_YOUNG), 0;
    uint32_t cold = pp->cold_ms << (pp->backoff ? bounces : 0);
    if (!since || (int32_t)(now_ms - since) < (int32_t)cold || (up && (int32_t)(now_ms - up) < (int32_t)cold))
        return pk_skip(SK_YOUNG), 0;
    uint64_t need = sizeof(PackBlob) + pk_cbound + (pp->verify ? PS5_MEM_UNIT : 0);
    if (__atomic_load_n(&pk_store_bytes, __ATOMIC_RELAXED) + need > pp->store_cap) return pk_skip(SK_STORE), 1;

    // The copy under the lock (another thread may release the unit), the compression outside it.
    unit_lock(u);
    int why = pk_check_locked(u, frozen_fn, fz);
    if (why < 0) memcpy(pk_copy, (const void *)(uintptr_t)a, PS5_MEM_UNIT);
    unit_unlock(u);
    if (why >= 0) return pk_skip(why), 0;
    const uint64_t *q = (const uint64_t *)pk_copy;
    bool zero = true;
    for (size_t k = 0; k < PS5_MEM_UNIT / sizeof *q && zero; ++k) zero = !q[k];
    PackBlob *b = NULL;
    size_t c = 0;
    if (!zero) {   // all zero: released with no blob, as the zero reclaim does; it commits again zero-filled
        uint64_t t0 = plat_now_ns();
        c = ZSTD_compressCCtx(pk_cctx, pk_cbuf, pk_cbound, pk_copy, PS5_MEM_UNIT, 1);
        __atomic_add_fetch(&pk_pack_ns, plat_now_ns() - t0, __ATOMIC_RELAXED);
        if (ZSTD_isError(c) || c * 100 > (size_t)PS5_MEM_UNIT * PK_MAX_PCT) {
            __atomic_store_n(&u->pk_fail_freeze, fz, __ATOMIC_RELAXED);
            return pk_skip(SK_RATIO), 0;
        }
        uint32_t bslot = 0, rslot = 0;
        b = pks_alloc(sizeof *b + c, &bslot);
        uint8_t *raw = pp->verify && b ? pks_alloc(PS5_MEM_UNIT, &rslot) : NULL;
        if (!b || (pp->verify && !raw)) {
            pks_free(b), pks_free(raw);
            return pk_skip(SK_STORE), 1;
        }
        *b = (PackBlob){.raw = raw, .csize = (uint32_t)c, .bytes = bslot + rslot};
        memcpy(b->data, pk_cbuf, c);
        if (raw) memcpy(raw, pk_copy, PS5_MEM_UNIT);
    }

    unit_lock(u);
    why = pk_check_locked(u, frozen_fn, fz);
    bool done = false;
    if (why < 0) {
        int rc = ps5_mem_decommit(&u->backing);
        if (!u->backing.committed) {
            done = true;
            __atomic_store_n(&u->pack, b, __ATOMIC_RELEASE);
            __atomic_sub_fetch(&backed_units[kind], 1, __ATOMIC_RELAXED);
            if (b) {
                __atomic_add_fetch(&packed_units[kind], 1, __ATOMIC_RELAXED);
                __atomic_add_fetch(&pk_packed_bytes, c, __ATOMIC_RELAXED);
                __atomic_add_fetch(&pk_store_bytes, b->bytes, __ATOMIC_RELAXED);
            }
        }
        if (rc) sdk_ok("pack release", a, PS5_MEM_UNIT, rc);
    }
    unit_unlock(u);
    if (!done) {
        if (b) pks_free(b->raw), pks_free(b);
        return pk_skip(why >= 0 ? why : SK_OTHER), 0;
    }
    // What the pool got back: the unit less the store's slots for it (with MKW_PACK_VERIFY's copy, nothing).
    *freed += !b ? PS5_MEM_UNIT : b->bytes < PS5_MEM_UNIT ? PS5_MEM_UNIT - b->bytes : 0;
    if (b) {
        __atomic_add_fetch(&pk_packs, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&pk_raw_in, PS5_MEM_UNIT, __ATOMIC_RELAXED);
        __atomic_add_fetch(&pk_comp_out, c, __ATOMIC_RELAXED);
        __atomic_store_n(&pk_ever, 1, __ATOMIC_RELAXED);
    } else __atomic_add_fetch(&pk_zeros, 1, __ATOMIC_RELAXED);
    return 0;
}

// The units of HEAP and ALIAS ranges at [lo, hi), in address order; 1 when the pass stops (target, time or store),
// with pk_cursor where the next starts.
static int pk_walk(uint64_t lo, uint64_t hi, const PlatPackParams *pp, int (*frozen_fn)(uint64_t),
                   uint32_t (*since_fn)(uint64_t), uint64_t deadline, uint64_t *freed) {
    uint32_t now_ms = (uint32_t)plat_now_ms();
    unsigned n = 0;
    for (VmRange *r = __atomic_load_n(&ranges, __ATOMIC_ACQUIRE); r; r = __atomic_load_n(&r->next, __ATOMIC_ACQUIRE)) {
        if (r->base >= hi) break;
        if (r->end <= lo || r->host || !r->chunks || (r->kind != RANGE_HEAP && r->kind != RANGE_ALIAS)) continue;
        for (size_t c = 0; c < r->chunk_count; ++c) {
            uint64_t cbase = r->base + ((uint64_t)c << CHUNK_SHIFT);
            if (cbase >= hi) return 0;
            if (cbase + CHUNK_SIZE <= lo) continue;
            Ps5MixedBlock *chunk = __atomic_load_n(&r->chunks[c], __ATOMIC_ACQUIRE);
            if (!chunk) continue;
            for (size_t i = 0; i < CHUNK_UNITS; ++i) {
                Ps5MixedBlock *u = &chunk[i];
                uint64_t a = cbase + (uint64_t)i * PS5_MEM_UNIT;
                if (a < lo) continue;
                if (a >= hi || a >= r->end) break;
                if (*freed >= pp->target_bytes || (!(++n & 63) && plat_now_ns() > deadline) ||
                    pk_unit(u, r->kind, pp, frozen_fn, since_fn, now_ms, freed)) {
                    pk_cursor = a + PS5_MEM_UNIT;
                    return 1;
                }
            }
        }
    }
    return 0;
}

uint64_t plat_mem_pack(const PlatPackParams *pp) {
    plat_mem_pack_collect();
    int (*frozen_fn)(uint64_t) = __atomic_load_n(&g_frozen_hook, __ATOMIC_ACQUIRE);
    uint32_t (*since_fn)(uint64_t) = __atomic_load_n(&g_frozen_ms_hook, __ATOMIC_ACQUIRE);
    if (!pp || !pp->target_bytes || !frozen_fn || !since_fn || !pk_init()) return 0;
    memset(pk_pass_skip, 0, sizeof pk_pass_skip);
    pk_pass_frozen = 0;
    uint64_t freed = 0, t0 = plat_now_ns(), deadline = pp->budget_us ? t0 + pp->budget_us * 1000ull : UINT64_MAX;
    // The store is direct memory (pks_alloc), so no floor of flexible memory holds the pass back any more.
    uint64_t start = pk_cursor;
    if (!pk_walk(start, UINT64_MAX, pp, frozen_fn, since_fn, deadline, &freed) &&
        !pk_walk(0, start, pp, frozen_fn, since_fn, deadline, &freed))
        pk_cursor = 0;
    for (int i = 0; i < PK_SKIPS; i++) __atomic_store_n(&pk_last_skip[i], pk_pass_skip[i], __ATOMIC_RELAXED);
    __atomic_store_n(&pk_last_frozen, pk_pass_frozen, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pk_passes, 1, __ATOMIC_RELAXED);
    return freed;
}

// MKW_PACK_TEXFREE (vk_mem_upload.c pack_step): one pass over [lo, hi) only, the main pass's cursor left where it is.
// The caller freezes the regions first; the units' own checks (frozen, GPU, early-read, ratio, store) hold as in
// plat_mem_pack.
uint64_t plat_mem_pack_range(uint64_t lo, uint64_t hi, const PlatPackParams *pp) {
    int (*frozen_fn)(uint64_t) = __atomic_load_n(&g_frozen_hook, __ATOMIC_ACQUIRE);
    uint32_t (*since_fn)(uint64_t) = __atomic_load_n(&g_frozen_ms_hook, __ATOMIC_ACQUIRE);
    if (!pp || !pp->target_bytes || lo >= hi || !frozen_fn || !since_fn || !pk_init()) return 0;
    uint64_t freed = 0, cursor = pk_cursor;
    uint64_t deadline = pp->budget_us ? plat_now_ns() + pp->budget_us * 1000ull : UINT64_MAX;
    pk_walk(lo, hi, pp, frozen_fn, since_fn, deadline, &freed);
    pk_cursor = cursor;
    return freed;
}

void plat_mem_pack_collect(void) {
    for (PackBlob *b = __atomic_exchange_n(&pk_free, NULL, __ATOMIC_ACQUIRE), *next; b; b = next) {
        next = b->next;
        __atomic_sub_fetch(&pk_store_bytes, b->bytes, __ATOMIC_RELAXED);
        pks_free(b->raw);
        pks_free(b);
    }
}

static void pk_backed_note(char *out, size_t cap) {
    if (!cap) return;
    out[0] = 0;
    if (!__atomic_load_n(&pk_ever, __ATOMIC_RELAXED)) return;
    size_t units = 0, flex = 0;
    for (int i = 0; i < 3; i++) units += __atomic_load_n(&packed_units[i], __ATOMIC_RELAXED);
    int frc = sceKernelAvailableFlexibleMemorySize(&flex);
    snprintf(out, cap,
             ", packed %zu MiB in %.1f MiB (store %.1f MiB in %llu direct-memory slabs of 1 MiB; flex free %zu MiB)",
             (units * PS5_MEM_UNIT) >> 20, __atomic_load_n(&pk_packed_bytes, __ATOMIC_RELAXED) / 1048576.0,
             __atomic_load_n(&pk_store_bytes, __ATOMIC_RELAXED) / 1048576.0,
             (unsigned long long)__atomic_load_n(&pks_slabs, __ATOMIC_RELAXED), frc == 0 ? flex >> 20 : 0);
}

void plat_mem_pack_summary(char *out, size_t cap) {
    if (!cap) return;
    out[0] = 0;
    if (!__atomic_load_n(&pk_ever, __ATOMIC_RELAXED) && !__atomic_load_n(&pk_passes, __ATOMIC_RELAXED)) return;
#define X(v) __atomic_exchange_n(&(v), 0, __ATOMIC_RELAXED)
#define L(v) (unsigned long long)__atomic_load_n(&(v), __ATOMIC_RELAXED)
    uint64_t packs = X(pk_packs), zeros = X(pk_zeros), raw = X(pk_raw_in), comp = X(pk_comp_out), pns = X(pk_pack_ns);
    uint64_t passes = X(pk_passes), un = X(pk_unpacks), uns = X(pk_unpack_ns), umax = X(pk_unpack_max_ns);
    uint64_t uc[PLAT_BACK_N];
    for (int i = 0; i < PLAT_BACK_N; i++) uc[i] = X(pk_unpack_cause[i]);
    uint64_t vchk = X(pk_verify_checked), vbad = X(pk_verify_bad), vaddr = X(pk_verify_addr), fbad = L(pk_fill_bad);
    size_t units = 0;
    for (int i = 0; i < 3; i++) units += __atomic_load_n(&packed_units[i], __ATOMIC_RELAXED);
    uint64_t pb = __atomic_load_n(&pk_packed_bytes, __ATOMIC_RELAXED);
    int o = snprintf(out, cap,
                     "%zu units, %.1f MiB in %.1f MiB (%.2f); since the last: %llu passes, %llu packed (%.1f -> %.1f MiB, "
                     "%.0f us each), %llu zero released, %llu unpacked (mean %.0f us, max %.0f us; write fault %llu, read "
                     "fault %llu, runtime read %llu, import %llu, commit %llu); last pass: %llu frozen units, skipped gpu "
                     "%llu, early %llu, young %llu, ratio %llu (+%llu known), store %llu, other %llu; verify %llu checked, "
                     "%llu differed (first %#llx); %llu failed to decompress; unpacked by:",
                     units, (units * (double)PS5_MEM_UNIT) / 1048576.0, pb / 1048576.0,
                     units ? pb / (units * (double)PS5_MEM_UNIT) : 0.0, (unsigned long long)passes,
                     (unsigned long long)packs, raw / 1048576.0, comp / 1048576.0, packs ? pns / 1e3 / packs : 0.0,
                     (unsigned long long)zeros, (unsigned long long)un, un ? uns / 1e3 / un : 0.0, umax / 1e3,
                     (unsigned long long)uc[PLAT_BACK_WRITE_FAULT], (unsigned long long)uc[PLAT_BACK_READ_FAULT],
                     (unsigned long long)uc[PLAT_BACK_RT_READ], (unsigned long long)uc[PLAT_BACK_IMPORT],
                     (unsigned long long)uc[PLAT_BACK_COMMIT], L(pk_last_frozen), L(pk_last_skip[SK_GPU]),
                     L(pk_last_skip[SK_EARLY]), L(pk_last_skip[SK_YOUNG]), L(pk_last_skip[SK_RATIO]),
                     L(pk_last_skip[SK_RATIO_KNOWN]), L(pk_last_skip[SK_STORE]), L(pk_last_skip[SK_OTHER]),
                     (unsigned long long)vchk, (unsigned long long)vbad, (unsigned long long)vaddr,
                     (unsigned long long)fbad);
    for (int i = 0; i < PK_THREADS && o > 0 && (size_t)o < cap; i++) {
        if (!__atomic_load_n(&pk_thr[i].tid, __ATOMIC_ACQUIRE)) break;
        uint64_t n = X(pk_thr[i].n);
        if (n) o += snprintf(out + o, cap - (size_t)o, " %s %llu", pk_thr[i].name, (unsigned long long)n);
    }
#undef X
#undef L
}

// ---- MKW_COLD_CENSUS: how much of the game's memory sits unwritten, by 64 KiB unit (count only, nothing changes)
// MKW_PACK packs only units of a frozen 2 MiB region (mem_frozen), and freeze candidates come only from the regions the
// replay checks for live textures and buffers (vk_mem_gen.c snap_regions): in four 1.8.0 Grand Prix runs (24/11/2026)
// it never found a frozen unit, while ~1.2 GiB of one pool is earlier courses' data (memgrow2, 16/11/2026). This
// measures what a unit-level cold tracker could pack before one is written: each committed unit of the heap and alias
// ranges is hashed in turn on the governor's thread (no page protection, so neither the game nor the write watch sees
// it), and a unit whose hash did not change for N seconds counts as cold for N. A sample of the cold ones is compressed
// as plat_mem_pack would (ZSTD level 1, PK_MAX_PCT) to estimate what packing them would give back. Reads are not seen:
// a cold unit the game still reads would unpack on its first read.
enum { CC_BUCKETS = 5, CC_SAMPLE_EVERY = 16, CC_FEED_PER_STEP = 32 };   // regions frozen a tick at most
static const uint32_t cc_bucket_s[CC_BUCKETS] = {0, 30, 60, 120, 300};   // age at least this many seconds
static uint64_t cc_cursor;
static uint64_t cc_cycle_t0;
static uint64_t cc_units[3][CC_BUCKETS], cc_new[3], cc_zero[3], cc_skip[3], cc_packed[3];
static uint64_t cc_s_units, cc_s_raw, cc_s_comp, cc_s_fit, cc_ns, cc_cycles;

static uint64_t cc_hash(const uint64_t *q) {   // four independent lanes: ~1 byte a cycle less than a serial chain
    uint64_t a = 0x9e3779b97f4a7c15ull, b = 0xc2b2ae3d27d4eb4full, c = 0x165667b19e3779f9ull, d = 0x27d4eb2f165667c5ull;
    for (size_t k = 0; k < PS5_MEM_UNIT / sizeof *q; k += 4) {
        a = (a ^ q[k]) * 0x100000001b3ull, b = (b ^ q[k + 1]) * 0x100000001b3ull;
        c = (c ^ q[k + 2]) * 0x100000001b3ull, d = (d ^ q[k + 3]) * 0x100000001b3ull;
    }
    return a ^ (b >> 1) ^ (c << 3) ^ (d >> 5);
}

// One unit, its lock held: 1 when it was read (*h its hash, *zero), 0 when it cannot be read without a fault or a thaw.
static int cc_read_locked(Ps5MixedBlock *u, uint64_t a, uint64_t *h, bool *zero, bool sample) {
    if (!u->backing.committed || !u->live_pages || prot_watch_has_gpu(a, PS5_MEM_UNIT)) return 0;
    for (size_t p = 0; p < PS5_MEM_UNIT / PAGE; ++p)
        if (u->pages[p] < PS5_GUEST_READ || prot_watch_is_early(a + p * PAGE)) return 0;
    const uint64_t *q = (const uint64_t *)(uintptr_t)u->backing.address;
    *zero = true;
    for (size_t k = 0; k < PS5_MEM_UNIT / sizeof *q && *zero; ++k) *zero = !q[k];
    *h = *zero ? 0 : cc_hash(q);
    if (sample && !*zero) memcpy(pk_copy, q, PS5_MEM_UNIT);
    return 1;
}

// What cc_unit found: the cold feed freezes a 2 MiB region only when none of its units is CC_HOT or CC_SKIP.
enum { CC_NONE, CC_PACKED, CC_SKIP, CC_ZERO, CC_HOT, CC_COLD };

// One unit; cold for the feed when unchanged for at least freeze_s seconds. want_sample: run ZSTD ratio sampling
// for MKW_COLD_CENSUS diagnostics (skipped when only feeding MKW_PACK, avoiding extra CPU/L3 work mid-race).
static int cc_unit(Ps5MixedBlock *u, RangeKind kind, uint64_t a, uint32_t now_ms, uint32_t freeze_s, bool want_sample) {
    if (__atomic_load_n(&u->pack, __ATOMIC_ACQUIRE)) return cc_packed[kind]++, CC_PACKED;
    if (!__atomic_load_n(&u->backing.committed, __ATOMIC_ACQUIRE)) return CC_NONE;
    int32_t d = u->cc_changed_ms ? (int32_t)(now_ms - u->cc_changed_ms) : 0;   // signed: a stored | 1 can be 1 ms ahead
    uint32_t age_s = d > 0 ? (uint32_t)d / 1000u : 0;
    bool sample = want_sample && u->cc_changed_ms && age_s >= cc_bucket_s[1] && !((a / PS5_MEM_UNIT) % CC_SAMPLE_EVERY);
    uint64_t h = 0;
    bool zero = false;
    unit_lock(u);
    int ok = cc_read_locked(u, a, &h, &zero, sample);
    unit_unlock(u);
    if (!ok) return cc_skip[kind]++, CC_SKIP;
    if (zero) return cc_zero[kind]++, CC_ZERO;
    uint32_t h32 = (uint32_t)(h ^ h >> 32) | 1;
    if (!u->cc_changed_ms || u->cc_hash != h32) {   // first sight, or written since the last
        if (!u->cc_changed_ms) cc_new[kind]++;
        else cc_units[kind][0]++;
        u->cc_hash = h32, u->cc_changed_ms = now_ms | 1;
        return CC_HOT;
    }
    int b = CC_BUCKETS - 1;
    while (b > 0 && age_s < cc_bucket_s[b]) b--;
    cc_units[kind][b]++;
    if (sample) {
        size_t c = ZSTD_compressCCtx(pk_cctx, pk_cbuf, pk_cbound, pk_copy, PS5_MEM_UNIT, 1);
        if (!ZSTD_isError(c)) {
            cc_s_units++, cc_s_raw += PS5_MEM_UNIT, cc_s_comp += c;
            if (c * 100 <= (size_t)PS5_MEM_UNIT * PK_MAX_PCT) cc_s_fit++;
        }
    }
    return age_s >= freeze_s ? CC_COLD : CC_HOT;
}

// The cold feed (MKW_PACK): the 2 MiB region the walk is in, judged once the walk leaves it.
static uint64_t cc_reg = UINT64_MAX;
static int cc_reg_cold, cc_reg_bad;
static uint64_t cc_feed_frozen, cc_feed_refused, cc_feed_capped;

static void cc_region_end(int (*freeze_fn)(uint64_t), unsigned *budget) {
    if (cc_reg != UINT64_MAX && freeze_fn && cc_reg_cold && !cc_reg_bad) {
        int (*frozen_fn)(uint64_t) = __atomic_load_n(&g_frozen_hook, __ATOMIC_ACQUIRE);
        int already = frozen_fn && frozen_fn(cc_reg);
        if (!already && !*budget) cc_feed_capped++;   // the next walk offers it again
        else if (!already) {
            (*budget)--;
            if (freeze_fn(cc_reg)) cc_feed_frozen++;
            else cc_feed_refused++;
        }
    }
    cc_reg = UINT64_MAX, cc_reg_cold = cc_reg_bad = 0;
}

static void cc_summary(char *out, size_t cap, uint64_t cycle_ns) {
    const double M = (double)PS5_MEM_UNIT / 1048576.0;
    uint64_t cold[CC_BUCKETS] = {0}, all = 0;
    for (int k = 1; k < 3; k++)
        for (int b = 0; b < CC_BUCKETS; b++) cold[b] += cc_units[k][b], all += cc_units[k][b];
    // MiB unchanged for at least 30 s; what packing those would free at the sample's ratio, units that fit counted only
    double c30 = 0;
    for (int b = 1; b < CC_BUCKETS; b++) c30 += (double)cold[b] * M;
    double fit = cc_s_units ? (double)cc_s_fit / (double)cc_s_units : 0.0;
    double ratio = cc_s_raw ? (double)cc_s_comp / (double)cc_s_raw : 0.0;
    uint64_t heap30 = 0, alias30 = 0;
    for (int b = 1; b < CC_BUCKETS; b++) heap30 += cc_units[RANGE_HEAP][b], alias30 += cc_units[RANGE_ALIAS][b];
    snprintf(out, cap,
             "cycle %llu in %.1f s (%.1f ms of hashing); heap+alias units by time unchanged: <30 s %.1f MiB, "
             "30-60 s %.1f, 60-120 s %.1f, 120-300 s %.1f, >=300 s %.1f MiB (heap %.1f / alias %.1f MiB >=30 s); "
             "new %.1f, all zero %.1f, unreadable %.1f, packed %.1f MiB; cold sample %llu units: %.0f%% compress to "
             "<=%d%%, ratio %.2f -> packing the >=30 s ones would free ~%.1f MiB; cold feed: %llu regions frozen, %llu "
             "refused, %llu left for the next walk",
             (unsigned long long)cc_cycles, cycle_ns / 1e9, cc_ns / 1e6, cold[0] * M, cold[1] * M, cold[2] * M,
             cold[3] * M, cold[4] * M, heap30 * M, alias30 * M, (cc_new[1] + cc_new[2]) * M,
             (cc_zero[1] + cc_zero[2]) * M, (cc_skip[1] + cc_skip[2]) * M, (cc_packed[1] + cc_packed[2]) * M,
             (unsigned long long)cc_s_units, fit * 100.0, PK_MAX_PCT, ratio, c30 * fit * (1.0 - ratio),
             (unsigned long long)cc_feed_frozen, (unsigned long long)cc_feed_refused,
             (unsigned long long)cc_feed_capped);
    cc_feed_frozen = cc_feed_refused = cc_feed_capped = 0;
    (void)all;
    memset(cc_units, 0, sizeof cc_units), memset(cc_new, 0, sizeof cc_new), memset(cc_zero, 0, sizeof cc_zero);
    memset(cc_skip, 0, sizeof cc_skip), memset(cc_packed, 0, sizeof cc_packed);
    cc_s_units = cc_s_raw = cc_s_comp = cc_s_fit = cc_ns = 0;
}

int plat_mem_cold_census_step(uint32_t budget_us, uint32_t freeze_s, int (*freeze_fn)(uint64_t), char *out,
                              size_t cap) {
    if (!pk_init()) return 0;
    uint64_t t0 = plat_now_ns(), deadline = t0 + (uint64_t)budget_us * 1000ull;
    if (!cc_cycle_t0) cc_cycle_t0 = t0;
    uint32_t now_ms = (uint32_t)plat_now_ms();
    unsigned n = 0, feed = CC_FEED_PER_STEP;
    int done = 1;   // the walk reached the end: a cycle is complete
    for (VmRange *r = __atomic_load_n(&ranges, __ATOMIC_ACQUIRE); r && done;
         r = __atomic_load_n(&r->next, __ATOMIC_ACQUIRE)) {
        if (r->end <= cc_cursor || r->host || !r->chunks || (r->kind != RANGE_HEAP && r->kind != RANGE_ALIAS)) continue;
        for (size_t c = 0; c < r->chunk_count && done; ++c) {
            uint64_t cbase = r->base + ((uint64_t)c << CHUNK_SHIFT);
            if (cbase + CHUNK_SIZE <= cc_cursor) continue;
            Ps5MixedBlock *chunk = __atomic_load_n(&r->chunks[c], __ATOMIC_ACQUIRE);
            if (!chunk) continue;
            for (size_t i = 0; i < CHUNK_UNITS; ++i) {
                uint64_t a = cbase + (uint64_t)i * PS5_MEM_UNIT;
                if (a < cc_cursor) continue;
                if (a >= r->end) break;
                if (!(++n & 15) && plat_now_ns() > deadline) { cc_cursor = a, done = 0; break; }
                if (a >> 21 != cc_reg >> 21) cc_region_end(freeze_fn, &feed), cc_reg = a & ~((1ull << 21) - 1);
                int k = cc_unit(&chunk[i], r->kind, a, now_ms, freeze_s, out != NULL && cap > 0);
                if (k == CC_COLD) cc_reg_cold++;
                else if (k == CC_HOT || k == CC_SKIP) cc_reg_bad = 1;
            }
        }
    }
    cc_ns += plat_now_ns() - t0;
    if (!done) return 0;
    cc_region_end(freeze_fn, &feed);
    uint64_t now = plat_now_ns();
    cc_cycles++;
    if (out && cap) cc_summary(out, cap, now - cc_cycle_t0);
    else {
        cc_feed_frozen = cc_feed_refused = cc_feed_capped = 0;
        memset(cc_units, 0, sizeof cc_units), memset(cc_new, 0, sizeof cc_new), memset(cc_zero, 0, sizeof cc_zero);
        memset(cc_skip, 0, sizeof cc_skip), memset(cc_packed, 0, sizeof cc_packed);
        cc_s_units = cc_s_raw = cc_s_comp = cc_s_fit = cc_ns = 0;
    }
    cc_cursor = 0, cc_cycle_t0 = now;
    return out && cap ? 1 : 0;
}

uint64_t plat_mem_backed_bytes(uint64_t addr, uint64_t size) {
    if (!size || size > UINT64_MAX - addr) return 0;
    uint64_t end = addr + size;
    uint64_t at = addr & ~(uint64_t)(PS5_MEM_UNIT - 1);
    uint64_t total = 0;
    while (at < end) {
        VmRange *r = find_range_lockfree(at);
        if (!r || r->host) {
            if (r && r->end > at) {
                at = (r->end + PS5_MEM_UNIT - 1) & ~(uint64_t)(PS5_MEM_UNIT - 1);
                continue;
            }
            at += PS5_MEM_UNIT;
            continue;
        }
        Ps5MixedBlock *u = find_unit_lockfree(r, at);
        if (u) {
            if (__atomic_load_n(&u->backing.committed, __ATOMIC_RELAXED)) {
                uint64_t lo = at > addr ? at : addr;
                uint64_t hi = (at + PS5_MEM_UNIT) < end ? (at + PS5_MEM_UNIT) : end;
                if (hi > lo) total += hi - lo;
            }
            at += PS5_MEM_UNIT;
        } else {
            uint64_t next_chunk = r->base + (((at - r->base) >> CHUNK_SHIFT) + 1) * (uint64_t)CHUNK_SIZE;
            at = next_chunk < r->end ? next_chunk : r->end;
        }
    }
    return total;
}

uint64_t plat_mem_release_backing(uint64_t addr, uint64_t size, PlatBackingDiscarded discarded) {
    if (!size || size > UINT64_MAX - addr || addr > UINT64_MAX - (PS5_MEM_UNIT - 1)) return 0;
    uint64_t freed = 0;
    uint64_t lo = (addr + PS5_MEM_UNIT - 1) & ~(uint64_t)(PS5_MEM_UNIT - 1);
    uint64_t hi = (addr + size) & ~(uint64_t)(PS5_MEM_UNIT - 1);
    for (uint64_t at = lo; at < hi;) {
        VmRange *r = find_range_lockfree(at);
        if (!r || r->host) {
            if (r && r->end > at) {
                at = (r->end + PS5_MEM_UNIT - 1) & ~(uint64_t)(PS5_MEM_UNIT - 1);
                continue;
            }
            at += PS5_MEM_UNIT;
            continue;
        }
        Ps5MixedBlock *u = find_unit_lockfree(r, at);
        if (!u) {
            uint64_t next_chunk = r->base + (((at - r->base) >> CHUNK_SHIFT) + 1) * (uint64_t)CHUNK_SIZE;
            at = next_chunk < r->end ? next_chunk : r->end;
            continue;
        }
        int unmapped = 0;
        unit_lock(u);
        if (u->backing.reserved && u->backing.committed && u->live_pages &&
            !prot_watch_has_gpu(u->backing.address, PS5_MEM_UNIT)) {
            bool early = false;
            for (uint64_t p = u->backing.address; p < u->backing.address + PS5_MEM_UNIT; p += PS5_MEM_PAGE) {
                if (prot_watch_is_early(p)) { early = true; break; }
            }
            if (!early) {
                int result = ps5_mem_decommit(&u->backing);
                unmapped = !u->backing.committed;
                if (unmapped) {
                    __atomic_sub_fetch(&backed_units[r->kind], 1, __ATOMIC_RELAXED);
                    if (!u->backing.allocated) freed += PS5_MEM_UNIT;
                }
                if (result) sdk_ok("release backing", at, PS5_MEM_UNIT, result);
            }
        }
        unit_unlock(u);
        if (unmapped && discarded && !discarded(at, at + PS5_MEM_UNIT)) break;
        at += PS5_MEM_UNIT;
    }
    return freed;
}

// MKW_POOL_RELEASE (vk_memory.c): a finalized pool's whole units that hold only zeros. ps5_mem_commit zero-fills a unit
// it commits again (ps5_memory.c), so the game reads what it left there. Task 32: releasing every unit of a finalized
// pool discarded the character cards' bytes, which MKW keeps after the finalize and samples again (magenta bands).
uint64_t plat_mem_release_zero_backing(uint64_t addr, uint64_t size) {
    if (!size || size > UINT64_MAX - addr || addr > UINT64_MAX - (PS5_MEM_UNIT - 1)) return 0;
    uint64_t freed = 0;
    uint64_t lo = (addr + PS5_MEM_UNIT - 1) & ~(uint64_t)(PS5_MEM_UNIT - 1);
    uint64_t hi = (addr + size) & ~(uint64_t)(PS5_MEM_UNIT - 1);
    for (uint64_t at = lo; at < hi; at += PS5_MEM_UNIT) {
        VmRange *r = find_range_lockfree(at);
        if (!r || r->host) continue;
        Ps5MixedBlock *u = find_unit_lockfree(r, at);
        if (!u || !__atomic_load_n(&u->backing.committed, __ATOMIC_ACQUIRE)) continue;
        unit_lock(u);
        bool ok = u->backing.reserved && u->backing.committed && u->live_pages &&
                  !prot_watch_has_gpu(u->backing.address, PS5_MEM_UNIT);
        for (size_t p = 0; ok && p < PS5_MEM_UNIT / PAGE; ++p)   // readable and not early-watched: the scan reads it
            ok = u->pages[p] >= PS5_GUEST_READ && !prot_watch_is_early(u->backing.address + p * PAGE);
        const uint64_t *q = (const uint64_t *)(uintptr_t)u->backing.address;
        for (size_t k = 0; ok && k < PS5_MEM_UNIT / sizeof(uint64_t); ++k) ok = !q[k];
        if (ok) {
            int result = ps5_mem_decommit(&u->backing);
            if (!u->backing.committed) {
                __atomic_sub_fetch(&backed_units[r->kind], 1, __ATOMIC_RELAXED);
                if (!u->backing.allocated) freed += PS5_MEM_UNIT;
            }
            if (result) sdk_ok("release zero backing", at, PS5_MEM_UNIT, result);
        }
        unit_unlock(u);
    }
    return freed;
}

void plat_mem_back_causes(uint64_t out[PLAT_BACK_N]) {
    for (int i = 0; i < PLAT_BACK_N; i++)
        out[i] = (uint64_t)__atomic_load_n(&backed_by_cause[i], __ATOMIC_RELAXED) * PS5_MEM_UNIT;
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_install();
    __atomic_store_n(&g_fault_hook, fn, __ATOMIC_RELEASE);
}
void plat_mem_watch_hook(int (*fn)(uint64_t addr, uint64_t rip)) {
    fault_install();
    __atomic_store_n(&g_watch_hook, fn, __ATOMIC_RELEASE);
}
void plat_crash_hook(void (*fn)(const PlatCrash *c)) {
    fault_install();
    __atomic_store_n(&g_crash_hook, fn, __ATOMIC_RELEASE);
}

static bool unmap(uint64_t address, uint64_t end) {
    return address == end || sdk_ok("shutdown unmap", address, end - address,
                                    sceKernelMunmap((void *)(uintptr_t)address, end - address));
}

static bool release_ranges(VmRange *range) {
    bool ok = true;
    while (range) {
        VmRange *next = range->next;
        uint64_t at = range->base;
        if (!range->host && range->chunks) {
            for (size_t c = 0; c < range->chunk_count; ++c) {
                Ps5MixedBlock *chunk = range->chunks[c];
                if (!chunk) continue;
                for (size_t i = 0; i < CHUNK_UNITS; ++i) {
                    Ps5MemBlock *unit = &chunk[i].backing;
                    uint64_t address = unit->address;
                    if (!unit->reserved && !unit->committed && !address) break;
                    if (unit->committed) __atomic_sub_fetch(&backed_units[range->kind], 1, __ATOMIC_RELAXED);
                    ok &= unmap(at, address);
                    // A slot lost on decommit has reserved=false. Release only its
                    // remaining physical or relocated ownership, never the hole.
                    ok &= sdk_ok("shutdown unit", address, PS5_MEM_UNIT, ps5_mem_release(unit));
                    at = address + PS5_MEM_UNIT;
                }
                free(chunk);
            }
            free((void *)range->chunks);
        }
        ok &= unmap(at, range->end);
        free(range);
        range = next;
    }
    return ok;
}

bool ps5_guest_memory_shutdown(void) {
    pthread_mutex_lock(&vm_lock);
    if (stopped) {
        pthread_mutex_unlock(&vm_lock);
        return false;
    }
    stopped = true;
    bool ok = release_ranges(ranges);
    ok &= release_ranges(cleanup_pending);
    ranges = cleanup_pending = NULL;
    prot_watch_shutdown();
    pthread_mutex_unlock(&vm_lock);
    return ok;
}
