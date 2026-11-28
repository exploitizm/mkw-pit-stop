// CPU backing for the PS5 platform. PS5's direct-memory unit is larger than
// its protection page, and both are larger than a guest page.
#include "ps5_memory.h"
#include "prot_watch.h"
#include "platform.h"

#include <ps5platform/kernel.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(PS5_MEM_PAGE == PS5_KERNEL_PAGE_SIZE, "unexpected PS5 protection page");
_Static_assert(PS5_MEM_UNIT == PS5_KERNEL_DIRECT_ALIGNMENT, "unexpected PS5 direct-memory unit");

static const int cpu_rw = PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE;

static int discard_relocation(Ps5MemBlock *block, void *placed) {
    block->relocated = placed;
    int result = sceKernelMunmap(placed, PS5_MEM_UNIT);
    if (!result) block->relocated = NULL;
    return result ? result : PS5_MEM_RELOCATED;
}

static int reserve_exact(Ps5MemBlock *block, uintptr_t address) {
    void *placed = (void *)address;
    int result = sceKernelReserveVirtualRange(&placed, PS5_MEM_UNIT, 0, PS5_MEM_UNIT);
    if (result || placed == (void *)address) return result;
    return discard_relocation(block, placed);
}

static bool zp_give(int64_t physical);

// ---- MKW_FILL_WINDOW (default 0, 1 in the PS5 profile; ps5_mem_fill_window): a unit's first contents (an unpack's bytes, a
// fresh unit's zeros) are written through a slot of one shared 2 MiB window instead of a private map of its own, and
// the window is unmapped once per 32 units. Each private map's unmap cost ~42 us alone (MKW_MAP_PROBE, 28/11/2026)
// and more under load: the race's 50-73 ms unpack frames carried 17 ms of private maps beside 35 ms of decompression
// (dr_1 frame 8474), read faults of the vk replay thread on units MKW_PACK had packed a lap before. A slot keeps its
// unit mapped (unused) until the flush, so a unit's physical memory is never given back or recycled while a slot still
// holds it: release_physical flushes first. Spin-locked: fills run in the fault handler. Lock order: unit, then fw.
enum { FW_UNITS = 32 };
static volatile int fw_on;
static char fw_spin;
static void *fw_va;                 // fw_spin: the window, NULL none yet (or lost)
static int64_t fw_phys[FW_UNITS];   // fw_spin: the units in its slots
static int fw_n;                    // fw_spin
static uint64_t fw_flushes;
static void fw_lock(void) { while (__atomic_test_and_set(&fw_spin, __ATOMIC_ACQUIRE)) {} }
static void fw_unlock(void) { __atomic_clear(&fw_spin, __ATOMIC_RELEASE); }
static void fw_flush_locked(void) {
    if (!fw_n) return;
    size_t win = (size_t)FW_UNITS * PS5_MEM_UNIT;
    if (sceKernelMunmap(fw_va, win)) {   // the slots stay mapped: no more fills through it, and none released while aliased
        static int said;
        if (!said++) fprintf(stderr, "PS5 memory: fill window: unmapping it failed; fills go back to private maps\n");
        __atomic_store_n(&fw_on, 0, __ATOMIC_RELEASE);
        return;   // fw_n kept: release_physical still finds those units (and fails to flush them: leaked, never reused)
    }
    __atomic_add_fetch(&fw_flushes, 1, __ATOMIC_RELAXED);
    fw_n = 0;
    void *again = fw_va;
    if (sceKernelReserveVirtualRange(&again, win, 0, PS5_MEM_UNIT) || again != fw_va) {
        if (again && again != fw_va) sceKernelMunmap(again, win);
        fw_va = NULL;   // reserved afresh at the next fill
    }
}
// false: a slot still maps the unit and the window could not be unmapped (the unit must stay allocated)
static bool fw_forget(int64_t physical) {
    if (!__atomic_load_n(&fw_n, __ATOMIC_ACQUIRE)) return true;
    bool ok = true;
    fw_lock();
    for (int i = 0; i < fw_n; i++)
        if (fw_phys[i] == physical) {
            fw_flush_locked();
            ok = !fw_n;
            break;
        }
    fw_unlock();
    return ok;
}

static int release_physical(Ps5MemBlock *block) {
    if (!block->allocated) return 0;
    if (!fw_forget(block->physical)) return PS5_MEM_RELOCATED;   // still mapped in the fill window: kept
    if (zp_give(block->physical)) {   // MKW_ZERO_POOL: kept, zeroed off the game's path, handed to the next commit
        block->allocated = false;
        return 0;
    }
    int result = sceKernelReleaseDirectMemory(block->physical, PS5_MEM_UNIT);
    if (!result) block->allocated = false;
    return result;
}

int ps5_mem_reserve(Ps5MemBlock *block, uintptr_t address) {
    if (block->address || block->relocated || !address || address % PS5_MEM_UNIT || address > UINTPTR_MAX - PS5_MEM_UNIT)
        return PS5_MEM_BAD_REQUEST;
    int result = reserve_exact(block, address);
    if (!result) {
        block->address = address;
        block->reserved = true;
    }
    return result;
}

static int probe_any_mapping(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    void *address = NULL;
    report->any_attempted = true;
    report->any_status = sceKernelMapDirectMemory(&address, PS5_MEM_UNIT, cpu_rw, 0,
                                                 block->physical, PS5_MEM_UNIT);
    report->any_address = address;
    if (report->any_status) return 0;
    if (!address) return PS5_MEM_RELOCATED;

    // Keep the physical allocation live until its temporary map is gone.
    report->any_unmap_status = sceKernelMunmap(address, PS5_MEM_UNIT);
    if (report->any_unmap_status && address != (void *)block->address) block->relocated = address;
    if (address != (void *)block->address) return report->any_unmap_status;
    // An OS-chosen map should never overlap the reservation. A successful
    // unmap in that case would leave no owned range for the fixed test.
    if (!report->any_unmap_status) block->reserved = false;
    return report->any_unmap_status ? report->any_unmap_status : PS5_MEM_RELOCATED;
}

// The unit's first contents (zeros, or a packed unit's bytes) are written through a private map of it, and only then is
// it mapped at its guest address. Zeroing after the fixed map let another thread write the unit in between (it no
// longer faults, so it never waits for the unit's lock) and lost that write: the title's MetaDataBase::setData crash
// (19/11/2026), a SharedBuffer header the game had just written read back as 0. Two more calls per unit backed.
// Step times (ps5_mem_commit_times): bench of 28/11/2026, an 83.6 ms frame carried 83.8 ms of unpacks, and the fault
// storms of streamed zstd output were all fresh units committed, with no duration accounted for them at all.
static uint64_t commit_ns[PS5_COMMIT_N];
#define COMMIT_ADD(k, v) __atomic_add_fetch(&commit_ns[k], (v), __ATOMIC_RELAXED)
void ps5_mem_commit_times(uint64_t out[PS5_COMMIT_N]) {
    for (int i = 0; i < PS5_COMMIT_N; i++) out[i] = __atomic_load_n(&commit_ns[i], __ATOMIC_RELAXED);
}

// MKW_FILL_WINDOW: the same through a slot of the shared window; 1 when the window could not take it (no change made).
static int fill_window(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx,
                       int *result) {
    if (!__atomic_load_n(&fw_on, __ATOMIC_ACQUIRE)) return 1;
    uint64_t t0 = plat_now_ns();
    fw_lock();
    size_t win = (size_t)FW_UNITS * PS5_MEM_UNIT;
    if (!fw_va) {
        void *v = NULL;
        if (!sceKernelReserveVirtualRange(&v, win, 0, PS5_MEM_UNIT) && v) fw_va = v;
    }
    if (fw_va && fw_n == FW_UNITS) fw_flush_locked();
    if (!fw_va || fw_n == FW_UNITS || !__atomic_load_n(&fw_on, __ATOMIC_ACQUIRE)) {
        fw_unlock();
        return 1;
    }
    void *slot = (char *)fw_va + (size_t)fw_n * PS5_MEM_UNIT, *at = slot;
    int rc = sceKernelMapDirectMemory(&at, PS5_MEM_UNIT, cpu_rw, PS5_KERNEL_MAP_FIXED, block->physical, PS5_MEM_UNIT);
    if (rc || at != slot) {
        if (!rc && at && sceKernelMunmap(at, PS5_MEM_UNIT)) block->relocated = at;   // as fill_private does
        fw_unlock();
        if (block->relocated) {
            *result = PS5_MEM_RELOCATED;
            return 0;
        }
        return 1;
    }
    fw_phys[fw_n] = block->physical;
    __atomic_store_n(&fw_n, fw_n + 1, __ATOMIC_RELEASE);
    uint64_t t1 = plat_now_ns();
    int r = fill ? fill(ctx, slot) : (memset(slot, 0, PS5_MEM_UNIT), 0);
    uint64_t t2 = plat_now_ns();
    fw_unlock();
    COMMIT_ADD(PS5_COMMIT_PRIVMAP, t1 - t0);
    COMMIT_ADD(PS5_COMMIT_FILL, t2 - t1);
    if (report) report->private_status = 0;
    *result = r;
    return 0;
}

int ps5_mem_fill_window(int on) {
    __atomic_store_n(&fw_on, on ? 1 : 0, __ATOMIC_RELEASE);
    if (!on) {   // the slots go now: nothing lingers mapped while off
        fw_lock();
        fw_flush_locked();
        fw_unlock();
    }
    return 0;
}
uint64_t ps5_mem_fill_window_flushes(void) { return __atomic_exchange_n(&fw_flushes, 0, __ATOMIC_RELAXED); }

static int fill_private(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx) {
    int wres;
    if (!fill_window(block, report, fill, ctx, &wres)) return wres;
    void *at = NULL;
    uint64_t t0 = plat_now_ns();
    int result = sceKernelMapDirectMemory(&at, PS5_MEM_UNIT, cpu_rw, 0, block->physical, PS5_MEM_UNIT);
    uint64_t t1 = plat_now_ns();
    if (report) report->private_status = result;
    if (result) return result;
    if (!at) return PS5_MEM_RELOCATED;
    if (fill) result = fill(ctx, at);
    else memset(at, 0, PS5_MEM_UNIT);   // the direct-memory allocator can return a previously used unit
    uint64_t t2 = plat_now_ns();
    int unmapped = sceKernelMunmap(at, PS5_MEM_UNIT);
    COMMIT_ADD(PS5_COMMIT_PRIVMAP, (t1 - t0) + (plat_now_ns() - t2));
    COMMIT_ADD(PS5_COMMIT_FILL, t2 - t1);
    if (report) report->private_unmap_status = unmapped;
    if (unmapped) block->relocated = at;   // still mapped: ps5_mem_release unmaps it, the physical unit is kept
    return result ? result : unmapped;
}

static int zp_take(int64_t *physical, bool fill);

static int commit(Ps5MemBlock *block, Ps5MemCommitReport *report, bool probe_any, int (*fill)(void *ctx, void *unit),
                  void *ctx) {
    if (report) *report = (Ps5MemCommitReport){0};
    if (!block->reserved || block->committed || block->allocated || block->relocated) return PS5_MEM_BAD_REQUEST;
    uint64_t c0 = plat_now_ns();
    // MKW_ZERO_POOL: a fresh unit the pool's thread zeroed already needs neither the allocation nor the private map; a
    // fill (an unpack) writes its own contents, so it takes a recycled unit not zeroed yet and skips the allocation only.
    int pooled = probe_any ? 0 : zp_take(&block->physical, fill != NULL);
    int result = pooled ? 0 : sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), PS5_MEM_UNIT,
                                                            PS5_MEM_UNIT, PS5_KERNEL_DIRECT_TYPE_CPU, &block->physical);
    COMMIT_ADD(PS5_COMMIT_ALLOC, plat_now_ns() - c0);
    if (report) *report = (Ps5MemCommitReport){.attempted = true, .allocation_status = result};
    if (result) return result;
    block->allocated = true;
    if (report) report->physical = block->physical;
    if (probe_any) {
        result = probe_any_mapping(block, report);
        if (result) return result;
    }

    result = pooled == 1 && !fill ? 0 : fill_private(block, report, fill, ctx);
    if (result) {
        int cleanup = block->relocated ? 0 : release_physical(block);
        return cleanup ? cleanup : result;
    }

    // MAP_FIXED replaces the single reservation this block already owns. Use
    // the direct-unit alignment seen in Hookshot a3551c6's guest mappings.
    void *mapped = (void *)block->address;
    uint64_t f0 = plat_now_ns();
    result = sceKernelMapDirectMemory(&mapped, PS5_MEM_UNIT, cpu_rw, PS5_KERNEL_MAP_FIXED,
                                      block->physical, PS5_MEM_UNIT);
    uint64_t f1 = plat_now_ns();
    COMMIT_ADD(PS5_COMMIT_FIXED, f1 - f0);
    if (report) {
        report->fixed_attempted = true;
        report->fixed_address = mapped;
        report->mapping_status = result;
    }
    if (!result && mapped != (void *)block->address) result = discard_relocation(block, mapped);
    if (result) {
        int cleanup = block->relocated ? 0 : release_physical(block);
        return cleanup ? cleanup : result;
    }
    block->committed = true;
    if (!fill) COMMIT_ADD(PS5_COMMIT_FRESH_NS, f1 - c0), COMMIT_ADD(PS5_COMMIT_FRESH_N, 1);
    return 0;
}

// ---- MKW_ZERO_POOL (default 0; ps5_mem_zero_pool): direct-memory units kept zeroed ahead of the commits
// that need them. Pool-churn logs of 28/11/2026 (slice0_*, slice1_*, 1.8.0 GP): the game maps and touches 5-21 MiB
// pools and frees them ~0.1 s later; the worst race frames (113-122 ms) were 600-700 fresh units backed in one frame,
// ~166 us each on the faulting thread: allocation ~13, private map and unmap ~97 (only to zero it before the guest
// sees it), memset ~7, fixed map ~49. A released unit (decommit, release, MKW_PACK's) goes to the pool instead of back
// to the system; the "zeropool" thread zeroes it (32 at a time through one window, zp_zero_batch), and tops the pool
// up with fresh allocations while direct memory is plentiful. A commit then pays the fixed map alone. Scored 28/11/2026
// with MKW_DECOMMIT_RUN (drz_* against dr_*): no clear gain over DECOMMIT_RUN alone, so it stays off. Zeroed before the
// fixed map, as fill_private's comment requires. Nothing maps a pooled unit: its guest map and private map are gone
// when it comes in (a still-mapped unit stays `relocated` and never reaches release_physical), and change() refuses
// to decommit a range a GPU import still holds. Spin-locked like the units: commits run in the fault handler.
// Zeroed only up to ZP_READY: at first every recycled unit was zeroed, ~17.5k a race, though only ~6.5k fresh commits
// needed zeros; the other ~11k went to unpacks, which write their own bytes through a private map anyway, and the 22k
// extra map calls doubled the race's slow protection changes (zp1_2, zs_1 against zp0_*, 28/11/2026: frames over 20 ms
// 236-253 -> 322-351; the kernel serialises the process's map calls).
enum { ZP_CAP = 768,            // 48 MiB: two of the game's 21 MiB pools in flight
       ZP_READY = 448,          // zeroed units kept for fresh commits (28 MiB: one pool and a half)
       ZP_TOP = 192 };          // fresh units allocated ahead (12 MiB) when the pool runs low
// Both against sceKernelAvailableDirectMemorySize, the largest free area: 320-450 MiB through a 1.8.0 GP while all
// areas held 925-1184 MiB (zp1_1, 28/11/2026; at 512 the pool gave every unit back and never served a commit). The
// governor's floor on the same number is 128 MiB (vk_mem_upload.c DMEM_FLOOR_BYTES).
#ifndef ZP_TOP_AVAIL
#define ZP_TOP_AVAIL (256ull << 20)    // fresh units allocated ahead only while this much is free
#endif
#ifndef ZP_TRIM_AVAIL
#define ZP_TRIM_AVAIL (160ull << 20)   // below it the pool gives its units back to the system (host tests: 0)
#endif
static volatile int zp_on;
static int zp_started;
static char zp_spin;
static int64_t zp_ready[ZP_CAP], zp_dirty[ZP_CAP];   // zp_spin
static int zp_nready, zp_ndirty;                     // zp_spin
static uint64_t zp_st[5];   // taken zeroed, taken dirty, missed, recycled, zeroed by the thread
static void zp_lock(void) { while (__atomic_test_and_set(&zp_spin, __ATOMIC_ACQUIRE)) {} }
static void zp_unlock(void) { __atomic_clear(&zp_spin, __ATOMIC_RELEASE); }

// 1 a zeroed unit, 2 a recycled one not zeroed yet (its first contents still to write), 0 none. A fill takes a zeroed
// one only when no other is left.
static int zp_take(int64_t *physical, bool fill) {
    if (!__atomic_load_n(&zp_on, __ATOMIC_ACQUIRE)) return 0;
    int got = 0;
    zp_lock();
    if (fill && zp_ndirty) *physical = zp_dirty[--zp_ndirty], got = 2;
    else if (zp_nready) *physical = zp_ready[--zp_nready], got = 1;
    else if (zp_ndirty) *physical = zp_dirty[--zp_ndirty], got = 2;
    zp_unlock();
    __atomic_add_fetch(&zp_st[got ? got - 1 : 2], 1, __ATOMIC_RELAXED);
    return got;
}

static bool zp_give(int64_t physical) {
    if (!__atomic_load_n(&zp_on, __ATOMIC_ACQUIRE)) return false;
    bool kept = false;
    zp_lock();
    if (zp_nready + zp_ndirty < ZP_CAP) zp_dirty[zp_ndirty++] = physical, kept = true;
    zp_unlock();
    if (kept) __atomic_add_fetch(&zp_st[3], 1, __ATOMIC_RELAXED);
    return kept;
}

static uint64_t zp_avail(void) {
    int64_t phys = 0;
    size_t avail = 0;
    return sceKernelAvailableDirectMemorySize(0, INT64_MAX, PS5_MEM_UNIT, &phys, &avail) == 0 ? avail : UINT64_MAX;
}

static bool zp_zero(int64_t physical) {   // false: the private map failed (the unit is released) or stayed (leaked)
    void *at = NULL;
    if (sceKernelMapDirectMemory(&at, PS5_MEM_UNIT, cpu_rw, 0, physical, PS5_MEM_UNIT) || !at) {
        sceKernelReleaseDirectMemory(physical, PS5_MEM_UNIT);
        return false;
    }
    memset(at, 0, PS5_MEM_UNIT);
    if (sceKernelMunmap(at, PS5_MEM_UNIT)) {
        static int said;
        if (!said++) fprintf(stderr, "PS5 memory: zero pool: unmapping a zeroed unit failed; the unit is left mapped\n");
        return false;
    }
    return true;
}

// Up to 32 units zeroed through one 2 MiB window: each mapped into its slot (~2 us), zeroed, then one unmap for all
// (14.5 us against ~42 for each unit's own, MKW_MAP_PROBE 28/11/2026). Zeroing every recycled unit through its own
// private map, ~17.5k a race, lowered the race's 1 % low by 2-4 fps: each unmap interrupts every core. p[] keeps the
// zeroed ones (the count returned); the others are given back to the system.
enum { ZP_WIN_UNITS = 32 };
static void *zp_win;   // the thread's window, reserved while idle; NULL: none yet (or lost)
static int zp_zero_batch(int64_t *p, int n) {
    if (!zp_win) {
        void *v = NULL;
        if (!sceKernelReserveVirtualRange(&v, (size_t)ZP_WIN_UNITS * PS5_MEM_UNIT, 0, PS5_MEM_UNIT) && v) zp_win = v;
    }
    if (!zp_win) {   // unit by unit
        int k = 0;
        for (int i = 0; i < n; i++)
            if (zp_zero(p[i])) p[k++] = p[i];
        return k;
    }
    bool in[ZP_WIN_UNITS] = {0};
    for (int i = 0; i < n; i++) {
        void *slot = (char *)zp_win + (size_t)i * PS5_MEM_UNIT, *at = slot;
        int rc = sceKernelMapDirectMemory(&at, PS5_MEM_UNIT, cpu_rw, PS5_KERNEL_MAP_FIXED, p[i], PS5_MEM_UNIT);
        if (!rc && at == slot) {
            memset(slot, 0, PS5_MEM_UNIT);
            in[i] = true;
        } else {
            if (!rc && at && sceKernelMunmap(at, PS5_MEM_UNIT)) continue;   // still mapped elsewhere: kept, unused
            sceKernelReleaseDirectMemory(p[i], PS5_MEM_UNIT);
        }
    }
    size_t win = (size_t)ZP_WIN_UNITS * PS5_MEM_UNIT;
    if (sceKernelMunmap(zp_win, win)) {   // still mapped: the units stay there, unused
        static int said;
        if (!said++) fprintf(stderr, "PS5 memory: zero pool: unmapping its window failed; its units are left mapped\n");
        zp_win = NULL;
        return 0;
    }
    void *again = zp_win;
    if (sceKernelReserveVirtualRange(&again, win, 0, PS5_MEM_UNIT) || again != zp_win) {
        if (again && again != zp_win) sceKernelMunmap(again, win);
        zp_win = NULL;   // reserved afresh at the next batch
    }
    int k = 0;
    for (int i = 0; i < n; i++)
        if (in[i]) p[k++] = p[i];
    return k;
}

static void *zp_main(void *arg) {
    (void)arg;
    uint64_t avail = UINT64_MAX, avail_t = 0;
    for (;;) {
        uint64_t now = plat_now_ns();
        if (now - avail_t >= 20000000ull) avail = zp_avail(), avail_t = now;   // ~20 us a call: 50 a second
        int on = __atomic_load_n(&zp_on, __ATOMIC_ACQUIRE);
        int64_t batch[ZP_WIN_UNITS], give = 0;
        int n = 0, what = 0;   // 1 zero recycled units, 2 give one back, 3 allocate some ahead
        zp_lock();
        if (!on || avail < ZP_TRIM_AVAIL) {
            if (zp_ndirty) give = zp_dirty[--zp_ndirty], what = 2;
            else if (zp_nready) give = zp_ready[--zp_nready], what = 2;
        } else if (zp_ndirty && zp_nready < ZP_READY) {
            while (n < ZP_WIN_UNITS && zp_ndirty && zp_nready + n < ZP_READY) batch[n++] = zp_dirty[--zp_ndirty];
            what = 1;
        } else if (zp_nready < ZP_TOP && avail >= ZP_TOP_AVAIL) {
            n = ZP_TOP - zp_nready < ZP_WIN_UNITS ? ZP_TOP - zp_nready : ZP_WIN_UNITS;
            what = 3;
        }
        zp_unlock();
        if (!what) {
            plat_sleep_ns(1000000);
            continue;
        }
        if (what == 2) {
            sceKernelReleaseDirectMemory(give, PS5_MEM_UNIT);
            continue;
        }
        if (what == 3) {
            int got = 0;
            while (got < n && !sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), PS5_MEM_UNIT,
                                                             PS5_MEM_UNIT, PS5_KERNEL_DIRECT_TYPE_CPU, &batch[got]))
                got++;
            if (got < n) avail = 0, avail_t = plat_now_ns();   // out of memory: no more ahead until the next look
            else avail = avail > (uint64_t)got * PS5_MEM_UNIT ? avail - (uint64_t)got * PS5_MEM_UNIT : 0;
            n = got;
        }
        n = zp_zero_batch(batch, n);
        __atomic_add_fetch(&zp_st[4], (uint64_t)n, __ATOMIC_RELAXED);
        int kept = 0;
        zp_lock();
        while (kept < n && zp_nready + zp_ndirty < ZP_CAP) zp_ready[zp_nready++] = batch[kept++];
        zp_unlock();
        for (; kept < n; kept++) sceKernelReleaseDirectMemory(batch[kept], PS5_MEM_UNIT);
    }
    return NULL;
}

// The runtime's own spawn where it is linked (ps5_threads.c: the default CPU mask, a signal stack, the name); the host
// model of the tests has none.
extern int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) __attribute__((weak));
static void zp_entry(void *arg) { zp_main(arg); }

int ps5_mem_zero_pool(int on) {
    if (on && !zp_started) {
        if (plat_thread_spawn) {
            if (!plat_thread_spawn(zp_entry, NULL, "zeropool", 0)) return -1;
        } else {
            pthread_t t;
            if (pthread_create(&t, NULL, zp_main, NULL)) return -1;
            pthread_detach(t);
        }
        zp_started = 1;
    }
    __atomic_store_n(&zp_on, on ? 1 : 0, __ATOMIC_RELEASE);   // off: the thread gives the units back
    return 0;
}

uint64_t ps5_mem_zero_pool_bytes(void) {
    zp_lock();
    uint64_t n = (uint64_t)(zp_nready + zp_ndirty);
    zp_unlock();
    return n * PS5_MEM_UNIT;
}

void ps5_mem_zero_pool_stats(uint64_t out[7]) {
    for (int i = 0; i < 5; i++) out[i] = __atomic_exchange_n(&zp_st[i], 0, __ATOMIC_RELAXED);
    zp_lock();
    out[5] = (uint64_t)zp_nready, out[6] = (uint64_t)zp_ndirty;
    zp_unlock();
}

// MKW_MAP_PROBE=1 (once, at the first frame): what one fixed map of 2 MiB of contiguous direct memory costs against
// 32 maps of 64 KiB, and whether 64 KiB of a 2 MiB allocation can be unmapped and released on its own. Decides whether
// a fresh pool's units can be backed several to a call: with the zero pool, 748 fresh units in one frame still took
// 55 ms in their fixed maps alone (zu_1, 28/11/2026, ~74 us each).
void ps5_mem_map_probe(char *out, size_t cap) {
    enum { N = 32, BIG = N * PS5_MEM_UNIT };
    void *va = NULL;
    if (sceKernelReserveVirtualRange(&va, BIG, 0, BIG) || !va) {
        snprintf(out, cap, "no reservation");
        return;
    }
    uint64_t best[6] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};   // alloc, map 1, unmap 1, map 32, unmap 32, touch
    int rc_part_unmap = 0, rc_part_rel = 0, rc_rest = 0;
    for (int round = 0; round < 4; round++) {
        int64_t phys = 0;
        uint64_t t0 = plat_now_ns();
        if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), BIG, PS5_MEM_UNIT, PS5_KERNEL_DIRECT_TYPE_CPU,
                                          &phys)) {
            snprintf(out, cap, "no 2 MiB of direct memory");
            sceKernelMunmap(va, BIG);
            return;
        }
        uint64_t t1 = plat_now_ns(), t[6] = {t1 - t0};
        void *at = va;
        int rc = sceKernelMapDirectMemory(&at, BIG, cpu_rw, PS5_KERNEL_MAP_FIXED, phys, PS5_MEM_UNIT);
        uint64_t t2 = plat_now_ns();
        t[1] = t2 - t1;
        if (!rc) {
            for (int i = 0; i < N; i++) ((volatile char *)va)[(size_t)i * PS5_MEM_UNIT] = 1;
            t[5] = plat_now_ns() - t2;
            t2 = plat_now_ns();
            rc = sceKernelMunmap(va, BIG);
            t[2] = plat_now_ns() - t2;
        }
        void *again = va;
        if (!rc) rc = sceKernelReserveVirtualRange(&again, BIG, 0, BIG) || again != va;   // as reserve_exact
        uint64_t t3 = plat_now_ns();
        for (int i = 0; !rc && i < N; i++) {
            void *u = (char *)va + (size_t)i * PS5_MEM_UNIT;
            rc = sceKernelMapDirectMemory(&u, PS5_MEM_UNIT, cpu_rw, PS5_KERNEL_MAP_FIXED, phys + (int64_t)i * PS5_MEM_UNIT,
                                          PS5_MEM_UNIT);
        }
        uint64_t t4 = plat_now_ns();
        t[3] = t4 - t3;
        for (int i = 0; !rc && i < N; i++) rc = sceKernelMunmap((char *)va + (size_t)i * PS5_MEM_UNIT, PS5_MEM_UNIT);
        t[4] = plat_now_ns() - t4;
        if (rc) {
            snprintf(out, cap, "round %d failed (%#x)", round, rc);
            sceKernelReleaseDirectMemory(phys, BIG);
            sceKernelMunmap(va, BIG);
            return;
        }
        for (int k = 0; k < 6; k++) if (t[k] < best[k]) best[k] = t[k];
        if (round == 3) {   // a part on its own: map whole, unmap and release the first unit, then the rest
            again = va;
            sceKernelReserveVirtualRange(&again, BIG, 0, BIG);
            at = va;
            if (!sceKernelMapDirectMemory(&at, BIG, cpu_rw, PS5_KERNEL_MAP_FIXED, phys, PS5_MEM_UNIT)) {
                rc_part_unmap = sceKernelMunmap(va, PS5_MEM_UNIT);
                rc_part_rel = sceKernelReleaseDirectMemory(phys, PS5_MEM_UNIT);
                ((volatile char *)va)[PS5_MEM_UNIT] = 2;   // the rest still mapped and writable
                sceKernelMunmap((char *)va + PS5_MEM_UNIT, BIG - PS5_MEM_UNIT);
                rc_rest = sceKernelReleaseDirectMemory(phys + PS5_MEM_UNIT, BIG - PS5_MEM_UNIT);
                if (rc_part_rel) sceKernelReleaseDirectMemory(phys, PS5_MEM_UNIT);
            } else sceKernelReleaseDirectMemory(phys, BIG);
        } else {
            sceKernelReleaseDirectMemory(phys, BIG);
            again = va;
            sceKernelReserveVirtualRange(&again, BIG, 0, BIG);
        }
    }
    sceKernelMunmap(va, BIG);
    snprintf(out, cap, "best of 4, us: alloc 2 MiB %.1f; one fixed map of 2 MiB %.1f, its unmap %.1f, "
            "first touch of its 32 units %.1f; 32 fixed maps of 64 KiB %.1f, their unmaps %.1f; part unmap %#x, part "
            "release %#x, rest release %#x", best[0] / 1e3, best[1] / 1e3, best[2] / 1e3, best[5] / 1e3,
            best[3] / 1e3, best[4] / 1e3, rc_part_unmap, rc_part_rel, rc_rest);
}

int ps5_mem_commit(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    return commit(block, report, false, NULL, NULL);
}

int ps5_mem_commit_fill(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx) {
    return commit(block, report, false, fill, ctx);
}

int ps5_mem_commit_probe(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    if (!report) return PS5_MEM_BAD_REQUEST;
    return commit(block, report, true, NULL, NULL);
}

static _Thread_local Ps5ProtBatch *tl_batch;   // ps5_mem_batch_begin
static void batch_flush(Ps5ProtBatch *b);

int ps5_mem_protect_native(Ps5MemBlock *block, size_t offset, size_t bytes, Ps5MemProt protection) {
    const int flags[] = {0, PS5_KERNEL_PROT_CPU_READ, PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE};
    if (!block->committed || !bytes || offset >= PS5_MEM_UNIT || bytes > PS5_MEM_UNIT - offset ||
        offset % PS5_MEM_PAGE || bytes % PS5_MEM_PAGE || (unsigned)protection >= sizeof flags / sizeof *flags)
        return PS5_MEM_BAD_REQUEST;
    int prot = flags[protection];
    if (protection > 0 && prot_watch_has_gpu(block->address + offset, bytes))
        prot |= PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
    Ps5ProtBatch *b = tl_batch;
    if (!b) return sceKernelMprotect((void *)(block->address + offset), bytes, prot);
    for (uintptr_t p = block->address + offset, end = p + bytes; p < end; p += PS5_MEM_PAGE) {
        if (b->n && (p < b->base || (p - b->base) / PS5_MEM_PAGE >= PS5_PROT_BATCH_PAGES)) batch_flush(b);
        if (!b->n) b->base = p;
        int i = (int)((p - b->base) / PS5_MEM_PAGE);
        while (b->n <= i) b->prot[b->n++] = -1;   // pages between that change nothing
        b->prot[i] = (int16_t)prot;
    }
    return 0;
}

static void batch_flush(Ps5ProtBatch *b) {
    for (int i = 0; i < b->n;) {
        int j = i + 1;
        while (j < b->n && b->prot[j] == b->prot[i]) j++;
        if (b->prot[i] >= 0) {
            int r = sceKernelMprotect((void *)(b->base + (uintptr_t)i * PS5_MEM_PAGE), (size_t)(j - i) * PS5_MEM_PAGE,
                                      b->prot[i]);
            if (r && !b->err) b->err = r;
        }
        i = j;
    }
    b->n = 0;
}

void ps5_mem_batch_begin(Ps5ProtBatch *batch) {
    batch->n = 0, batch->err = 0, batch->base = 0;
    tl_batch = batch;
}

int ps5_mem_batch_flush(void) {
    Ps5ProtBatch *b = tl_batch;
    if (!b) return 0;
    batch_flush(b);
    int e = b->err;
    b->err = 0;
    return e;
}

Ps5ProtBatch *ps5_mem_batch_swap(Ps5ProtBatch *batch) {
    Ps5ProtBatch *prev = tl_batch;
    tl_batch = batch;
    return prev;
}

int ps5_mem_batch_end(void) {
    int e = ps5_mem_batch_flush();
    tl_batch = NULL;
    return e;
}

int ps5_mem_protect(Ps5MemBlock *block, size_t offset, size_t bytes, bool writable) {
    return ps5_mem_protect_native(block, offset, bytes, writable ? PS5_MEM_RW : PS5_MEM_READ);
}

int ps5_mem_decommit(Ps5MemBlock *block) {
    if (!block->committed) return PS5_MEM_BAD_REQUEST;
    int result = sceKernelMunmap((void *)block->address, PS5_MEM_UNIT);
    if (result) return result;
    block->reserved = false;
    block->committed = false;

    // Reacquire without replacement. If someone took the hole, release only
    // our physical unit and leave the handle in a releasable state.
    result = reserve_exact(block, block->address);
    if (!result) block->reserved = true;
    int cleanup = release_physical(block);
    return cleanup ? cleanup : result;
}

int ps5_mem_decommit_run(Ps5MemBlock *const *blocks, int n) {
    if (n <= 0) return PS5_MEM_BAD_REQUEST;
    uintptr_t base = blocks[0]->address;
    for (int i = 0; i < n; i++)
        if (!blocks[i]->committed || blocks[i]->relocated || blocks[i]->address != base + (uintptr_t)i * PS5_MEM_UNIT)
            return PS5_MEM_BAD_REQUEST;
    size_t bytes = (size_t)n * PS5_MEM_UNIT;
    int result = sceKernelMunmap((void *)base, bytes);
    if (result) return result;   // nothing changed
    for (int i = 0; i < n; i++) blocks[i]->reserved = false, blocks[i]->committed = false;
    // The run's reservation back in one call where it lands in place; unit by unit (reserve_exact) where it does not.
    void *placed = (void *)base;
    result = sceKernelReserveVirtualRange(&placed, bytes, 0, PS5_MEM_UNIT);
    if (!result && placed != (void *)base) result = sceKernelMunmap(placed, bytes) ? PS5_MEM_RELOCATED : 1;
    if (!result)
        for (int i = 0; i < n; i++) blocks[i]->reserved = true;
    int first = 0;
    for (int i = 0; i < n; i++) {
        int r = 0;
        if (result) {
            r = reserve_exact(blocks[i], blocks[i]->address);
            if (!r) blocks[i]->reserved = true;
        }
        int cleanup = release_physical(blocks[i]);
        if (!first) first = r ? r : cleanup;
    }
    return result == PS5_MEM_RELOCATED ? result : first;
}

int ps5_mem_release(Ps5MemBlock *block) {
    if (block->relocated) {
        int result = sceKernelMunmap(block->relocated, PS5_MEM_UNIT);
        if (result) return result;
        block->relocated = NULL;
    }
    if (block->reserved) {
        int result = sceKernelMunmap((void *)block->address, PS5_MEM_UNIT);
        if (result) return result;
        block->reserved = false;
        block->committed = false;
    }
    int result = release_physical(block);
    if (!result) *block = (Ps5MemBlock){0};
    return result;
}

void *ps5_mem_map_host(size_t bytes, size_t align, int64_t *physical) {
    if (!bytes || bytes % PS5_MEM_UNIT || !align || align % PS5_MEM_UNIT) return NULL;
#ifdef __PROSPERO__
    int64_t phys = 0;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, PS5_MEM_UNIT,
                                      PS5_KERNEL_DIRECT_TYPE_CPU, &phys))
        return NULL;
    void *at = NULL;
    if (sceKernelMapDirectMemory(&at, bytes, cpu_rw, 0, phys, align) || !at) {
        sceKernelReleaseDirectMemory(phys, bytes);
        return NULL;
    }
    *physical = phys;
    return at;
#else
    *physical = -1;
    return aligned_alloc(align, bytes);
#endif
}

int ps5_mem_unmap_host(void *at, size_t bytes, int64_t physical) {
#ifdef __PROSPERO__
    int result = sceKernelMunmap(at, bytes);
    if (result) return result;   // still mapped: the memory stays ours
    return sceKernelReleaseDirectMemory(physical, bytes);
#else
    (void)bytes, (void)physical;
    free(at);
    return 0;
#endif
}
