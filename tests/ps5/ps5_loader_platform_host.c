// Linux memory mappings model the SDK's 16 KiB protection and 64 KiB backing.
#define _GNU_SOURCE   // (dladdr)
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <ps5platform/kernel.h>

#include "guest_layout.h"
#include "platform.h"
#include "ps5_guest_memory.h"
#include "ps5_memory.h"
#include "prot_watch.h"
#include "memory_int.h"
#include "host.h"

enum { UNIT = PS5_KERNEL_DIRECT_ALIGNMENT, PAGE = 0x1000, ERROR = -41 };
uintptr_t plat_exe_base(void) { return 0; }

static const uint64_t sample = RT_ALIAS_BASE + (1ull << 30);
static const uint64_t unexpected = 176ull << 30;
static const uint64_t hint = 175ull << 30;
static bool units[16];
static unsigned allocations;
static bool lose_slot, foreign, pending;

static void *reserve_at(uint64_t address, size_t bytes, int protection) {
    void *p = mmap((void *)(uintptr_t)address, bytes, protection, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p == (void *)(uintptr_t)address);
    return p;
}

static uintptr_t windows[8][2];   // the system-chosen reservations: a unit is filled there, so it may be stale first
static int nwindows;
static bool in_window(uintptr_t a) {
    for (int i = 0; i < nwindows; i++)
        if (a >= windows[i][0] && a < windows[i][1]) return true;
    return false;
}

int32_t sceKernelReserveVirtualRange(void **address, size_t bytes, int flags, size_t alignment) {
    uint64_t at = (uintptr_t)*address;
    assert(!flags && alignment == UNIT && !(at % UNIT) && !(bytes % UNIT));
    if (!at) {   // the system's choice (the zero pool's and the fill's windows), aligned as asked
        uint8_t *q = mmap(NULL, bytes + UNIT, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        assert(q != MAP_FAILED);
        uint8_t *a = (uint8_t *)(((uintptr_t)q + UNIT - 1) & ~(uintptr_t)(UNIT - 1));
        if (a > q) munmap(q, (size_t)(a - q));
        munmap(a + bytes, (size_t)(q + bytes + UNIT - (a + bytes)));
        *address = a;
        if (nwindows < 8) windows[nwindows][0] = (uintptr_t)a, windows[nwindows++][1] = (uintptr_t)a + bytes;
        return 0;
    }
    for (int i = 0; i < nwindows; i++)   // a window reserved again in place
        if (at == windows[i][0]) {
            *address = reserve_at(at, bytes, PROT_NONE);
            return 0;
        }
    if (at == hint) {
        *address = reserve_at(unexpected, bytes, PROT_NONE);
        pending = true;
        return 0;
    }
    if (at == sample && lose_slot) {
        foreign = true;
        *(volatile unsigned char *)reserve_at(at, bytes, PROT_READ | PROT_WRITE) = 0x6d;
        return ERROR;
    }
    *address = reserve_at(at, bytes, PROT_NONE);
    return 0;
}

int64_t sceKernelGetDirectMemorySize(void) { return sizeof units * UNIT; }

int32_t sceKernelAvailableDirectMemorySize(int64_t lo, int64_t hi, size_t align, int64_t *phys, size_t *avail) {
    (void)align;   // as the console seems to: the largest free run of the range
    size_t first = (size_t)lo / UNIT, last = (uint64_t)hi / UNIT < sizeof units ? (size_t)((uint64_t)hi / UNIT) : sizeof units;
    size_t best = 0, best_at = 0, run = 0;
    for (size_t i = first; i < last; ++i) {
        run = units[i] ? 0 : run + 1;
        if (run > best) best = run, best_at = i + 1 - run;
    }
    if (phys) *phys = (int64_t)(best_at * UNIT);
    if (avail) *avail = best * UNIT;
    return 0;
}

int32_t sceKernelAvailableFlexibleMemorySize(size_t *avail) {
    if (avail) *avail = 256ull << 20;
    return 0;
}

int32_t sceKernelConfiguredFlexibleMemorySize(size_t *cfg) {
    if (cfg) *cfg = 512ull << 20;
    return 0;
}

int sceKernelGetPageTableStats(int *cpu_total, int *cpu_free, int *gpu_total, int *gpu_free) {
    if (cpu_total) *cpu_total = 16384;
    if (cpu_free) *cpu_free = 12000;
    if (gpu_total) *gpu_total = 16384;
    if (gpu_free) *gpu_free = 12000;
    return 0;
}

// Physical memory is a shared file, so a unit mapped twice (ps5_memory.c's private fill map, then the fixed map) shows
// the same bytes in both. A unit allocated again holds a stale pattern until it is filled.
static const uint64_t STALE = 0xbabababababababaull;
static int phys_fd = -1;
static uint8_t *phys;

static void phys_init(void) {
    if (phys) return;
    phys_fd = memfd_create("ps5-direct", 0);
    assert(phys_fd >= 0 && !ftruncate(phys_fd, (off_t)(sizeof units * UNIT)));
    phys = mmap(NULL, sizeof units * UNIT, PROT_READ | PROT_WRITE, MAP_SHARED, phys_fd, 0);
    assert(phys != MAP_FAILED);
}

int32_t sceKernelAllocateDirectMemory(int64_t lo, int64_t hi, size_t bytes, size_t alignment, int type, int64_t *offset) {
    assert(!lo && hi == sceKernelGetDirectMemorySize() && bytes == UNIT && alignment == UNIT);
    assert(type == PS5_KERNEL_DIRECT_TYPE_CPU);
    phys_init();
    for (size_t i = 0; i < sizeof units; ++i) {
        if (units[i]) continue;
        memset(phys + i * UNIT, 0xba, UNIT);
        units[i] = true;
        ++allocations;
        *offset = (int64_t)(i * UNIT);
        return 0;
    }
    return ERROR;
}

int32_t sceKernelReleaseDirectMemory(int64_t offset, size_t bytes) {
    assert(offset >= 0 && (size_t)offset < sizeof units * UNIT && bytes == UNIT && units[offset / UNIT]);
    units[offset / UNIT] = false;
    --allocations;
    return 0;
}

static volatile unsigned private_maps;   // bumped in the fault handler too

int32_t sceKernelMapDirectMemory(void **address, size_t bytes, int protection, int flags, int64_t offset, size_t align) {
    assert(bytes == UNIT && align == UNIT && units[offset / UNIT]);
    assert(protection == (PROT_READ | PROT_WRITE));
    if (!flags) {   // OS-chosen: the private map that writes the unit's first contents
        assert(!*address);
        void *p = mmap(NULL, bytes, protection, MAP_SHARED, phys_fd, offset);
        assert(p != MAP_FAILED);
        *address = p;
        ++private_maps;
        return 0;
    }
    assert(flags == PS5_KERNEL_MAP_FIXED);
    assert((uintptr_t)*address != unexpected && (!foreign || (uintptr_t)*address != sample));
    // Nothing stale may be seen at the guest address, even for an instant: the unit is filled before this map.
    const uint64_t *w = (const uint64_t *)(phys + offset);
    if (in_window((uintptr_t)*address)) ++private_maps;   // the fill window's slot: the map that writes first contents
    for (size_t k = 0; !in_window((uintptr_t)*address) && k < UNIT / sizeof *w; ++k) assert(w[k] != STALE);
    void *p = mmap(*address, bytes, protection, MAP_SHARED | MAP_FIXED, phys_fd, offset);
    assert(p == *address);
    return 0;
}

int32_t sceKernelMunmap(void *address, size_t bytes) {
    uint64_t start = (uintptr_t)address;
    assert(!foreign || start >= sample + UNIT || start + bytes <= sample);
    if (start == unexpected && pending) {
        pending = false;
        return ERROR;
    }
    return munmap(address, bytes);
}

static unsigned early_protect_calls, early_protect_failures;

int32_t sceKernelMprotect(const void *address, size_t bytes, int protection) {
    assert(!((uintptr_t)address % PS5_KERNEL_PAGE_SIZE) && !(bytes % PS5_KERNEL_PAGE_SIZE));
    if (early_protect_failures) {
        assert(early_protect_calls < 8);
        unsigned call = early_protect_calls++;
        if (early_protect_failures & (1u << call)) {
            if (call == 1) assert(!mprotect((void *)address, PS5_KERNEL_PAGE_SIZE, PROT_NONE));
            return ERROR;
        }
    }
    return mprotect((void *)address, bytes, protection & (PROT_READ | PROT_WRITE));
}

int32_t sceKernelQueryMemoryProtection(void *address, void **start, void **end, uint32_t *protection) {
    FILE *maps = fopen("/proc/self/maps", "r");
    assert(maps);
    char line[256], perms[5];
    unsigned long lo, hi;
    while (fgets(line, sizeof line, maps)) {
        if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3 || (uintptr_t)address < lo ||
            (uintptr_t)address >= hi) continue;
        *start = (void *)lo;
        *end = (void *)hi;
        *protection = (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0);
        fclose(maps);
        return 0;
    }
    fclose(maps);
    return ERROR;
}

void catchReturnFromMain(int status) { exit(status); }

static void state(uint64_t address, uint64_t limit, int expected, uint64_t next) {
    uint64_t end;
    assert(plat_mem_state(address, limit, &end) == expected && end == next);
}

static uint64_t hook_fault_addr, hook_watch_addr, hook_watch_rip, hook_crash_addr;
static int hook_fault_ret, hook_watch_ret;

static int test_fault_hook(uint64_t addr) {
    hook_fault_addr = addr;
    if (hook_fault_ret) plat_mem_protect(addr, PAGE, PLAT_PROT_RW);
    return hook_fault_ret;
}

static int test_watch_hook(uint64_t addr, uint64_t rip) {
    hook_watch_addr = addr;
    hook_watch_rip = rip;
    if (hook_watch_ret) plat_mem_protect(addr, PAGE, PLAT_PROT_RW);
    return hook_watch_ret;
}

static void test_oom_crash_hook(const PlatCrash *c) {
    // Verify fault_hook and watch_hook were NOT called when backing failed, and crash_hook was called.
    assert(hook_fault_addr == 0 && hook_watch_addr == 0);
    assert(c->addr == RT_LOW_BASE && c->access == 1);
    _exit(42);
}

static void check_early_page(uint64_t addr, uint32_t prot, int early) {
    void *lo, *hi;
    uint32_t actual;
    assert(!sceKernelQueryMemoryProtection((void *)(uintptr_t)addr, &lo, &hi, &actual));
    assert(actual == prot && prot_watch_is_early(addr) == early);
}

PlatMutex frz_lock;
_Thread_local int frz_held;
unsigned rt_trace;   // memory_int.h: frz_take notes frz_lock's holder only with MKW_TRACE=perf (memory.c)
void frz_note(uintptr_t site) { (void)site; }
void frz_unnote(void) {}
uint8_t *trapped;

static uint64_t test_watch_addr;
static int expect_rollback_fatal;

_Thread_local struct RtThread *tcur;
int kern_fault_in_game(uint64_t rip) { (void)rip; return 0; }
void mem_early_ev_note(int kind, uint64_t a, uint64_t w, uint64_t done, uint64_t pc) {
    (void)kind; (void)a; (void)w; (void)done; (void)pc;
}

const char *cfg_get(const char *key) { return !strcmp(key, "MKW_EARLY_READ") ? "1" : NULL; }
int rt_knob(const char *name, const char *def) { (void)name; return atoi(def); }
int rt_ab(void) { return 0; }
void rt_log(const char *fmt, ...) { (void)fmt; }
void hang_reporter(void (*report)(char *, int)) { (void)report; }
void prof_names_init(void) {}
int mem_trapped_init(void) { return trapped != NULL; }
int watched(uint64_t addr, uint64_t size) {
    return addr == test_watch_addr && size == 3 * PS5_KERNEL_PAGE_SIZE;
}
void mem_base_fill_locked(uint64_t addr) { (void)addr; abort(); }
void mem_spill_fill_locked(uint64_t p, int why) { (void)p, (void)why; abort(); }   // no spill on the PS5
void rt_fatal(const char *fmt, ...) {
    assert(expect_rollback_fatal && strstr(fmt, "rollback failed") && early_protect_calls == 4);
    const uint64_t addr = test_watch_addr;
    const size_t ep = PS5_KERNEL_PAGE_SIZE;
    check_early_page(addr, PROT_READ | PROT_WRITE, 0);
    check_early_page(addr + ep, PROT_NONE, 1);
    check_early_page(addr + 2 * ep, PROT_READ | PROT_WRITE, 1);
    assert(mem_early_pages == 3 * (int64_t)(ep / PAGE));
    for (uint64_t p = addr; p < addr + 3 * ep; p += PAGE) assert(trapped[p / PAGE] == 2);
    puts("early batch: runtime stopped before fallback with all guest trap entries intact");
    fflush(stdout);
    _exit(43);
}

static void test_early_batch(uint64_t addr) {
    const size_t ep = PS5_KERNEL_PAGE_SIZE;
    const unsigned rw = PROT_READ | PROT_WRITE;
    assert(plat_mem_commit(addr, 3 * ep));
    assert(plat_mem_back(addr, 3 * ep));
    // One backing unit with two native protection runs. A failed second run must also be restored.
    plat_mem_gpu_import(addr + ep, 2 * ep, 1);
    early_protect_calls = 0;
    early_protect_failures = 1u << 1;
    assert(plat_mem_early_protect(addr, 3 * ep, 1, 0) == PLAT_EARLY_FAILED);
    assert(early_protect_calls == 4);
    early_protect_failures = 0;
    for (unsigned i = 0; i < 3; ++i) check_early_page(addr + i * ep, rw, 0);
    puts("early batch: second run failed, both restores succeeded, three pages RW and unwatched");
    test_watch_addr = addr;
    size_t trap_size = (addr + 3 * ep) / PAGE;
    trapped = mmap(NULL, trap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(trapped != MAP_FAILED);
    fflush(stdout);
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        expect_rollback_fatal = 1;
        early_protect_calls = 0;
        early_protect_failures = (1u << 1) | (1u << 3);
        mem_early_watch(addr, 3 * ep, 1, 0);
        _exit(99);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 43);

    early_protect_calls = 0;
    early_protect_failures = 1u << 1;
    mem_early_watch(addr, 3 * ep, 1, 0);
    assert(early_protect_calls == 7);
    early_protect_failures = 0;
    for (unsigned i = 0; i < 3; ++i) check_early_page(addr + i * ep, PROT_NONE, 1);
    for (uint64_t p = addr; p < addr + 3 * ep; p += PAGE) assert(trapped[p / PAGE] == 2);
    assert(mem_early_pages == 3 * (int64_t)(ep / PAGE));
    mem_early_resolve_hold(addr, 3 * ep);
    for (unsigned i = 0; i < 3; ++i) check_early_page(addr + i * ep, rw, 0);
    for (uint64_t p = addr; p < addr + 3 * ep; p += PAGE) assert(trapped[p / PAGE] == 0);
    assert(mem_early_pages == 0);
    assert(!munmap(trapped, trap_size));
    trapped = NULL;
    puts("early batch: successful rollback allowed exactly three single-page retries");

    early_protect_calls = 0;
    early_protect_failures = (1u << 1) | (1u << 3);
    assert(plat_mem_early_protect(addr, 3 * ep, 1, 0) == PLAT_EARLY_UNRESTORED);
    assert(early_protect_calls == 4);
    early_protect_failures = 0;
    check_early_page(addr, rw, 0);
    check_early_page(addr + ep, PROT_NONE, 1);
    check_early_page(addr + 2 * ep, rw, 1);
    puts("early batch: second arm and second restore failed; native RW/NONE/RW, watch bits 0/1/1");
    assert(plat_mem_early_protect(addr, 3 * ep, 0, 0) == PLAT_EARLY_OK);
    plat_mem_gpu_import(addr + ep, 2 * ep, 0);
    // A closed page opened for reading only, as memory_early.c's page copies do; closing it moves its counter once.
    assert(plat_mem_early_protect(addr, ep, 1, 0) == PLAT_EARLY_OK);
    assert(!plat_mem_early_open_read(addr + 1, ep) && !plat_mem_early_open_read(addr, 2 * ep));
    assert(plat_mem_early_open_read(addr, ep));
    check_early_page(addr, PROT_READ, 1);
    uint32_t gen = plat_mem_early_gen(addr);
    assert(plat_mem_early_protect(addr, ep, 1, 0) == PLAT_EARLY_OK);
    assert(plat_mem_early_gen(addr) == gen + 1);
    check_early_page(addr, PROT_NONE, 1);
    assert(plat_mem_early_protect(addr, ep, 0, 0) == PLAT_EARLY_OK);
    assert(plat_mem_decommit(addr, 3 * ep));
}

static uint64_t discarded_lo, discarded_hi;
static int note_discard(uint64_t lo, uint64_t hi) {
    assert(!discarded_lo);
    discarded_lo = lo;
    discarded_hi = hi;
    return 1;
}

static void test_backing_release(uint64_t addr) {
    assert(plat_mem_commit(addr, 3 * UNIT));
    assert(plat_mem_back(addr, 3 * UNIT));
    volatile unsigned char *edge = (void *)(uintptr_t)addr;
    volatile unsigned char *imported = (void *)(uintptr_t)(addr + 2 * UNIT);
    *edge = 0xa5;
    *imported = 0x5a;
    plat_mem_gpu_import(addr + UNIT, 2 * UNIT, 1);
    assert(plat_mem_release_backing(addr + PAGE, 3 * UNIT - PAGE, note_discard) == 0);
    assert(!discarded_lo && *edge == 0xa5 && *imported == 0x5a && allocations == 3);
    plat_mem_gpu_import(addr + UNIT, UNIT, 0);
    assert(plat_mem_release_backing(addr + PAGE, 3 * UNIT - PAGE, note_discard) == UNIT);
    assert(discarded_lo == addr + UNIT && discarded_hi == addr + 2 * UNIT);
    assert(*edge == 0xa5 && *imported == 0x5a && allocations == 2);
    plat_mem_gpu_import(addr + 2 * UNIT, UNIT, 0);
    assert(plat_mem_decommit(addr, 3 * UNIT));
    puts("GPU-only backing: zero reclaimed for imported units; one eligible unit reclaimed exactly 65536 bytes");
}

// MKW_PACK: frozen units packed (compressible), released (zero) or skipped (random bytes, GPU-imported); a read
// unpacks the same bytes; a page decommitted while packed reads zero once committed again; a unit decommitted whole drops
// its blob.
static uint64_t pack_lo, pack_hi;
static uint32_t pack_since;
static int pack_frozen(uint64_t a) { return a >= pack_lo && a < pack_hi ? 7 : 0; }
static uint32_t pack_frozen_ms(uint64_t a) { return pack_frozen(a) ? pack_since : 0; }
uint32_t plat_thread_id(void) { return 1; }
int plat_sample_self_name(char *out, size_t cap) {
    snprintf(out, cap, "test");
    return 1;
}

static void fill_text(uint8_t *p, uint32_t seed) {
    for (size_t i = 0; i < UNIT; i++) p[i] = (uint8_t)("packed guest unit "[(i + seed) % 18] + (i / 4096));
}

static void test_pack(uint64_t addr) {
    static uint8_t want[4][UNIT];
    fill_text(want[0], 0);                                       // packs
    memset(want[1], 0, UNIT);                                    // released, no blob
    for (size_t i = 0; i < UNIT; i++) want[2][i] = (uint8_t)(rand() >> 7);   // does not compress: kept
    fill_text(want[3], 5);                                       // GPU-imported: kept, then packed once released
    assert(plat_mem_commit(addr, 4 * UNIT) && plat_mem_back(addr, 4 * UNIT) && allocations == 4);
    for (int u = 0; u < 4; u++) memcpy((void *)(uintptr_t)(addr + u * UNIT), want[u], UNIT);
    assert(plat_mem_protect(addr, 4 * UNIT, PLAT_PROT_READ));   // as a freeze leaves them
    plat_mem_gpu_import(addr + 3 * UNIT, UNIT, 1);
    pack_lo = addr, pack_hi = addr + 4 * UNIT;
    pack_since = ((uint32_t)plat_now_ms() - 10000) | 1;          // frozen 10 s ago
    plat_mem_frozen_hook(pack_frozen);
    plat_mem_frozen_ms_hook(pack_frozen_ms);
    PlatPackParams pp = {.target_bytes = 64 * UNIT, .store_cap = 64ull << 20, .cold_ms = 5000, .verify = 1};
    char line[1024];
    plat_mem_pack_summary(line, sizeof line);                    // nothing yet: no pass ran
    assert(!line[0]);
    // Two units' backing released; the pool gets back only the zero one's: with MKW_PACK_VERIFY the packed one's store
    // slots (its blob and a whole copy) take more than the unit.
    assert(plat_mem_pack(&pp) == UNIT && allocations == 2);
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "1 units, 0.1 MiB") && strstr(line, "1 packed") && strstr(line, "1 zero released"));
    assert(strstr(line, "4 frozen units, skipped gpu 1") && strstr(line, "ratio 1 "));
    uint64_t runs[4][2];
    // The packed unit holds data, the released zero unit does not.
    assert(plat_mem_populated(addr, 4 * UNIT, runs, 4) == 2 && runs[0][0] == addr && runs[0][1] == addr + UNIT &&
           runs[1][0] == addr + 2 * UNIT && runs[1][1] == addr + 4 * UNIT);
    // A read of the packed unit faults and unpacks it; the zero unit commits zero-filled. Both are written through a
    // private map before the guest address shows them (the map model checks that nothing stale is left).
    unsigned maps_before = private_maps;
    for (int u = 0; u < 3; u++) assert(!memcmp((void *)(uintptr_t)(addr + u * UNIT), want[u], UNIT));
    assert(allocations == 4 && private_maps == maps_before + 2);
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "0 units") && strstr(line, "1 unpacked") && strstr(line, "read fault 1") &&
           strstr(line, "1 checked, 0 differed") && strstr(line, "0 failed") && strstr(line, "test 1"));
    // Just unpacked: young again. The zero unit is packed again (released), the random one is known not to compress.
    assert(plat_mem_pack(&pp) == UNIT && allocations == 3);
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "young 1") && strstr(line, "(+1 known)"));
    // Released by the GPU: packs; a page decommitted while packed reads zero once committed again, the rest unpacks.
    plat_mem_gpu_import(addr + 3 * UNIT, UNIT, 0);
    assert(plat_mem_pack(&pp) == 0 && allocations == 2);   // packed (backing released), nothing net with the copy
    assert(plat_mem_decommit(addr + 3 * UNIT + PAGE, PAGE) && plat_mem_commit(addr + 3 * UNIT + PAGE, PAGE));
    assert(plat_mem_back(addr + 3 * UNIT, UNIT) && allocations == 3);   // a runtime path unpacks too
    memset(want[3] + PAGE, 0, PAGE);
    assert(!memcmp((void *)(uintptr_t)(addr + 3 * UNIT), want[3], UNIT));
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "runtime read 1") && strstr(line, "1 checked, 0 differed"));
    // Packed, then decommitted whole: the blob goes, nothing is unpacked.
    pack_since = ((uint32_t)plat_now_ms() - 10000) | 1;
    PlatPackParams young = pp;
    young.cold_ms = 0;
    plat_sleep_ns(3000000);   // the unpack times are made odd, up to 1 ms ahead
    plat_mem_pack(&young);   // with the verify copy, packing gives nothing net back (counted 0)
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "1 packed"));
    assert(plat_mem_decommit(addr, 4 * UNIT));
    plat_mem_pack_collect();
    plat_mem_pack_summary(line, sizeof line);
    assert(strstr(line, "0 units") && strstr(line, "0 unpacked") && !allocations);
    plat_mem_frozen_hook(NULL);
    plat_mem_frozen_ms_hook(NULL);
    puts("pack: compressible unit packed and read back, zero unit released, random and GPU units kept; zeroed page and "
         "whole decommit honoured");
}

// MKW_PACK and the write watch (issue 11, 0.3.0: menu text garbled after a Grand Prix): a unit whose pages the watch had
// armed is packed, its region thawed, then unpacked by a read or by a write. Every later write to it must still show in
// plat_mem_written: the unpack maps the unit read-write by the guest's modes, and an armed page left so takes writes
// with no fault, so the texture there never learns of them.
static void test_pack_watch(uint64_t addr) {
    // plat_mem_written counts 4 KiB guest pages; the watch arms its own (16 KiB on the console).
    const uint64_t wp = plat_mem_watch_page(), per = wp / PAGE, max = UNIT / PAGE;
    assert(UNIT / wp >= 2);
    static uint8_t text[UNIT];
    uint64_t got[UNIT / PAGE];
    fill_text(text, 3);
    assert(plat_mem_commit(addr, UNIT) && plat_mem_back(addr, UNIT));
    memcpy((void *)(uintptr_t)addr, text, UNIT);
    pack_lo = addr, pack_hi = addr + UNIT;
    plat_mem_frozen_hook(pack_frozen);
    plat_mem_frozen_ms_hook(pack_frozen_ms);
    for (int by_write = 0; by_write < 2; by_write++) {
        plat_mem_written(addr, UNIT, got, max, 1);   // armed: every page read-only, nothing written
        assert(plat_mem_written(addr, UNIT, got, max, 0) == 0);
        assert(plat_mem_protect(addr, UNIT, PLAT_PROT_READ));   // frozen
        pack_since = ((uint32_t)plat_now_ms() - 10000) | 1;
        plat_sleep_ns(3000000);   // past the last unpack's time (odd, up to 1 ms ahead)
        PlatPackParams pp = {.target_bytes = 64 * UNIT, .store_cap = 64ull << 20, .cold_ms = 0};
        assert(plat_mem_pack(&pp) > 0);   // the unit less its blob's store slots
        assert(plat_mem_protect(addr, UNIT, PLAT_PROT_RW));   // thawed while packed: no backing to protect
        volatile uint8_t *g = (volatile uint8_t *)(uintptr_t)addr;
        if (by_write) g[5] ^= 1;                                    // unpacked by a write to the first watch page
        else assert(g[5] == text[5]);                               // unpacked by a read
        assert(plat_mem_written(addr, UNIT, got, max, 0) == (by_write ? per : 0));
        g[wp + 7] ^= 1;   // the second watch page, already backed: its write must reach the watch
        uint64_t n = plat_mem_written(addr, UNIT, got, max, 0);
        assert(n == (by_write ? 2 : 1) * per && got[n - per] == addr + wp);
        if (by_write) g[5] ^= 1;
        g[wp + 7] ^= 1;
        assert(!memcmp((const void *)(uintptr_t)addr, text, UNIT));
    }
    plat_mem_frozen_hook(NULL);
    plat_mem_frozen_ms_hook(NULL);
    assert(plat_mem_decommit(addr, UNIT));
    plat_mem_pack_collect();
    puts("pack and write watch: an armed unit unpacked by a read or a write reports every later write");
}

static int feed_n;
static uint64_t feed_at;
static int feed_stub(uint64_t a) { return feed_n++, feed_at = a, 1; }

// MKW_COLD_CENSUS: counts only. 8 text units, 2 zero and 2 unreadable text ones (PROT_NONE): the first walk sees the
// text ones for the first time, the second finds them unchanged (<30 s), and after a write one of them changed again.
static void test_cold_census(uint64_t addr) {
    enum { TEXT = 8, ZERO = 2, NONE = 2, ALL = TEXT + ZERO + NONE };   // the model has 16 units
    static uint8_t buf[UNIT];
    assert(plat_mem_commit(addr, ALL * UNIT) && plat_mem_back(addr, ALL * UNIT));
    for (int u = 0; u < TEXT; u++) fill_text(buf, u), memcpy((void *)(uintptr_t)(addr + u * UNIT), buf, UNIT);
    memset((void *)(uintptr_t)(addr + TEXT * UNIT), 0, ZERO * UNIT);
    for (int u = TEXT + ZERO; u < ALL; u++)
        fill_text(buf, 100 + u), memcpy((void *)(uintptr_t)(addr + u * UNIT), buf, UNIT);
    unsigned before = allocations;
    assert(plat_mem_protect(addr + (TEXT + ZERO) * UNIT, NONE * UNIT, PLAT_PROT_NONE));
    char line[1024];
    while (!plat_mem_cold_census_step(1000000, 60, NULL, line, sizeof line)) {}
    assert(strstr(line, "new 0.5, all zero 0.1, unreadable 0.1"));
    while (!plat_mem_cold_census_step(1000000, 60, NULL, line, sizeof line)) {}
    assert(strstr(line, "<30 s 0.5 MiB") && strstr(line, "new 0.0,"));
    ((volatile uint8_t *)(uintptr_t)addr)[17] ^= 0xff;   // one unit written
    while (!plat_mem_cold_census_step(1000000, 60, NULL, line, sizeof line)) {}
    assert(strstr(line, "<30 s 0.5 MiB") && allocations == before);   // nothing backed, released or packed by it
    // The cold feed (MKW_PACK): a 2 MiB region is offered only when none of its units is unreadable or changed. The
    // unreadable units made readable are new in the next walk, so the region goes in the one after.
    assert(plat_mem_protect(addr + (TEXT + ZERO) * UNIT, NONE * UNIT, PLAT_PROT_RW));
    feed_n = 0;
    while (!plat_mem_cold_census_step(1000000, 0, feed_stub, line, sizeof line)) {}
    assert(!feed_n && strstr(line, "0 regions frozen"));
    while (!plat_mem_cold_census_step(1000000, 0, feed_stub, line, sizeof line)) {}
    assert(feed_n == 1 && feed_at == (addr & ~((2ull << 20) - 1)) && strstr(line, "1 regions frozen, 0 refused"));
    assert(plat_mem_decommit(addr, ALL * UNIT));
    puts("cold census: new, zero and unreadable units counted, unchanged ones aged, nothing backed or released; "
         "a region offered to the feed once all its units are unchanged");
}

// prot_watch_dirty (plat_mem_written_any): 4 KiB sub-page spans (<16 KiB native page or ending mid-page) must check
// the covering 16 KiB page instead of truncating the page count to zero.
static void test_prot_watch_subpage(uint64_t addr) {
    uint64_t pages[64];
    assert(plat_mem_commit(addr, UNIT) && plat_mem_back(addr, UNIT));
    assert(prot_watch_written(addr, UNIT, pages, 64, 1, NULL) != UINT64_MAX);
    assert(prot_watch_dirty(addr, PAGE) == 0);
    assert(prot_watch_dirty(addr, PS5_KERNEL_PAGE_SIZE + PAGE) == 0);
    *(volatile uint8_t *)(uintptr_t)addr = 0x11;
    assert(prot_watch_dirty(addr, PAGE) == 1);
    assert(prot_watch_written(addr, UNIT, pages, 64, 1, NULL) != UINT64_MAX);
    assert(prot_watch_dirty(addr, PS5_KERNEL_PAGE_SIZE + PAGE) == 0);
    *(volatile uint8_t *)(uintptr_t)(addr + PS5_KERNEL_PAGE_SIZE) = 0x22;
    assert(prot_watch_dirty(addr, PS5_KERNEL_PAGE_SIZE + PAGE) == 1);
    assert(plat_mem_decommit(addr, UNIT));
    puts("prot_watch_dirty: sub-16 KiB and tail-4 KiB spans report writes accurately");
}

// MKW_ZERO_POOL: a decommitted unit stays allocated in the pool; the next commit takes it zeroed (whether the pool's
// thread zeroed it first or not) and allocates nothing; turned off, the pool gives every unit back. The build sets the
// pool's memory thresholds to 0 and "never", so the model's 1 MiB neither trims it nor gets units allocated ahead.
static unsigned live_allocations(void) { return __atomic_load_n(&allocations, __ATOMIC_ACQUIRE); }
static uint64_t pool_ready(void) {
    uint64_t st[7];
    ps5_mem_zero_pool_stats(st);
    return st[5];
}
static void test_zero_pool(uint64_t addr) {
    assert(!ps5_mem_zero_pool(1));
    unsigned before = live_allocations();
    for (int round = 0; round < 2; round++) {
        assert(plat_mem_commit(addr, 2 * UNIT) && plat_mem_back(addr, 2 * UNIT));
        assert(live_allocations() == before + 2);
        memset((void *)(uintptr_t)addr, 0x77, 2 * UNIT);
        assert(plat_mem_decommit(addr, 2 * UNIT));
        assert(live_allocations() == before + 2);   // kept, not released
        if (!round)   // the first time the thread zeroes both first; the second time the commit may meet one unzeroed
            for (int i = 0; i < 5000 && pool_ready() < 2; i++) plat_sleep_ns(1000000);
        assert(plat_mem_commit(addr, 2 * UNIT) && plat_mem_back(addr, 2 * UNIT));
        assert(live_allocations() == before + 2);   // taken from the pool
        const uint64_t *w = (const uint64_t *)(uintptr_t)addr;
        for (size_t k = 0; k < 2 * UNIT / sizeof *w; k++) assert(!w[k]);
        assert(plat_mem_decommit(addr, 2 * UNIT));
    }
    assert(!ps5_mem_zero_pool(0));
    for (int i = 0; i < 5000 && live_allocations() != before; i++) plat_sleep_ns(1000000);
    assert(live_allocations() == before);
    puts("zero pool: decommitted units recycled zeroed, no allocation; off gives them back");
}

// MKW_DECOMMIT_RUN: units with backing in a row released by one unmap, a half-covered unit page by page; the run's
// pages read zero when committed again, its physical units are given back and its guest pages are reserved again.
static void test_decommit_run(uint64_t addr) {
    assert(!plat_mem_decommit_run_mode(1));
    unsigned before = live_allocations();
    assert(plat_mem_commit(addr, 4 * UNIT) && plat_mem_back(addr, 4 * UNIT));
    assert(live_allocations() == before + 4);
    memset((void *)(uintptr_t)addr, 0x5c, 4 * UNIT);
    // three whole units and half of the fourth
    assert(plat_mem_decommit(addr, 3 * UNIT + UNIT / 2));
    assert(live_allocations() == before + 1);
    state(addr, addr + 3 * UNIT + UNIT / 2, PLAT_MEM_RESERVED, addr + 3 * UNIT + UNIT / 2);
    assert(((volatile uint8_t *)(uintptr_t)addr)[3 * UNIT + UNIT / 2] == 0x5c);   // the rest of the fourth kept
    assert(plat_mem_commit(addr, 3 * UNIT) && plat_mem_back(addr, 3 * UNIT));
    const uint64_t *w = (const uint64_t *)(uintptr_t)addr;
    for (size_t k = 0; k < 3 * UNIT / sizeof *w; k++) assert(!w[k]);
    assert(plat_mem_decommit(addr, 4 * UNIT));
    assert(live_allocations() == before);
    assert(!plat_mem_decommit_run_mode(0));
    puts("decommit run: three units released by one unmap, a half unit page by page; zero when committed again");
}

// MKW_FILL_WINDOW: fresh units zeroed and packed units' bytes written through the shared window read the same at their
// guest addresses; a unit decommitted while its slot still maps it is released only after the window's flush; off,
// the window goes and nothing stays allocated.
static void test_fill_window(uint64_t addr) {
    assert(!ps5_mem_fill_window(1));
    unsigned before = live_allocations();
    for (int round = 0; round < 8; round++) {   // 8 x 6 units: past the window's 32 slots
        assert(plat_mem_commit(addr, 6 * UNIT) && plat_mem_back(addr, 6 * UNIT));
        const uint64_t *w = (const uint64_t *)(uintptr_t)addr;
        for (size_t k = 0; k < 6 * UNIT / sizeof *w; k++) assert(!w[k]);
        memset((void *)(uintptr_t)addr, 0x3d, 6 * UNIT);
        assert(plat_mem_decommit(addr, 6 * UNIT));
        assert(live_allocations() == before);
    }
    assert(ps5_mem_fill_window_flushes() >= 1);
    assert(!ps5_mem_fill_window(0));
    puts("fill window: 48 units zeroed through the window, released after its flushes, nothing left allocated");
}

static void test_memory(void) {
    uint64_t boundary = sample + UNIT, tail = boundary - PAGE;
    uint64_t alias_end = RT_ALIAS_BASE + RT_ALIAS_SIZE;
    assert(plat_mem_reserve(RT_MODULE_BASE, RT_KERNEL_BASE - RT_MODULE_BASE, 0));
    assert(plat_mem_reserve(RT_KERNEL_BASE, RT_KERNEL_SIZE, 0));
    assert(plat_mem_reserve(RT_HOSTFN_BASE, RT_HOSTFN_SIZE, 0));
    assert(plat_mem_reserve(RT_ALIAS_BASE, RT_ALIAS_SIZE, 1));
    assert(plat_mem_reserve(RT_LOW_BASE, RT_LOW_SIZE, 1));
    test_early_batch(RT_ALIAS_BASE);
    test_backing_release(RT_ALIAS_BASE + 4 * UNIT);
    test_pack(RT_ALIAS_BASE + 8 * UNIT);
    test_pack_watch(RT_ALIAS_BASE + 24 * UNIT);
    test_cold_census(RT_ALIAS_BASE + 32 * UNIT);
    test_prot_watch_subpage(RT_ALIAS_BASE + 48 * UNIT);
    test_zero_pool(RT_ALIAS_BASE + 52 * UNIT);
    test_decommit_run(RT_ALIAS_BASE + 56 * UNIT);
    test_fill_window(RT_ALIAS_BASE + 64 * UNIT);
    assert(!allocations && !plat_mem_reserve(sample, UNIT, 0));
    assert(!plat_mem_reserve(hint, UNIT, 0));
    assert(!plat_mem_reserve(unexpected, UNIT, 0) && !plat_mem_commit(unexpected, PAGE));

    state(alias_end, RT_LOW_BASE + PAGE, PLAT_MEM_FREE, RT_LOW_BASE);
    state(RT_ALIAS_BASE, alias_end, PLAT_MEM_RESERVED, alias_end);
    // Demand backing: commit records RW state without allocating physical DMEM yet.
    assert(plat_mem_commit(tail, PAGE) && allocations == 0);
    assert(!plat_mem_protect(tail, 2 * PAGE, PLAT_PROT_READ));
    state(tail, boundary, PLAT_PROT_RW, boundary);
    assert(plat_mem_commit(tail, 2 * PAGE) && allocations == 0);
    state(RT_ALIAS_BASE, alias_end, PLAT_MEM_RESERVED, tail);
    state(tail, alias_end, PLAT_PROT_RW, boundary + PAGE);
    // First touch triggers SIGSEGV demand-backing for the unit, zeroing it first.
    *(volatile unsigned char *)(uintptr_t)(boundary - 1) = 0xa5;
    assert(allocations == 1);
    *(volatile unsigned char *)(uintptr_t)boundary = 0x5a;
    assert(allocations == 2);
    assert(plat_mem_protect(tail, PAGE, PLAT_PROT_READ));
    unsigned char bytes[2] = {0};
    assert(plat_mem_read(boundary - 1, bytes, sizeof bytes) && bytes[0] == 0xa5 && bytes[1] == 0x5a);
    assert(plat_mem_protect(tail, PAGE, PLAT_PROT_NONE));
    assert(!plat_mem_read(boundary - 1, bytes, sizeof bytes) && bytes[0] == 0xa5 && bytes[1] == 0x5a);
    uint64_t written, populated[1][2];
    assert(!plat_mem_readable(UINT64_MAX, 2));
    assert(plat_mem_written(tail, PAGE, &written, 1, 0) == 1 && written == tail);
    assert(plat_mem_populated(tail, PAGE, populated, 1) == 1 && populated[0][0] == tail && populated[0][1] == tail + PAGE);

    // Protect-before-back + explicit plat_mem_back() + decommit before back.
    assert(plat_mem_commit(RT_MODULE_BASE, 3 * PAGE) && allocations == 2);
    assert(plat_mem_protect(RT_MODULE_BASE, 2 * PAGE, PLAT_PROT_READ) && allocations == 2);
    state(RT_MODULE_BASE, RT_MODULE_BASE + UNIT, PLAT_PROT_READ, RT_MODULE_BASE + 2 * PAGE);
    assert(plat_mem_back(RT_MODULE_BASE, 3 * PAGE) && allocations == 3);
    *(volatile unsigned char *)(uintptr_t)(RT_MODULE_BASE + 2 * PAGE) = 0x21;
    assert(plat_mem_read(RT_MODULE_BASE + 2 * PAGE, bytes, 1) && bytes[0] == 0x21);
    assert(plat_mem_decommit(RT_MODULE_BASE, 3 * PAGE) && allocations == 2);
    state(RT_MODULE_BASE, RT_MODULE_BASE + UNIT, PLAT_MEM_RESERVED, RT_MODULE_BASE + UNIT);

    // Fault and watch hooks fire on protection violations of backed pages.
    plat_mem_fault_hook(test_fault_hook);
    plat_mem_watch_hook(test_watch_hook);
    hook_fault_ret = 1;
    *(volatile unsigned char *)(uintptr_t)tail = 0x11; // tail is PLAT_PROT_NONE -> fault_hook
    assert(hook_fault_addr == tail);
    assert(plat_mem_protect(tail, PAGE, PLAT_PROT_NONE));
    hook_fault_ret = 0;
    hook_watch_ret = 1;
    *(volatile unsigned char *)(uintptr_t)tail = 0x12; // fault_hook returns 0 -> watch_hook
    assert(hook_watch_addr == tail && hook_watch_rip != 0);
    plat_mem_fault_hook(NULL);
    plat_mem_watch_hook(NULL);
    (void)hook_crash_addr;

    // Partial decommit clears only the slice being reused, and recommit comes back zeroed.
    assert(plat_mem_commit(tail - PAGE, PAGE));
    *(volatile unsigned char *)(uintptr_t)(tail - PAGE) = 0x31;
    assert(plat_mem_decommit(tail, PAGE) && allocations == 2);
    assert(plat_mem_commit(tail, PAGE) && allocations == 2);
    assert(plat_mem_read(tail, bytes, 1) && !bytes[0]);
    assert(plat_mem_read(tail - PAGE, bytes, 1) && bytes[0] == 0x31);
    assert(plat_mem_decommit(tail - PAGE, PAGE));
    lose_slot = true;
    assert(!plat_mem_decommit(tail, PAGE) && allocations == 1);
    state(sample, boundary + PAGE, PLAT_MEM_FREE, boundary);
    assert(!plat_mem_commit(tail, PAGE) && !plat_mem_readable(tail, 1));
    assert(plat_mem_read(boundary, bytes, 1) && bytes[0] == 0x5a);

    // Backing failure in fault_handler must NOT call fault_hook/watch_hook; it must invoke crash_hook and exit.
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        for (size_t i = 0; i < sizeof units; ++i) units[i] = true; // exhaust DMEM
        hook_fault_addr = 0;
        hook_watch_addr = 0;
        hook_fault_ret = 1; // would loop forever if fault_handler fell through to thaw_fault
        hook_watch_ret = 1;
        plat_mem_fault_hook(test_fault_hook);
        plat_mem_watch_hook(test_watch_hook);
        plat_crash_hook(test_oom_crash_hook);
        assert(plat_mem_commit(RT_LOW_BASE, PAGE));
        *(volatile unsigned char *)(uintptr_t)RT_LOW_BASE = 0x77;
        _exit(99);
    }
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 42);

    uint64_t host = (uintptr_t)plat_mem_alloc(3 * PS5_KERNEL_PAGE_SIZE);
    assert(host);
    uint64_t aligned = (host + PS5_KERNEL_PAGE_SIZE - 1) & ~(uint64_t)(PS5_KERNEL_PAGE_SIZE - 1);
    assert(!plat_mem_protect(aligned, PAGE, PLAT_PROT_READ));
    assert(plat_mem_protect(aligned, PS5_KERNEL_PAGE_SIZE, PLAT_PROT_READ));
    state(aligned, aligned + PS5_KERNEL_PAGE_SIZE, PLAT_PROT_READ, aligned + PS5_KERNEL_PAGE_SIZE);
    void *small = plat_aligned_alloc(53, 64);
    assert(small && !((uintptr_t)small % 64));
    plat_aligned_free(small);

    assert(ps5_guest_memory_shutdown() && !allocations);
    assert(!plat_mem_reserve(RT_LOW_BASE, UNIT, 0));
    assert(*(volatile unsigned char *)(uintptr_t)sample == 0x6d);
    assert(!munmap((void *)(uintptr_t)sample, UNIT));
    void *lo, *hi;
    uint32_t prot;
    assert(sceKernelQueryMemoryProtection((void *)(uintptr_t)unexpected, &lo, &hi, &prot));
    assert(sceKernelQueryMemoryProtection((void *)(uintptr_t)boundary, &lo, &hi, &prot));
}

static void *test_contended_lock(void *arg) {
    PlatMutex *lock = arg;
    assert(!plat_mutex_trylock(lock));
    assert(plat_mutex_trylock_shared(lock));
    plat_mutex_unlock_shared(lock);
    return NULL;
}

// MKW_PS5_FAST_LOCK under load: writers and readers on a few locks. A lost wakeup hangs here (the test's timeout); a
// broken exclusion shows in the counters, which only the holders of the lock may touch.
enum { FL_LOCKS = 3, FL_THREADS = 8, FL_ROUNDS = 60000 };
static PlatMutex fl_lock[FL_LOCKS];
static uint64_t fl_a[FL_LOCKS], fl_b[FL_LOCKS], fl_writes;
static void *fast_lock_worker(void *arg) {
    uint64_t r = (uintptr_t)arg * 2654435761u + 1;
    for (int i = 0; i < FL_ROUNDS; i++) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        int k = (int)(r % FL_LOCKS);
        if (r >> 60 < 5) {
            plat_mutex_lock(&fl_lock[k]);
            fl_a[k]++;
            if (!(r >> 20 & 15)) sched_yield();
            fl_b[k]++;
            __atomic_fetch_add(&fl_writes, 1, __ATOMIC_RELAXED);
            plat_mutex_unlock(&fl_lock[k]);
        } else if (r >> 56 & 1 || !plat_mutex_trylock_shared(&fl_lock[k])) {
            plat_mutex_lock_shared(&fl_lock[k]);
            assert(fl_a[k] == fl_b[k]);
            if (!(r >> 20 & 31)) sched_yield();
            assert(fl_a[k] == fl_b[k]);
            plat_mutex_unlock_shared(&fl_lock[k]);
        } else {
            assert(fl_a[k] == fl_b[k]);
            plat_mutex_unlock_shared(&fl_lock[k]);
        }
    }
    return NULL;
}
static void test_fast_lock(void) {
    plat_lock_fast_mode(1);
    pthread_t t[FL_THREADS];
    for (uintptr_t i = 0; i < FL_THREADS; i++) assert(!pthread_create(&t[i], NULL, fast_lock_worker, (void *)(i + 1)));
    for (int i = 0; i < FL_THREADS; i++) assert(!pthread_join(t[i], NULL));
    uint64_t sum = 0;
    for (int k = 0; k < FL_LOCKS; k++) assert(fl_a[k] == fl_b[k]), sum += fl_a[k];
    assert(sum == fl_writes && sum);
    plat_lock_fast_mode(0);
    puts("fast lock: eight threads, three locks: no hang, exclusion held");
}

// The log's total of free direct memory adds up every free area, not only the largest.
static void test_dmem_scan(void) {
    bool saved[sizeof units];
    memcpy(saved, units, sizeof units);
    memset(units, 1, sizeof units);
    // Free: unit 1, units 3-4 and units 6-9 (the model's direct memory is 1 MiB, so the MiB total only rounds to 0).
    units[1] = units[3] = units[4] = units[6] = units[7] = units[8] = units[9] = false;
    char out[384];
    plat_mem_backed_summary(out, sizeof out);
    assert(strstr(out, "all free areas: 0 MiB in 3,"));
    // MKW_BENCH's sampler (plat_mem_pressure) adds up the same areas on its own scan buffer, and reads the largest
    PlatMemPressure p;
    plat_mem_pressure(&p);
    assert(p.dmem_free == 7 * (int64_t)UNIT && p.dmem_largest == 4 * (int64_t)UNIT);
    assert(p.guest_backed >= 0 && p.flex_free == (256ll << 20));
    assert(p.pt_cpu_free == 12000 && p.pt_cpu_total == 16384 && p.pt_gpu_free == 12000 && p.pt_gpu_total == 16384);
    memcpy(units, saved, sizeof units);
    puts("dmem scan: three free areas found, by the log and by plat_mem_pressure");
}

static void test_io_and_mutex(void) {
    uint64_t size;
    uint8_t *bytes = plat_file_read("binary", &size);
    assert(bytes && size == 8 && !memcmp(bytes, "mkw\0test", 9));
    free(bytes);
    assert(!plat_file_read("empty", &size) && !size);
    assert(!plat_file_read(".", &size) && !size);
    plat_file_mkdir_p("test\\nested");
    assert(plat_dir_exists("test/nested") && !plat_file_exists("test/nested"));
    assert(plat_file_size("binary") == 8 && plat_file_size("missing") < 0);
    uint64_t before = plat_now_ns();
    plat_sleep_ns(1000);
    assert(plat_now_ns() >= before + 1000);

    PlatMutex lock = PLAT_MUTEX_INIT;
    pthread_t thread;
    plat_mutex_lock_shared(&lock);
    assert(!pthread_create(&thread, NULL, test_contended_lock, &lock));
    assert(!pthread_join(thread, NULL));
    plat_mutex_unlock_shared(&lock);
    assert(plat_mutex_trylock(&lock));
    plat_mutex_unlock(&lock);
}

// plat_fault_step (fault_guard.c, MKW_EARLY_SIBLING_STEP): a fault hook opens the page and asks for one instruction;
// the access completes with the page's bytes, and the page is closed again right after it, so the next access faults.
static uint64_t step_page;
static volatile unsigned step_faults, step_closes;   // (changed in the signal handlers: volatile, or -O2 keeps them in registers across the faulting read)
static void step_close(uint64_t page) {
    assert(page == step_page && !mprotect((void *)(uintptr_t)page, PS5_KERNEL_PAGE_SIZE, PROT_NONE));
    step_closes++;
}
static int step_hook(uint64_t addr) {
    if (addr - step_page >= PS5_KERNEL_PAGE_SIZE) return 0;
    step_faults++;
    assert(plat_fault_step(step_close, step_page));
    assert(!mprotect((void *)(uintptr_t)step_page, PS5_KERNEL_PAGE_SIZE, PROT_READ | PROT_WRITE));
    return 1;
}

static void test_fault_step(void) {
    plat_mem_fault_hook(step_hook);   // (installs the handlers, SIGTRAP's and its self-test too)
    assert(plat_fault_step_init() == 1);
    void *m = mmap(NULL, PS5_KERNEL_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(m != MAP_FAILED);
    step_page = (uintptr_t)m;
    ((volatile unsigned char *)m)[PAGE + 5] = 0x5a;
    assert(!mprotect(m, PS5_KERNEL_PAGE_SIZE, PROT_NONE));
    for (unsigned i = 1; i <= 3; i++) {
        assert(((volatile unsigned char *)m)[PAGE + 5] == 0x5a);
        assert(step_faults == i && step_closes == i);
    }
    plat_mem_fault_hook(NULL);
    assert(!munmap(m, PS5_KERNEL_PAGE_SIZE));
    puts("fault step: three reads of a closed page went through, each closing it again after its instruction");
}

// plat_fault_emulate_load (fault_guard.c, MKW_EARLY_SIBLING_EMU): a load from a closed page is emulated by the fault
// hook, which reads through the protection (here: opens and closes the page around the copy, as memory_early.c's
// sibling_read does) and the page stays closed: every read faults. The forms of the PS5 race's six reader sites, then
// SIB addressing, sign extension and a byte load that keeps the register's upper bytes.
static uint64_t emu_page;
static volatile unsigned emu_faults;
static int emu_read(uint64_t ea, size_t n, void *dst) {
    assert(!mprotect((void *)(uintptr_t)emu_page, PS5_KERNEL_PAGE_SIZE, PROT_READ));
    memcpy(dst, (const void *)(uintptr_t)ea, n);
    assert(!mprotect((void *)(uintptr_t)emu_page, PS5_KERNEL_PAGE_SIZE, PROT_NONE));
    return 1;
}
static int emu_hook(uint64_t addr) {
    if (addr - emu_page >= PS5_KERNEL_PAGE_SIZE) return 0;
    emu_faults++;
    assert(plat_fault_emulate_load(addr, 0, emu_read));
    return 1;
}

static void test_fault_emulate(void) {
    // the code bytes from the executable's file, as the PS5 reads them (its text is execute-only): this test's ELF
    Dl_info self;
    assert(dladdr((void *)test_fault_emulate, &self) && self.dli_fbase);
    assert(plat_code_file_init("/proc/self/exe", (uintptr_t)self.dli_fbase) && plat_code_check());
    assert(!plat_code_file_init("/proc/self/maps", 0));   // (not an ELF: refused, the file reads kept)
    plat_mem_fault_hook(emu_hook);
    uint8_t *m = mmap(NULL, PS5_KERNEL_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(m != MAP_FAILED);
    emu_page = (uintptr_t)m;
    uint8_t *b = m + PAGE;   // (the sibling 4 KiB page)
    for (int i = 0; i < 0x200; i++) b[i] = (uint8_t)(0x80 + i);
    memcpy(b + 0x10, &(uint64_t){0x1122334455667788ull}, 8);
    memcpy(b + 0x100, &(uint64_t){0x99aabbccddeeff00ull}, 8);
    memcpy(m + 0x3000 + 0x8a0, &(uint32_t){0xdeadbeef}, 4);
    assert(!mprotect(m, PS5_KERNEL_PAGE_SIZE, PROT_NONE));
    uint64_t r, base = (uintptr_t)b;
    unsigned f = emu_faults;
    __asm__ volatile("movq %1, %%rax\n\tmovq 0x10(%%rax), %%rax\n\tmovq %%rax, %0" : "=r"(r) : "r"(base) : "rax", "memory");
    assert(r == 0x1122334455667788ull && emu_faults == ++f);   // 48 8b 40 10
    __asm__ volatile("movq %1, %%rdi\n\tmovq $-1, %%r9\n\tmovzbl 0x4a(%%rdi), %%r9d\n\tmovq %%r9, %0" : "=r"(r) : "r"(base) : "rdi", "r9", "memory");
    assert(r == (uint8_t)(0x80 + 0x4a) && emu_faults == ++f);   // 44 0f b6 4f 4a
    __asm__ volatile("movq %1, %%r13\n\tmovq $-1, %%rax\n\tmovl 0x58a0(%%r13), %%eax\n\tmovq %%rax, %0" : "=r"(r) : "r"((uintptr_t)m - 0x58a0 + 0x38a0) : "rax", "r13", "memory");
    assert(r == 0xdeadbeef && emu_faults == ++f);   // 41 8b 85 a0 58 00 00
    __asm__ volatile("movq %1, %%rax\n\tmovq $-1, %%rcx\n\tmovzwl 0x8(%%rax), %%ecx\n\tmovq %%rcx, %0" : "=r"(r) : "r"(base) : "rax", "rcx", "memory");
    assert(r == (uint16_t)(0x88 | 0x89 << 8) && emu_faults == ++f);   // 0f b7 48 08
    __asm__ volatile("movq %1, %%rcx\n\tmovq 0x100(%%rcx), %%rcx\n\tmovq %%rcx, %0" : "=r"(r) : "r"(base) : "rcx", "memory");
    assert(r == 0x99aabbccddeeff00ull && emu_faults == ++f);   // 48 8b 89 00 01 00 00
    __asm__ volatile("movq %1, %%rcx\n\tmovq $-1, %%r8\n\tmovl 0x10(%%rcx), %%r8d\n\tmovq %%r8, %0" : "=r"(r) : "r"(base) : "rcx", "r8", "memory");
    assert(r == 0x55667788 && emu_faults == ++f);   // 44 8b 41 10
    __asm__ volatile("movq %1, %%r8\n\tmovq $2, %%rdi\n\tmovq 0x0(%%r8,%%rdi,8), %%rdi\n\tmovq %%rdi, %0" : "=r"(r) : "r"(base) : "rdi", "r8", "memory");
    assert(r == 0x1122334455667788ull && emu_faults == ++f);   // SIB: base + index * 8
    __asm__ volatile("movq %1, %%rax\n\tmovsbq 0x4a(%%rax), %%rax\n\tmovq %%rax, %0" : "=r"(r) : "r"(base) : "rax", "memory");
    assert(r == (uint64_t)(int64_t)(int8_t)(0x80 + 0x4a) && emu_faults == ++f);   // movsx, REX.W
    __asm__ volatile("movq %1, %%rax\n\tmovabsq $0x0102030405060708, %%r10\n\tmovb 0x4a(%%rax), %%r10b\n\tmovq %%r10, %0" : "=r"(r) : "r"(base) : "rax", "r10", "memory");
    assert(r == (0x0102030405060700ull | (uint8_t)(0x80 + 0x4a)) && emu_faults == ++f);   // a byte, upper bytes kept
    plat_mem_fault_hook(NULL);
    assert(!munmap(m, PS5_KERNEL_PAGE_SIZE));
    puts("fault emulate: nine loads from a closed page, their code read from the executable file, read its bytes into their registers, the page closed throughout");
}

int main(void) {
    test_memory();
    test_io_and_mutex();
    test_fast_lock();
    test_dmem_scan();
    test_fault_step();
    test_fault_emulate();
}
