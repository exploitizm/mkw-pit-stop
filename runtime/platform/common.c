// The parts of platform.h built on its locks or its other calls rather than on the OS: recursive mutexes, one-time
// initialisation, events and the clean-up of torn atomic writes. Every OS shares them.
#include <stdio.h>
#include <string.h>

#include "platform.h"

// The present thread used this policy locally. Share it with image acquisition: millisecond Sleep overshoots by
// several ms with TIMER_1MS=0 on Windows, while a shorter high-resolution guard can still wake after the deadline.
// Keep twice the platform's expected late wake-up as margin; yield there rather than occupying a core with PAUSE.
void plat_sleep_until_ns(uint64_t deadline) {
    const uint64_t guard = 2ull * PLAT_WAIT_LATE_NS;
    uint64_t now = plat_now_ns();
    if (deadline > now && deadline - now > guard + guard / 2) plat_wait_ns(deadline - now - guard);
    while (plat_now_ns() < deadline) plat_yield();
}

// ---- recursive mutex: owner is the holder's thread id (0 when free); only the holder writes depth.

void plat_rmutex_lock(PlatRMutex *m) {
    uint32_t me = plat_thread_id();
    if (__atomic_load_n(&m->owner, __ATOMIC_RELAXED) == me) {
        m->depth++;
        return;
    }
    plat_mutex_lock_from(&m->m, (uintptr_t)__builtin_return_address(0));
    __atomic_store_n(&m->owner, me, __ATOMIC_RELAXED);
    m->depth = 1;
}

int plat_rmutex_trylock(PlatRMutex *m) {
    uint32_t me = plat_thread_id();
    if (__atomic_load_n(&m->owner, __ATOMIC_RELAXED) == me) {
        m->depth++;
        return 1;
    }
    if (!plat_mutex_trylock(&m->m)) return 0;
    __atomic_store_n(&m->owner, me, __ATOMIC_RELAXED);
    m->depth = 1;
    return 1;
}

void plat_rmutex_unlock(PlatRMutex *m) {
    if (--m->depth) return;
    __atomic_store_n(&m->owner, 0, __ATOMIC_RELAXED);
    plat_mutex_unlock(&m->m);
}

// ---- once: state 0 not run, 1 running, 2 done. Waiters for a running initialiser sleep on one process-wide
// condition variable; initialisers are rare and short.

static PlatMutex once_lock = PLAT_MUTEX_INIT;
static PlatCond once_done = PLAT_COND_INIT;

void plat_once(PlatOnce *o, void (*fn)(void *arg), void *arg) {
    if (__atomic_load_n(&o->state, __ATOMIC_ACQUIRE) == 2) return;
    int32_t expected = 0;
    if (__atomic_compare_exchange_n(&o->state, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        fn(arg);
        plat_mutex_lock(&once_lock);
        __atomic_store_n(&o->state, 2, __ATOMIC_RELEASE);
        plat_cond_broadcast(&once_done);
        plat_mutex_unlock(&once_lock);
        return;
    }
    plat_mutex_lock(&once_lock);
    while (__atomic_load_n(&o->state, __ATOMIC_ACQUIRE) != 2) plat_cond_wait(&once_done, &once_lock);
    plat_mutex_unlock(&once_lock);
}

// ---- events

void plat_event_init(PlatEvent *e, int manual, int initially_set) {
    plat_mutex_init(&e->m);
    plat_cond_init(&e->c);
    e->manual = manual;
    e->set = initially_set;
}

void plat_event_set(PlatEvent *e) {
    plat_mutex_lock(&e->m);
    e->set = 1;
    if (e->manual) plat_cond_broadcast(&e->c);
    else plat_cond_signal(&e->c);
    plat_mutex_unlock(&e->m);
}

void plat_event_wait(PlatEvent *e) {
    plat_mutex_lock(&e->m);
    while (!e->set) plat_cond_wait(&e->c, &e->m);
    if (!e->manual) e->set = 0;
    plat_mutex_unlock(&e->m);
}

int plat_event_wait_ns(PlatEvent *e, uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    plat_mutex_lock(&e->m);
    while (!e->set) {
        uint64_t now = plat_now_ns();
        if (now >= end) break;
        plat_cond_wait_ns(&e->c, &e->m, end - now);
    }
    int got = e->set;
    if (got && !e->manual) e->set = 0;
    plat_mutex_unlock(&e->m);
    return got;
}

typedef struct { char dir[PLAT_PATH_MAX * 2]; const char *base; size_t base_len; int n; } Stale;

static int stale_tmp(const char *name, void *ctx) {
    Stale *s = ctx;
    unsigned pid, tid;
    char end;
    if (strncmp(name, s->base, s->base_len) || name[s->base_len] != '.') return 1;
    if (sscanf(name + s->base_len, ".%u.%u.tm%c", &pid, &tid, &end) != 3 || end != 'p') return 1;
    if (pid == plat_process_id() || plat_process_alive(pid)) return 1;
    char path[PLAT_PATH_MAX * 3];
    snprintf(path, sizeof path, "%s/%s", s->dir, name);
    plat_file_delete(path);
    s->n++;
    return 1;
}

int plat_file_atomic_cleanup(const char *path) {
    Stale s = {0};
    const char *cut = plat_last_sep(path);
    if (cut) snprintf(s.dir, sizeof s.dir, "%.*s", (int)(cut - path), path);
    else snprintf(s.dir, sizeof s.dir, ".");
    s.base = cut ? cut + 1 : path;
    s.base_len = strlen(s.base);
    plat_file_list(s.dir, ".tmp", stale_tmp, &s);
    return s.n;
}

__attribute__((weak)) void plat_threads_log(void (*emit)(const char *line)) { (void)emit; }
__attribute__((weak)) int plat_thread_priority_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_tsc_clock_mode(int on) { (void)on; return -1; }
__attribute__((weak)) void plat_cond_probe(void) {}
__attribute__((weak)) void plat_lock_prof_mode(int on) { (void)on; }
__attribute__((weak)) int plat_lock_fast_mode(int on) { return (void)on, 0; }
__attribute__((weak)) int plat_pw_batch_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_pw_hot_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_pw_fault_batch_mode(int on) { (void)on; return -1; }
__attribute__((weak)) uint64_t plat_mem_write_watch_hot_skips(int reset) { (void)reset; return 0; }
__attribute__((weak)) void plat_mem_pw_stats(uint64_t *calls, uint64_t *pages, int reset) {
    (void)reset;
    if (calls) *calls = 0;
    if (pages) *pages = 0;
}
__attribute__((weak)) int plat_mem_back_hold_as(uint64_t addr, size_t n, int cause) {
    (void)cause;
    return plat_mem_back_hold(addr, n);
}
__attribute__((weak)) uint64_t plat_mem_backed_bytes(uint64_t addr, uint64_t size) {
    (void)addr; (void)size;
    return 0;
}
__attribute__((weak)) uint64_t plat_mem_release_backing(uint64_t addr, uint64_t size,
                                                          PlatBackingDiscarded discarded) {
    (void)addr; (void)size; (void)discarded;
    return 0;
}
__attribute__((weak)) void plat_mem_protect_stats(void (*emit)(uintptr_t, uint64_t, uint64_t, void *), void *ctx) {
    (void)emit; (void)ctx;
}
__attribute__((weak)) int plat_mem_unpack_batch_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_mem_zero_pool_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_mem_decommit_run_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_mem_fill_window_mode(int on) { (void)on; return -1; }
__attribute__((weak)) int plat_mem_map_probe(char *out, size_t cap) { (void)out, (void)cap; return 0; }
__attribute__((weak)) void plat_mem_zero_pool_stats(uint64_t out[7]) { memset(out, 0, 7 * sizeof *out); }
__attribute__((weak)) int plat_mem_storm_take(char *out, size_t cap, uint32_t min_faults) {
    (void)out, (void)cap, (void)min_faults;
    return 0;
}
__attribute__((weak)) void plat_mem_cost_ns(uint64_t out[PLAT_MEM_COST_N]) {
    for (int i = 0; i < PLAT_MEM_COST_N; i++) out[i] = 0;
}
__attribute__((weak)) uint64_t plat_mem_release_zero_backing(uint64_t addr, uint64_t size) {
    (void)addr; (void)size;
    return 0;
}
__attribute__((weak)) void plat_mem_back_causes(uint64_t out[PLAT_BACK_N]) {
    memset(out, 0, sizeof(uint64_t) * PLAT_BACK_N);
}
__attribute__((weak)) size_t plat_mem_early_page_size(void) { return 4096; }
__attribute__((weak)) size_t plat_mem_watch_page(void) { return 4096; }
__attribute__((weak)) PlatEarlyResult plat_mem_early_protect(uint64_t addr, size_t n, int on, int restore_prot) {
    if (plat_mem_protect(addr, n, on ? PLAT_PROT_NONE : restore_prot)) return PLAT_EARLY_OK;
    return on ? PLAT_EARLY_FAILED : PLAT_EARLY_UNRESTORED;
}
__attribute__((weak)) int plat_mem_early_open_read(uint64_t addr, size_t n) { return (void)addr, (void)n, 0; }
__attribute__((weak)) uint32_t plat_mem_early_gen(uint64_t addr) { return (void)addr, 0; }
__attribute__((weak)) void plat_mem_early_hold_hook(void (*fn)(uint64_t addr, size_t n)) { (void)fn; }
__attribute__((weak)) void plat_mem_early_resolve(uint64_t addr, size_t n) { (void)addr, (void)n; }
__attribute__((weak)) void plat_mem_thaw_hook(void (*fn)(uint64_t addr, size_t n)) { (void)fn; }
__attribute__((weak)) void plat_mem_frozen_hook(int (*fn)(uint64_t addr)) { (void)fn; }
__attribute__((weak)) void plat_mem_backed_summary(char *out, size_t cap) { if (cap) out[0] = 0; }
__attribute__((weak)) void plat_mem_pressure(PlatMemPressure *p) { *p = (PlatMemPressure){-1, -1, -1, -1, -1, -1, -1, -1}; }
__attribute__((weak)) uint64_t plat_mem_reclaim_zeros(uint64_t target_bytes, uint64_t *scanned_bytes, int cache) {
    (void)target_bytes, (void)cache;
    if (scanned_bytes) *scanned_bytes = 0;
    return 0;
}
__attribute__((weak)) uint64_t plat_mem_pack(const PlatPackParams *p) { (void)p; return 0; }
__attribute__((weak)) uint64_t plat_mem_pack_range(uint64_t lo, uint64_t hi, const PlatPackParams *p) {
    (void)lo, (void)hi, (void)p;
    return 0;
}
__attribute__((weak)) void plat_mem_pack_collect(void) {}
__attribute__((weak)) void plat_mem_pack_summary(char *out, size_t cap) { if (cap) out[0] = 0; }
__attribute__((weak)) int plat_mem_cold_census_step(uint32_t budget_us, uint32_t freeze_s, int (*freeze_fn)(uint64_t),
                                                   char *out, size_t cap) {
    (void)budget_us, (void)freeze_s, (void)freeze_fn;
    if (cap) out[0] = 0;
    return 0;
}
__attribute__((weak)) void plat_mem_frozen_ms_hook(uint32_t (*fn)(uint64_t addr)) { (void)fn; }

// sigsample.c's, where it is not built (Windows).
__attribute__((weak)) void plat_sample_add_self(const char *name) { (void)name; }
__attribute__((weak)) void plat_sample_remove_self(void) {}
__attribute__((weak)) int64_t plat_sample_cpu_named(const char *name) { (void)name; return -1; }
__attribute__((weak)) uint64_t plat_sample_cpu_tid(uint32_t tid) { (void)tid; return 0; }
__attribute__((weak)) int plat_sample_slot_name(int slot, char *out, size_t cap) { (void)slot, (void)out, (void)cap; return 0; }
__attribute__((weak)) int plat_sample_start(uint32_t period_us, PlatSampleHook hook) { (void)period_us, (void)hook; return 0; }
__attribute__((weak)) void plat_sample_stop(void) {}
__attribute__((weak)) int plat_mem_copy_guarded(void *dst, const void *src, size_t n) { memcpy(dst, src, n); return 1; }
__attribute__((weak)) void plat_fault_guarded(void) {}
// fault_guard.c's decoder and single step, which it builds only on Linux and the PS5: Windows has neither (0, so
// memory_early.c waits for a watched page's neighbour as before b907279, whose Windows link failed without these).
__attribute__((weak)) int plat_code_init(void) { return 0; }
__attribute__((weak)) const char *plat_code_why(int *err) { if (err) *err = 0; return "no decoder here"; }
__attribute__((weak)) int plat_fault_emulate_load(uint64_t fault_addr, int extra, int (*rd)(uint64_t, size_t, void *)) {
    return (void)fault_addr, (void)extra, (void)rd, 0;
}
__attribute__((weak)) int plat_fault_step(void (*fn)(uint64_t), uint64_t arg) { return (void)fn, (void)arg, 0; }
// Background threads apart from the frame's fast cores (linux.c; win32.c's does nothing): the PS5 has no hybrid CPU,
// so every thread already runs on the same cores (vk_backend.c keeps MKW_FAST_CORES off there).
__attribute__((weak)) void plat_thread_background(void) {}
// The same for the fast-core mask: every processor alike ("& all" in the callers keeps the process's own), and the C
// heap trim after the prewarm's SPIR-V release (gfx_shader.c): the PS5's libc has no malloc_trim, so its pages stay.
__attribute__((weak)) uint64_t plat_cpu_fast(void) { return ~0ull; }
__attribute__((weak)) void plat_heap_trim(void) {}
__attribute__((weak)) int plat_heap_trim_available(void) { return 1; }
__attribute__((weak)) int plat_sample_stacks(void (*emit)(const char *, uint32_t, void *const *, int, void *), void *ctx) {
    (void)emit, (void)ctx;
    return 0;
}
__attribute__((weak)) int plat_sample_self_name(char *out, size_t cap) { (void)out, (void)cap; return 0; }
__attribute__((weak)) void plat_sample_hold(int on) { (void)on; }
