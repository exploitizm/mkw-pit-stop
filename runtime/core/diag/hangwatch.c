// Hang watch: where each runtime thread is when the game stops presenting. The PS5's run 5 (13/11/2026) stopped after
// "sparse bind 2" with no crash, no "GPU has not reached submission" from RADV and the kernel log still flowing: the
// game's threads were blocked and nothing said where. Every thread that calls hang_mark gets a slot holding its last
// checkpoint (a static string: "queue lock: waiting", "vkQueueBindSparse"...) and when it was set. A watchdog thread
// wakes every WAKE_MS; once the present counter (hang_frame) has not moved for MKW_HANG_WATCH seconds it logs every
// slot once, then the reporters' values (semaphores, counters), and a line when presents resume. It reads the slots
// with atomic loads and takes no lock of the runtime's: a lock the hung threads hold cannot hold the report too.
//
//   MKW_HANG_WATCH=<s>   seconds (fractions allowed) without a present before the report; 0 turns it off. Default 10 on the PS5 (no
//                        debugger attaches to a title there), 0 elsewhere (gdb, and llvmpipe runs present every 25 s).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <pthread.h>
#endif
#ifdef __PROSPERO__
#include <pthread_np.h>
#endif

#include "cfg.h"
#include "hangwatch.h"
#include "host.h"
#include "prof.h"
#include "platform/platform.h"

#define SLOTS 128
#define WAKE_MS 2000
#define REPORTERS 8

typedef struct {
    const char *where;   // atomic
    uint64_t since;      // atomic: plat_now_ns when where was set
    uint64_t val;        // atomic: value involved when has_val is set
    int has_val;         // atomic
    uint32_t tid;
    char name[24];
} Slot;

static Slot slots[SLOTS];
static int nslots;                       // atomic: slots handed out
static _Thread_local int my_slot = -1;   // -2: no slot left
static void (*reporters[REPORTERS])(char *, int);
static int nreporters;
static uint64_t frame_seen;              // atomic: hang_frame's last frame
static int started;

#if defined(__PROSPERO__)
// Exported by the PS5 libkernel (PS5_PayloadSDK sce_stubs/libkernel.c) but not declared in the SDK's headers.
int pthread_getname_np(pthread_t thread, char *name, size_t len);
#endif

static void thread_name(char *buf, size_t n) {
    buf[0] = 0;
#if defined(__PROSPERO__) || defined(__linux__)
    // The PS5 libkernel exports pthread_getname_np, not FreeBSD's pthread_get_name_np (the console link failed on it).
    pthread_getname_np(pthread_self(), buf, n);
#endif
    if (!buf[0]) snprintf(buf, n, "?");
}

uint32_t hang_tid(void) {
    static _Thread_local uint32_t tid;
    if (!tid) tid = plat_thread_id();
    return tid;
}

static const char *mark_set(const char *where, int has_val, uint64_t val) {
    int i = my_slot;
    if (i == -1) {
        i = __atomic_fetch_add(&nslots, 1, __ATOMIC_RELAXED);
        if (i >= SLOTS) {
            my_slot = -2;
            return NULL;
        }
        slots[i].tid = hang_tid();
        thread_name(slots[i].name, sizeof slots[i].name);
        my_slot = i;
    }
    if (i < 0) return NULL;
    const char *prev = __atomic_load_n(&slots[i].where, __ATOMIC_RELAXED);
    __atomic_store_n(&slots[i].since, plat_now_ns(), __ATOMIC_RELAXED);
    __atomic_store_n(&slots[i].val, val, __ATOMIC_RELAXED);
    __atomic_store_n(&slots[i].has_val, has_val, __ATOMIC_RELAXED);
    __atomic_store_n(&slots[i].where, where, __ATOMIC_RELEASE);
    return prev;
}

const char *hang_mark(const char *where) { return mark_set(where, 0, 0); }
const char *hang_mark_val(const char *where, uint64_t val) { return mark_set(where, 1, val); }

void hang_reporter(void (*fn)(char *buf, int n)) {
    int i = __atomic_fetch_add(&nreporters, 1, __ATOMIC_RELAXED);
    if (i < REPORTERS) __atomic_store_n(&reporters[i], fn, __ATOMIC_RELEASE);
}

// Each runtime thread's stack at the report (MKW_HANG_STACKS, default 1): the interrupted rip, then the frames below
// the signal frame, named (prof_pc_describe). Threads idle on a condition variable are left out.
static void stack_line(const char *name, uint32_t tid, void *const *f, int n, void *ctx) {
    (void)ctx;
    char line[1600], d[200];
    int o = snprintf(line, sizeof line, "hang watch: stack %-16s tid %u:", name, tid);
    // f[0] the interrupted rip; f[1] the handler, f[2] the signal trampoline, f[3] the interrupted function again (glibc's
    // backtrace from a handler): its callers follow from f[4]
    for (int i = 0; i < n && o < (int)sizeof line - 220; i = i ? i + 1 : 4) {
        prof_pc_describe((uint64_t)(uintptr_t)f[i], d, sizeof d);
        o += snprintf(line + o, sizeof line - (size_t)o, "%s %s", i ? " <" : "", d);
    }
    rt_log("%s", line);
}

static void report(uint64_t frame, uint64_t stalled_ns) {
    uint64_t now = plat_now_ns();
    int n = __atomic_load_n(&nslots, __ATOMIC_ACQUIRE);
    if (n > SLOTS) n = SLOTS;
    rt_log("hang watch: no present for %.1f s (frame %llu); %d threads have checkpoints:", stalled_ns / 1e9,
           (unsigned long long)frame, n);
    for (int i = 0; i < n; i++) {
        const char *w = __atomic_load_n(&slots[i].where, __ATOMIC_ACQUIRE);
        uint64_t since = __atomic_load_n(&slots[i].since, __ATOMIC_RELAXED);
        int hv = __atomic_load_n(&slots[i].has_val, __ATOMIC_RELAXED);
        uint64_t v = hv ? __atomic_load_n(&slots[i].val, __ATOMIC_RELAXED) : 0;
        if (!w) continue;   // registered, then set back to nothing: idle
        if (hv)
            rt_log("hang watch:   %-16s tid %u: %s (%llu) for %.1f s", slots[i].name, slots[i].tid, w,
                   (unsigned long long)v, since && now > since ? (now - since) / 1e9 : 0.0);
        else
            rt_log("hang watch:   %-16s tid %u: %s for %.1f s", slots[i].name, slots[i].tid, w,
                   since && now > since ? (now - since) / 1e9 : 0.0);
    }
    int nr = __atomic_load_n(&nreporters, __ATOMIC_ACQUIRE);
    for (int i = 0; i < nr && i < REPORTERS; i++) {
        void (*fn)(char *, int) = __atomic_load_n(&reporters[i], __ATOMIC_ACQUIRE);
        char buf[512];
        if (!fn) continue;
        buf[0] = 0;
        fn(buf, (int)sizeof buf);
        if (buf[0]) rt_log("hang watch:   %s", buf);
    }
    static int stacks = -1;
    if (stacks < 0) stacks = cfg_get("MKW_HANG_STACKS") ? atoi(cfg_get("MKW_HANG_STACKS")) : 1;
    if (stacks) {
        prof_names_init();
        plat_sample_stacks(stack_line, NULL);
    }
}

static void watch_main(void *arg) {
    uint64_t limit_ms = (uint64_t)(uintptr_t)arg, limit = limit_ms * 1000000ull;
    // Four looks per limit (at most every WAKE_MS, at least every 50 ms): with one look every 2 s a stall had to span
    // two looks, so the 0.6-1 s hitches of a PS5 race (run vo, 16/11/2026) were never reported.
    uint32_t wake = limit_ms / 4 < WAKE_MS ? (uint32_t)(limit_ms / 4) : WAKE_MS;
    if (wake < 50) wake = 50;
    uint64_t last = __atomic_load_n(&frame_seen, __ATOMIC_ACQUIRE), moved = plat_now_ns();
    // A stall is reported at `limit`, then again every REPEAT_S while it lasts (at most MAX_REPORTS times): console run 6
    // reported once 10 s into a slow load, and the real hang that followed (after "sparse bind 2") went unreported.
    enum { REPEAT_S = 30, MAX_REPORTS = 8 };
    int reported = 0;
    uint64_t next = limit;
    for (;;) {
        plat_sleep_ms(wake);
        uint64_t f = __atomic_load_n(&frame_seen, __ATOMIC_ACQUIRE), now = plat_now_ns();
        if (f != last) {
            if (reported) rt_log("hang watch: presents resumed at frame %llu after %.1f s", (unsigned long long)f,
                                 (now - moved) / 1e9);
            last = f, moved = now, reported = 0, next = limit;
        } else if (reported < MAX_REPORTS && now - moved >= next) {
            report(f, now - moved);
            reported++;
            next += REPEAT_S * 1000000000ull;
        }
    }
}

// ---- MKW_STALL_TRACE=<ms> (default 0, off): what every thread's checkpoint was through each frame of <ms> or more.
// A sampler thread reads every slot's checkpoint once a millisecond into a ring of STALL_RING samples; a present that
// comes <ms> or more after the previous one asks it for the frame's window, and it logs one line per thread with the
// milliseconds spent at each checkpoint ("(none)": no checkpoint set, a guest thread running its own code or a runtime
// thread between marks). Only the sampler touches the ring. For the 50-90 ms frames of the PS5 race whose cost ledger
// shows only 8-10 ms of replay and nothing else (28/11/2026): which thread the frame waited on, and on what.
#define STALL_RING 256
static const char *st_ring[STALL_RING][SLOTS];
static uint64_t st_ts[STALL_RING];
static uint32_t st_at;                         // samples taken (the sampler's)
static uint64_t st_req_t0, st_req_t1, st_req_frame;   // atomic: a window to report (t1 set last)
static uint64_t st_last_present;

static void stall_dump(uint64_t t0, uint64_t t1, uint64_t frame) {
    static int dumps;
    if (dumps++ >= 80) return;
    int n = __atomic_load_n(&nslots, __ATOMIC_ACQUIRE);
    if (n > SLOTS) n = SLOTS;
    uint32_t last = st_at;
    rt_log("stall trace: frame %llu, %.1f ms (checkpoint ms by thread, one sample a ms):", (unsigned long long)frame,
           (t1 - t0) / 1e6);
    for (int i = 0; i < n; i++) {
        const char *lab[6] = {0};
        int cnt[6] = {0}, nl = 0, other = 0, total = 0;
        for (uint32_t k = 0; k < STALL_RING && k < last; k++) {
            uint32_t s = (last - 1 - k) % STALL_RING;
            if (st_ts[s] < t0) break;
            if (st_ts[s] > t1) continue;
            const char *w = st_ring[s][i];
            total++;
            int j = 0;
            while (j < nl && lab[j] != w) j++;
            if (j < nl) cnt[j]++;
            else if (nl < 6) lab[nl] = w, cnt[nl++] = 1;
            else other++;
        }
        if (!total) continue;
        // a thread that sat at one idle checkpoint the whole frame says nothing about the stall
        if (nl == 1 && lab[0] && (strstr(lab[0], "idle") || strstr(lab[0], "sleep"))) continue;
        char line[600];
        int o = snprintf(line, sizeof line, "stall trace:   %-16s tid %u:", slots[i].name, slots[i].tid);
        for (int j = 0; j < nl && o < (int)sizeof line - 80; j++)
            o += snprintf(line + o, sizeof line - (size_t)o, " %s %d;", lab[j] ? lab[j] : "(none)", cnt[j]);
        if (other && o < (int)sizeof line - 30) snprintf(line + o, sizeof line - (size_t)o, " other %d", other);
        rt_log("%s", line);
    }
}

static void stall_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_sleep_ns(1000000);
        uint32_t s = st_at % STALL_RING;
        int n = __atomic_load_n(&nslots, __ATOMIC_ACQUIRE);
        if (n > SLOTS) n = SLOTS;
        for (int i = 0; i < n; i++) st_ring[s][i] = __atomic_load_n(&slots[i].where, __ATOMIC_ACQUIRE);
        st_ts[s] = plat_now_ns();
        st_at++;
        uint64_t t1 = __atomic_exchange_n(&st_req_t1, 0, __ATOMIC_ACQUIRE);
        if (t1) stall_dump(__atomic_load_n(&st_req_t0, __ATOMIC_RELAXED), t1, __atomic_load_n(&st_req_frame, __ATOMIC_RELAXED));
    }
}

static void stall_frame(uint64_t frame) {
    static int64_t ms = -1;
    if (ms < 0) {
        const char *e = cfg_get("MKW_STALL_TRACE");
        ms = e && *e ? atol(e) : 0;
        if (ms > 0 && !plat_thread_spawn(stall_main, NULL, "stall trace", 0)) ms = 0;
        if (ms > 0) rt_log("stall trace: each frame of %lld ms or more logs its threads' checkpoints (MKW_STALL_TRACE)", (long long)ms);
    }
    uint64_t now = plat_now_ns(), prev = st_last_present;
    st_last_present = now;
    if (ms <= 0 || !prev || now - prev < (uint64_t)ms * 1000000ull) return;
    __atomic_store_n(&st_req_t0, prev, __ATOMIC_RELAXED);
    __atomic_store_n(&st_req_frame, frame, __ATOMIC_RELAXED);
    __atomic_store_n(&st_req_t1, now, __ATOMIC_RELEASE);
}

void hang_frame(uint64_t frame) {
    __atomic_store_n(&frame_seen, frame, __ATOMIC_RELEASE);
    stall_frame(frame);
    if (__atomic_load_n(&started, __ATOMIC_RELAXED)) return;
    started = 1;   // the present thread only
#ifdef __PROSPERO__
    const char *def = "10";
#else
    const char *def = "0";
#endif
    const char *e = cfg_get("MKW_HANG_WATCH");
    double s = atof(e && *e ? e : def);   // seconds, fractions allowed (0.5)
    long ms = (long)(s * 1000.0 + 0.5);
    if (ms <= 0) return;
    if (!plat_thread_spawn(watch_main, (void *)(uintptr_t)ms, "hang watch", 0)) {
        rt_log("hang watch: cannot start its thread");
        return;
    }
    rt_log("hang watch: every thread's checkpoint is logged after %.2f s without a present (MKW_HANG_WATCH)", s);
}
