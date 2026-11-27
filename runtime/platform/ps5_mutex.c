// Native reader/writer locks and condition variables for the PS5 platform.
// Timed waits: run 23's probe proved pthread_condattr_setclock(CLOCK_MONOTONIC) works on PS5
// (mono_setclock_rc=0, mono_timedwait=10.065 ms rc=60, plat_cond_wait_ns=10.109 ms). Wait deadlines use
// CLOCK_MONOTONIC via plat_now_ns() (TSC-backed by default). pthread_cond_reltimedwait_np is not used (its PS5
// signature is unverified and hung at frame 1 in run 22).
// MKW_LOCK_PROF=1 records lock contention and contended hold times for plat_lock_report (as linux_sync.c).
// MKW_PS5_FAST_LOCK=1 (main.c, plat_lock_fast_mode) makes the locks created after it user-space atomic reader/writer
// locks that sleep on a pthread condition variable when contended (0: pthread_rwlock_t). Rules of the fast lock: word
// holds the writer bit, the reader count and LK_S, "a thread may be asleep"; a thread going to sleep sets LK_S and looks
// again at what fetch_or returned, so an unlock between its check and the bit cannot be missed (every unlock that
// empties the lock and finds LK_S wakes the sleepers). Writers in the slow path are counted in writers_waiting, and new
// readers hold back for them, so a stream of readers cannot starve a writer; LK_S itself never stops anyone.
#include "platform.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fast_lock_on, pi_mode, locks_made;

int plat_lock_fast_mode(int on) {
    __atomic_store_n(&fast_lock_on, on ? 1 : 0, __ATOMIC_RELEASE);
    return __atomic_load_n(&locks_made, __ATOMIC_ACQUIRE);
}

int plat_lock_prio_inherit_mode(int mode) {
    __atomic_store_n(&pi_mode, mode >= 0 && mode <= 2 ? mode : 0, __ATOMIC_RELEASE);
    return __atomic_load_n(&locks_made, __ATOMIC_ACQUIRE);
}

static int init_mutex_pi(pthread_mutex_t *m, int want_pi) {
    if (!want_pi) return pthread_mutex_init(m, NULL);
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr)) return pthread_mutex_init(m, NULL);
#ifdef PTHREAD_PRIO_INHERIT
    int prc = pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    if (prc) {
        static int warned;
        if (!__atomic_exchange_n(&warned, 1, __ATOMIC_RELAXED))
            fprintf(stderr, "ps5_mutex: PTHREAD_PRIO_INHERIT unsupported (%d: %s), falling back to default mutex\n",
                    prc, strerror(prc));
        pthread_mutexattr_destroy(&attr);
        return pthread_mutex_init(m, NULL);
    }
#endif
    int rc = pthread_mutex_init(m, &attr);
    pthread_mutexattr_destroy(&attr);
    return rc;
}

// ---- MKW_LOCK_PROF (ported from linux_sync.c)

#define LP_N 512
typedef struct { volatile uintptr_t a, ra, lk; volatile int64_t ns, n; } LpEnt;
static LpEnt lp[LP_N], lp_hold[LP_N];
static int lp_on = -1;

void plat_lock_prof_mode(int on) { __atomic_store_n(&lp_on, on ? 1 : 0, __ATOMIC_RELEASE); }

#define LH_DEPTH 16
static _Thread_local struct { const void *m; uintptr_t ra; uint64_t t; } lh[LH_DEPTH];
static _Thread_local int nlh;

static uint64_t lp_begin(void) {
    if (lp_on < 0) {
        const char *e = getenv("MKW_LOCK_PROF");
        lp_on = e && *e && *e != '0';
    }
    return lp_on ? plat_now_ns() : 0;
}

static void lp_add(LpEnt *tbl, const void *m, uint64_t t0, uintptr_t ra) {
    uintptr_t a = (uintptr_t)m, key = a ^ ra * 0x9E3779B97F4A7C15ull;
    uint64_t d = plat_now_ns() - t0;
    for (uint32_t i = (uint32_t)((key >> 3) * 0x9E3779B1u) % LP_N, k = 0; k < LP_N; k++, i = (i + 1) % LP_N) {
        uintptr_t cur = __atomic_load_n(&tbl[i].a, __ATOMIC_RELAXED);
        if (!cur && __atomic_compare_exchange_n(&tbl[i].a, &cur, key, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            cur = key, tbl[i].ra = ra, tbl[i].lk = a;
        if (cur == key && tbl[i].ra == ra) {
            __atomic_fetch_add(&tbl[i].ns, (int64_t)d, __ATOMIC_RELAXED);
            __atomic_fetch_add(&tbl[i].n, 1, __ATOMIC_RELAXED);
            return;
        }
    }
}

static void lh_push(const void *m, uintptr_t ra) {
    if (nlh == LH_DEPTH) memmove(lh, lh + 1, sizeof lh - sizeof *lh), nlh--;
    lh[nlh].m = m, lh[nlh].ra = ra, lh[nlh++].t = plat_now_ns();
}

static void lh_pop(const void *m, int slept) {
    for (int i = nlh - 1; i >= 0; i--)
        if (lh[i].m == m) {
            if (slept) lp_add(lp_hold, m, lh[i].t, lh[i].ra);
            lh[i] = lh[--nlh];
            return;
        }
}

int plat_lock_report(int holds, char *buf, int size) {
    LpEnt *t = holds ? lp_hold : lp;
    int k = 0;
    for (int top = 0; top < 10; top++) {
        int best = -1;
        for (int i = 0; i < LP_N; i++)
            if (t[i].a && t[i].ns > 0 && (best < 0 || t[i].ns > t[best].ns)) best = i;
        if (best < 0) break;
        if (k < size)
            k += snprintf(buf + k, (size_t)(size - k), "%s exe+%#lx from exe+%#lx %.1f ms (%lld)", top ? ";" : "",
                          (unsigned long)(t[best].lk - plat_exe_base()), (unsigned long)(t[best].ra - plat_exe_base()),
                          t[best].ns / 1e6, (long long)t[best].n);
        t[best].ns = -t[best].ns;
    }
    for (int i = 0; i < LP_N; i++) t[i].ns = 0, t[i].n = 0;
    return k;
}

// ---- locks

#define LK_W (1u << 31)
#define LK_S (1u << 30)
#define LK_R (LK_S - 1)
#define LK_SPINS 40

typedef struct {
    int fast;
    int pi_mtx;   // MKW_PS5_PRIO_INHERIT=2: exclusive PTHREAD_PRIO_INHERIT pthread_mutex_t in m
    pthread_rwlock_t rw;
    volatile uint32_t word;
    volatile uint32_t waiters;           // threads between taking m to sleep and leaving (m)
    volatile uint32_t writers_waiting;   // of them, writers (changed under m, read without it)
    pthread_mutex_t m;
    pthread_cond_t cv;
} Ps5Lock;

static Ps5Lock *get_lock(PlatMutex *mutex) {
    void *lock = __atomic_load_n(&mutex->p, __ATOMIC_ACQUIRE);
    if (lock) return lock;

    Ps5Lock *created = calloc(1, sizeof *created);
    if (!created) abort();
    int pi = __atomic_load_n(&pi_mode, __ATOMIC_ACQUIRE);
    created->fast = __atomic_load_n(&fast_lock_on, __ATOMIC_ACQUIRE);
    created->pi_mtx = !created->fast && pi >= 2;
    if (created->fast) {
        if (init_mutex_pi(&created->m, pi >= 1) || pthread_cond_init(&created->cv, NULL)) abort();
    } else if (created->pi_mtx) {
        if (init_mutex_pi(&created->m, 1)) abort();
    } else {
        if (pthread_rwlock_init(&created->rw, NULL)) abort();
    }
    if (__atomic_compare_exchange_n(&mutex->p, &lock, created, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        __atomic_add_fetch(&locks_made, 1, __ATOMIC_ACQ_REL);
        return created;
    }
    if (created->fast) {
        pthread_cond_destroy(&created->cv);
        pthread_mutex_destroy(&created->m);
    } else if (created->pi_mtx) {
        pthread_mutex_destroy(&created->m);
    } else {
        pthread_rwlock_destroy(&created->rw);
    }
    free(created);
    return lock;
}

void plat_mutex_init(PlatMutex *mutex) {
    mutex->p = NULL;
    (void)get_lock(mutex);
}

// After an unlock that found LK_S: everyone asleep looks again. The bit goes when no one is left (under m, where it is set).
static void fast_wake(Ps5Lock *l) {
    pthread_mutex_lock(&l->m);
    if (__atomic_load_n(&l->waiters, __ATOMIC_RELAXED) == 0) __atomic_fetch_and(&l->word, ~LK_S, __ATOMIC_ACQ_REL);
    pthread_cond_broadcast(&l->cv);
    pthread_mutex_unlock(&l->m);
}

// m held: sleeps until the lock has none of the blocking bits (and, for a reader, no writer is waiting).
static void fast_sleep(Ps5Lock *l, uint32_t blocking, int reader) {
    for (;;) {
        uint32_t s = __atomic_load_n(&l->word, __ATOMIC_ACQUIRE);
        int writers = reader && __atomic_load_n(&l->writers_waiting, __ATOMIC_RELAXED);
        if (!(s & blocking) && !writers) return;
        if (!(s & LK_S)) {
            s = __atomic_fetch_or(&l->word, LK_S, __ATOMIC_ACQ_REL);
            if (!(s & blocking) && !writers) return;   // released before the bit went in: no one would wake this thread
        }
        pthread_cond_wait(&l->cv, &l->m);
    }
}

static void lock_at(PlatMutex *mutex, uintptr_t ra) {
    Ps5Lock *l = get_lock(mutex);
    if (l->fast) {
        uint64_t t0 = 0;
        for (int spin = 0;; spin++) {
            uint32_t s = __atomic_load_n(&l->word, __ATOMIC_RELAXED);
            if (!(s & (LK_W | LK_R))) {
                if (__atomic_compare_exchange_n(&l->word, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                    if (t0) lp_add(lp, mutex, t0, ra);
                    if (lp_on > 0) lh_push(mutex, ra);
                    return;
                }
                continue;
            }
            if (!spin) t0 = lp_begin();
            if (spin < LK_SPINS) { plat_pause(); continue; }
            pthread_mutex_lock(&l->m);
            __atomic_fetch_add(&l->waiters, 1, __ATOMIC_ACQ_REL);
            __atomic_fetch_add(&l->writers_waiting, 1, __ATOMIC_RELAXED);
            fast_sleep(l, LK_W | LK_R, 0);
            __atomic_fetch_sub(&l->writers_waiting, 1, __ATOMIC_RELAXED);
            __atomic_fetch_sub(&l->waiters, 1, __ATOMIC_ACQ_REL);
            pthread_mutex_unlock(&l->m);
        }
    }
    if (l->pi_mtx) {
        uint64_t t0 = 0;
        if (lp_on != 0 && pthread_mutex_trylock(&l->m) == 0) {
            if (lp_on > 0) lh_push(mutex, ra);
            return;
        }
        if (lp_on != 0) t0 = lp_begin();
        if (pthread_mutex_lock(&l->m)) abort();
        if (t0) lp_add(lp, mutex, t0, ra);
        if (lp_on > 0) lh_push(mutex, ra);
        return;
    }
    uint64_t t0 = 0;
    if (lp_on != 0 && pthread_rwlock_trywrlock(&l->rw) == 0) {
        if (lp_on > 0) lh_push(mutex, ra);
        return;
    }
    if (lp_on != 0) t0 = lp_begin();
    if (pthread_rwlock_wrlock(&l->rw)) abort();
    if (t0) lp_add(lp, mutex, t0, ra);
    if (lp_on > 0) lh_push(mutex, ra);
}

void plat_mutex_lock(PlatMutex *mutex) { lock_at(mutex, (uintptr_t)__builtin_return_address(0)); }
void plat_mutex_lock_from(PlatMutex *mutex, uintptr_t caller) { lock_at(mutex, caller); }

int plat_mutex_trylock(PlatMutex *mutex) {
    Ps5Lock *l = get_lock(mutex);
    if (l->fast) {
        uint32_t s = __atomic_load_n(&l->word, __ATOMIC_RELAXED);
        while (!(s & (LK_W | LK_R)))
            if (__atomic_compare_exchange_n(&l->word, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
        return 0;
    }
    if (l->pi_mtx) return pthread_mutex_trylock(&l->m) == 0;
    return pthread_rwlock_trywrlock(&l->rw) == 0;
}

void plat_mutex_unlock(PlatMutex *mutex) {
    Ps5Lock *l = get_lock(mutex);
    if (l->fast) {
        uint32_t s = __atomic_fetch_and(&l->word, ~LK_W, __ATOMIC_RELEASE);
        if (lp_on > 0) lh_pop(mutex, (s & LK_S) != 0);
        if (s & LK_S) fast_wake(l);
        return;
    }
    if (lp_on > 0) lh_pop(mutex, 0);
    if (l->pi_mtx) {
        if (pthread_mutex_unlock(&l->m)) abort();
        return;
    }
    if (pthread_rwlock_unlock(&l->rw)) abort();
}

void plat_mutex_lock_shared(PlatMutex *mutex) {
    Ps5Lock *l = get_lock(mutex);
    uintptr_t ra = (uintptr_t)__builtin_return_address(0);
    if (l->fast) {
        uint64_t t0 = 0;
        for (int spin = 0;; spin++) {
            uint32_t s = __atomic_load_n(&l->word, __ATOMIC_RELAXED);
            if (!(s & LK_W) && !__atomic_load_n(&l->writers_waiting, __ATOMIC_RELAXED)) {
                if (__atomic_compare_exchange_n(&l->word, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                    if (t0) lp_add(lp, mutex, t0, ra);
                    return;
                }
                continue;
            }
            if (!spin) t0 = lp_begin();
            if (spin < LK_SPINS) { plat_pause(); continue; }
            pthread_mutex_lock(&l->m);
            __atomic_fetch_add(&l->waiters, 1, __ATOMIC_ACQ_REL);
            fast_sleep(l, LK_W, 1);
            __atomic_fetch_sub(&l->waiters, 1, __ATOMIC_ACQ_REL);
            pthread_mutex_unlock(&l->m);
        }
    }
    if (l->pi_mtx) {
        uint64_t t0 = 0;
        if (lp_on != 0 && pthread_mutex_trylock(&l->m) == 0) return;
        if (lp_on != 0) t0 = lp_begin();
        if (pthread_mutex_lock(&l->m)) abort();
        if (t0) lp_add(lp, mutex, t0, ra);
        return;
    }
    uint64_t t0 = 0;
    if (lp_on != 0 && pthread_rwlock_tryrdlock(&l->rw) == 0) return;
    if (lp_on != 0) t0 = lp_begin();
    if (pthread_rwlock_rdlock(&l->rw)) abort();
    if (t0) lp_add(lp, mutex, t0, ra);
}

int plat_mutex_trylock_shared(PlatMutex *mutex) {
    Ps5Lock *l = get_lock(mutex);
    if (l->fast) {
        uint32_t s = __atomic_load_n(&l->word, __ATOMIC_RELAXED);
        while (!(s & LK_W) && !__atomic_load_n(&l->writers_waiting, __ATOMIC_RELAXED))
            if (__atomic_compare_exchange_n(&l->word, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
        return 0;
    }
    if (l->pi_mtx) return pthread_mutex_trylock(&l->m) == 0;
    return pthread_rwlock_tryrdlock(&l->rw) == 0;
}

void plat_mutex_unlock_shared(PlatMutex *mutex) {
    Ps5Lock *l = get_lock(mutex);
    if (l->fast) {
        uint32_t s = __atomic_sub_fetch(&l->word, 1, __ATOMIC_RELEASE);
        if ((s & LK_S) && !(s & LK_R)) fast_wake(l);
        return;
    }
    if (l->pi_mtx) {
        if (pthread_mutex_unlock(&l->m)) abort();
        return;
    }
    if (pthread_rwlock_unlock(&l->rw)) abort();
}

// ---- condition variables

typedef struct {
    pthread_mutex_t gate;
    pthread_cond_t cond;
} Cond;

static Cond *condition(PlatCond *cond) {
    void *value = __atomic_load_n(&cond->p, __ATOMIC_ACQUIRE);
    if (value) return value;

    Cond *created = malloc(sizeof *created);
    pthread_condattr_t attr;
    int want_pi = __atomic_load_n(&pi_mode, __ATOMIC_ACQUIRE) >= 1;
    if (!created || init_mutex_pi(&created->gate, want_pi) || pthread_condattr_init(&attr) ||
        pthread_condattr_setclock(&attr, CLOCK_MONOTONIC) || pthread_cond_init(&created->cond, &attr))
        abort();
    pthread_condattr_destroy(&attr);
    if (__atomic_compare_exchange_n(&cond->p, &value, created, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return created;
    pthread_cond_destroy(&created->cond);
    pthread_mutex_destroy(&created->gate);
    free(created);
    return value;
}

void plat_cond_init(PlatCond *cond) {
    cond->p = NULL;
    (void)condition(cond);
}

static int wait_cond(PlatCond *cond, PlatMutex *mutex, uint64_t ns, int timed) {
    Cond *state = condition(cond);
    // The gate makes dropping the lock and entering the pthread wait atomic with respect to a notification.
    if (pthread_mutex_lock(&state->gate)) abort();
    plat_mutex_unlock(mutex);
    int result;
    if (!timed) {
        result = pthread_cond_wait(&state->cond, &state->gate);
    } else {
        uint64_t now = plat_now_ns(), max_ns = (uint64_t)3600 * 24 * 365 * 1000000000ull;
        if (ns > max_ns) ns = max_ns;
        uint64_t target = ns > UINT64_MAX - now ? UINT64_MAX : now + ns;
        struct timespec deadline = {.tv_sec = (time_t)(target / 1000000000ull), .tv_nsec = (long)(target % 1000000000ull)};
        result = pthread_cond_timedwait(&state->cond, &state->gate, &deadline);
    }
    if (pthread_mutex_unlock(&state->gate)) abort();
    plat_mutex_lock(mutex);
    if (result && result != ETIMEDOUT) abort();
    return result == 0;
}

void plat_cond_wait(PlatCond *cond, PlatMutex *mutex) { (void)wait_cond(cond, mutex, 0, 0); }

int plat_cond_wait_ns(PlatCond *cond, PlatMutex *mutex, uint64_t ns) {
    return wait_cond(cond, mutex, ns, 1);
}

static void notify(PlatCond *cond, int all) {
    Cond *state = condition(cond);
    if (pthread_mutex_lock(&state->gate)) abort();
    int result = all ? pthread_cond_broadcast(&state->cond) : pthread_cond_signal(&state->cond);
    if (pthread_mutex_unlock(&state->gate) || result) abort();
}

void plat_cond_signal(PlatCond *cond) { notify(cond, 0); }
void plat_cond_broadcast(PlatCond *cond) { notify(cond, 1); }

void plat_cond_probe(void) {
    static int probed;
    if (__atomic_exchange_n(&probed, 1, __ATOMIC_ACQ_REL)) return;

    pthread_mutex_t m;
    pthread_cond_t c_mono;
    pthread_condattr_t a;
    pthread_mutex_init(&m, NULL);
    pthread_condattr_init(&a);
    int attr_rc = pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
    pthread_cond_init(&c_mono, &a);
    pthread_condattr_destroy(&a);

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t nsec = (uint64_t)ts.tv_nsec + 10000000ull;
    ts.tv_sec += (time_t)(nsec / 1000000000ull);
    ts.tv_nsec = (long)(nsec % 1000000000ull);
    pthread_mutex_lock(&m);
    uint64_t t0 = plat_now_ns();
    int mono_rc = pthread_cond_timedwait(&c_mono, &m, &ts);
    uint64_t mono_el = plat_now_ns() - t0;
    pthread_mutex_unlock(&m);
    pthread_cond_destroy(&c_mono);
    pthread_mutex_destroy(&m);

    PlatMutex pm = {0};
    PlatCond pc = {0};
    plat_mutex_lock(&pm);
    uint64_t t1 = plat_now_ns();
    int wait_ret = plat_cond_wait_ns(&pc, &pm, 10000000ull);
    uint64_t wait_el = plat_now_ns() - t1;
    plat_mutex_unlock(&pm);

    fprintf(stderr, "ps5 cond probe (10 ms): mono_setclock_rc=%d mono_timedwait=%.3f ms (rc=%d), plat_cond_wait_ns=%.3f ms (ret=%d)\n",
            attr_rc, mono_el / 1e6, mono_rc, wait_el / 1e6, wait_ret);
}
