// Adapted from MacNativeGames games/totk/runtime/threads.c @ b75be8c (MIT, Copyright (c) 2026 MacNativeGames
// contributors; itself from d2r-mobile dm/host/hle_thread.c), d2r-mobile dm/host/thread.c @ 53e7c5b (MIT, Copyright
// (c) 2026 jhaizhou) and LAS Recomp runtime/kernel.c @ da35b83 (MIT, Copyright (c) 2026 Peiikai); the preemption of
// virtual cores from MacNativeGames threads.c @ 33e8fa8 (MIT, as above).
// Threads, synchronization, virtual cores and fibers of the Horizon kernel.
//
// One guest thread = one Windows thread with a 256 MiB reserved host stack (every guest BL is a host call) and its own
// Cpu. The SDK's nn::os runs as lifted code; only the kernel under it is the host. Every wait hangs on one global lock
// G with one condition variable per thread; wake order is the kernel's: lower priority value first, then first come.
// Semantics follow the public Horizon kernel (KConditionVariable, KAddressArbiter, KSynchronization).
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "hangwatch.h"
#include "kernel_int.h"
#include "platform/platform.h"

const char *const s_name[S_N] = {"off", "run", "core", "host", "sleep", "sync", "cv", "lock", "addr"};
PlatMutex G = PLAT_MUTEX_INIT;
HandleSlot g_h[MAX_H];
_Thread_local RtThread *tcur;
RtThread *g_all, *g_wait, *g_main;
uint64_t g_next_id = 0x51, g_wseq, g_created;
uint64_t g_main_stack;


uint64_t kern_now_ns(void) { return plat_now_ns(); }
int st_of(int kind) {
    return kind == W_LOCK ? S_LOCK : kind == W_CONDVAR ? S_CV : kind == W_ADDRESS ? S_ADDR : S_SYNC;
}

// ---- handles (caller holds G)
uint32_t h_new(int type, void *obj) {
    for (int i = 1; i < MAX_H; i++)
        if (!g_h[i].type) {
            g_h[i].type = type;
            g_h[i].obj = obj;
            return 0x1000 + i;
        }
    rt_fatal("kernel: handle table full (%d handles)", MAX_H);
}

void *h_peek(uint32_t h, int type) {
    if (h == H_CUR_THREAD && type == O_THREAD) return tcur;
    unsigned i = h - 0x1000;
    if (i == 0 || i >= MAX_H || g_h[i].type != type) return NULL;
    return g_h[i].obj;
}

void *h_get(uint32_t h, int type) {
    void *o = h_peek(h, type);
    if (!o) rt_log("kernel: invalid handle %#x (want kind %d)", h, type);
    return o;
}


// ---- wait primitives: a wait list and the lock that guards it (G and g_wait; an address queue for WaitForAddress)
static void wait_unlink(RtThread **list, RtThread *t) {
    for (RtThread **p = list; *p; p = &(*p)->next)
        if (*p == t) {
            *p = t->next;
            break;
        }
    t->next = NULL;
}

// The waiter to wake in kernel order. Lock waits match by lock address, condvar waits by key, arbiter waits by
// address.
RtThread *pick_in(RtThread *list, int kind, uint64_t a) {
    RtThread *b = NULL;
    for (RtThread *t = list; t; t = t->next) {
        if (t->wkind != kind || (kind == W_CONDVAR ? t->wkey : t->waddr) != a) continue;
        if (!b || t->prio < b->prio || (t->prio == b->prio && t->wseq < b->wseq)) b = t;
    }
    return b;
}

RtThread *pick(int kind, uint64_t a) { return pick_in(g_wait, kind, a); }

void wake_addr_prepare(RtThread **list, RtThread *t, uint32_t res) {
    wait_unlink(list, t);
    t->wkind = W_NONE;
    t->wres = res;
    t->waker = tcur ? tcur->handle : 0;
    if ((rt_trace & RT_TRACE_SCHED) || g_svc_stats) t->wake_at = kern_now_ns();
    if (g_crit_on) crit_add(CRIT_WAKE, t->handle, t->waker, 0, 0, t->wake_at);
}

void wake_addr_commit(RtThread *t) {
    // Signal cv BEFORE setting woke=1 (release), while holding the queue lock, so a spinning waiter seeing
    // woke=1 cannot continue before plat_cond_signal finishes with t->cv.
    if (!t->w_nosig) plat_cond_signal(&t->cv);
    __atomic_store_n(&t->woke, 1, __ATOMIC_RELEASE);
}

void wake_in(RtThread **list, RtThread *t, uint32_t res) {
    wake_addr_prepare(list, t, res);
    wake_addr_commit(t);
}

void wake(RtThread *t, uint32_t res) { wake_in(&g_wait, t, res); }

void wake_sync_waiters(void) {
    for (RtThread *w = g_wait; w;) {
        RtThread *n = w->next;
        if (w->wkind == W_SYNC) wake(w, (uint32_t)-1);
        w = n;
    }
}

static void wait_enqueue(RtThread **list, RtThread *t, int kind, uint64_t seq) {
    t->wkind = kind;
    t->woke = 0;
    t->wseq = seq;
    t->next = *list;
    *list = t;
}

// The wait re-checks the deadline against the clock, because it may also end early. The time left is rounded up to
// whole ms (Windows' SleepConditionVariableSRW counts ms; totk-pc-recomp did the same on Linux), so a 100 us timeout
// ends after 1 ms. MKW_WAIT_NS=1 waits the exact time (Linux futex): three in-race A/B runs gave +3.8, -0.9 and
// +1.0 ms per frame (waitns-ab1..3, 06/11/2026), and with the address queues +0.7 mean, -1.0 median (waitns-ab4): no
// gain over the noise, so it stays off.
static int g_wait_ns = -1;

// Sleeps until woken or ns pass (ns < 0: no limit), holding m, which guards list (t is on it).
static uint32_t wait_woken(RtThread *t, int64_t ns, PlatMutex *m, RtThread **list) {
    const char *hm_prev = hang_mark("kernel: waiting (svc)");   // MKW_STALL_TRACE (hangwatch.c)
    uint64_t now = kern_now_ns();
    int forever = ns < 0 || (uint64_t)ns > UINT64_MAX - now;
    uint64_t deadline = forever ? 0 : now + (uint64_t)ns;
    while (!t->woke) {
        if (forever) {
            plat_cond_wait(&t->cv, m);
            continue;
        }
        now = kern_now_ns();
        if (now >= deadline) {
            wait_unlink(list, t);
            t->wkind = W_NONE;
            if (rt_trace & RT_TRACE_SCHED) t->ho_wake_at = 0;
            hang_mark(hm_prev);
            return R_TIMEOUT;
        }
        if (g_wait_ns < 0) g_wait_ns = rt_knob("WAIT_NS", "0");
        uint64_t left = deadline - now;
        if (!RT_KNOB_ON(g_wait_ns)) left = (left + 999999) / 1000000 * 1000000;
        plat_cond_wait_ns(&t->cv, m, left);
    }
    if (t->wake_at) {
        t->last_wake_at = t->wake_at;
        if (rt_trace & RT_TRACE_SCHED) t->ho_wake_at = t->wake_at;
        uint64_t d = kern_now_ns() - t->wake_at;
        t->lw_ns += d;
        t->lw_n++;
        if (d > t->lw_max) t->lw_max = d;
        t->wake_at = 0;
    } else if (rt_trace & RT_TRACE_SCHED) {
        t->ho_wake_at = 0;
    }
    hang_mark(hm_prev);
    return t->wres;
}

uint32_t block_inner(RtThread *t, int kind, int64_t ns) {   // holding G
    wait_enqueue(&g_wait, t, kind, ++g_wseq);
    if (ns == 0) {
        wait_unlink(&g_wait, t);
        t->wkind = W_NONE;
        if (rt_trace & RT_TRACE_SCHED) t->ho_wake_at = 0;
        return R_TIMEOUT;
    }
    return wait_woken(t, ns, &G, &g_wait);
}

// ---- address queues: WaitForAddress / SignalToAddress wait and wake under the lock of one of ADDR_QUEUES lists,
// by address, instead of G and the single list g_wait that every other wait shares. They were the largest part of
// the waits for G in a race (~470 ms per 1.6 s, MKW_LOCK_PROF, waitns-ab2, 06/11/2026), and
// every priority lookup (eff_prio) walked them on g_wait. The address's value is still read and
// changed only under its queue's lock, as before under G. G is taken only to give the virtual core up and back.
// MKW_ADDR_QUEUES=0 keeps them on G and g_wait as before; read once, since a run cannot switch with threads waiting.
#define ADDR_QUEUES 256
static struct { PlatMutex m; RtThread *head; } g_aq[ADDR_QUEUES];
static int g_aq_on = -1;
static volatile uint64_t g_aseq;
static int g_addr_spin = -1;
static uint64_t g_addr_spin_ns;

AddrQueue addr_queue(uint64_t addr) {
    if (__atomic_load_n(&g_addr_spin, __ATOMIC_ACQUIRE) < 0) {
        lock();
        if (g_aq_on < 0) g_aq_on = rt_knob("ADDR_QUEUES", "1") == 1;
        if (g_addr_spin < 0) {
            const char *us = cfg_get("MKW_ADDR_SPIN_US");
            g_addr_spin_ns = (us && *us ? strtoull(us, NULL, 0) : 20ull) * 1000ull;
            __atomic_store_n(&g_addr_spin, rt_knob("ADDR_SPIN", "0"), __ATOMIC_RELEASE);
        }
        unlock();
    }
    if (!g_aq_on) return (AddrQueue){&G, &g_wait};
    unsigned i = (unsigned)(((addr >> 2) * 0x9E3779B97F4A7C15ull) >> 56) % ADDR_QUEUES;
    return (AddrQueue){&g_aq[i].m, &g_aq[i].head};
}

// WaitForAddress's wait: enters holding q.m, returns with q.m unlocked.
uint32_t addr_block(RtThread *t, AddrQueue q, int64_t ns) {
    if (q.m == &G) {
        uint32_t r = block(t, W_ADDRESS, ns);
        unlock();
        return r;
    }
    if (ns == 0) {
        plat_mutex_unlock(q.m);
        return R_TIMEOUT;
    }
    uint64_t addr = t->waddr;
    t->waker = 0;
    if (rt_trace & RT_TRACE_SCHED) {
        t->wake_at = 0;
        t->ho_wake_at = 0;
        t->ho_grant_at = 0;
        t->ho_kind = HO_UNCONT;
        t->ho_gated = 0;
        t->ho_pre = 0;
    }
    wait_enqueue(q.list, t, W_ADDRESS, __atomic_add_fetch(&g_aseq, 1, __ATOMIC_RELAXED));
    if (RT_KNOB_ON(g_addr_spin) && g_addr_spin_ns > 0 && t->cur_core >= 0 && !t->preempt) {
        uint64_t enter_ns = ns > 0 ? kern_now_ns() : 0;
        uint64_t lim = (ns > 0 && (uint64_t)ns < g_addr_spin_ns) ? (uint64_t)ns : g_addr_spin_ns;
        uint64_t end = (enter_ns ? enter_ns : kern_now_ns()) + lim;
        t->w_nosig = 1;
        plat_mutex_unlock(q.m);
        while (!__atomic_load_n(&t->woke, __ATOMIC_ACQUIRE)) {
            if (kern_now_ns() >= end || __atomic_load_n(&t->preempt, __ATOMIC_RELAXED) ||
                ((__atomic_load_n(&g_core_wait_mask, __ATOMIC_ACQUIRE) >> t->cur_core) & 1u))
                break;
            plat_pause();
        }
        plat_mutex_lock(q.m);
        t->w_nosig = 0;
        if (__atomic_load_n(&t->woke, __ATOMIC_ACQUIRE)) {
            uint32_t res = t->wres;
            uint32_t wk = t->waker;
            uint64_t wake_ts = t->wake_at;
            t->wake_at = 0;
            plat_mutex_unlock(q.m);
            if ((rt_trace & RT_TRACE_SCHED) && wake_ts) {
                uint64_t now = kern_now_ns();
                uint64_t tot = now > wake_ts ? now - wake_ts : 0;
                t->lw_ns += tot;
                t->lw_n++;
                if (tot > t->lw_max) t->lw_max = tot;
                ho_note_addr(t, addr, wk, tot, 0, 0, 0, 0, tot, HO_SPIN, 0);
            }
            return res;
        }
        if (ns > 0) {
            uint64_t spent = kern_now_ns() - enter_ns;
            if (spent >= (uint64_t)ns) {
                wait_unlink(q.list, t);
                t->wkind = W_NONE;
                plat_mutex_unlock(q.m);
                if (rt_trace & RT_TRACE_SCHED) ho_note_addr(t, addr, 0, 0, 0, 0, 0, 0, 0, HO_SPIN, 1);
                return R_TIMEOUT;
            }
            ns -= (int64_t)spent;
        }
    }
    plat_mutex_unlock(q.m);
    lock();
    core_put(t);
    acct(t, S_ADDR);
    t->nblk++;
    unlock();
    uint64_t t_block = g_svc_stats ? kern_now_ns() : 0;
    t->last_wake_at = 0;
    plat_mutex_lock(q.m);
    uint32_t r = wait_woken(t, ns, q.m, q.list);
    uint64_t t_woke = ((rt_trace & RT_TRACE_SCHED) || t_block) ? kern_now_ns() : 0;
    uint64_t wake_ts = t->ho_wake_at;
    uint32_t wk = t->waker;
    plat_mutex_unlock(q.m);
    if ((rt_trace & RT_TRACE_SCHED) || __atomic_load_n(&t->cur_core, __ATOMIC_ACQUIRE) < 0) {
        lock();
        uint64_t t_glock = (rt_trace & RT_TRACE_SCHED) ? kern_now_ns() : 0;
        core_take(t);
        uint64_t t_core = (rt_trace & RT_TRACE_SCHED) ? kern_now_ns() : 0;
        uint64_t grant_ts = t->ho_grant_at;
        int ho_kind = t->ho_kind;
        int ho_gated = t->ho_gated;
        unlock();
        if (rt_trace & RT_TRACE_SCHED) {
            uint64_t t_done = kern_now_ns();
            if (wake_ts) {
                uint64_t host_ns = t_woke > wake_ts ? t_woke - wake_ts : 0;
                uint64_t glock_ns = t_glock > t_woke ? t_glock - t_woke : 0;
                uint64_t cwait_ns = 0, cgrant_ns = 0;
                if (ho_gated && grant_ts && grant_ts >= t_glock && grant_ts <= t_core) {
                    cwait_ns = grant_ts - t_glock;
                    cgrant_ns = t_core - grant_ts;
                } else {
                    cgrant_ns = t_core > t_glock ? t_core - t_glock : 0;
                }
                uint64_t post_ns = t_done > t_core ? t_done - t_core : 0;
                uint64_t tot_ns = host_ns + glock_ns + cwait_ns + cgrant_ns + post_ns;
                ho_note_addr(t, addr, wk, host_ns, glock_ns, cwait_ns, cgrant_ns, post_ns, tot_ns, ho_kind, 0);
            } else if (r == R_TIMEOUT) {
                ho_note_addr(t, addr, 0, 0, 0, 0, 0, 0, 0, ho_kind, 1);
            }
        }
    }
    if (t_block && r != R_TIMEOUT && t->last_wake_at) {   // the core is back: the thread runs game code again
        uint64_t now = kern_now_ns(), w = t->last_wake_at;
        kern_wait_note(w > t_block ? w - t_block : 0, now > w ? now - w : 0, t_woke > w ? t_woke - w : 0, t->handle);
    }
    return r;
}

// ---- locks: tag word = owner handle | WAITERS_BIT

// Release the lock at addr: hand it to the best waiter, or clear it.
void lock_handoff(uint64_t addr) {
    RtThread *w = pick(W_LOCK, addr);
    int more = 0;
    if (!w) {
        st32(addr, 0);
        return;
    }
    for (RtThread *t = g_wait; t; t = t->next)
        if (t != w && t->wkind == W_LOCK && t->waddr == addr) {
            more = 1;
            break;
        }
    st32(addr, w->wtag | (more ? WAITERS_BIT : 0));
    wake(w, 0);
}

// A condvar waiter takes its lock: directly if free (and wakes), else it becomes a lock waiter behind the others.
void lock_acquire_for(RtThread *t, uint64_t addr, uint32_t tag) {
    uint32_t v = ld32(addr);
    for (;;) {
        if (v == 0) {
            if (cas32(addr, &v, tag)) {
                wake(t, 0);
                return;
            }
            continue;
        }
        if (cas32(addr, &v, v | WAITERS_BIT)) break;
    }
    acct(t, S_LOCK);
    t->wobj = v & ~WAITERS_BIT;
    t->wkind = W_LOCK;
    t->waddr = addr;
    t->wtag = tag;
    t->wseq = ++g_wseq;
}

// ---- threads
RtThread *thr_new(int prio, int core, uint64_t mask) {
    RtThread *t = calloc(1, sizeof *t);
    Cpu *c = plat_aligned_alloc(sizeof(Cpu), 64);
    if (!t || !c) rt_fatal("kernel: thread allocation");
    memset(c, 0, sizeof *c);
    plat_cond_init(&t->cv);
    t->prio = prio;
    t->core = t->ideal = core;
    t->mask = mask;
    t->cur_core = -1;
    // TLR (0x200) and the "current thread object" the SDK reads through TLR+0x1F8, in one guest-visible block.
    t->tlr = mem_kernel_alloc(TLR_SIZE + THREAD_OBJ_SIZE, TLR_SIZE);
    *(uint64_t *)(t->tlr + 0x1F8) = (uint64_t)(uintptr_t)(t->tlr + TLR_SIZE);
    t->tls = mem_kernel_alloc(TLS_SIZE, 16);
    c->tpidrro = (uint64_t)(uintptr_t)t->tlr;
    c->tpidr = (uint64_t)(uintptr_t)t->tls;
    c->excl_addr = ~0ull;
    c->thread = t;
    t->cpu = t->cur = c;
    return t;
}

void thr_link(RtThread *t) {   // holding G
    t->handle = h_new(O_THREAD, t);
    t->id = g_next_id++;
    t->all = g_all;
    g_all = t;
    g_created++;
}

void thr_free(RtThread *t) {   // holding G; t has exited and its handle is closed
    for (RtThread **p = &g_all; *p; p = &(*p)->all)
        if (*p == t) {
            *p = t->all;
            break;
        }
    mem_kernel_free(t->tlr, TLR_SIZE + THREAD_OBJ_SIZE);
    mem_kernel_free(t->tls, TLS_SIZE);
    plat_aligned_free(t->cpu);
    free(t);
}

// The end of every guest thread, on its own host context after the unwind: give the core up, signal the thread
// object, free it if its handle is already closed. t is not touched after the unlock.
static void thread_finish(RtThread *t) {
    if (t->base_fiber) plat_fiber_unself();
    t->base_fiber = NULL;
    tcur = NULL;
    lock();
    core_put(t);
    acct(t, S_OFF);
    if (t->hthr) plat_intr_free(t->hthr);   // no waiter interrupts a thread without a core
    t->hthr = NULL;
    t->exited = 1;
    wake_sync_waiters();
    if (t->closed) thr_free(t);
    unlock();
}

// ExitThread: back to the thread start with a non-unwinding longjmp; no SEH unwinding through lifted frames.
// (__builtin_setjmp is not usable: on Win64 LLVM saves the frame address, which is not rbp when the frame register
// is offset, and the landing code then reads its locals through the wrong base.)
// The first few exits are logged: a Windows fast fail in longjmp leaves nothing else in the log (see win32_mem.c).
__attribute__((noreturn)) void exit_now(RtThread *t) {
    static int logged;
    if (__atomic_fetch_add(&logged, 1, __ATOMIC_RELAXED) < 3) rt_log("kernel: a guest thread exits (longjmp to its start)");
    longjmp(t->exit_jmp, 1);
}

// The calling host thread, for ap_try (MKW_ASYNC_PREEMPT) and dg_sample (MKW_CORE_DIAG).
static PlatIntr *ap_open_self(void) {
    if (!cores_enabled() || (!g_ap_int && g_diag <= 0)) return NULL;
    PlatIntr *h = plat_intr_self();
    if (!h) rt_log("kernel: no interruption handle for this thread: it is not preempted asynchronously");
    return h;
}

void thread_main(void *p) {
    RtThread *t = p;
    tcur = t;
    t->tid = plat_thread_id();
    PlatIntr *self = ap_open_self();
    lock();
    t->hthr = self;
    while (!t->started) plat_cond_wait(&t->cv, &G);
    core_take(t);
    unlock();
    Cpu *c = t->cpu;
    c->sp = t->stack_top & ~15ull;
    if (!setjmp(t->exit_jmp)) {
        rt_call(c, t->entry, 1, t->arg);
        char d[128];
        rt_fatal("guest thread %#x: entry %s returned (Horizon threads end with ExitThread)", t->handle,
                 ldr_describe(t->entry, d, sizeof d));
    }
    thread_finish(tcur);
    return;
}

uint64_t kern_threads_created(void) {
    lock();
    uint64_t n = g_created;
    unlock();
    return n;
}

Cpu *kern_cpu(void) { return tcur ? tcur->cur : NULL; }

uint32_t kern_main_thread(Cpu **cpu) {
    if (tcur) {
        if (!tcur->is_main) rt_fatal("kern_main_thread on guest thread %#x, not the main thread", tcur->handle);
        *cpu = tcur->cpu;
        return tcur->handle;
    }
    if (g_main) rt_fatal("kern_main_thread: the main thread already runs on another host thread");
    const RtGuestLayout *L = mem_layout();
    uint64_t stack = L->stack.base + L->stack.size - MAIN_STACK;
    if (!plat_mem_commit(stack, MAIN_STACK))
        rt_fatal("kernel: commit the main thread stack at %#llx failed", (unsigned long long)stack);
    RtThread *t = thr_new(MAIN_PRIO, MAIN_CORE, APP_MASK);
    t->started = 1;
    t->is_main = 1;
    t->stack_top = stack + MAIN_STACK;
    t->cpu->sp = t->stack_top;
    t->tid = plat_thread_id();
    perf_main_thread();
    PlatIntr *self = ap_open_self();
    lock();
    t->hthr = self;
    thr_link(t);
    g_main = t;
    g_main_stack = stack;
    core_take(t);
    unlock();
    tcur = t;
    *cpu = t->cpu;
    return t->handle;
}

void kern_main_stack(uint64_t *base, uint64_t *size) {
    uint64_t b = __atomic_load_n(&g_main_stack, __ATOMIC_ACQUIRE);
    *base = b;
    *size = b ? MAIN_STACK : 0;
}

static void guest_main(void *p) {
    void (*fn)(Cpu *) = (void (*)(Cpu *))p;
    Cpu *c;
    kern_main_thread(&c);
    if (!setjmp(tcur->exit_jmp)) fn(c);
    thread_finish(tcur);
    return;
}

void kern_run_on_guest_main(void (*fn)(Cpu *c)) {
    PlatThread *h = plat_thread_new_ex(guest_main, (void *)fn, "guest main", HOST_STACK);
    if (!h) rt_fatal("kernel: cannot start the guest main thread");
    plat_thread_join(h);
}

// The OS timer ends a wait late by up to PLAT_WAIT_LATE_NS (Windows' high-resolution waitable timer: 0.3-0.5 ms; the
// SDK's 20 us polling sleeps, main thread and job workers, about 1.5 per frame each, took 460-515 us). Sleeps up to
// twice that spin on the clock instead (1 ms on Windows, 120 us on Linux); the virtual core is already given up.
#define SLEEP_SPIN_NS (2 * PLAT_WAIT_LATE_NS)
void sleep_ns(RtThread *t, int64_t ns) {
    (void)t;
    if (ns <= SLEEP_SPIN_NS) {
        uint64_t end = kern_now_ns() + (uint64_t)ns;
        while (kern_now_ns() < end) plat_pause();
        return;
    }
    plat_wait_ns((uint64_t)ns);
}

// ---- handles and events for host services (svc.c's transfer memory and the service overrides)
uint32_t kern_handle_new(int kind, void *obj) {
    if (kind != KERN_TMEM) rt_fatal("kern_handle_new: kind %d", kind);
    lock();
    uint32_t h = h_new(kind, obj);
    unlock();
    return h;
}

void *kern_handle_get(uint32_t h, int kind) {
    lock();
    void *o = h_peek(h, kind);
    unlock();
    return o;
}

uint32_t kern_event_new(void) {
    Event *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("kernel: event allocation");
    e->refs = 1;
    lock();
    uint32_t hw = h_new(O_EVENT_W, e);
    unlock();
    return hw;
}

// Copy-handle transfer of the readable end: each call gives the client its own handle to the same event.
uint32_t kern_event_readable(uint32_t hw) {
    lock();
    Event *e = h_peek(hw, O_EVENT_W);
    if (!e) rt_fatal("kern_event_readable: %#x is not a writable event handle", hw);
    e->refs++;
    uint32_t hr = h_new(O_EVENT_R, e);
    unlock();
    return hr;
}

void kern_event_signal(uint32_t hw) {
    lock();
    Event *e = h_peek(hw, O_EVENT_W);
    if (!e) rt_fatal("kern_event_signal: %#x is not a writable event handle", hw);
    e->signaled = 1;
    wake_sync_waiters();
    unlock();
}

// System events of host services: the SDK's own nn::os::CreateSystemEvent creates the kernel event (SVC 0x45) on a
// guest thread; the writable handle is kept, so any thread (host audio tick included) can signal it later.
static struct { uint64_t ev; uint32_t wh; } g_se[512];
static int g_nse;
static PlatMutex g_se_lock = PLAT_MUTEX_INIT;

static uint32_t se_find(uint64_t ev) {
    plat_mutex_lock_shared(&g_se_lock);
    uint32_t wh = 0;
    for (int i = 0; i < g_nse && !wh; i++)
        if (g_se[i].ev == ev) wh = g_se[i].wh;
    plat_mutex_unlock_shared(&g_se_lock);
    return wh;
}

// A SystemEventType's state byte (+0x28: 1 an event, 2 an inter-process event, 0 not initialized), as
// nn::os::TryWaitSystemEvent reads it (sdk+0x141e00).
enum { SE_STATE = 0x28 };

// An address already bound is bound again when the game destroyed the event since and handed the same storage over
// again (its state byte is 0 again): MKW's local wireless finalizes ldn, destroys its state change event and attaches
// it again after ldn::Initialize; skipping the bind left it destroyed and the next TryWaitSystemEvent aborted (PS5 and
// Windows, 17/11/2026). The old handles were closed by the game's DestroySystemEvent.
void kern_bind_system_event(Cpu *c, uint64_t ev, int autoclear) {
    int again = 0;
    if (se_find(ev)) {
        if (LD8(ev + SE_STATE) != 0) return;
        again = 1;
    }
    if (!tcur || !c) rt_fatal("kern_bind_system_event(%#llx) on a host thread", (unsigned long long)ev);
    static uint64_t create, getw;
    if (!create) {
        create = ldr_symbol("_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb");
        getw = ldr_symbol("_ZN2nn2os30GetWritableHandleOfSystemEventEPKNS0_15SystemEventTypeE");
        if (!create || !getw) rt_fatal("kern_bind_system_event: the SDK's system event functions are not loaded");
    }
    uint32_t r = (uint32_t)rt_call(c, create, 3, ev, (uint64_t)(autoclear ? 1 : 0), (uint64_t)1);
    if (r) rt_fatal("kern_bind_system_event: CreateSystemEvent(%#llx) failed %#x", (unsigned long long)ev, r);
    uint32_t wh = (uint32_t)rt_call(c, getw, 1, ev);
    plat_mutex_lock(&g_se_lock);
    int i = 0;
    while (again && i < g_nse && g_se[i].ev != ev) i++;
    if (!again || i == g_nse) {
        if (g_nse == (int)(sizeof g_se / sizeof *g_se)) rt_fatal("kern_bind_system_event: table full");
        i = g_nse++;
    }
    g_se[i].ev = ev;
    g_se[i].wh = wh;
    plat_mutex_unlock(&g_se_lock);
}

// An event not bound yet is bound on the spot with manual clear, which needs a guest thread.
void kern_signal_system_event(uint64_t ev) {
    uint32_t wh = se_find(ev);
    if (!wh) {
        kern_bind_system_event(kern_cpu(), ev, 0);
        wh = se_find(ev);
    }
    kern_event_signal(wh);
}
