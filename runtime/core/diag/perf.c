// Frame timing (MKW_TRACE=perf, the frame-time summary, the performance overlay's counts), the A/B switches (rt_ab,
// rt_knob) and MKW_ONE_PER_CORE. The sampling profiler (MKW_PROFILE) is the OS's (prof_win.c); without one the weak
// prof_sampler_* below do nothing.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "hangwatch.h"
#include "host.h"
#include "perfmon.h"
#include "prof.h"
#include "platform/platform.h"

volatile uint32_t prof_render_tid, prof_main_tid, prof_replay_tid;

__attribute__((weak)) int prof_sampler_start(void) {
    if (cfg_get("MKW_PROFILE")) rt_log("prof: MKW_PROFILE: no sampling profiler on this OS");
    return 0;
}
__attribute__((weak)) void prof_sampler_write(void) {}
__attribute__((weak)) void prof_names_init(void) {}
__attribute__((weak)) const char *prof_pc_describe(uint64_t pc, char *buf, int n) {
    snprintf(buf, (size_t)n, "exe+%#llx", (unsigned long long)(pc - plat_exe_base()));
    return buf;
}
__attribute__((weak)) void prof_sampler_frame(uint64_t frame, uint64_t now, uint64_t d) { (void)frame, (void)now, (void)d; }

// MKW_TRACE=perf, MKW_FLIGHT=1 or the full performance overlay: event counts are taken. Wait times are always taken
// (frame-time summary). perf_base: the part fixed at start.
static int perf_on, perf_base;
static int overlay;   // performance overlay mode (perf_set_overlay): 0 off, 1 fps, 2 full

void perf_set_overlay(int mode) {
    __atomic_store_n(&overlay, mode, __ATOMIC_RELAXED);
    __atomic_store_n(&perf_on, perf_base || mode == 2, __ATOMIC_RELAXED);
}

int perf_gpu_timing(void) { return (rt_trace & RT_TRACE_PERF) || __atomic_load_n(&overlay, __ATOMIC_RELAXED) == 2; }

static volatile int64_t gpu_busy_total;

void perf_gpu_busy(uint64_t ns) { plat_add64(&gpu_busy_total, (int64_t)ns); }

uint64_t perf_gpu_busy_total(void) { return (uint64_t)__atomic_load_n(&gpu_busy_total, __ATOMIC_RELAXED); }

void prof_start(void) {
    flight_init();
    ctl_init();
    atexit(perf_summary_flush);   // boot.c's _exit path goes through prof_write instead
    perf_base = (rt_trace & RT_TRACE_PERF) || flight_on;
    perf_on = perf_base || overlay == 2;
    if (prof_sampler_start()) atexit(prof_write);   // a --seconds run ends in _exit: boot.c calls prof_write itself first
}

void prof_write(void) {   // everything a run writes at exit for its diagnostics
    perf_summary_flush();
    impcensus_write();
    prof_sampler_write();
}

// ---- MKW_TRACE=perf: frame interval and the time the render thread spends waiting, logged every 60 presents. The
// flight recorder (MKW_FLIGHT=1, flight.c) takes the same numbers per frame.
static uint64_t wait_ns[PERF_N], frame_ns[PERF_N], worst_ns[PERF_N];   // window totals, this frame, the slowest frame

uint64_t perf_begin(void) { return kern_now_ns(); }

void perf_replay_thread(void) { prof_replay_tid = plat_thread_id(); }
void perf_main_thread(void) { prof_main_tid = plat_thread_id(); }

void perf_end(int kind, uint64_t t0) {
    if (!t0) return;
    uint32_t tid = plat_thread_id();
    if (tid != prof_render_tid && tid != prof_replay_tid) return;
    uint64_t d = kern_now_ns() - t0;
    if (tid == prof_render_tid && !prof_replay_tid) {   // one thread adds: no atomics on the synchronous path
        wait_ns[kind] += d;
        frame_ns[kind] += d;
        return;
    }
    __atomic_fetch_add(&wait_ns[kind], d, __ATOMIC_RELAXED);
    __atomic_fetch_add(&frame_ns[kind], d, __ATOMIC_RELAXED);
}

static volatile int64_t other_wait_ns;     // perf_gpu_wait on threads other than the render thread, over the window
static volatile int64_t render_wait_ns;    // the same on the render thread
static volatile int64_t other_replay_ns, render_replay_ns;   // part of each spent waiting for replay to submit
void (*perf_gpu_stats)(PerfGpu *out);

void perf_gpu_wait_split(uint64_t t0, uint64_t rep_ns) {
    if (!t0 || !__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) return;
    int64_t d = (int64_t)(kern_now_ns() - t0);
    int64_t r = (int64_t)rep_ns > d ? d : (int64_t)rep_ns;
    int is_render = plat_thread_id() == prof_render_tid;
    plat_add64(is_render ? &render_wait_ns : &other_wait_ns, d);
    if (r) plat_add64(is_render ? &render_replay_ns : &other_replay_ns, r);
}

void perf_gpu_wait(uint64_t t0) { perf_gpu_wait_split(t0, 0); }

static volatile int64_t replay_ns;         // perf_replay: summed over the window, any thread
static volatile int64_t frame_replay_ns;   // the same since the previous present (frame-time summary)

void perf_replay(uint64_t t0) {
    if (!t0) return;
    int64_t d = (int64_t)(kern_now_ns() - t0);
    plat_add64(&frame_replay_ns, d);
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) plat_add64(&replay_ns, d);
}

// The per-frame cost ledger (host.h), read and reset at each present. A cost is charged to the present that follows
// it: with the game a frame ahead of the replay, a slow frame's ledger holds the game's next frame and the replay of
// its own, which is what ran in its period.
static volatile int64_t ledger[PERF_LG_N];
static const char *const ledger_name[PERF_LG_N] = {"sync_replay", "sync_gpu", "texlock", "upload", "astc",
                                                   "precheck",    "early",    "gov",     "capture", "sparse"};

void perf_ledger(int kind, uint64_t ns) {
    if (ns && __atomic_load_n(&perf_on, __ATOMIC_RELAXED)) plat_add64(&ledger[kind], (int64_t)ns);
}

// The ledger since the last present, with the replay's time (rep_ns) and the platform's page protection and PACK
// unpacks (running totals, differenced here), into out as " name ms" pairs (under 0.05 ms left out); resets it and adds
// it to the window's sums, which ledger_window prints every 60 presents as the normal frame's costs.
enum { LG_REPLAY = PERF_LG_N, LG_PLAT, LG_ALL = LG_PLAT + PLAT_MEM_COST_N };
static uint64_t lg_win[LG_ALL];

static void ledger_fmt(char *out, size_t cap, const uint64_t *ns, double div) {
    static const char *const extra[LG_ALL - PERF_LG_N] = {"replay",      "prot_fault", "prot_arm", "prot_hold",
                                                          "prot_change", "prot_early", "unpack",   "prot_calls"};
    size_t o = 0;
    out[0] = 0;
    for (int i = 0; i < LG_ALL && o < cap; i++) {
        int count = i == LG_PLAT + PLAT_MEM_COST_PROT_CALLS;   // a count, not ns
        double ms = (double)ns[i] / (count ? 1.0 : 1e6) / div;
        if (ms >= 0.05) o += (size_t)snprintf(out + o, cap - o, " %s %.2f", i < PERF_LG_N ? ledger_name[i] : extra[i - PERF_LG_N], ms);
    }
}

static void ledger_take(char *out, size_t cap, uint64_t rep_ns) {
    static uint64_t plat_prev[PLAT_MEM_COST_N];
    uint64_t pc[PLAT_MEM_COST_N] = {0}, ns[LG_ALL];
    plat_mem_cost_ns(pc);
    for (int i = 0; i < PERF_LG_N; i++) ns[i] = (uint64_t)plat_xchg64(&ledger[i], 0);
    ns[LG_REPLAY] = rep_ns;
    for (int i = 0; i < PLAT_MEM_COST_N; i++) ns[LG_PLAT + i] = pc[i] - plat_prev[i];
    memcpy(plat_prev, pc, sizeof pc);
    for (int i = 0; i < LG_ALL; i++) lg_win[i] += ns[i];
    ledger_fmt(out, cap, ns, 1);
}

static void ledger_window(uint64_t frame, uint64_t n) {
    char b[384];
    ledger_fmt(b, sizeof b, lg_win, (double)n);
    rt_log("perf: frame %llu: costs ms/f:%s", (unsigned long long)frame, b);
    memset(lg_win, 0, sizeof lg_win);
}

static volatile int64_t counts[PERF_CNT_N];   // running totals: the perf log and the flight recorder take differences

void perf_count(int kind) {
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) plat_inc64(&counts[kind]);
}

// ---- Frame-time summary, always on: every 60 s of play and at exit one line of frame count, mean fps, 1% low and the
// frames over 25/50/100 ms, then the worst frames of the period with the render thread's waits in them. Per frame: a
// histogram increment, the compare against the worst list, a memset of the wait array. No allocation.
#define FT_BINS 1024              // 0.25 ms bins up to 256 ms; slower frames land in the last bin
#define FT_WORST 5
#define FT_PERIOD_NS 60000000000ull
typedef struct { uint64_t frame, ns, wait[PERF_N], replay; } FtWorst;
static struct {
    uint64_t t0, frames, sum, over25, over50, over100;
    uint32_t hist[FT_BINS];
    FtWorst worst[FT_WORST];   // slowest first
    int nworst;
} ft;
static uint64_t summary_t0;   // start of the current period (0: not started)

static void ft_report(uint64_t now, const char *tag) {
    if (ft.frames && ft.sum) {
        uint64_t cut = (ft.frames + 99) / 100, acc = 0, low = 0;   // the slowest 1% of frames
        for (int b = FT_BINS - 1; b >= 0; b--) {
            acc += ft.hist[b];
            if (acc >= cut) { low = (uint64_t)(b + 1) * 250000ull; break; }   // upper edge of the bin
        }
        rt_log("frametime: %s: %llu frames in %.1f s, mean %.1f fps, 1%% low %.1f fps, over 25 ms %llu, 50 ms %llu, 100 ms %llu",
               tag, (unsigned long long)ft.frames, (now - ft.t0) / 1e9, ft.frames * 1e9 / ft.sum, low ? 1e9 / low : 0.0,
               (unsigned long long)ft.over25, (unsigned long long)ft.over50, (unsigned long long)ft.over100);
        for (int i = 0; i < ft.nworst; i++) {
            const FtWorst *w = &ft.worst[i];
            rt_log("frametime:   worst frame %llu: %.1f ms; pipeline build %.1f, texture upload %.1f (astc %.1f), replay %.1f, fence/present wait %.1f, pacing %.1f",
                   (unsigned long long)w->frame, w->ns / 1e6, w->wait[PERF_PIPE] / 1e6,
                   w->wait[PERF_TEX] / 1e6, w->wait[PERF_ASTC] / 1e6, w->replay / 1e6,
                   (w->wait[PERF_FENCE] + w->wait[PERF_DRAIN]) / 1e6, w->wait[PERF_PACE] / 1e6);
        }
    }
    memset(&ft, 0, sizeof ft);
    ft.t0 = now;
}

__attribute__((weak)) void gfx_exit_log(void) {}

void perf_summary_flush(void) {   // at exit (prof_write)
    if (summary_t0) {
        ft_report(kern_now_ns(), "exit");
        gfx_exit_log();
    }
    summary_t0 = 0;
}

// Returns the replay time since the previous present.
static uint64_t ft_frame(uint64_t frame, uint64_t now, uint64_t d) {
    int64_t rep = plat_xchg64(&frame_replay_ns, 0);
    if (!summary_t0) {   // the first present starts the period; there is no interval yet
        summary_t0 = now;
        ft.t0 = now;
        return (uint64_t)rep;
    }
    ft.frames++;
    ft.sum += d;
    uint64_t b = d / 250000ull;
    ft.hist[b < FT_BINS ? b : FT_BINS - 1]++;
    if (d > 25000000ull) ft.over25++;
    if (d > 50000000ull) ft.over50++;
    if (d > 100000000ull) ft.over100++;
    if (d > 25000000ull && (ft.nworst < FT_WORST || d > ft.worst[FT_WORST - 1].ns)) {
        int i = ft.nworst < FT_WORST ? ft.nworst++ : FT_WORST - 1;
        while (i > 0 && ft.worst[i - 1].ns < d) ft.worst[i] = ft.worst[i - 1], i--;
        ft.worst[i].frame = frame;
        ft.worst[i].ns = d;
        memcpy(ft.worst[i].wait, frame_ns, sizeof frame_ns);
        ft.worst[i].replay = (uint64_t)rep;
    }
    if (now - summary_t0 >= FT_PERIOD_NS) {
        ft_report(now, "60 s");
        summary_t0 = now;
    }
    return (uint64_t)rep;
}

enum { FENCE_SITE_SLOTS = 64 };
typedef struct {
    const char *site;
    uint64_t pc;
    int frame_rel;
    int is_render;
    uint64_t ns;
    uint64_t count;
} FenceSiteSlot;
static FenceSiteSlot fence_sites[FENCE_SITE_SLOTS];

void perf_fence_site(const char *site, uint64_t pc, int frame_rel, uint64_t ns) {
    if (!site || !(rt_trace & RT_TRACE_PERF)) return;
    int is_render = prof_render_tid && plat_thread_id() == prof_render_tid ? 1 : 0;
    for (int i = 0; i < FENCE_SITE_SLOTS; i++) {
        const char *cur = __atomic_load_n(&fence_sites[i].site, __ATOMIC_ACQUIRE);
        if (!cur) {
            FenceSiteSlot *s = &fence_sites[i];
            if (__atomic_compare_exchange_n(&s->site, &cur, (const char *)1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                s->pc = pc;
                s->frame_rel = frame_rel;
                s->is_render = is_render;
                __atomic_store_n(&s->site, site, __ATOMIC_RELEASE);
                __atomic_fetch_add(&s->ns, ns, __ATOMIC_RELAXED);
                __atomic_fetch_add(&s->count, 1, __ATOMIC_RELAXED);
                return;
            }
            cur = __atomic_load_n(&fence_sites[i].site, __ATOMIC_ACQUIRE);
        }
        if (cur == (const char *)1) continue;
        if (cur == site && fence_sites[i].pc == pc && fence_sites[i].frame_rel == frame_rel &&
            fence_sites[i].is_render == is_render) {
            __atomic_fetch_add(&fence_sites[i].ns, ns, __ATOMIC_RELAXED);
            __atomic_fetch_add(&fence_sites[i].count, 1, __ATOMIC_RELAXED);
            return;
        }
    }
}

static void perf_fence_sites_report(uint64_t frame, uint64_t n) {
    typedef struct { const char *site; uint64_t pc; int frame_rel; int is_render; uint64_t ns, count; } Snap;
    Snap list[FENCE_SITE_SLOTS];
    int m = 0;
    for (int i = 0; i < FENCE_SITE_SLOTS; i++) {
        const char *site = __atomic_load_n(&fence_sites[i].site, __ATOMIC_ACQUIRE);
        if (!site || site == (const char *)1) continue;
        uint64_t ns = __atomic_exchange_n(&fence_sites[i].ns, 0, __ATOMIC_RELAXED);
        uint64_t cnt = __atomic_exchange_n(&fence_sites[i].count, 0, __ATOMIC_RELAXED);
        if (!cnt) continue;
        list[m++] = (Snap){site, fence_sites[i].pc, fence_sites[i].frame_rel, fence_sites[i].is_render, ns, cnt};
    }
    if (!m || !n) return;
    for (int i = 0; i < m - 1; i++) {
        int best = i;
        for (int j = i + 1; j < m; j++)
            if (list[j].ns > list[best].ns) best = j;
        if (best != i) { Snap t = list[i]; list[i] = list[best]; list[best] = t; }
    }
    char line[1024];
    int pos = 0, show = m < 8 ? m : 8;
    for (int i = 0; i < show; i++) {
        char fbuf[8], pcbuf[96];
        if (list[i].frame_rel < 0) snprintf(fbuf, sizeof fbuf, "f?");
        else snprintf(fbuf, sizeof fbuf, "f%d", list[i].frame_rel);
        if (list[i].pc) {
            char d[80];
            snprintf(pcbuf, sizeof pcbuf, "@%s", ldr_describe(list[i].pc, d, sizeof d));
        } else {
            pcbuf[0] = '\0';
        }
        int r = snprintf(line + pos, sizeof line - (size_t)pos, "%s%s[%s,%s]%s: %.2f ms/f (%.1f/f)",
                         i ? "; " : "", list[i].site, list[i].is_render ? "render" : "other", fbuf, pcbuf,
                         list[i].ns / 1e6 / n, (double)list[i].count / n);
        if (r < 0 || pos + r >= (int)sizeof line) break;
        pos += r;
    }
    rt_log("perf: frame %llu: fence sites (%d): %s", (unsigned long long)frame, m, line);
}

// The second line of the 60-frame window: GPU busy time (timestamps), the time threads waited for GPU fences, VRAM and
// process memory, and which side holds the frame back. pace: the frame waited for pacing (vsync, the frame limiter) at
// least 15% of the window; gpu: the GPU was busy for 80% of the frame interval or more (without timestamps: threads
// waited for fences for half the interval or more); cpu otherwise.
static void perf_gpu_line(uint64_t frame, double interval_ms, double pace_share, uint64_t n) {
    PerfGpu g;
    memset(&g, 0, sizeof g);
    if (perf_gpu_stats) perf_gpu_stats(&g);
    double rw = plat_xchg64(&render_wait_ns, 0) / 1e6 / n, ow = plat_xchg64(&other_wait_ns, 0) / 1e6 / n;
    double rr = plat_xchg64(&render_replay_ns, 0) / 1e6 / n, or_ = plat_xchg64(&other_replay_ns, 0) / 1e6 / n;
    double gpu_ms = g.have_gpu ? g.gpu_ns / 1e6 / n : 0;
    char gs[96], vs[96];
    if (g.have_gpu) snprintf(gs, sizeof gs, "gpu avg %.2f ms (%llu parts timed, %llu not)", gpu_ms,
                             (unsigned long long)g.gpu_parts, (unsigned long long)g.gpu_missed);
    else snprintf(gs, sizeof gs, "gpu avg n/a");
    if (g.have_vram) snprintf(vs, sizeof vs, "vram %.0f of %.0f MiB%s", g.vram_used / 1048576.0, g.vram_budget / 1048576.0,
                              g.vram_exact ? "" : " (own allocations of the heap size)");
    else snprintf(vs, sizeof vs, "vram n/a");
    uint64_t res, prv;
    plat_process_memory(&res, &prv);
    double ws = res / 1048576.0, priv = prv / 1048576.0;
    const char *bound = pace_share >= 0.15 ? "pace"
                      : g.have_gpu ? (gpu_ms >= 0.8 * interval_ms ? "gpu" : "cpu")
                      : (rw + ow >= 0.5 * interval_ms ? "gpu" : "cpu");
    rt_log("perf: frame %llu: %s; fence wait per frame: render thread %.2f ms (replay %.2f, gpu %.2f), other threads %.2f ms (replay %.2f, gpu %.2f); %s; ram working set "
           "%.0f MiB, private %.0f MiB; bound: %s",
           (unsigned long long)frame, gs, rw, rr, rw - rr, ow, or_, ow - or_, vs, ws, priv, bound);
    perf_fence_sites_report(frame, n);
}

// MKW_AB_INVERT=1 swaps the blocks (B A A B): a race's stretches are not equally heavy (sl-ab1, 06/11/2026: texture
// queries, which no switch changed, 8 % more in B), so a run and its inverted twin put each stretch once on each side.
int rt_ab(void) {
    static int64_t period = -1;   // us of world clock; 0: off
    static int invert;
    if (period < 0) {
        const char *e = cfg_get("MKW_AB"), *i = cfg_get("MKW_AB_INVERT");
        period = e && *e ? (int64_t)(atof(e) * 1e6) : 0;
        invert = i && *i && strcmp(i, "0");
        if (period && invert) rt_log("perf: A/B blocks in the order B A A B (MKW_AB_INVERT)");
    }
    int64_t w = period ? frame_step_world_us() : -1;
    if (w < 0) return 0;   // before the world clock: A
    int64_t k = w / period % 4;
    return (k == 1 || k == 2) != invert;
}

int rt_knob(const char *name, const char *def) {
    char key[64];
    snprintf(key, sizeof key, "MKW_%s", name);
    const char *e = cfg_get(key);
    if (!e || !*e) e = def;
    return !strcmp(e, "ab") ? 2 : strcmp(e, "0") != 0;
}

// MKW_ONE_PER_CORE=1: the process keeps the first logical processor of each physical core, so busy threads (the
// render thread, the guest cores, the job workers) never share a core's execution units with another; "ab" switches
// it with the A/B blocks. Off by default until measured here; Savage Wind Recomp
// reported +4-10% from the same rule on an 8-core SMT CPU.
static void cpu_placement(void) {
    static int mode = -1, applied = -1;
    static uint64_t all, one;
    if (mode < 0) {
        mode = rt_knob("ONE_PER_CORE", "0");
        int cores = plat_cpu_cores(&all, &one);
        if (mode && one && one != all)
            rt_log("cpu: %d cores, %d logical processors: one per core%s", cores, __builtin_popcountll(all),
                   mode == 2 ? " in the B blocks" : "");
        else
            mode = 0;
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    applied = on;
    plat_process_affinity(on ? one : all);
}

// MKW_THREAD_PRIO=1: the runtime's threads run at the priorities they ask for (present and vsync highest, replay above
// normal, shadow, headroom and pipeline builds below normal); 0, the default: all at the process's priority; "ab"
// switches it with the A/B blocks. Only the PS5 can turn them off (plat_thread_priority_mode); elsewhere they always
// apply and the switch does nothing. Off on the PS5 until a console A/B shows a gain: run 16, the first run with
// them, had the replay at 31-33 ms against 22 ms in run 15 (14/11/2026).
static void prio_placement(void) {
    static int mode = -1, applied = -1;
    if (mode < 0) {
        mode = rt_knob("THREAD_PRIO", "0");
        if (plat_thread_priority_mode(RT_KNOB_ON(mode)) < 0) mode = 0, applied = 0;
        else if (mode) rt_log("cpu: runtime thread priorities on%s (MKW_THREAD_PRIO)", mode == 2 ? " in the B blocks" : "");
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    if (applied >= 0) plat_thread_priority_mode(on);
    applied = on;
}

// MKW_PS5_TSC_CLOCK=1 (PS5 default, run 21: -1.0 ms/f) switches plat_now_ns to rdtsc + sceKernelGetTscFrequency.
static void tsc_clock_placement(void) {
    static int mode = -1, applied = -1;
    if (mode < 0) {
        mode = rt_knob("PS5_TSC_CLOCK", "1");
        if (plat_tsc_clock_mode(RT_KNOB_ON(mode)) < 0) mode = 0, applied = 0;
        else if (mode == 2) rt_log("time: PS5 TSC clock in the B blocks (MKW_PS5_TSC_CLOCK=ab)");
        else if (!mode) rt_log("time: PS5 TSC clock off (MKW_PS5_TSC_CLOCK=0)");
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    if (applied >= 0) plat_tsc_clock_mode(on);
    applied = on;
}

static void pw_batch_placement(void) {
    static int mode = -1, applied = -1;
    if (mode < 0) {
        mode = rt_knob("PS5_PW_BATCH", "1");
        if (plat_pw_batch_mode(RT_KNOB_ON(mode)) < 0) mode = 0, applied = 0;
        else if (mode == 2) rt_log("mem: PS5 prot_watch mprotect batching in the B blocks (MKW_PS5_PW_BATCH=ab)");
        else if (!mode) rt_log("mem: PS5 prot_watch mprotect batching off (MKW_PS5_PW_BATCH=0)");
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    if (applied >= 0) plat_pw_batch_mode(on);
    applied = on;
}

// Page-protection calls by caller (platform/ps5_guest_memory.c), every 600 frames with MKW_TRACE=perf: the 60 fps work
// found ~10 ms a frame of the game's threads in them (15/11/2026). exe+ offsets name with tools/run/sym.py --ps5.
typedef struct { uintptr_t ra; uint64_t n, ns; } PStatRow;
typedef struct { PStatRow r[64]; int n; } PStatTab;
static void pstat_emit(uintptr_t ra, uint64_t n, uint64_t ns, void *ctx) {
    PStatTab *t = ctx;
    if (t->n < 64) t->r[t->n++] = (PStatRow){ra, n, ns};
}
static void pstat_report(uint64_t frame, uint64_t frames) {
    PStatTab t = {0};
    plat_mem_protect_stats(pstat_emit, &t);
    if (!t.n || !frames) return;
    for (int i = 1; i < t.n; i++)   // by time, longest first
        for (int j = i; j > 0 && t.r[j].ns > t.r[j - 1].ns; j--) { PStatRow x = t.r[j]; t.r[j] = t.r[j - 1]; t.r[j - 1] = x; }
    uint64_t tot = 0, totn = 0;
    for (int i = 0; i < t.n; i++) tot += t.r[i].ns, totn += t.r[i].n;
    rt_log("perf: frame %llu: page protection %.1f calls/f, %.3f ms/f (by caller, top 12):", (unsigned long long)frame,
           (double)totn / frames, tot / 1e6 / frames);
    uintptr_t base = plat_exe_base();
    for (int i = 0; i < t.n && i < 12; i++) {
        char who[48];
        if (t.r[i].ra <= 3) snprintf(who, sizeof who, "%s", t.r[i].ra == 1 ? "write watch: fault unprotect" : t.r[i].ra == 3 ? "write watch: hold unprotect" : "write watch: arm");
        else snprintf(who, sizeof who, "exe+%#llx", (unsigned long long)(t.r[i].ra - base));
        rt_log("perf:   %-32s %8.1f calls/f %8.3f ms/f (%.1f us each)", who, (double)t.r[i].n / frames,
               t.r[i].ns / 1e6 / frames, t.r[i].ns / 1e3 / t.r[i].n);
    }
}

// MKW_PW_HOT (default 1 on the PS5, "ab" to compare): pages the game writes every arm cycle stay writable instead of
// being armed again to fault (prot_watch.c plat_pw_hot_mode). Linux's write watch is off unless MKW_PROT_WATCH.
static void pw_hot_placement(void) {
    static int mode = -1, applied = -1;
    if (mode < 0) {
#ifdef __PROSPERO__
        mode = rt_knob("PW_HOT", "1");
#else
        mode = rt_knob("PW_HOT", "0");
#endif
        if (plat_pw_hot_mode(RT_KNOB_ON(mode)) < 0) mode = 0, applied = 0;
        else rt_log("mem: write watch: hot pages stay writable %s (MKW_PW_HOT)", mode == 2 ? "in the B blocks" : mode ? "on" : "off");
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    if (applied >= 0) plat_pw_hot_mode(on);
    applied = on;
}

// MKW_PW_FAULT_BATCH (default 1 on the PS5, "ab" to compare): sequential write-watch faults inside the same 2 MiB unit
// unprotect a ramping contiguous run of armed pages (up to 512 KiB) in one mprotect call (prot_watch.c).
static void pw_fault_batch_placement(void) {
    static int mode = -1, applied = -1;
    if (mode < 0) {
#ifdef __PROSPERO__
        mode = rt_knob("PW_FAULT_BATCH", "1");
#else
        mode = rt_knob("PW_FAULT_BATCH", "0");
#endif
        if (plat_pw_fault_batch_mode(RT_KNOB_ON(mode)) < 0) mode = 0, applied = 0;
        else rt_log("mem: write watch: sequential fault batching %s (MKW_PW_FAULT_BATCH)", mode == 2 ? "in the B blocks" : mode ? "on" : "off");
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    if (applied >= 0) plat_pw_fault_batch_mode(on);
    applied = on;
}

void perf_present(uint64_t frame) {
    static uint64_t last, sum, max, n, win_start, ab_b;
    static int64_t win_prev[PERF_CNT_N];
    prof_render_tid = plat_thread_id();
    if (rt_trace & RT_TRACE_SCHED) kern_sched_report(frame);
    seqwatch_frame(frame);
    bench_frame(frame);   // MKW_BENCH: the race's frame times, and the exit at its finish
    hang_frame(frame);   // MKW_HANG_WATCH: the watchdog compares this frame
    if (!seqwatch_window()) hostfn_time_report(frame);
    uint64_t now = kern_now_ns(), d = last ? now - last : 0;
    prof_sampler_frame(frame, now, last ? d : 0);
    pgo_frame(frame);   // MKW_PGO_AT (pgo.c)
    if ((rt_trace & RT_TRACE_PERF) && frame && frame % 600 == 0) pstat_report(frame, 600), kern_svc_stats_report(frame, 600);
    {   // MKW_PROFILE_FN=1 (kernel_diag.c), from MKW_PROFILE_FROM's frame: a report every 600 frames
        static int fn_mode = -1, fn_started;
        static uint64_t fn_from, fn_last;
        if (fn_mode < 0) {
            const char *e = cfg_get("MKW_PROFILE_FN"), *f = cfg_get("MKW_PROFILE_FROM");
            fn_mode = e && *e && strcmp(e, "0");
            fn_from = f ? strtoull(f, NULL, 10) : 0;
        }
        if (fn_mode == 1 && !fn_started && frame >= fn_from) {
            const char *u = cfg_get("MKW_PROFILE_US");
            fn_started = kern_fn_profile_start(u ? (uint32_t)atoi(u) : 1000);
            fn_last = frame;
            if (!fn_started) fn_mode = 0;
        }
        if (fn_started && frame >= fn_last + 600) kern_fn_profile_report(frame, frame - fn_last), fn_last = frame;
    }
    uint64_t rep = ft_frame(frame, now, d);   // d is 0 on the first present: it only starts the period
    ab_b += rt_ab();
    static int cond_probed, lock_prof_init;
    if (!cond_probed) { cond_probed = 1; plat_cond_probe(); }
    if (!lock_prof_init) {
        lock_prof_init = 1;
        const char *lp = cfg_get("MKW_LOCK_PROF");
        plat_lock_prof_mode(lp && atoi(lp) != 0);
    }
    cpu_placement();
    prio_placement();
    tsc_clock_placement();
    pw_batch_placement();
    pw_hot_placement();
    pw_fault_batch_placement();
    if (__atomic_load_n(&overlay, __ATOMIC_RELAXED))   // the overlay's only cost while it is off: this branch
        perfmon_frame(now, d, frame_ns, rep, __atomic_load_n(&counts[PERF_CNT_TEX_UP], __ATOMIC_RELAXED));
    kern_core_diag_frame(frame, d);
    if (flight_on) flight_frame(frame, now, d, frame_ns, counts);
    if (!(rt_trace & RT_TRACE_PERF)) {
        memset(frame_ns, 0, sizeof frame_ns);
        last = now;
        return;
    }
    static uint64_t slow_ns;   // MKW_PERF_SLOW_MS (default 50): frames at least this long get a line of their own
    if (!slow_ns) {
        const char *e = cfg_get("MKW_PERF_SLOW_MS");
        slow_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 50) * 1000000ull;
    }
    char costs[384], storm[512];
    ledger_take(costs, sizeof costs, rep);   // every present, so a slow frame's ledger is its own period's
    if (plat_mem_storm_take(storm, sizeof storm, 64))   // 64+ write faults: ~3x a normal frame's (protsplit, 28/11)
        rt_log("perf: fault storm frame %llu: %s", (unsigned long long)frame, storm);
    if (last) {
        sum += d;
        n++;
        if (d > max) max = d, memcpy(worst_ns, frame_ns, sizeof worst_ns);
        if (d >= slow_ns)
            rt_log("perf: slow frame %llu: %.1f ms; blocked pipeline %.1f texture %.1f astc %.1f fence %.1f pacing %.1f; "
                   "costs ms:%s; ab=%c",
                   (unsigned long long)frame, d / 1e6, frame_ns[PERF_PIPE] / 1e6,
                   frame_ns[PERF_TEX] / 1e6, frame_ns[PERF_ASTC] / 1e6, frame_ns[PERF_FENCE] / 1e6,
                   frame_ns[PERF_PACE] / 1e6, costs, rt_ab() ? 'B' : 'A');
    }
    memset(frame_ns, 0, sizeof frame_ns);
    last = now;
    if (!win_start) win_start = now;
    if (frame % 60) return;
    uint64_t span = now - win_start;
    int64_t win[PERF_CNT_N];
    for (int i = 0; i < PERF_CNT_N; i++) {
        int64_t v = __atomic_load_n(&counts[i], __ATOMIC_RELAXED);
        win[i] = v - win_prev[i];
        win_prev[i] = v;
    }
    int64_t async = win[PERF_CNT_PIPE_ASYNC], skips = win[PERF_CNT_PIPE_SKIP], up = win[PERF_CNT_TEX_UP];
    int64_t reup = win[PERF_CNT_TEX_REUP], waited = win[PERF_CNT_PIPE_WAIT];
    int64_t oneshot = win[PERF_CNT_ONESHOT_WAIT];
    uint64_t shader_keys = gfx_shader_count();
    uint64_t pw_calls = 0, pw_pages = 0;
    plat_mem_pw_stats(&pw_calls, &pw_pages, 1);
    if (pw_calls && n)
        rt_log("perf: frame %llu: prot_watch arm %.1f mprotect calls/f (%.1f pages/f, %.1f pages/call)",
               (unsigned long long)frame, (double)pw_calls / n, (double)pw_pages / n, (double)pw_pages / pw_calls);
    if (n && span)
        rt_log("perf: frame %llu: interval avg %.2f ms max %.2f ms (%.1f fps); render thread blocked: pacing %.0f%% fence %.0f%% drain %.0f%% pipeline build %.0f%% texture %.0f%% (astc %.0f%%); pipelines built in background %lld, draws skipped pending %lld, waited %lld, one-shot waited %lld; textures uploaded %lld re-uploaded %lld; slowest frame: texture %.1f ms (astc %.1f) pipeline %.1f ms fence %.1f ms; shaders: %llu loaded; render busy avg %.2f ms; replay avg %.2f ms; ab %llu/%llu; world %.1f s",
               (unsigned long long)frame, sum / 1e6 / n, max / 1e6, n * 1e9 / sum, 100.0 * wait_ns[PERF_PACE] / span,
               100.0 * wait_ns[PERF_FENCE] / span, 100.0 * wait_ns[PERF_DRAIN] / span,
               100.0 * wait_ns[PERF_PIPE] / span,
               100.0 * wait_ns[PERF_TEX] / span, 100.0 * wait_ns[PERF_ASTC] / span, (long long)async, (long long)skips, (long long)waited, (long long)oneshot,
               (long long)up, (long long)reup, worst_ns[PERF_TEX] / 1e6,
               worst_ns[PERF_ASTC] / 1e6, worst_ns[PERF_PIPE] / 1e6, worst_ns[PERF_FENCE] / 1e6,
               (unsigned long long)shader_keys,
               (span > wait_ns[PERF_PACE] ? span - wait_ns[PERF_PACE] : 0) / 1e6 / n,
               plat_xchg64(&replay_ns, 0) / 1e6 / n, (unsigned long long)ab_b, (unsigned long long)n, frame_step_world_us() / 1e6);
    if (n && span) perf_gpu_line(frame, sum / 1e6 / n, (double)wait_ns[PERF_PACE] / span, n);
    if (n) ledger_window(frame, n);
    // MKW_LOCK_PROF=1: the waits per lock in this window (vk_sync.c's line needs 1,200 submissions; the boot has fewer),
    // and who held a lock while another thread waited on it
    if (n && span) {
        char lb[1024];
        if (plat_lock_report(0, lb, sizeof lb))
            rt_log("perf: frame %llu: lock waits (MKW_LOCK_PROF):%s", (unsigned long long)frame, lb);
        if (plat_lock_report(1, lb, sizeof lb))
            rt_log("perf: frame %llu: lock holds while waited on (MKW_LOCK_PROF):%s", (unsigned long long)frame, lb);
    }
    sum = max = n = ab_b = 0;
    memset(wait_ns, 0, sizeof wait_ns);
    win_start = now;
}
