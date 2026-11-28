#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "prof.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_mem_internal.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- page write generations
// Two-level table over the guest address space (below 2^39): 256 MiB chunks of 65,536 page generations, allocated
// when a page of the chunk is first stamped (GEN_*, vk_mem_internal.h). 0 = never stamped.
uint32_t *gen_chunk[GEN_CHUNKS];
// The same table for stamps of CPU origin only: write-watch queries, host writes (vk_guest_written). A GPU write
// reported at its fence (vk_gpu_written_stamp) stamps gen_chunk, so the texture checks and the shadow see it, but not
// this one: the mirror already holds those bytes (the GPU wrote them there) and syncs by this table, so it does not
// upload them again (NX2navi 6c971b7: 57 of 60 MiB a frame; here 11 of ~15, mir-reup, 14/11/2026). A CPU write after
// the GPU's stamps both, so a page both wrote still goes up.
uint32_t *cpu_gen_chunk[GEN_CHUNKS];
uint32_t gen_now;                 // last generation handed out
PlatMutex gen_lock = PLAT_MUTEX_INIT;
static uint64_t gen_pages[1024];
// The newest generation stamped on any page of each 2 MiB region (the write-watch snapshot regions below): the mirror
// sync skips regions with nothing newer than its pool's mirror (REG_SHIFT).
uint32_t reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];

// Every write-watch query of the generation table (gen_lock held exclusively): calls, bytes and time since the last
// mirror summary, all consumers together; the queries that found written pages (they also flush TLBs) apart.
uint64_t ww_bytes, ww_pages;
static uint64_t ww_calls, ww_ticks, ww_replay_ticks, ww_chunk_ticks, ww_dirty_calls, ww_dirty_ticks;
static _Thread_local int is_chunk_thread;

void vk_mem_chunk_thread_mark(void) { is_chunk_thread = 1; }

int ww_peek;   // gen_lock: ww_query leaves the written pages unprotected (a hot shadow pool, MKW_SHADOW_HOT)

static uint64_t ww_query(uint64_t addr, uint64_t size) {
    uint64_t t0 = qpc();
    uint64_t n = (ww_peek ? mem_written_peek : mem_written_pages)(addr, size, gen_pages, sizeof gen_pages / sizeof *gen_pages);
    uint64_t dt = qpc() - t0;
    ww_ticks += dt, ww_calls++, ww_bytes += size;
    if (plat_thread_id() == prof_replay_tid) ww_replay_ticks += dt;
    else if (is_chunk_thread) ww_chunk_ticks += dt;
    if (n && n != UINT64_MAX) ww_dirty_calls++, ww_dirty_ticks += dt, ww_pages += n;
    return n;
}

// Frozen regions: a 2 MiB region whose pages nothing wrote for freeze_idle submissions is made read-only
// (memory.c mem_freeze), and write-watch queries skip it until a write or a file read thaws it. Most of the pool memory
// the mirror and the texture checks cover is static (meshes, textures), and a query costs about 1 us per MiB (4-5 on an
// 8th-gen i7), so this keeps the queries to the regions the CPU writes. The region need not lie wholly in pools: NVN
// pools are small and not 2 MiB aligned, so most regions also hold other guest memory, and a rule that refused those
// left 340 of 436 queried MiB a frame unfrozen in town although nothing had written them for 300 submissions. The
// other memory is safe to freeze for the same reason pools are: the only kernel writes into guest memory are file
// reads, which thaw their buffer first (fs_host.c, romfs.c), and every user-mode write faults and thaws.
#define FREEZE_PER_SUB 32
int freeze_idle;                       // 0: off
uint32_t ww_ep = 1;                    // gen_lock: the submission being recorded (vk_mem_mirror_begin)
static uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last stamp
static uint32_t reg_tried[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: ww_ep of its last freeze candidacy
static uint32_t *fcand;                       // gen_lock: regions to try at the next freeze pass
static int nfcand, cfcand;
static uint64_t n_frz_refused;                // gen_lock: candidates mem_freeze refused

uint32_t *gen_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!gen_chunk[ci]) {
        if (!create) return NULL;
        gen_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!gen_chunk[ci]) rt_fatal("vk: out of memory for page generations");
    }
    return &gen_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// gen_lock held exclusively; g is the newest generation handed out; cpu 0 for a GPU write reported at its fence.
static void stamp_page(uint64_t a, uint32_t g, int cpu) {
    uint32_t *s = gen_slot(a, 1);
    if (!s) return;
    *s = g;
    reg_max[a >> REG_SHIFT] = g;
    reg_last[a >> REG_SHIFT] = ww_ep;
    if (!cpu) return;
    uint64_t ci = a >> GEN_CHUNK_SHIFT;
    if (!cpu_gen_chunk[ci]) {
        cpu_gen_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!cpu_gen_chunk[ci]) rt_fatal("vk: out of memory for page generations");
    }
    cpu_gen_chunk[ci][(a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] = g;
}

static void stamp_range(uint64_t lo, uint64_t hi, uint32_t g, int cpu) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page(a, g, cpu);
}

// Stamps that found gen_lock taken wait here (gstamp_lock) and go in before anyone reads stamps: every taker of gen_lock
// goes through gen_lock_x or gen_lock_s. The completion thread stamped each GPU write under gen_lock at the fence and
// waited ~1.3 ms there while a shadow sync queried the write watch, 1.47 s in a 12 s race (rt-n3, MKW_LOCK_PROF), so
// the fence was done that much later. A stamp applied later is one made after the wait the lock would have cost.
typedef struct { uint64_t lo, hi; int cpu; } GStamp;
static GStamp *gstamp;
static int ngstamp, cgstamp;
static volatile int32_t gstamp_n;           // ngstamp, read without the lock
static PlatMutex gstamp_lock = PLAT_MUTEX_INIT;   // after gen_lock

static void gen_drain(void) {   // gen_lock exclusive
    if (!__atomic_load_n(&gstamp_n, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&gstamp_lock);
    for (int i = 0; i < ngstamp; i++) stamp_range(gstamp[i].lo, gstamp[i].hi, ++gen_now, gstamp[i].cpu);
    ngstamp = 0;
    __atomic_store_n(&gstamp_n, 0, __ATOMIC_RELEASE);
    plat_mutex_unlock(&gstamp_lock);
}

void gen_lock_x(void) {
    plat_mutex_lock(&gen_lock);
    gen_drain();
}

void gen_lock_s(void) {
    for (;;) {
        plat_mutex_lock_shared(&gen_lock);
        if (!__atomic_load_n(&gstamp_n, __ATOMIC_ACQUIRE)) return;
        plat_mutex_unlock_shared(&gen_lock);
        gen_lock_x();
        plat_mutex_unlock(&gen_lock);
    }
}

static void written_stamp(uint64_t addr, uint64_t size, int cpu) {
    if (!size || nvn_gpu_only(addr)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    if (plat_mutex_trylock(&gen_lock)) {
        gen_drain();
        stamp_range(lo, hi, ++gen_now, cpu);
        plat_mutex_unlock(&gen_lock);
        return;
    }
    plat_mutex_lock(&gstamp_lock);
    if (ngstamp == cgstamp) vk_mem_grow((void **)&gstamp, &cgstamp, sizeof *gstamp);
    gstamp[ngstamp++] = (GStamp){lo, hi, cpu};
    __atomic_store_n(&gstamp_n, ngstamp, __ATOMIC_RELEASE);
    plat_mutex_unlock(&gstamp_lock);
}

void vk_guest_written(uint64_t addr, uint64_t size) { written_stamp(addr, size, 1); }
void vk_gpu_written_stamp(uint64_t addr, uint64_t size) { written_stamp(addr, size, 0); }

int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    gen_lock_x();
    uint32_t g = ++gen_now;
    int tracked = 1, frozen = 1;
    for (uint64_t a = lo & ~((1ull << REG_SHIFT) - 1); a < hi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    while (!frozen) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) { tracked = 0; break; }
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g, 1);
        if (n < sizeof gen_pages / sizeof *gen_pages) break;   // a full answer may have left pages for the next call
    }
    int changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
    }
    plat_mutex_unlock(&gen_lock);
    *now = g;
    return tracked ? changed : -1;
}

// Write-watch snapshots per submission: the guest address space in 2 MiB regions, each queried at most once per epoch
// (a guest submission), at the first check that covers it. A query costs about 1.3 us plus 0.55 us per MiB, so one per
// region instead of one per texture. The snapshot only adds observations to the generation table: a page written
// before it carries a generation newer than any check before it, and a page written after it keeps its OS bit until
// the next query over it (at the latest the next epoch's snapshot). SNAP_SHIFT is REG_SHIFT.
uint32_t snap_at[SNAP_REGIONS];      // epoch of the region's last snapshot; 0 never
uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t n_snap_checks, n_snap_calls, n_snap_untracked;

// gen_lock held exclusively. Queries the regions r0..r1 that were not snapped in epoch, one query per run of them;
// returns the number of queries.
uint64_t snap_regions(uint64_t r0, uint64_t r1, uint32_t epoch) {
    uint32_t g = 0;
    uint64_t calls = 0;
    for (uint64_t r = r0; r <= r1;) {
        if (snap_at[r] == epoch) { r++; continue; }
        if (mem_frozen(r << SNAP_SHIFT)) {   // nothing in it changed since its freeze
            snap_at[r] = epoch, snap_untracked[r] = 0;
            r++;
            continue;
        }
        uint64_t k = r;   // one query over a run of regions not seen in this epoch
        while (k + 1 <= r1 && snap_at[k + 1] != epoch && !mem_frozen((k + 1) << SNAP_SHIFT)) k++;
        uint64_t a = r << SNAP_SHIFT, b = (k + 1) << SNAP_SHIFT;
        int tracked = 1;
        if (!g) g = ++gen_now;
        for (;;) {
            uint64_t n = ww_query(a, b - a);
            if (n == UINT64_MAX) { tracked = 0; break; }
            for (uint64_t j = 0; j < n; j++) stamp_page(gen_pages[j], g, 1);
            if (n < sizeof gen_pages / sizeof *gen_pages) break;
        }
        calls++;
        for (uint64_t j = r; j <= k; j++) {
            snap_at[j] = epoch, snap_untracked[j] = (uint8_t)!tracked;
            if (tracked && freeze_idle && ww_ep - reg_last[j] >= (uint32_t)freeze_idle &&
                ww_ep - reg_tried[j] >= (uint32_t)freeze_idle) {
                reg_tried[j] = ww_ep;
                if (nfcand == cfcand) vk_mem_grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (uint32_t)j;
            }
        }
        r = k + 1;
    }
    return calls;
}

uint64_t vk_snap_q_ns, vk_snap_lock_ns;   // MKW_TRACE=perf: queries made by checks, and waits for gen_lock (precheck)

int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    int excl = 0, perf = (rt_trace & RT_TRACE_PERF) != 0;
    uint64_t t0 = perf ? kern_now_ns() : 0;
    gen_lock_s();
    if (perf) __atomic_add_fetch(&vk_snap_lock_ns, kern_now_ns() - t0, __ATOMIC_RELAXED);
    for (uint64_t r = r0; r <= r1; r++)
        if (snap_at[r] != epoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        uint64_t t1 = perf ? kern_now_ns() : 0;
        gen_lock_x();
        uint64_t t2 = perf ? kern_now_ns() : 0;
        n_snap_calls += snap_regions(r0, r1, epoch);
        if (perf)
            __atomic_add_fetch(&vk_snap_lock_ns, t2 - t1, __ATOMIC_RELAXED),
                __atomic_add_fetch(&vk_snap_q_ns, kern_now_ns() - t2, __ATOMIC_RELAXED);
    }
    int untracked = 0, changed = 0;
    for (uint64_t r = r0; r <= r1; r++) untracked |= snap_untracked[r];
    // Pages only in regions stamped after since (reg_max; REG_SHIFT is SNAP_SHIFT): an unchanged texture of 4 MiB was
    // 1,024 page reads, at about 1,100 texture checks a frame.
    for (uint64_t r = r0; !untracked && r <= r1 && !changed; r++) {
        if (reg_max[r] <= since) continue;
        uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
        for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) changed = 1;
        }
    }
    *now = gen_now;   // every observation so far is in the table
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (untracked) __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    return untracked ? -1 : changed;
}

int mem_written_any(uint64_t addr, uint64_t size);   // memory.c (MKW_SNAP_SCOPED)

// MKW_SNAP_SCOPED (default 0, 1 in the PS5 profile; "ab"): a texture's check looks at its own span only. Its pages' generations newer than
// since (a GPU write, a host write, an earlier query) say changed; then the write watch is peeked over the span
// (mem_written_any, nothing protected): a page that may hold an unreported write sends the check to vk_mem_changed,
// which queries and re-arms only the span before the texels are read; otherwise every page is armed and unwritten,
// so a write from now on faults and is seen by the next check, as with the snapshot. Why: the snapshot queried and
// re-armed whole 2 MiB regions, and of the written pages it re-armed 98 % (normal frames) and 65 % (the precheck's
// spikes) lay in no texture the precheck looked up: 0.7 ms of sceKernelMprotect a frame, 2-9 ms in spikes, on the
// replay thread under tex_lock (MKW_BURST_PROBE runs, PS5, 24/11/2026). Pages no texture covers stay written and
// writable: nothing re-arms them, and the game's writes to them stop faulting. The peek and the generations are read
// under gen_lock shared, so no query (exclusive) can report and re-arm a page between them. In-race A/B, two races
// (MKW_AB=6, plain and inverted): sceKernelMprotect arms 0.77 -> 0.24-0.28 ms a frame, the game's write faults
// 13-14 -> 2.7-4.2, uploads unchanged, precheck spikes (> 4 ms) 11.0 -> 4.8 % and 6.9 -> 2.9 % of frames, the frame
// -0.28 / -0.14 ms and busy -0.55 / -0.84 ms (scoped-1p, 24/11/2026). A 1.8.0 race with it on, watched on screen by
// the user, showed no stale texture (a missed write would show only there): on in the PS5 profile (cfg.c). Off
// elsewhere: without a peek (plat_mem_written_any) every check would query its span.
// vk_mem_changed_scoped's answer when it needs nothing but reads: 0 or 1 as it would answer, -2 when it would query or
// re-arm the write watch (written pages, a span not watched or past the snapshot regions), which must stay on the
// caller's serial path. Any thread, with the texture lock held by the caller's job (MKW_PRECHECK_PAR, vk_tex_slots.c).
int vk_mem_changed_peek(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> REG_SHIFT, r1 = (hi - 1) >> REG_SHIFT;
    if (r1 >= SNAP_REGIONS) return -2;
    gen_lock_s();
    uint32_t g = gen_now;
    int changed = 0, all_frozen = 1;
    for (uint64_t r = r0; r <= r1; r++) {
        if (all_frozen && !mem_frozen(r << REG_SHIFT)) all_frozen = 0;
        if (!changed && reg_max[r] > since) {
            uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
            for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
                uint32_t *s = gen_slot(a, 0);
                if (s && *s > since) changed = 1;
            }
        }
    }
    int w = all_frozen ? 0 : mem_written_any(lo, hi - lo);
    plat_mutex_unlock_shared(&gen_lock);
    if (w) return -2;
    *now = g;
    return changed;
}

int vk_mem_changed_scoped(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> REG_SHIFT, r1 = (hi - 1) >> REG_SHIFT;
    if (r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    gen_lock_s();
    uint32_t g = gen_now;
    int changed = 0, all_frozen = 1;
    for (uint64_t r = r0; r <= r1; r++) {
        if (all_frozen && !mem_frozen(r << REG_SHIFT)) all_frozen = 0;
        if (!changed && reg_max[r] > since) {
            uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
            for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
                uint32_t *s = gen_slot(a, 0);
                if (s && *s > since) changed = 1;
            }
        }
    }
    int w = all_frozen ? 0 : mem_written_any(lo, hi - lo);
    plat_mutex_unlock_shared(&gen_lock);
    if (w < 0) return vk_mem_changed(addr, size, since, now);   // not watched: it answers -1, the hash decides
    if (!w) {
        *now = g;
        return changed;
    }
    // May hold an unreported write: query and re-arm the span rounded out to whole native pages (a query leaves a
    // page it covers only in part written and unarmed, prot_watch_written's sub_partial, so a span that began or ended
    // inside a 16 KiB page found it written at every check and uploaded again each time: 13 uploads a frame instead of
    // 1.2 in the first A/B, 24/11/2026), then judge only the span's own pages.
    uint64_t wp = plat_mem_watch_page();
    uint64_t nlo = lo & ~(wp - 1), nhi = (hi + wp - 1) & ~(wp - 1);
    gen_lock_x();
    uint32_t g2 = ++gen_now;
    int tracked = 1, frozen = 1;
    for (uint64_t a = nlo & ~((1ull << REG_SHIFT) - 1); a < nhi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    if (!frozen) tracked = query_stamp(nlo, nhi, g2);
    changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
    }
    plat_mutex_unlock(&gen_lock);
    *now = g2;
    return tracked ? changed : -1;
}

// Per-subresource dirty check against the page generation table already populated by vk_mem_changed / vk_mem_changed_snap:
// takes gen_lock_s() once for all subresources of a re-uploaded texture so unchanged mip levels / array layers are skipped.
void vk_mem_ranges_written_since(uint64_t base, const uint64_t *offs, const uint64_t *sizes, uint8_t *dirty, uint32_t n,
                                 uint32_t since) {
    gen_lock_s();
    for (uint32_t i = 0; i < n; i++) {
        if (!sizes[i]) { dirty[i] = 0; continue; }
        uint64_t lo = (base + offs[i]) & ~0xfffull, hi = (base + offs[i] + sizes[i] + 0xfff) & ~0xfffull;
        uint64_t r0 = lo >> REG_SHIFT, r1 = (hi - 1) >> REG_SHIFT;
        int ch = 0;
        if (r1 >= SNAP_REGIONS) {
            ch = 1;   // outside the region table: conservative
        } else {
            for (uint64_t r = r0; r <= r1 && !ch; r++) {
                if (reg_max[r] <= since) continue;
                uint64_t a = (r << REG_SHIFT) > lo ? (r << REG_SHIFT) : lo;
                uint64_t b = ((r + 1) << REG_SHIFT) < hi ? ((r + 1) << REG_SHIFT) : hi;
                for (; a < b && !ch; a += 1ull << GEN_PAGE_SHIFT) {
                    uint32_t *s = gen_slot(a, 0);
                    if (s && *s > since) ch = 1;
                }
            }
        }
        dirty[i] = (uint8_t)ch;
    }
    plat_mutex_unlock_shared(&gen_lock);
}

void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked) {
    *checks = __atomic_exchange_n(&n_snap_checks, 0, __ATOMIC_RELAXED);
    *calls = __atomic_exchange_n(&n_snap_calls, 0, __ATOMIC_RELAXED);
    *untracked = __atomic_exchange_n(&n_snap_untracked, 0, __ATOMIC_RELAXED);
}

// gen_lock held exclusively. 0: the range is not write-watched.
int query_stamp(uint64_t lo, uint64_t hi, uint32_t g) {
    for (;;) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) return 0;
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g, 1);
        if (n < sizeof gen_pages / sizeof *gen_pages) return 1;
        if (ww_peek) lo = gen_pages[n - 1] + (1ull << GEN_PAGE_SHIFT);   // the pages stay written: go on past them
    }
}
// gen_lock exclusive: freezes up to FREEZE_PER_SUB candidate regions still idle. The query after the
// protection collects the writes made before it took effect.
void freeze_pass(void) {
    for (int done = 0; nfcand && done < FREEZE_PER_SUB;) {
        uint64_t j = fcand[--nfcand], a = j << REG_SHIFT;
        if (ww_ep - reg_last[j] < (uint32_t)freeze_idle || mem_frozen(a)) continue;
        done++;
        if (!mem_freeze(a)) { n_frz_refused++; continue; }
        query_stamp(a, a + (1ull << REG_SHIFT), ++gen_now);
    }
}

// MKW_PACK's cold feed (ps5_guest_memory.c plat_mem_cold_census_step, the governor's thread): a 2 MiB region whose
// units were all unwritten for MKW_PACK_FEED_S is frozen as freeze_pass freezes its candidates, the query after the
// protection collecting the writes made before it took effect. freeze_pass's candidates come only from regions the
// replay checks (snap_regions), so the game's own cold memory was never frozen and MKW_PACK found nothing to pack in
// four 1.8.0 Grand Prix runs (24/11/2026). 1 when the region is frozen.
int vk_mem_cold_freeze(uint64_t a) {
    if (mem_frozen(a)) return 1;
    gen_lock_x();
    int ok = mem_freeze(a);
    if (ok) query_stamp(a, a + (1ull << REG_SHIFT), ++gen_now);
    else n_frz_refused++;
    plat_mutex_unlock(&gen_lock);
    return ok;
}

// With MKW_TRACE=perf, beside the mirror summary (vk_mem_census): what every write-watch consumer cost since the last.
void ww_log(uint64_t frame) {
    static uint64_t ww_last;
    gen_lock_x();
    uint64_t wc = ww_calls, wb = ww_bytes, wt = ww_ticks, wrt = ww_replay_ticks, wct = ww_chunk_ticks, dc = ww_dirty_calls, dt = ww_dirty_ticks, wp = ww_pages;
    ww_pages = 0;
    uint64_t nr = n_frz_refused;
    ww_calls = ww_bytes = ww_ticks = ww_replay_ticks = ww_chunk_ticks = ww_dirty_calls = ww_dirty_ticks = n_frz_refused = 0;
    plat_mutex_unlock(&gen_lock);
    double wf = frame > ww_last ? (double)(frame - ww_last) : 1.0;
    ww_last = frame;
    uint64_t fz, ff, ft, fo;
    mem_freeze_stats(&fz, &ff, &ft, &fo);
    uint64_t ww_faults = plat_mem_write_watch_faults(1), ww_rep = plat_mem_write_watch_repeats(1),
             ww_hot = plat_mem_write_watch_hot_skips(1);
    if (rt_trace & RT_TRACE_PERF)
        rt_log("mem: write-watch frame %llu: per frame, every consumer: %.1f queries over %.1f MiB, %.3f ms (replay %.3f, chunks %.3f, %.1f found "
               "%.0f written pages, %.3f ms); frozen %.0f MiB; since the last line: freezes %llu, thaws by a write %llu, by a file read "
               "or mapping change %llu, refused %llu; write-watch faults %llu (%llu on pages that faulted in their previous cycle too), arms skipped for hot pages %llu",
               (unsigned long long)frame, wc / wf, wb / wf / 1048576.0, qpc_ms(wt) / wf, qpc_ms(wrt) / wf, qpc_ms(wct) / wf, dc / wf, wp / wf, qpc_ms(dt) / wf,
               fz * 2.0, (unsigned long long)ff, (unsigned long long)ft, (unsigned long long)fo, (unsigned long long)nr,
               (unsigned long long)ww_faults, (unsigned long long)ww_rep,
               (unsigned long long)ww_hot);
}

// vk_mem_shutdown, after every pool went.
void gen_reset(void) {
    gen_lock_x();
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) {
        free(gen_chunk[i]), gen_chunk[i] = NULL;
        free(cpu_gen_chunk[i]), cpu_gen_chunk[i] = NULL;
    }
    plat_mutex_unlock(&gen_lock);
}
