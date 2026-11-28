#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_tex_internal.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- released entries, eviction and the owned list
// Out of memory with no released entry left: live textures that mirror CPU-visible storage (uploaded again from it
// on their next use), that nothing looked up in the last EVICT_IDLE submissions and the GPU did not write, go oldest
// use first. By the draw bindings' check alone (checked), a texture the replay used only as a copy source
// (vk_texture_get) looked idle while a part being recorded could name it: the likely cause of a view destroyed with
// that part not yet sent (MKW_GPU_GUARD, tsvp1 frame 1 477 and tsvp2, 08/11/2026, both after evictions). The race's downscaled render targets (bloom, the final composite's inputs) found the heap full of decoded
// ASTC textures and got no image: a black race (run-race4, 31/10/2026).
#define EVICT_IDLE 64
static Tex *rel_oldest_droppable(void);   // below: released entries a trim may take (rel_keep)
static uint64_t n_evicted, evicted_bytes;

uint32_t tex_evict_idle(void) {
    static int32_t v = -1;
    int32_t val = __atomic_load_n(&v, __ATOMIC_RELAXED);
    if (val < 0) {
        const char *e = cfg_get("MKW_TEX_EVICT_IDLE");
        val = e && *e && atoi(e) > 0 ? atoi(e) : EVICT_IDLE;
        __atomic_store_n(&v, val, __ATOMIC_RELAXED);
    }
    return (uint32_t)val;
}

VkResult evict_idle(VkCtx *c, const VkImageCreateInfo *ci, const VmaAllocationCreateInfo *ai, VkImage *img,
                    VmaAllocation *alloc) {
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    uint32_t idle = tex_evict_idle();
    for (int round = 0; r == VK_ERROR_OUT_OF_DEVICE_MEMORY && round < 4; round++) {
        Tex *cand[64];
        int n = 0;
        for (uint32_t i = 0; i < by_key.cap && n < 64; i++) {
            Tex *e = by_key.vals ? by_key.vals[i] : NULL;
            if (e && e->alloc && e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty && !e->released && !e->pin &&
                (uint32_t)(submit_epoch - e->seen) > idle)
                cand[n++] = e;
        }
        if (!n) break;
        for (int i = 0; i < n; i++) {
            VmaAllocationInfo vi = {0};
            vmaGetAllocationInfo(c->vma, cand[i]->alloc, &vi);
            n_evicted++, evicted_bytes += vi.size;
            destroy_idle(cand[i]);
        }
        vk_mem_collect();
        r = vma_image(c, ci, ai, img, alloc);
    }
    if (n_evicted && !(n_evicted & (n_evicted - 1)))
        rt_log("vk: %llu idle mirrored textures evicted for room (%.0f MiB)", (unsigned long long)n_evicted,
               evicted_bytes / 1048576.0);
    return r;
}

// vk_staging out of memory: released entries, oldest first (as create_tex), then idle mirrored textures (evict_idle's
// rule), worth at least bytes, when the texture lock is free or this thread's (a thread holding it may wait for the
// upload lock the caller holds). 1 if any went. tt4 (09/11/2026) died on a staging buffer with 1 249 released entries
// holding 946 MiB: the idle ones alone were looked at.
// The trylock succeeds on the thread that holds the lock, the recursive case: upload() asking for staging trimmed the
// very entry it was uploading (looked up last more than idle submissions ago) and its caller went on with the freed
// Tex. make_fresh marks the entry seen before it uploads and pins it (tex_pin); a pinned entry is skipped here.
uint64_t vk_texture_trim(uint64_t bytes) {
    VkCtx *c = vk_ctx();
    if (!c || !plat_rmutex_trylock(&tex_lock)) return 0;
    TL_TOOK();
    uint64_t freed = 0;
    for (Tex *r; freed < bytes && (r = rel_oldest());) {
        VmaAllocationInfo vi = {0};
        if (r->alloc) vmaGetAllocationInfo(c->vma, r->alloc, &vi);
        freed += vi.size;
        destroy_idle(r);
    }
    uint32_t idle = tex_evict_idle();
    for (uint32_t i = 0; i < by_key.cap && freed < bytes; i++) {
        Tex *e = by_key.vals ? by_key.vals[i] : NULL;
        if (!e || !e->alloc || !e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty || e->released || e->pin ||
            (uint32_t)(submit_epoch - e->seen) <= idle)
            continue;
        VmaAllocationInfo vi = {0};
        vmaGetAllocationInfo(c->vma, e->alloc, &vi);
        freed += vi.size, n_evicted++, evicted_bytes += vi.size;
        destroy_idle(e);
        i--;   // the table shifted an entry into this slot
    }
    unlock();
    return freed;
}

// The memory governor's trim (vk_mem_governor, hard level, MKW_GOV_TOTAL): mirrored textures idle for
// MKW_GOV_EVICT_IDLE submissions (default 600, a few seconds; the out-of-memory paths keep EVICT_IDLE), oldest use
// first, GOV_EVICT_BATCH at a time with tex_lock released between batches. vk_texture_trim walks the table in hash
// order with the lock held throughout: in a race (tls-new4, 19/11/2026) 42 such passes evicted 10,184 MiB of textures
// seen half a second before, held tex_lock 6-16 ms each, and the replay uploaded them again moments later.
#define GOV_EVICT_IDLE 600
#define GOV_EVICT_BATCH 16
typedef struct { TexKey key; uint32_t seen; } EvictCand;

static int cand_by_seen(const void *a, const void *b) {
    uint32_t x = ((const EvictCand *)a)->seen, y = ((const EvictCand *)b)->seen;
    return (int32_t)(x - y) < 0 ? -1 : (int32_t)(x - y) > 0;
}

static int gov_evictable(const Tex *e, uint32_t idle) {
    return e && e->alloc && e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty && !e->released && !e->pin &&
           (uint32_t)(submit_epoch - e->seen) > idle;
}

uint64_t vk_texture_trim_gov(uint64_t bytes) {
    VkCtx *c = vk_ctx();
    if (!c || !bytes) return 0;
    static int32_t idle_v = -1;
    if (idle_v < 0) {
        const char *s = cfg_get("MKW_GOV_EVICT_IDLE");
        idle_v = s && *s && atoi(s) > 0 ? atoi(s) : GOV_EVICT_IDLE;
    }
    uint32_t idle = (uint32_t)idle_v;
    if (!plat_rmutex_trylock(&tex_lock)) return 0;
    TL_TOOK();
    uint32_t cap = by_key.n + 1, n = 0;
    EvictCand *cand = malloc(sizeof *cand * cap);
    if (cand)
        for (uint32_t i = 0; i < by_key.cap && n < cap; i++) {
            Tex *e = by_key.vals ? by_key.vals[i] : NULL;
            if (gov_evictable(e, idle)) cand[n].key = e->key, cand[n].seen = e->seen, n++;
        }
    unlock();
    if (!cand) return 0;
    qsort(cand, n, sizeof *cand, cand_by_seen);
    uint64_t freed = 0;
    for (uint32_t k = 0; k < n && freed < bytes;) {
        if (!plat_rmutex_trylock(&tex_lock)) break;   // the replay holds it: the next pass carries on
        TL_TOOK();
        for (uint32_t b = 0; b < GOV_EVICT_BATCH && k < n && freed < bytes; b++, k++) {
            Tex *e = nvn_hm_get(&by_key, &cand[k].key);   // looked up again: it may have gone or been used since
            if (!gov_evictable(e, idle)) continue;
            VmaAllocationInfo vi = {0};
            vmaGetAllocationInfo(c->vma, e->alloc, &vi);
            freed += vi.size, n_evicted++, evicted_bytes += vi.size;
            destroy_idle(e);
        }
        unlock();
    }
    free(cand);
    return freed;
}

uint64_t vk_texture_trim_released(uint64_t keep_bytes) {
    VkCtx *c = vk_ctx();
    if (!c || !plat_rmutex_trylock(&tex_lock)) return 0;
    TL_TOOK();
    uint64_t before = released_bytes;
    for (Tex *r; released_bytes > keep_bytes && (r = rel_oldest_droppable());)
        destroy_idle(r);
    uint64_t freed = before > released_bytes ? before - released_bytes : 0;
    unlock();
    return freed;
}

int owned_lower(uint64_t a) {
    int lo = 0, hi = nowned;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (owned[m]->pub.addr < a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

void own_add(Tex *e) {
    if (e->listed) return;
    plat_mutex_lock(&owned_lock);
    if (nowned == cowned) {
        cowned = cowned ? 2 * cowned : 256;
        if (!(owned = realloc(owned, sizeof *owned * (size_t)cowned))) rt_fatal("vk: out of memory for the owned-texture list");
    }
    int i = owned_lower(e->pub.addr);
    memmove(owned + i + 1, owned + i, sizeof *owned * (size_t)(nowned - i));
    owned[i] = e;
    __atomic_store_n(&nowned, nowned + 1, __ATOMIC_RELEASE);
    if (e->t.L.size > max_owned) max_owned = e->t.L.size;
    e->listed = 1;
    plat_mutex_unlock(&owned_lock);
}

static void own_remove(Tex *e) {
    if (!e->listed) return;
    plat_mutex_lock(&owned_lock);
    for (int i = owned_lower(e->pub.addr); i < nowned && owned[i]->pub.addr == e->pub.addr; i++)
        if (owned[i] == e) {
            memmove(owned + i, owned + i + 1, sizeof *owned * (size_t)(nowned - i - 1));
            __atomic_store_n(&nowned, nowned - 1, __ATOMIC_RELEASE);
            break;
        }
    e->listed = 0;
    plat_mutex_unlock(&owned_lock);
}

void rel_append(Tex *e) {
    VkCtx *c = vk_ctx();
    VmaAllocationInfo vi = {0};
    if (e->alloc && c) vmaGetAllocationInfo(c->vma, e->alloc, &vi);
    e->rel_bytes = vi.size, released_bytes += vi.size;
    e->rel_next = NULL;
    e->rel_prev = rel_tail;
    if (rel_tail) rel_tail->rel_next = e;
    else rel_head = e;
    rel_tail = e;
}

// Released entries are never the one a caller is uploading (taking one back clears released), but the sweeps that walk
// them run inside staging that ran short too: a pinned one stays, as in vk_texture_trim.
Tex *rel_oldest(void) {
    Tex *r = rel_head;
    while (r && r->pin) r = r->rel_next;
    return r;
}

// A released image the GPU wrote in the last REL_HOT frames is kept by the count, the memory cap, the overwrite rule
// (up to REL_KEEP_OVER) and the governor's trim; only running out of memory takes it (rel_oldest). It holds what the
// GPU drew there, which the console keeps in that memory, and the game still uses it. The race's screen-tiled volumes (56x32-88x48 by 256
// slices, 3.5-6 MiB; released and taken back every frame, each frame reading what the last one wrote) were dropped by
// those rules, the governor's trim most of all, and made again cleared to zero: the lit scene black for a frame with
// the HUD drawn, 37-91 times a race on the PS5. With this, 0 in a race, free direct memory at least 646 MiB (runs
// tr-trace, tr2, fix6 with MKW_ODD_CHECK, 21/11/2026; 0 also with new images left uncleared, MKW_FIX_ZERO=0, where the
// new one happened to get the old memory). Keyed on the game taking it back, it never engaged: each went before that.
#define REL_HOT 600
#define REL_KEEP_OVER (16ull << 20)   // drop_overwritten heeds rel_keep up to this size
static int rel_keep(const Tex *e) {
    return !e->pub.mirrored && (e->pub.gpu_owned || e->gpu_dirty) && !e->sparse && e->gpu_frame &&
           vk_exec_presented + 1 - e->gpu_frame <= REL_HOT;
}
static Tex *rel_oldest_droppable(void) {
    Tex *r = rel_head;
    while (r && (r->pin || rel_keep(r))) r = r->rel_next;
    return r;
}

void rel_unlink(Tex *e) {
    released_bytes -= e->rel_bytes, e->rel_bytes = 0;
    if (e->rel_prev) e->rel_prev->rel_next = e->rel_next;
    else rel_head = e->rel_next;
    if (e->rel_next) e->rel_next->rel_prev = e->rel_prev;
    else rel_tail = e->rel_prev;
    e->rel_prev = e->rel_next = NULL;
}

void destroy_tex(Tex *e) {
    own_remove(e);
    tex_gen_bump(e->key.addr);
    nvn_hm_del(&by_key, &e->key);
    Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head == e) {
        if (e->next_at) nvn_hm_put(&by_addr, &e->key.addr, e->next_at);
        else nvn_hm_del(&by_addr, &e->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == e) { p->next_at = e->next_at; break; }
    }
    if (e->released) nreleased--, rel_unlink(e);
    vk_mem_defer_use(e->use);   // MKW_GPU_GUARD: the image and its views
    vk_exec_image_gone(e->pub.image);
    // A sparse image also waits for its binds: they are queue operations the parts sent do not cover (vk_sparse.c).
    uint64_t binds = e->sparse ? vk_sparse_destroy(e->pub.image) : 0, prev = 0;
    if (binds) prev = vk_mem_defer_fence(0), vk_mem_defer_fence(prev > binds ? prev : binds);
    vk_defer_image(e->pub.image, e->alloc);
    if (e->rb) vk_defer_buffer(e->rb, e->rba);   // a readback the parts sent may still write (readback_record_locked)
    if (binds) vk_mem_defer_fence(prev);
    vk_mem_defer_use(0);
    free(e->gpu_base);
    free(e);
}

// A released entry is not named by the submission being recorded (taking one back clears released), nor an idle one:
// they go once the parts already sent are done (vk_mem_defer_idle).
// Not when the submission being recorded looked it up (seen): a released entry replaced at a new descriptor of its
// address (entry_of) was still named by the part being recorded, and its view went with the parts already sent only
// (MKW_GPU_GUARD stack, gstack1 08/11/2026). Then it waits for that submission (the replay records one at a time).
void destroy_idle(Tex *e) {
    if (e->seen == submit_epoch) {
        vk_mem_defer_fence(vk_sync_issued() + 1);
        destroy_tex(e);
        vk_mem_defer_fence(0);
        return;
    }
    vk_mem_defer_idle(1);
    destroy_tex(e);
    vk_mem_defer_idle(0);
}

// Released entries with another descriptor at the same address are replaced; past RELEASED_MAX released entries the
// oldest one goes at each release (it comes back on demand, a mirrored one re-uploaded from its storage). TotK
// finalizes and initializes transient textures every frame, so a released entry is mostly taken back within frames.
// Dropping all of them at once when the count passed 4096 destroyed thousands of images in one frame: 333-422 ms
// frames every 40-50 s on a GTX 1050 Ti, then the textures still in use were created and uploaded again.
// The count alone let them hold 1 603 MiB on an 8 GB RTX 4070 Laptop (2 032 entries, at the character select after the
// title's menus, 12/11/2026): with the rest, 7 333 MiB against the driver's 7 180 MiB budget, so Windows moved memory
// out of the GPU and the game ran at 0.24x (the GPU 80-444 ms a frame at 33 W); the ctl "trim" of them brought it back
// to 1.00x at once, under the budget. Their memory is bounded too: MKW_TEX_RELEASED_MIB (default a twelfth of the
// device-local budget at start, 256 MiB to 2 GiB: ~600 MiB on 8 GB, 2 GiB from 24 GB), the oldest recreatable ones
// going first.
#define RELEASED_MAX 2048
static uint64_t released_cap(void) {
    static uint64_t cap;
    if (cap) return cap;
    const char *e = cfg_get("MKW_TEX_RELEASED_MIB");
    uint64_t used, budget;
    int exact;
    if (e && *e) cap = (uint64_t)atoll(e) << 20;
#ifdef MKW_PS5_NATIVE
    // The PS5 shares its 12 GiB between the guest and the GPU: 256 MiB, as before the cap followed the budget (run
    // memgrow, 16/11/2026, the console ran out of direct memory over a session).
    else cap = 256ull << 20;
    (void)used, (void)budget, (void)exact;
#else
    else if (vk_mem_vram(&used, &budget, &exact) && budget) cap = budget / 12;
    else cap = 512ull << 20;
#endif
    if (!(e && *e)) cap = cap < (256ull << 20) ? 256ull << 20 : cap > (2048ull << 20) ? 2048ull << 20 : cap;
    if (!cap) cap = 1;   // MKW_TEX_RELEASED_MIB=0: none kept
    rt_log("vk: released textures keep at most %d entries and %llu MiB (MKW_TEX_RELEASED_MIB)", RELEASED_MAX,
           (unsigned long long)(cap >> 20));
    return cap;
}

// Past the count the oldest goes, as before. Past the memory only entries whose contents come back with them go: a
// mirror of CPU-visible storage the GPU never wrote, re-uploaded on demand (evict_idle's rule). A render target or
// another GPU-written image taken back at the same address would come back without what was drawn there, which the
// console keeps (the same memory); the count's rule takes that risk only past 2 048 entries. On the 8 GB laptop the
// recreatable ones were enough: the cap was never passed in two cups (no "over the cap" line).
static int recreatable(const Tex *e) { return e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty; }

// A released GPU-written entry is kept for what was drawn in it, which the console keeps in the same memory. But only
// until another image is written over that memory: then the console holds the newer contents there, and the entry has
// nothing of its own left to come back with. 1.8.0 makes its shadow map again at each scene, at a new place in the
// render-buffer pool 0x24000 bytes past the last (ph4, 16/11/2026: 4096x4096 D16, 96 MiB each, 61 addresses in 4
// minutes of the title's demo runs): every one overlaps the next, all were kept as GPU-written, and the released ones
// held 9.9 GB until the device ran out of memory (amdgpu: "Not enough memory for command submission", then device
// lost); with the console's map, 3.3 GB after 50 s of a two-player race (sp1). Those whose memory a later write
// (a newer wr_fence; an equal one is not counted) covers in part go, oldest first and only until the cap holds: the
// game takes many back at the same key, and each one dropped is made again (the first version dropped every one it
// found: 2 522 shadow maps made in rel1 against 204, the turbo's speed 1.67x against 1.90x). Writers larger than
// WRITER_SPAN are left out of the search (the window it scans before each entry): they are textures, not render
// targets.
#define WRITER_SPAN (512ull << 20)
typedef struct { uint64_t lo, hi, fence; } Span;
static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Span *)a)->lo, y = ((const Span *)b)->lo;
    return (x > y) - (x < y);
}
static uint64_t overwritten_n, overwritten_bytes;
static void drop_overwritten(void) {
    Span *w = malloc(sizeof *w * (size_t)(by_key.n + 1));
    Tex **gone = malloc(sizeof *gone * (size_t)(nreleased + 1));
    if (!w || !gone) { free(w), free(gone); return; }
    int nw = 0, ng = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && e->wr_fence && e->t.L.size && e->t.L.size <= WRITER_SPAN)
            w[nw++] = (Span){e->key.addr, e->key.addr + e->t.L.size, e->wr_fence};
    }
    qsort(w, (size_t)nw, sizeof *w, by_lo);
    uint64_t over = released_bytes - released_cap(), freed = 0;   // the caller: over the cap
    for (Tex *e = rel_head; e && freed < over; e = e->rel_next) {
        // rel_keep only up to REL_KEEP_OVER: the volumes it is for (8.6 MiB at most) are written through a second
        // descriptor of the same memory, which counts here as a newer image over them (with half coverage required
        // they still went, black frames 0-2 a race, fix7r1/r2), while a scene's shadow map (~96 MiB, written every
        // frame) is covered whole by the next scene's: kept, the released ones held 3 GB and free direct memory fell to
        // 68 MiB in a race (bug2c3, 21/11/2026).
        if (recreatable(e) || (rel_keep(e) && e->t.L.size <= REL_KEEP_OVER) || e->sparse || !e->wr_fence || !e->t.L.size)
            continue;
        uint64_t lo = e->key.addr, hi = lo + e->t.L.size, from = lo > WRITER_SPAN ? lo - WRITER_SPAN : 0;
        int a = 0, b = nw;   // the first writer starting at from or later
        while (a < b) {
            int m = (a + b) / 2;
            if (w[m].lo < from) a = m + 1;
            else b = m;
        }
        for (int k = a; k < nw && w[k].lo < hi; k++)
            if (w[k].hi > lo && w[k].fence > e->wr_fence) { gone[ng++] = e, freed += e->rel_bytes; break; }
    }
    for (int i = 0; i < ng; i++) overwritten_n++, overwritten_bytes += gone[i]->rel_bytes, destroy_idle(gone[i]);
    free(w), free(gone);
}

// The same rule for entries the game never released (issue 11). 1.0.0 initializes its race shadow map (3072x3072, 3
// layers of D16, 54 MiB) at a new place of the render-buffer pool without finalizing the last one: on
// Race_Coaster_A_00 the runtime made it 2 539 times in a few minutes and held up to 9 of them live, 501 MiB, while
// the dynamic-resolution targets around it moved (issue11-shapes-20261123-210136); with the guest near 5.3 GiB, the
// next 54 MiB image or a 4 MiB transient buffer found no memory (issue11-100-*, issue11-heaprel-*). On the console
// those maps are the same pool memory, and once a newer target is written over one, what it held is gone there too.
// So a live GPU-owned image goes when a later write (newer wr_fence) covers part of its storage and nothing looked
// it up for MKW_TEX_LIVE_DROP_IDLE submissions (default 4): it has nothing of its own left, and if the game names it
// again it is made again, as a released one would be. Writers at its own address are left to the alias copies
// (vk_tex_alias.c). With MKW_TEX_EVICT_IDLE's 64 the drop lagged the remaking: 9 maps live at once again and GPU-owned
// images at 1.1-1.3 GiB through the race (issue11-livedrop-20261123-212301, which finished with 68 MiB free).
// Run every 64 MiB of GPU-owned images made (tex_gpu_made), and before an image creation gives up.
static uint64_t live_dropped_n, live_dropped_bytes, gpu_made_since;
static Span *live_w_buf;
static Tex **live_gone_buf;
static uint32_t live_buf_cap;

void tex_evict_shutdown(void) {
    free(live_w_buf), free(live_gone_buf);
    live_w_buf = NULL, live_gone_buf = NULL, live_buf_cap = 0;
}

int drop_overwritten_live(void) {
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    uint32_t need_cap = by_key.n + 1;
    if (need_cap > live_buf_cap) {
        uint32_t ncap = live_buf_cap ? live_buf_cap : 1024;
        while (ncap < need_cap) ncap *= 2;
        Span *nw_buf = realloc(live_w_buf, sizeof *nw_buf * (size_t)ncap);
        Tex **ng_buf = realloc(live_gone_buf, sizeof *ng_buf * (size_t)ncap);
        if (!nw_buf || !ng_buf) {
            if (nw_buf) live_w_buf = nw_buf;
            if (ng_buf) live_gone_buf = ng_buf;
            return 0;
        }
        live_w_buf = nw_buf, live_gone_buf = ng_buf, live_buf_cap = ncap;
    }
    Span *w = live_w_buf;
    Tex **gone = live_gone_buf;
    int nw = 0, ng = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && e->wr_fence && e->t.L.size && e->t.L.size <= WRITER_SPAN)
            w[nw++] = (Span){e->key.addr, e->key.addr + e->t.L.size, e->wr_fence};
    }
    qsort(w, (size_t)nw, sizeof *w, by_lo);
    static int32_t idle = -1;
    if (idle < 0) {
        const char *v = cfg_get("MKW_TEX_LIVE_DROP_IDLE");
        idle = v && *v && atoi(v) > 0 ? atoi(v) : 4;
    }
    uint64_t freed = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (!e || e->released || !e->alloc || !e->pub.gpu_owned || e->pub.mirrored || e->sparse || e->pin ||
            !e->wr_fence || !e->t.L.size || (uint32_t)(submit_epoch - e->seen) <= (uint32_t)idle)
            continue;
        uint64_t lo = e->key.addr, hi = lo + e->t.L.size, from = lo > WRITER_SPAN ? lo - WRITER_SPAN : 0;
        int a = 0, b = nw;
        while (a < b) {
            int m = (a + b) / 2;
            if (w[m].lo < from) a = m + 1;
            else b = m;
        }
        for (int k = a; k < nw && w[k].lo < hi; k++)
            if (w[k].hi > lo && w[k].lo != lo && w[k].fence > e->wr_fence) { gone[ng++] = e; break; }
    }
    for (int i = 0; i < ng; i++) {
        VmaAllocationInfo vi = {0};
        vmaGetAllocationInfo(c->vma, gone[i]->alloc, &vi);
        freed += vi.size;
        destroy_idle(gone[i]);
    }
    if (ng) {
        live_dropped_n += (uint64_t)ng, live_dropped_bytes += freed;
        static int logs;
        static uint64_t logged_gib;
        if (logs++ < 8 || (live_dropped_bytes >> 30) > logged_gib)
            rt_log("vk: live GPU-owned images written over by a later image and idle: %d dropped (%llu MiB), %llu in "
                   "all (%llu MiB)", ng, (unsigned long long)(freed >> 20), (unsigned long long)live_dropped_n,
                   (unsigned long long)(live_dropped_bytes >> 20)),
                logged_gib = live_dropped_bytes >> 30;
    }
    return ng;
}

void tex_gpu_made(uint64_t bytes) {
    gpu_made_since += bytes;
    if (gpu_made_since < (64ull << 20)) return;
    gpu_made_since = 0;
    lock();   // recursive: image creation may hold it already
    drop_overwritten_live();
    unlock();
}

void sweep_released(void) {
    for (Tex *r; nreleased > RELEASED_MAX && (r = rel_oldest_droppable());) destroy_idle(r);
    if (released_bytes <= released_cap()) return;
    for (Tex *e = rel_head, *next; e && released_bytes > released_cap(); e = next) {
        next = e->rel_next;
        if (recreatable(e) && !e->pin) destroy_idle(e);   // a pinned one is in use (tex_pin)
    }
    // The search walks every entry: only once the kept ones grew 64 MiB since the last one (a release at a time
    // otherwise, on the replay thread).
    // MKW_TEX_OVERWRITTEN=0 (a measurement): kept as before, for comparing a run with and without.
    static uint64_t searched_at;
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_OVERWRITTEN", "1") != 0;
    if (on && released_bytes > released_cap() && released_bytes >= searched_at + (64ull << 20)) {
        uint64_t n0 = overwritten_n, b0 = overwritten_bytes;
        drop_overwritten();
        searched_at = released_bytes;
        static int logs;            // the first 8, then at each GiB more in all (ab1: ~4 000 dropped in 7 minutes)
        static uint64_t logged_gib;
        if (overwritten_n > n0 && (logs++ < 8 || (overwritten_bytes >> 30) > logged_gib))
            rt_log("vk: released textures written over by a later image: %llu dropped (%llu MiB), %llu in all (%llu MiB)",
                   (unsigned long long)(overwritten_n - n0), (unsigned long long)((overwritten_bytes - b0) >> 20),
                   (unsigned long long)overwritten_n, (unsigned long long)(overwritten_bytes >> 20)),
            logged_gib = overwritten_bytes >> 30;
    }
    static uint64_t logged_at;   // what stays over the cap: released images the GPU wrote (render targets and such)
    if (released_bytes > released_cap() && released_bytes > logged_at + (256ull << 20)) {
        logged_at = released_bytes;
        rt_log("vk: released textures hold %llu MiB, over the %llu MiB cap: the rest are GPU-written, kept",
               (unsigned long long)(released_bytes >> 20), (unsigned long long)(released_cap() >> 20));
    }
}

