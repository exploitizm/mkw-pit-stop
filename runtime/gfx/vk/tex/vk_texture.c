// Vulkan textures of the graphics layer: a VkImage cache keyed by storage
// GPU address plus descriptor, uploads through staging buffers with block-linear deswizzle and ASTC decode (vk_astc.c),
// re-uploads when page write generations (vk_memory.c) show the storage written, ReadTexels / WriteTexels, the texture
// and sampler descriptor slots, and VkSamplers. The cache state lives here; the parts, sharing vk_tex_internal.h:
// vk_tex_image.c (images, released entries), vk_tex_upload.c, vk_tex_alias.c (aliases, GPU writes into mirrored
// storage), vk_tex_slots.c (descriptor slots, samplers) and vk_tex_texel.c (ReadTexels / WriteTexels, CPU reads).
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn @ b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile): which textures mirror their pool memory (nvn_mtl.m tex_mirrored, 3347-3362), the
// "GPU-owned" rule that stops re-uploads once the GPU rendered into a mirrored texture (nvn_mtl.m tex_mtl, 3366-3372),
// the live texture at a storage address that a slot of a finalized texture follows (hle_nvn.c:373-393), and
// ReadTexels / WriteTexels (nvn_mtl.m:10105-10195). Image creation, upload regions and samplers are ported from LAS
// Recomp runtime/nvn/vulkan/textures.c:235-330, 778-844 @ da35b83a (MIT, Copyright (c) 2026 Peiikai).
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL. Uploads are recorded into the upload batch of vk_memory.c; the replay
// submits it (vk_upload_flush) before a submission that samples the textures.
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
#include "prof.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_tex_internal.h"
#include "platform/platform.h"

PlatRMutex tex_lock;
PlatOnce tex_once = PLAT_ONCE_INIT;
nvn_hmap by_key = {.klen = sizeof(TexKey)};
nvn_hmap by_addr = {.klen = 8};
TexSlot *tslots;
int ntslots;
SampSlot *sslots;
int nsslots;
nvn_hmap samplers = {.klen = sizeof(samp_par)};
int nreleased;
uint64_t released_bytes;
Tex *rel_head, *rel_tail;   // released entries in release order (sweep_released drops from the head)
uint64_t n_uploads, n_reuploads;
// Re-uploads since the last perf line (tex_lock): all, those after WriteTexels, and subresources skipped by MKW_TEX_REUP_PARTIAL.
uint64_t ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns, ru_skip_sub, ru_skip_bytes;
uint64_t tl_n, tl_ns, tl_max;       // lock() waits (MKW_TRACE=perf), vk_tex_internal.h
uint64_t n_entries_made;
volatile uint32_t tl_htid;           // the holder (vk_tex_internal.h tl_took)
volatile uint64_t tl_hra, tl_ht0;
const char *volatile tl_hname;
static _Thread_local int tl_depth;
static _Thread_local char tl_name[24];
__attribute__((noinline)) void tl_took(void) {
    if (tl_depth++) return;
    if (!tl_name[0] && !plat_sample_self_name(tl_name, sizeof tl_name)) snprintf(tl_name, sizeof tl_name, "?");
    tl_hname = tl_name, tl_htid = plat_thread_id(), tl_hra = (uint64_t)(uintptr_t)__builtin_return_address(0);
    tl_ht0 = plat_now_ns();
}
void tl_site(uint64_t ra) {
    if (tl_depth == 1) tl_hra = ra;
}
void tl_released(void) {
    if (tl_depth > 0 && !--tl_depth) tl_htid = 0;
}
void tl_long_wait(uint64_t dt, uint32_t htid, uint64_t hra, uint64_t ht0, const char *hname) {
    static volatile int32_t logged;
    if (__atomic_add_fetch(&logged, 1, __ATOMIC_RELAXED) > 400) return;
    char me[24] = "?", at[160] = "?", mine[160];
    plat_sample_self_name(me, sizeof me);
    if (hra) prof_pc_describe(hra, at, sizeof at);
    prof_pc_describe((uint64_t)(uintptr_t)__builtin_return_address(0), mine, sizeof mine);
    uint64_t now = plat_now_ns();
    rt_log("vk: tex lock: %s waited %.1f ms (frame %llu, from %s); held by %s tid %#x for %.1f ms, taken at %s", me,
           dt / 1e6, (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), mine,
           htid ? (hname ? hname : "?") : "(none noted)", htid, htid && ht0 && now > ht0 ? (now - ht0) / 1e6 : 0.0,
           htid ? at : "-");
}
volatile int tex_spin_on = 1;        // MKW_TEX_LOCK_SPIN (vk_tex_internal.h)
void vk_texture_spin_update(void) {   // per present (vk_backend.c), so "ab" follows the blocks
    static int mode = -1;
    if (mode < 0) {
        mode = rt_knob("TEX_LOCK_SPIN", "1");
        if (mode != 1) rt_log("vk: texture lock spins before sleeping%s (MKW_TEX_LOCK_SPIN)", mode ? " in the B blocks" : ": off");
    }
    int on = RT_KNOB_ON(mode);
    if (on != tex_spin_on) __atomic_store_n(&tex_spin_on, on, __ATOMIC_RELAXED);
}
uint64_t up_ph_ns[UPH_N], up_ph_n; // upload() phases (MKW_TRACE=perf)
UpSlow up_slow;
uint64_t rc_n, rc_bytes, rc_ns;   // copies out of CPU-written storage (vk_texture_region_copy), for the same perf line
// MKW_TEX_ALIAS (default on; =0: a new image over live storage starts empty): a new descriptor over the storage of a
// live image that the GPU wrote gets that image's content (alias_copy). Counts since the last log line.
int tex_alias = -1;
uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
uint32_t submit_epoch = 1;   // bumped per queue submit: draw bindings check a texture's storage once per submit
// Mirrored textures the GPU wrote into, sorted by address (vk_texture_cpu_stale, vk_texture_write_back). Changed under
// tex_lock and owned_lock; read under either.
PlatMutex owned_lock = PLAT_MUTEX_INIT;
Tex **owned;
int nowned, cowned;
uint64_t max_owned;   // largest storage size in owned[]
uint64_t gpu_writes;  // GPU writes into mirrored textures so far: Tex.gpu_seq
// Bumped whenever the entry a slot resolves to can change: an entry created or destroyed, released or taken back into
// use (the slot lookup and the follow of a finalized slot read only by_key, by_addr and the released flags).
uint32_t tex_gen = 1;
// Bumped with tex_gen and whenever a slot is registered: what vk_texture_by_id returns for any slot may have changed.
uint32_t slots_gen = 1;
uint32_t samp_gen = 1;   // bumped whenever a sampler slot is registered: vk_sampler_by_id may answer otherwise
uint64_t n_checks;    // draw-binding storage checks (make_fresh_snap) since the last perf line
// tex_gen per storage address (hashed): a slot resolves through the entries at its own address only (slot_resolve), so
// a slot's resolution stays valid while its address's generation does. TotK finalizes and initializes transient
// textures every frame: about 90 tex_gen and 160 slots_gen bumps per frame in the monsters fight, one every 16 draws,
// which emptied every cache keyed on the global generations.
uint32_t agen[AGEN_N];
// Per slot id, for vk_texture_slot_token without the lock: the registration serial (bits 16-47; 0: never registered)
// and the address generation index (bits 0-15), one word so a reader gets a consistent pair. Fixed-size (slot ids are
// 20 bits), unlike tslots, which moves when it grows.
uint64_t *sinfo;
// Per slot id, for vk_texture_checked without the lock: the submit_epoch of the slot's last freshness check (bits
// 32-63) and the agen its entry was resolved under (bits 0-31); 0 after a registration. Fixed-size like sinfo.
uint64_t *schk;
uint32_t reg_serial;

uint32_t agen_idx(uint64_t addr) { return (uint32_t)(addr * 0x9E3779B97F4A7C15ull >> 52); }

void tex_gen_bump(uint64_t addr) {
    if (!++tex_gen) tex_gen = 1;
    slots_gen++;
    uint32_t *g = &agen[agen_idx(addr)];
    __atomic_store_n(g, *g + 1, __ATOMIC_RELEASE);
}

volatile int32_t tex_ready;
void tex_init(void *p) {
    (void)p;
    __atomic_store_n(&tex_ready, 1, __ATOMIC_RELEASE);
    return;
}


// Flags that do not change where the texels are: two descriptors of one storage that differ only in them are one image.
// MKW writes from compute through an IMAGE descriptor and samples the same storage through another without the flag;
// as two images, the readers never saw the writes (the race's wave simulation and 3D LUTs, run-race33, 31/10/2026).
#define KEY_FREE_FLAGS (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL | \
                        NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL)

void make_key(const tex_t *t, TexKey *k) {
    memset(k, 0, sizeof *k);
    k->addr = (uintptr_t)t->mem;
    k->d = t->d;
    k->d.flags &= ~KEY_FREE_FLAGS;
    k->has_layout = t->has_layout;
    memcpy(k->layout, t->layout, 8);
}

// Keys equal apart from the descriptor's swizzle selectors: the same storage, format and geometry.
int same_but_swizzle(const TexKey *a, const TexKey *b) {
    TexKey x = *a, y = *b;
    x.d.swizzle_r = x.d.swizzle_g = x.d.swizzle_b = x.d.swizzle_a = 0;
    y.d.swizzle_r = y.d.swizzle_g = y.d.swizzle_b = y.d.swizzle_a = 0;
    return !memcmp(&x, &y, sizeof x);
}

// MNG nvn_mtl.m tex_mirrored: textures whose content the CPU provides in pool memory. Display, compressible, zcull,
// depth, multisample and GPU-only storage stays GPU-owned.
int tex_mirrored(const tex_t *t) {
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem)) return 0;
    if (t->has_layout) return 1;
    const pool_t *p = t->pool;
    if (!p || !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL |
                      NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL))
        return 0;
    if (t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;   // stencil / depth formats
    return 1;
}

void log_limited(volatile int32_t *n, const char *fmt, ...) {
    if (plat_inc32(n) > 16) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    rt_log("vk: %s", buf);
}

void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
// ---------------------------------------------------------------- statistics and shutdown
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads) {
    lock();
    if (images) *images = by_key.n;
    if (uploads) *uploads = n_uploads;
    if (reuploads) *reuploads = n_reuploads;
    unlock();
}

void vk_texture_census(VkTexCensus *s) {
    VkCtx *c = vk_ctx();
    uint32_t idle = tex_evict_idle();
    memset(s, 0, sizeof *s);
    lock();
    s->epoch = submit_epoch;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
            if (ai.memoryType < c->mem_props.memoryTypeCount &&
                !(c->mem_props.memoryTypes[ai.memoryType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                s->host_n++, s->host_bytes += bytes;
        }
        if (e->released) s->released++, s->own_released += e->alloc != NULL, s->bytes_released += bytes;
        else {
            s->live++, s->own_live += e->alloc != NULL, s->bytes_live += bytes;
            int fmt = e->t.d.format;
            if (fmt >= 0x79 && fmt <= 0x94) {
                s->astc_bytes += bytes, s->astc_n += bytes != 0;
                if (!e->f.astc_bc7) {
                    s->astc_rgba_bytes += bytes;
                    if (e->pub.layers > 1 || e->pub.depth > 1) s->astc_rgba_arr_bytes += bytes;
                }
            } else {
                if (e->pub.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
                    s->target_bytes += bytes, s->target_n += bytes != 0;
                if (e->pub.gpu_owned)
                    s->gpu_owned_bytes += bytes, s->gpu_owned_n += bytes != 0;
                else if (e->pub.mirrored)
                    s->mirrored_ro_bytes += bytes, s->mirrored_ro_n += bytes != 0;
                else
                    s->other_gpu_bytes += bytes, s->other_gpu_n += bytes != 0;
            }
            if (bytes) {
                if (e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty &&
                    (uint32_t)(submit_epoch - e->seen) > idle)
                    s->idle_evict_n++, s->idle_evict_bytes += bytes;
                const Tex *head = NULL;
                for (const Tex *p = nvn_hm_get(&by_addr, &e->key.addr); p; p = p->next_at)
                    if (!p->released && p->alloc) { head = p; break; }
                if (head && head != e) {
                    if (same_but_swizzle(&head->key, &e->key)) s->dup_swz_n++, s->dup_swz_bytes += bytes;
                    else s->dup_other_n++, s->dup_other_bytes += bytes;
                }
            }
            if (bytes && !(e->pub.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
                VkFormat vf = e->f.vk;
                int k = vf >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && vf <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK ? 0
                        : vf == VK_FORMAT_BC7_UNORM_BLOCK || vf == VK_FORMAT_BC7_SRGB_BLOCK ? 1
                        : vf >= VK_FORMAT_BC2_UNORM_BLOCK && vf <= VK_FORMAT_BC6H_SFLOAT_BLOCK ? 2 : 3;
                s->cls_bytes[k] += bytes, s->cls_n[k]++;
                uint32_t side = e->t.d.width > e->t.d.height ? e->t.d.width : e->t.d.height;
                int z = side < 512 ? 0 : side < 1024 ? 1 : 2;
                s->size_bytes[z] += bytes, s->size_n[z]++;
                uint32_t unseen = submit_epoch - e->seen;
                for (int j = 0; j < 3; j++)
                    if (unseen > (64u << (3 * j))) s->idle_bytes[j] += bytes, s->idle_n[j]++;
            }
            if (bytes > (64ull << 20)) s->big_bytes += bytes;
        }
    }
    unlock();
}

// Issue 11 (diagnosis): GPU-owned images by shape. 1.0.0's third race (Race_Coaster_A_00) died with GPU-owned textures
// at 1569 MiB against 870-905 MiB the rest of the Grand Prix, while 34 GiB of released images were written over in
// 1500 frames (4 GiB per 3000 frames on 1.8.0, same course). Which shapes those are and how fast they are made: own
// images made by shape since the last line (tex_shape_made, at creation) beside the live GPU-owned ones, logged when
// the live GPU-owned bytes pass their high mark by 128 MiB (MKW_TRACE=perf) and at the OOM census.
#define SHAPES 64
typedef struct { int32_t fmt; uint32_t w, h, layers, levels; uint64_t made_n, made_b, n, b, shared; } Shape;
static Shape shapes[SHAPES];
static int nshapes;
static PlatMutex shape_lock = PLAT_MUTEX_INIT;
static uint64_t shape_high;

static Shape *shape_of(int32_t fmt, uint32_t w, uint32_t h, uint32_t layers, uint32_t levels) {
    for (int i = 0; i < nshapes; i++)
        if (shapes[i].fmt == fmt && shapes[i].w == w && shapes[i].h == h && shapes[i].layers == layers &&
            shapes[i].levels == levels)
            return &shapes[i];
    if (nshapes == SHAPES) {   // the smallest made-since one goes
        int k = 0;
        for (int i = 1; i < SHAPES; i++) if (shapes[i].made_b < shapes[k].made_b) k = i;
        shapes[k] = shapes[--nshapes];
    }
    Shape *s = &shapes[nshapes++];
    memset(s, 0, sizeof *s);
    s->fmt = fmt, s->w = w, s->h = h, s->layers = layers, s->levels = levels;
    return s;
}

void tex_shape_made(int32_t fmt, uint32_t w, uint32_t h, uint32_t layers, uint32_t levels, uint64_t bytes) {
    plat_mutex_lock(&shape_lock);
    Shape *s = shape_of(fmt, w, h, layers, levels);
    s->made_n++, s->made_b += bytes;
    plat_mutex_unlock(&shape_lock);
}

static int by_live(const void *a, const void *b) {
    uint64_t x = ((const Shape *)a)->b + ((const Shape *)a)->made_b, y = ((const Shape *)b)->b + ((const Shape *)b)->made_b;
    return (x < y) - (x > y);
}

void vk_texture_gpu_shapes(const char *why, uint64_t gpu_owned_bytes) {
    if (gpu_owned_bytes && gpu_owned_bytes < shape_high + (128ull << 20)) return;
    if (gpu_owned_bytes > shape_high) shape_high = gpu_owned_bytes;
    VkCtx *c = vk_ctx();
    lock();   // before shape_lock: image creation holds the texture lock when it calls tex_shape_made
    plat_mutex_lock(&shape_lock);
    for (int i = 0; i < nshapes; i++) shapes[i].n = shapes[i].b = shapes[i].shared = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || e->released || !e->alloc || !e->pub.gpu_owned || !c) continue;
        VmaAllocationInfo ai;
        vmaGetAllocationInfo(c->vma, e->alloc, &ai);
        Shape *s = shape_of(e->t.d.format, e->pub.width, e->pub.height, e->pub.layers, e->pub.levels);
        s->n++, s->b += ai.size;
        int at = 0;
        for (const Tex *p = nvn_hm_get(&by_addr, &e->key.addr); p && at < 2; p = p->next_at) at++;
        s->shared += at > 1;
    }
    Shape top[SHAPES];
    int n = nshapes;
    memcpy(top, shapes, sizeof(Shape) * (size_t)n);
    for (int i = 0; i < nshapes; i++) shapes[i].made_n = shapes[i].made_b = 0;
    plat_mutex_unlock(&shape_lock);
    unlock();
    qsort(top, (size_t)n, sizeof *top, by_live);
    rt_log("mem: gpu-owned by shape (%s, high %.0f MiB): live n/MiB (sharing an address), made since the last line n/MiB",
           why, shape_high / 1048576.0);
    for (int i = 0; i < n && i < 14; i++)
        if (top[i].b || top[i].made_b)
            rt_log("mem:   fmt %#x %ux%u layers %u levels %u: live %llu %.1f MiB (%llu shared), made %llu %.1f MiB",
                   top[i].fmt, top[i].w, top[i].h, top[i].layers, top[i].levels, (unsigned long long)top[i].n,
                   top[i].b / 1048576.0, (unsigned long long)top[i].shared, (unsigned long long)top[i].made_n,
                   top[i].made_b / 1048576.0);
}

void vk_texture_range_census(uint64_t lo, uint64_t hi, VkTexRange *s) {
    VkCtx *c = vk_ctx();
    uint32_t idle = tex_evict_idle();
    memset(s, 0, sizeof *s);
    lock();
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || e->key.addr < lo || e->key.addr >= hi) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
        }
        nvn_texlayout L;
        nvn_tex_layout(&e->key.d, &L);
        // Storage counted once per address: the entry the address chain starts with.
        if (nvn_hm_get(&by_addr, &e->key.addr) == e) {
            s->n++, s->storage += L.size;
            s->backed += plat_mem_backed_bytes(e->key.addr, L.size);
        }
        if (e->released) s->released++;
        else s->live++;
        if (!e->released && (uint32_t)(submit_epoch - e->seen) > idle) s->idle++;
        if (!e->uploaded) s->never_up++;
        if (e->sparse) s->sparse_n++;
        s->vk_bytes += bytes;
        int fmt = e->key.d.format;
        if (fmt >= 0x79 && fmt <= 0x94) {
            if (e->f.astc_bc7) s->astc_bc7_n++, s->astc_bc7_vk += bytes, s->astc_bc7_storage += L.size;
            else s->astc_rgba_n++, s->astc_rgba_vk += bytes, s->astc_rgba_storage += L.size;
        } else s->other_n++, s->other_vk += bytes, s->other_storage += L.size;
    }
    unlock();
}

void vk_texture_shutdown(void) {
    VkCtx *c = vk_ctx();
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) list[n++] = by_key.vals[i];
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    tex_evict_shutdown();
    for (uint32_t i = 0; i < samplers.cap; i++)
        if (samplers.vals[i] && c) vkDestroySampler(c->device, (VkSampler)samplers.vals[i], NULL);
    free(by_key.keys), free(by_key.vals), free(by_addr.keys), free(by_addr.vals), free(samplers.keys), free(samplers.vals);
    by_key = (nvn_hmap){.klen = sizeof(TexKey)};
    by_addr = (nvn_hmap){.klen = 8};
    samplers = (nvn_hmap){.klen = sizeof(samp_par)};
    free(tslots), free(sslots);
    tslots = NULL, sslots = NULL, ntslots = nsslots = 0, nreleased = 0, released_bytes = 0;
    rel_head = rel_tail = NULL;
    if (sinfo) memset(sinfo, 0, sizeof *sinfo << 20);   // no slot is registered: tokens 0
    if (schk) memset(schk, 0, sizeof *schk << 20);
    unlock();
}
