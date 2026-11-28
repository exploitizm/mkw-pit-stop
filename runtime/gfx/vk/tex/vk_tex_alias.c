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

// ---------------------------------------------------------------- aliases (MKW_TEX_ALIAS)
// Bytes per block and block size of the image's VkFormat (ASTC decodes to RGBA8, RGB expands to RGBA); 0 when the
// VkFormat substitutes another one of unknown size.
static int img_block(const Tex *e, int *bw, int *bh) {
    if (e->f.astc_bc7) { *bw = *bh = 4; return e->f.astc_bc7 == 2 ? 8 : 16; }   // re-encoded as BC7 or BC1 (vk_astc_bc7)
    if (e->f.astc_bw) { *bw = *bh = 1; return 4; }
    int b = nvn_fmt_bpb(e->t.d.format, bw, bh);
    if (e->f.expand_rgb) { *bw = *bh = 1; return b / 3 * 4; }
    return e->f.vk == e->f.ideal ? b : 0;
}

// The new entry e can take src's content texel for texel: colour, one sample, the same shape and storage layout, and
// image formats of one block size.
static int alias_compatible(const Tex *e, const Tex *src) {
    const VkTexImage *p = &e->pub, *q = &src->pub;
    if (e->f.depth || e->f.stencil || src->f.depth || src->f.stencil || p->samples > 1 || q->samples > 1) return 0;
    if (p->type != q->type || p->width != q->width || p->height != q->height || p->depth != q->depth) return 0;
    if (e->t.L.size != src->t.L.size || nvn_tex_is_linear(&e->t.d) != nvn_tex_is_linear(&src->t.d)) return 0;
    int ew, eh, sw, sh, eb = img_block(e, &ew, &eh), sb = img_block(src, &sw, &sh);
    return eb && eb == sb && ew == sw && eh == sh;
}

// Maxwell reads the same bytes through both descriptors: src's content goes into the new image e at the replay's
// current position, before any later command of the submission uses e. Through a buffer, since bound images share
// their memory and image copies may not overlap; e leaves UNDEFINED between the two halves, so its first transition
// cannot clobber src's bytes before they are read. 0 when no replay is recording (e is then not initialised yet).
static int alias_copy(Tex *e, const Tex *src) {
    int reopen;
    VkCommandBuffer cb = vk_exec_inline_begin(&reopen);
    if (!cb) return 0;
    if (e->pub.drop != src->pub.drop) {   // vk_tex_image.c tex_drop: the levels do not line up; not copied
        static volatile int32_t n_drop;
        if (plat_inc32(&n_drop) <= 8)
            rt_log("vk: alias copy %#llx: levels dropped %u and %u differ: not copied", (unsigned long long)e->pub.addr,
                   e->pub.drop, src->pub.drop);
        vk_exec_inline_end(reopen);
        return 1;
    }
    int bw, bh, bpb = img_block(e, &bw, &bh);
    uint32_t levels = e->pub.levels < src->pub.levels ? e->pub.levels : src->pub.levels;
    uint32_t layers = e->pub.layers < src->pub.layers ? e->pub.layers : src->pub.layers, n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for an alias copy");
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(e->pub.width, lv), h = mip(e->pub.height, lv), z = mip(e->pub.depth, lv);
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1},
                                        .imageExtent = {w, h, z}};
            total += ((VkDeviceSize)(w + bw - 1) / bw * ((h + bh - 1) / bh) * z * bpb + 15) & ~15ull;
        }
    VkBuffer buf = vk_exec_inline_buffer(total);
    if (!buf) {
        free(rg);
        vk_exec_inline_end(reopen);
        return 0;
    }
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, src->pub.image, VK_IMAGE_LAYOUT_GENERAL, buf, n, rg);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdCopyBufferToImage(cb, buf, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_exec_inline_end(reopen);
    free(rg);
    return 1;
}

// ---- MKW_REPLAY_PAR: alias copies at their place in the frame
// The parallel replay's walk makes each new render target's image as it goes (vk_exec_par.c plan, OP_SET_RT), on the
// replay thread, whose command buffer is not where the frame uses the target: with chunks already sent (the pipelined
// walk) an alias copy recorded there ran between the chunks on the GPU and the ones still recording, after the frame
// had cleared the old image and before it drew into it. On the PS5 (18/11/2026) the race's lit scene came out black
// for one frame, the HUD and late particles drawn over it, or a stretch of ground washed out: 55 frames in a race with
// MKW_REPLAY_PAR_PIPE=1, 1 (a false positive) with 0 (pipeC-on, pipeC-off; MKW_BLACK_CHECK), 0 with no chunk threads.
// Without the pipe the copy went before every chunk: last frame's picture, unseen. The walk now notes the copy (the
// source and the target's replay position) and the recording that reaches that position makes it there: the chunk
// that starts at or before it, or the serial replay when the submission stays serial.
static _Thread_local uint64_t alias_defer_at;
void vk_texture_alias_defer(uint64_t pos) { alias_defer_at = pos; }
// per window of the alias perf line: copies deferred, made at their place, dropped (the source gone, or the
// submission ended first)
uint64_t n_alias_deferred, n_alias_placed, n_alias_dropped;

static int entry_alive_at(uint64_t addr, const Tex *x) {
    for (Tex *p = nvn_hm_get(&by_addr, &addr); p; p = p->next_at)
        if (p == x) return !p->released || p->pub.image != VK_NULL_HANDLE;
    return 0;
}

// e's deferred copy, if this recording has reached its place (texture lock held)
static void alias_place(Tex *e) {
    if (e->alias_epoch != submit_epoch) {   // the submission ended without a recording reaching it
        e->alias_src = NULL, n_alias_dropped++;
        return;
    }
    uint64_t pos = vk_exec_op_pos(), pos0 = vk_exec_op_pos0();
    if (pos < e->alias_pos || pos0 >= e->alias_pos) return;   // before its place, or a later chunk: not this one
    Tex *src = e->alias_src;
    e->alias_src = NULL;
    if (!entry_alive_at(e->pub.addr, src)) { n_alias_dropped++; return; }
    if (alias_copy(e, src)) n_alias_placed++;
    else n_alias_dropped++;
}

// The entries at a storage address, chained by next_at (ui/glyphs/glyphs.c re-uploads all of them). Texture lock held.
Tex *entries_at(uint64_t addr) { return nvn_hm_get(&by_addr, &addr); }

Tex *entry_of(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e) {
        if (tex_alias < 0) {
            tex_alias = cfg_bool("MKW_TEX_ALIAS");
            rt_log("vk: texture aliases: %s", tex_alias ? "a new image takes the live image's content" : "a new image starts empty");
        }
        if (!(e = create_tex(t, &k, tex_alias))) return NULL;
        perf_count(PERF_CNT_TEX_NEW);
        // released entries of another descriptor at this address are replaced by the new one; a live one holding
        // GPU-written content is the alias source, preferring the one that differs only in swizzle (same_but_swizzle)
        int alias = 0;
        Tex *src = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr), *nx; p; p = nx) {
            nx = p->next_at;
            if (p->released) { destroy_idle(p); continue; }
            alias = 1;
            if (tex_alias && (!src || (same_but_swizzle(&p->key, &k) && !same_but_swizzle(&src->key, &k))) &&
                alias_compatible(e, p) && tex_has_gpu_content(p))
                src = p;
        }
        if (alias) perf_count(PERF_CNT_TEX_ALIAS);
        if (tex_alias) {
            int deferred = 0;
            if (src && alias_defer_at) {   // the replay's walk: copied where the frame reaches the target (above)
                e->alias_src = src, e->alias_pos = alias_defer_at, e->alias_epoch = submit_epoch;
                deferred = 1, n_alias_deferred++;
            }
            if (src && (deferred || alias_copy(e, src))) {
                n_alias_copied++;
                if (e->pub.mirrored) {   // the content is src's GPU render, newer than the storage bytes
                    vk_mem_changed(e->pub.addr, e->t.L.size, 0, &e->wgen);
                    e->uploaded = 1;
                    e->pub.gpu_owned = 1;
                    e->rhash_nup = UINT32_MAX;   // the image is src's now, not the last upload's (MKW_REUP_SAME)
                    gpu_baseline(e);   // only a CPU write after this copy makes the storage the content again
                }
            } else {
                init_image(e, vk_upload_begin());
                vk_upload_end();
                if (alias) {
                    if (e->pub.mirrored) n_alias_upload++;
                    else n_alias_uninit++;
                }
            }
        }
        e->next_at = nvn_hm_get(&by_addr, &k.addr);
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
    }
    if (e->released) e->released = 0, nreleased--, rel_unlink(e), tex_gen_bump(k.addr);
    e->t = *t;
    if (e->alias_src && !alias_defer_at) alias_place(e);
    return e;
}

// vk_dump: upload bookkeeping of an image vk_texture_get / vk_texture_by_id returned. pages_written: whether its
// storage pages were written since the last check (1, 0, -1 not tracked); the query leaves that check's state alone.
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written) {
    lock();
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint32_t now;
    *uploads = e->n_up;
    *last_change = e->last_ch;
    *pages_written = e->pub.mirrored && e->uploaded ? vk_mem_changed(e->pub.addr, e->t.L.size, e->wgen, &now) : 0;
    unlock();
}

const VkTexImage *vk_texture_get(const tex_t *t) {
    if (!t || !t->mem) return NULL;
    lock();
    Tex *e = entry_of(t);
    if (e) make_fresh(e);   // marks it seen before it may upload (vk_texture_trim)
    if (e && vk_guard_on()) vk_guard_note(&e->use);
    unlock();
    return e ? &e->pub : NULL;
}

// Small render targets and pitch textures the CPU reads without NVN's invalidate (vk_texture_write_back_small);
// small_pending: one may be dirty.
int small_pending;

// Without creating an entry (entry_of would: a capture asked about every registered texture and made images for
// thousands nobody used).
int vk_texture_capture_state(const tex_t *t, uint32_t recent) {
    if (!t || !t->mem) return 0;
    TexKey k;
    make_key(t, &k);
    lock();
    Tex *e = nvn_hm_get(&by_key, &k);
    int r = 0;
    if (e && !e->released && e->pub.image) {
        if (e->pub.gpu_owned) r |= VK_TEXCAP_GPU;
        if (vk_sparse_is(e->pub.image)) r |= VK_TEXCAP_SPARSE;
        if (e->checked && (uint32_t)(submit_epoch - e->checked) <= recent) r |= VK_TEXCAP_RECENT;
    }
    unlock();
    return r;
}

// e was written by the GPU in the submission being recorded (texture lock held): no re-upload from storage, and a
// mirrored texture is written back to it (small ones at the fence, wb_at_fence). fresh: bring the entry up to date with
// its storage first (make_fresh), as a partial write keeps what the storage held before.
static void gpu_written_locked(Tex *e, int fresh) {
    if (fresh) make_fresh(e);
    e->pub.gpu_owned = 1;
    e->gpu_frame = vk_exec_presented + 1;   // (rel_keep)
    uint64_t f = vk_exec_recording_fence();
    if (f > e->wr_fence) e->wr_fence = f;
    e->rb_fence = 0, e->rb_other = f;   // any write but a dispatch's: the readback no longer follows the image's last one
    if (!e->pub.mirrored) return;
    e->gpu_dirty = 1;
    gpu_baseline(e);   // CPU writes up to here are older than this one
    gpu_base_note(e);
    e->gpu_seq = ++gpu_writes;
    e->gpu_fence = f;
    if (wb_at_fence(&e->t.d, e->t.L.size)) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    own_add(e);
}

// A texture the GPU wrote is gpu_owned: its image is newer than the storage, so make_fresh never uploads it again. A
// plugin's native image (nvn2.c) is the other way round: the game renders its placeholder into its own textures over
// the image's pool once (1.8.0's GameChat camera textures, built at boot over each native image's pool, flags 0x40),
// then samples them while the plugin rewrites the pixels. The host's rewrite takes them back: no write-back of the
// GPU render, and the next check uploads the storage (whose pages vk_guest_written just stamped).
void vk_texture_host_rewrote(uint64_t addr) {
    static volatile int32_t logged;
    lock();
    for (Tex *p = entries_at(addr); p; p = p->next_at) {
        if (p->released || !p->pub.mirrored || !p->pub.gpu_owned) continue;
        p->pub.gpu_owned = 0;
        p->gpu_dirty = 0;
        if (plat_inc32(&logged) <= 16)
            rt_log("vk: texture at %#llx (%ux%u fmt %#x flags %#x) takes its storage again over its GPU render",
                   (unsigned long long)addr, p->t.d.width, p->t.d.height, p->t.d.format, p->t.d.flags);
    }
    unlock();
}

// ret: the caller's return address, for MKW_TRACE_ADDR.
static void gpu_written(const tex_t *t, int fresh, uintptr_t ret) {
    if (!t || !t->mem) return;
    if (tracing((uintptr_t)t->mem)) {   // with the replay function that wrote it (copies, clears, render targets)
        // the caller as an offset in the executable (addr2line -f -e build/mkw)
        uintptr_t off = ret - plat_exe_base();
        rt_log("vk: texture %#llx %ux%ux%u fmt %#x written on the GPU%s, caller build/mkw+%#llx",
               (unsigned long long)trace_addr, t->d.width, t->d.height, t->d.depth, t->d.format,
               fresh ? "" : " (copy recorded after its freshness check)", (unsigned long long)off);
    }
    vk_exec_frame_written(t);
    lock();
    Tex *e = entry_of(t);
    if (e) gpu_written_locked(e, fresh);
    unlock();
}

void vk_texture_compute_written(int id, VkCommandBuffer cb) {
    // A compute dispatch's storage-image store lands in the image only; the storage keeps what the CPU put there.
    // The game's water collision height map (a 256x256 pitch-linear R16G16B16A16_SNORM texture Eglantine publishes to
    // guest memory after its compute pass) is read back by the CPU for tricks and wave riding: without this mark the
    // write-back paths (owned_dirty) never see it and physics reads a flat road while the GPU shows waves (#410).
    // As vk_texture_gpu_written below, resolved by slot: the dispatch binds by handle, not by tex_t. The dispatch is
    // recorded into the replay's command buffer, so make_fresh's upload lands before it. cb: that command buffer, where
    // the image is copied for the write-back right after the dispatch (readback_record_locked).
    if (id < 0 || id >= (1 << 20)) return;
    lock();
    int perf = (rt_trace & RT_TRACE_PERF) != 0;
    uint64_t t0 = perf ? kern_now_ns() : 0, t1 = 0, t2 = 0, t3 = 0;
    view_t v;
    int hv = 0;
    const VkTexImage *pub = vk_texture_by_id_locked(id, &v, &hv);
    if (pub) {
        Tex *e = (Tex *)pub;   // pub is Tex's first member
        uint64_t other = e->rb_other;
        if (perf) t1 = kern_now_ns();
        gpu_written_locked(e, 1);
        if (perf) t2 = kern_now_ns();
        e->rb_other = other;   // a dispatch: not another kind of write
        readback_record_locked(e, cb);
        if (perf) t3 = kern_now_ns();
        // MKW_TRACE=perf: a hold of 2 ms or more, by part (the lock-holder log of 28/11/2026 found holds of 33 ms here
        // three times a race, the replay waiting behind them)
        if (perf && t3 - t0 >= 2000000ull) {
            static volatile int32_t logged;
            if (__atomic_add_fetch(&logged, 1, __ATOMIC_RELAXED) <= 100)
                rt_log("vk: compute_written held tex_lock %.1f ms: check %.1f, gpu_written (make_fresh, baseline) %.1f, "
                       "readback %.1f; texture %#llx %ux%u fmt %#x, %llu bytes, uploads %u, mirrored %d",
                       (t3 - t0) / 1e6, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6,
                       (unsigned long long)e->pub.addr, e->t.d.width, e->t.d.height, e->t.d.format,
                       (unsigned long long)e->t.L.size, e->n_up, e->pub.mirrored);
        }
    }
    unlock();
}

// A write recorded into the replay's command buffer: make_fresh's upload, if any, goes into the upload batch, which runs
// before that command buffer, so the storage's bytes land first and the write over them.
void vk_texture_gpu_written(const tex_t *t) { gpu_written(t, 1, (uintptr_t)__builtin_return_address(0)); }

// A write recorded into the upload batch (vk_texture_astc_copy's fallback outside a replay, vk_texture_region_copy):
// an upload from make_fresh now would go into the same batch after the copy and put the storage's bytes over it (a
// mirrored destination whose pages the game wrote since its last check, or whose upload was deferred; zero storage then
// decodes as the error colour). The caller looked the destination up with vk_texture_get, which made it fresh before
// the copy was recorded.
void vk_texture_gpu_copied(const tex_t *t) { gpu_written(t, 0, (uintptr_t)__builtin_return_address(0)); }

void vk_texture_release(const tex_t *t) {
    if (!t || !t->mem) return;
    TexKey k;
    make_key(t, &k);
    lock();
    Tex *e = nvn_hm_get(&by_key, &k);
    if (e && !e->released) {
        e->released = 1;
        nreleased++;
        rel_append(e);
        tex_gen_bump(k.addr);
        sweep_released();
    }
    unlock();
}

void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc) {
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && ((e->key.addr >= lo && e->key.addr - lo < size) || (alloc && e->bound_to == alloc))) list[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    unlock();
}

