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
#include "ui/glyphs/glyphs.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- uploads
static uint32_t texel_alpha_one(int format, uint32_t *elem) {
    switch (format) {
    case 0x19: *elem = 1; return 0xFF;
    case 0x1A: *elem = 1; return 0x7F;
    case 0x1B: case 0x1C: *elem = 1; return 1;
    case 0x1D: *elem = 2; return 0x3C00;
    case 0x1E: *elem = 2; return 0xFFFF;
    case 0x1F: *elem = 2; return 0x7FFF;
    case 0x20: case 0x21: *elem = 2; return 1;
    case 0x22: *elem = 4; return 0x3F800000;
    default: *elem = 4; return 1;   // 0x23, 0x24
    }
}

// 3-component texels to 4 (n texels).
void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e, one = texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 3 * e, dst += 4 * e) {
        memcpy(dst, src, 3 * e);
        memcpy(dst + 3 * e, &one, e);
    }
}

void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e;
    texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 4 * e, dst += 3 * e) memcpy(dst, src, 3 * e);
}

// MKW_ASTC_BC_CHECK=N (diagnosis, task 40; N > 0 turns it on): each ASTC upload re-encoded as BC7 or BC1 (2D, arrays and cube maps, up to
// 64 MiB of texels) is read back through a blit to RGBA8 and compared with astc_decode_ldr of its blocks: what sampling
// the BC image gives against what the RGBA8 image gave. Sums for arrays and for single images apart (the arrays went to
// BC with their layers counted), and the first N textures one line each. Waits for the GPU at every upload.
static void bc_check(const Tex *e, const uint8_t *raw, const VkAstcRegion *ar, int n, int lines) {
    static volatile int32_t n_logged;
    static struct { uint64_t tex, texels, sum, worst_tex; int worst; double worst_psnr; } cls[2];   // [array]
    static PlatMutex lock = PLAT_MUTEX_INIT;
    VkCtx *c = vk_ctx();
    uint64_t texels = 0;
    for (int k = 0; k < n; k++) texels += (uint64_t)ar[k].w * ar[k].h * ar[k].slices;
    if (e->pub.type != VK_IMAGE_TYPE_2D || texels * 4 > (64ull << 20)) return;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = e->f.astc_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
                            .extent = {e->pub.width, e->pub.height, 1}, .mipLevels = e->pub.levels,
                            .arrayLayers = e->pub.layers, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage img;
    VmaAllocation alloc;
    if (vmaCreateImage(c->vma, &ci, &ai, &img, &alloc, NULL) != VK_SUCCESS) return;
    VkBuffer hb;
    VmaAllocation ha;
    VkCommandBuffer cb = vk_upload_begin();
    uint8_t *map = vk_staging(texels * 4, &hb, &ha);
    VkBufferImageCopy *cp = calloc((size_t)n, sizeof *cp);
    if (!cb || !map || !cp) rt_fatal("vk: MKW_ASTC_BC_CHECK: no staging");
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                                                   VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    barrier(cb, ALL_RW, ALL_RW);
    VkDeviceSize off = 0;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, r->level, r->layer, 1},
                          .srcOffsets = {{0, 0, 0}, {(int32_t)r->w, (int32_t)r->h, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, r->level, r->layer, 1},
                          .dstOffsets = {{0, 0, 0}, {(int32_t)r->w, (int32_t)r->h, 1}}};
        vkCmdBlitImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
        cp[k] = (VkBufferImageCopy){.bufferOffset = off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, r->level, r->layer, 1},
                                    .imageExtent = {r->w, r->h, 1}};
        off += (VkDeviceSize)r->w * r->h * 4;
    }
    barrier(cb, ALL_RW, ALL_RW);
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, hb, (uint32_t)n, cp);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    vk_upload_end();
    vk_upload_flush(1);
    vmaInvalidateAllocation(c->vma, ha, 0, VK_WHOLE_SIZE);
    uint32_t bw = e->f.astc_bw, bh = e->f.astc_bh;
    uint64_t sum = 0, sq = 0;
    int worst = 0;
    const uint8_t *got = map;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        size_t st = (size_t)r->w * r->h * 4;
        uint8_t *want = malloc(st);
        astc_decode_ldr(raw + r->src, r->w, r->h, bw, bh, e->f.astc_srgb, want);
        for (size_t i = 0; i < st; i++) {
            int d = abs((int)want[i] - (int)got[i]);
            sum += (uint64_t)d, sq += (uint64_t)d * (uint64_t)d;
            if (d > worst) worst = d;
        }
        free(want);
        got += st;
    }
    double mse = (double)sq / (double)(texels * 4), psnr = mse > 0 ? 10 * log10(255.0 * 255.0 / mse) : 99.0;
    int array = e->pub.layers > 1;
    plat_mutex_lock(&lock);
    cls[array].tex++, cls[array].texels += texels, cls[array].sum += sum;
    if (psnr < cls[array].worst_psnr || !cls[array].worst_tex)
        cls[array].worst_psnr = psnr, cls[array].worst = worst, cls[array].worst_tex = e->pub.addr;
    uint64_t all = cls[0].tex + cls[1].tex;
    if (plat_inc32(&n_logged) <= lines)
        rt_log("vk: ASTC BC check: %#llx %ux%u x%u layers, %u levels, ASTC %ux%u -> %s: mean error %.3f, largest %d, PSNR "
               "%.1f dB", (unsigned long long)e->pub.addr, e->pub.width, e->pub.height, e->pub.layers, e->pub.levels, bw, bh,
               e->f.astc_bc7 == 2 ? "BC1" : "BC7", (double)sum / (double)(texels * 4), worst, psnr);
    if (all % 32 == 0 || all < 4)
        for (int a = 0; a < 2; a++)
            if (cls[a].tex)
                rt_log("vk: ASTC BC check: %s: %llu textures, %.1f M texels, mean error %.3f per channel; worst PSNR %.1f dB "
                       "(%#llx, largest error %d)", a ? "arrays and cube maps" : "single images",
                       (unsigned long long)cls[a].tex, (double)cls[a].texels / 1e6,
                       (double)cls[a].sum / (double)(cls[a].texels * 4), cls[a].worst_psnr,
                       (unsigned long long)cls[a].worst_tex, cls[a].worst);
    plat_mutex_unlock(&lock);
    free(cp);
    vmaDestroyBuffer(c->vma, hb, ha);
    vk_defer_image(img, alloc);
}

// MKW_ASTC=check: the compute decoder's texels of n regions (gpu: the regions one after the other) against
// astc_decode_ldr of the same raw blocks (raw: the staging bytes the regions point into).
static void astc_check(const Tex *e, const uint8_t *raw, const VkAstcRegion *ar, int n, const uint8_t *gpu) {
    static volatile int32_t n_bad;
    static uint64_t n_tex, n_texels, n_diff_tex;
    uint32_t bw = e->f.astc_bw, bh = e->f.astc_bh;
    uint64_t diff = 0, first = UINT64_MAX;
    uint32_t want = 0, got = 0;
    int first_k = 0;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        size_t sb = (size_t)((r->w + bw - 1) / bw) * ((r->h + bh - 1) / bh) * 16, st = (size_t)r->w * r->h * 4;
        uint8_t *cpu = malloc(st);
        for (uint32_t z = 0; z < r->slices; z++, gpu += st) {
            astc_decode_ldr(raw + r->src + z * sb, r->w, r->h, bw, bh, e->f.astc_srgb, cpu);
            if (!memcmp(cpu, gpu, st)) continue;
            for (size_t i = 0; i < st / 4; i++)
                if (memcmp(cpu + 4 * i, gpu + 4 * i, 4)) {
                    if (first == UINT64_MAX) {
                        first = i, first_k = k;
                        memcpy(&want, cpu + 4 * i, 4), memcpy(&got, gpu + 4 * i, 4);
                    }
                    diff++;
                }
        }
        free(cpu);
        n_texels += (uint64_t)r->w * r->h * r->slices;
    }
    n_tex++;
    if (diff) {
        n_diff_tex++;
        log_limited(&n_bad, "ASTC check: texture %#llx fmt %#x %ux%u: %llu texels differ, first region %d texel %llu "
                    "cpu %08x gpu %08x", (unsigned long long)e->pub.addr, e->t.d.format, bw, bh,
                    (unsigned long long)diff, first_k, (unsigned long long)first, want, got);
    }
    if (n_tex == 1 || !(n_tex & 63) || diff)
        rt_log("vk: ASTC check: %llu textures, %llu texels compared, %llu textures differ", (unsigned long long)n_tex,
               (unsigned long long)n_texels, (unsigned long long)n_diff_tex);
}

// An ASTC texture re-encoded as BC7: the BC7 encoded offline for these storage bytes (vk_texpre.c) straight into the
// image. Without one, the storage is recorded for the offline encoder and 0 sends the upload on through the GPU encoder.
static int upload_texpre(Tex *e, uint32_t levels, uint32_t layers) {
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    int is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    uint32_t n = levels * layers, k = 0;
    VkTexpreRegion *pr = calloc(n, sizeof *pr);
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
            pr[k] = (VkTexpreRegion){w, h, sl, lv, ly};
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, lv, ly, 1},
                                        .imageExtent = {w, h, sl}};
            total += vk_texpre_region_bytes(&pr[k]);
        }
    uint64_t h = vk_texpre_hash(t->mem, t->L.size);
    int ok = 0;
    if (vk_texpre_has(h)) {
        VkBuffer sb;
        uint8_t *stg;
        VkCommandBuffer cb = vk_upload_begin_staging(total, &sb, (void **)&stg);
        if (cb) {   // no staging: upload() tries its own, smaller or deferred
            if (vk_texpre_load(h, stg, total)) {
                barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
                barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
                ok = 1;
            }
            vk_upload_end();
        }
    }
    if (!ok) vk_texpre_record(h, d, pr, (int)n, t->mem, t->L.size);
    free(pr), free(rg);
    return ok;
}

// Copies the whole storage into the image (every level and layer the image has).
// MKW_TRACE_ADDR=<hex> (diagnosis): the texture whose storage starts there: its GPU writes (vk_texture_gpu_written),
// uploads and WriteTexels, the first 64 events.
uint64_t trace_addr = 1;
static volatile int32_t traced;
// MKW_TRACE_ADDR=all (diagnosis): every texture, up to 20 000 lines: two replays of a capture told apart by their
// uploads and GPU writes (tools/test: a recording order that changes the image).
int tracing(uint64_t addr) {
    if (trace_addr == 1) {
        const char *e = cfg_get("MKW_TRACE_ADDR");
        trace_addr = e && !strcmp(e, "all") ? UINT64_MAX : e && *e ? strtoull(e, NULL, 16) : 0;
    }
    if (trace_addr == UINT64_MAX) return plat_inc32(&traced) <= 20000;
    return trace_addr && addr == trace_addr && plat_inc32(&traced) <= 64;
}

// The button icon sheet (ui/glyphs/glyphs.c): its texture object and the icons it was drawn with. When they change it is
// uploaded again, redrawn, at the next present (vk_backend.c, the game thread, after the queued replays). Waiting for its next storage check was not enough: after the first
// upload the binding memos never check it again while its pages stay clean (gl4, gl5, 06/11/2026).
static uint64_t g_glyph_addr;
static int g_glyph_known;
static uint32_t g_glyph_gen;
static int upload(Tex *e, uint32_t partial_since);

int vk_glyphs_due(void) {
    if (!__atomic_load_n(&g_glyph_known, __ATOMIC_ACQUIRE)) return 0;
    static uint64_t last_ns;
    uint64_t now = plat_now_ns();
    if (now - last_ns < 100000000ull) return 0;   // ten times a second: a device switch shows within a few frames
    last_ns = now;
    return glyphs_generation() != g_glyph_gen;
}

// Every live entry at the sheet's address: the game samples it through more than one (another view of the same
// storage); uploading the one entry_of found again changed nothing on screen (gl6, 06/11/2026).
// All of them pinned first: an upload that runs short of staging trims idle textures (vk_texture_trim), and the walk
// goes on through next_at.
void vk_glyphs_redo(void) {
    lock();
    g_glyph_gen = glyphs_generation();
    for (Tex *e = entries_at(g_glyph_addr); e; e = e->next_at) tex_pin(e);
    for (Tex *e = entries_at(g_glyph_addr); e; e = e->next_at)
        if (e->uploaded && !e->released && e->pub.mirrored && !e->pub.gpu_owned) upload(e, 0);
    for (Tex *e = entries_at(g_glyph_addr); e; e = e->next_at) tex_unpin(e);
    unlock();
}

// MKW_TEX_UP_LOG's content checks (diagnosis, task 32: magenta bands on the character-select cards, right of a step in
// every card, the shape of a byte range of a block-linear texture; ASTC blocks that are not valid 4x4 blocks decode to
// magenta). The cards (275x362 ASTC 4x4, 144 KiB, CPU-written storage) were uploaded up to 6 times in 12 frames as the
// game wrote them, never with a zero tail. Each ASTC upload of 64 KiB or more keeps a hash per 4 KiB page of what it
// read (the latest upload of an address replaces the earlier one); tex_up_recheck hashes the pages again 300 frames
// later: a page changed with no upload since is a write the runtime missed, and its offset places it on the card.
#define ZT_MAX 256
#define ZT_PAGES 64
static struct { uint64_t addr, size, ztail, frame; uint32_t n_up; int checked; uint64_t h[ZT_PAGES]; } zt[ZT_MAX];
static int nzt;
static PlatMutex zt_lock = PLAT_MUTEX_INIT;

// Offset of the first byte of [addr, addr + size)'s trailing run of zero bytes (size when the last byte is not zero).
static uint64_t zero_tail(uint64_t addr, uint64_t size) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)addr;
    uint64_t n = size;
    while (n && !p[n - 1]) n--;
    return n;
}

static void page_hashes(uint64_t addr, uint64_t size, uint64_t *h) {   // FNV-1a over each 4 KiB page of the range
    const uint8_t *p = (const uint8_t *)(uintptr_t)addr;
    for (uint64_t k = 0; k < ZT_PAGES && k * 4096 < size; k++) {
        uint64_t x = 0xcbf29ce484222325ull, end = (k + 1) * 4096 < size ? (k + 1) * 4096 : size;
        for (uint64_t i = k * 4096; i < end; i++) x = (x ^ p[i]) * 0x100000001b3ull;
        h[k] = x;
    }
}

static void zero_tail_note(uint64_t addr, uint64_t size, uint64_t ztail, uint64_t frame, uint32_t n_up) {
    if (size > (uint64_t)ZT_PAGES * 4096) size = (uint64_t)ZT_PAGES * 4096;
    plat_mutex_lock(&zt_lock);
    int i = 0;
    while (i < nzt && zt[i].addr != addr) i++;
    if (i == nzt && nzt == ZT_MAX)   // full: an entry already checked makes room (the menus' filled it before the cards)
        for (i = 0; i < nzt && !zt[i].checked; i++) {}
    if (i == nzt && nzt < ZT_MAX) nzt++;
    if (i < nzt) {
        zt[i].addr = addr, zt[i].size = size, zt[i].ztail = ztail, zt[i].frame = frame, zt[i].n_up = n_up;
        zt[i].checked = 0;
        page_hashes(addr, size, zt[i].h);
    }
    plat_mutex_unlock(&zt_lock);
}

// From the texture lock's perf report (vk_tex_slots.c), once a minute: each noted upload 300 frames or more old, read
// again if its storage is still backed.
void tex_zero_tail_recheck(uint64_t frame) {
    plat_mutex_lock(&zt_lock);
    for (int i = 0; i < nzt; i++) {
        if (zt[i].checked || frame < zt[i].frame + 300) continue;
        zt[i].checked = 1;
        if (plat_mem_backed_bytes(zt[i].addr, zt[i].size) < zt[i].size) {
            rt_log("vk: upload check %#llx (upload %u at frame %llu): storage no longer backed, not read",
                   (unsigned long long)zt[i].addr, zt[i].n_up, (unsigned long long)zt[i].frame);
            continue;
        }
        uint64_t h[ZT_PAGES];
        page_hashes(zt[i].addr, zt[i].size, h);
        uint64_t npg = (zt[i].size + 4095) / 4096, changed = 0, first = npg, last = 0;
        for (uint64_t k = 0; k < npg; k++)
            if (h[k] != zt[i].h[k]) changed++, first = k < first ? k : first, last = k;
        rt_log("vk: upload check %#llx (%.1f KiB, upload %u at frame %llu, zero tail %#llx) at frame %llu: %s; %llu of "
               "%llu pages changed (offsets %#llx-%#llx)",
               (unsigned long long)zt[i].addr, zt[i].size / 1024.0, zt[i].n_up, (unsigned long long)zt[i].frame,
               (unsigned long long)zt[i].ztail, (unsigned long long)frame,
               changed ? "CHANGED with no upload since" : "unchanged", (unsigned long long)changed,
               (unsigned long long)npg, (unsigned long long)(changed ? first * 4096 : 0),
               (unsigned long long)(changed ? (last + 1) * 4096 : 0));
    }
    plat_mutex_unlock(&zt_lock);
}

// Row-band partial re-upload for large single-level 2D color textures (MKW_TEX_REUP_BANDS, default 1, supports "ab").
// Targets dynamic atlases (e.g. MKW's 1024x8192 R8 font/atlas at 8 MiB, re-uploaded 25+ times per 240 s race at
// 12.5-18.5 ms each) by querying per-page write generations per contiguous row-band (linear) or block-row (block-linear)
// and uploading only the dirty strips. Returns 1 handled, 0 staging OOM (deferred), -1 fall back to full upload().
static int upload_bands(Tex *e, uint32_t partial_since, int bw, int bh, int bpb, int linear, uint64_t p0, uint64_t u0,
                        uint64_t frame_now) {
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    uint32_t drop = e->pub.drop;
    uint32_t w = mip(d->width, drop), h = mip(d->height, drop);
    uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
    if (w == 1024 && h == 1024 && (d->format == 0x25 || d->format == 0x38)) return -1;   // button glyph sheet

    uint64_t offs_stk[256], spans_stk[256];
    uint8_t dirty_stk[256];
    uint32_t nb = 0, band_rows = 0;
    unsigned th = 0, td = 0;
    uint64_t lv_off = 0, row_blk_bytes = 0;
    uint32_t stride = 0;

    if (linear) {
        if (drop || (d->target != 1 && d->target != 7)) return -1;
        stride = nvn_tex_linear_stride(d);
        if (stride < rb || (uint64_t)rows * stride > t->L.size) return -1;
        band_rows = (uint32_t)((65536u + stride - 1u) / stride);
        if (band_rows < 1) band_rows = 1;
        nb = (rows + band_rows - 1u) / band_rows;
        if (nb < 2) return -1;
    } else {
        if ((rb & 3u) || !vk_bl_deswizzle_ready()) return -1;
        nvn_tex_level_tiles(d, drop, &th, &td);
        if (td != 0) return -1;
        lv_off = nvn_tex_level_offset(d, drop);
        band_rows = 8u << th;
        uint64_t gobs_x = (rb + 63) / 64;
        nb = (rows + band_rows - 1u) / band_rows;
        row_blk_bytes = gobs_x * (512ull << th);
        if (nb < 2 || lv_off + (uint64_t)nb * row_blk_bytes > t->L.size) return -1;
    }

    uint64_t *offs = nb <= 256 ? offs_stk : malloc((size_t)nb * sizeof *offs);
    uint64_t *spans = nb <= 256 ? spans_stk : malloc((size_t)nb * sizeof *spans);
    uint8_t *dirty = nb <= 256 ? dirty_stk : malloc((size_t)nb);
    if (!offs || !spans || !dirty) {
        if (offs != offs_stk) free(offs);
        if (spans != spans_stk) free(spans);
        if (dirty != dirty_stk) free(dirty);
        return -1;
    }
    if (linear) {
        for (uint32_t b = 0; b < nb; b++) {
            uint32_t y0 = b * band_rows, ny = rows - y0 < band_rows ? rows - y0 : band_rows;
            offs[b] = (uint64_t)y0 * stride;
            spans[b] = (uint64_t)ny * stride;
        }
    } else {
        for (uint32_t b = 0; b < nb; b++) {
            offs[b] = lv_off + (uint64_t)b * row_blk_bytes;
            spans[b] = row_blk_bytes;
        }
    }
    vk_mem_ranges_written_since(e->pub.addr, offs, spans, dirty, nb, partial_since);
    if (offs != offs_stk) free(offs);
    if (spans != spans_stk) free(spans);

    uint32_t nd = 0, nruns = 0;
    for (uint32_t b = 0; b < nb; b++) {
        if (dirty[b]) {
            nd++;
            if (b == 0 || !dirty[b - 1]) nruns++;
        }
    }
    if (!nd) {
        if (dirty != dirty_stk) free(dirty);
        ru_skip_sub++, ru_skip_bytes += t->L.size;
        e->wt_n = 0, e->up_pending = 0;
        perf_end(PERF_TEX, p0);
        return 1;
    }
    if (nd == nb) {
        if (dirty != dirty_stk) free(dirty);
        return -1;   // everything dirty: let normal upload() handle it
    }

    uint64_t ph[UPH_N] = {0}, pt = u0;
#define UB_PHASE(i) do { uint64_t now_ = kern_now_ns(); ph[i] += now_ - pt; pt = now_; } while (0)
    UB_PHASE(UPH_TEXPRE);
    vk_upload_flush_over(64ull << 20);
    UB_PHASE(UPH_FLUSH);

    VkBufferImageCopy *rg = linear ? calloc(nruns, sizeof *rg) : NULL;
    VkBlDeswRegion *br = linear ? NULL : calloc(nruns, sizeof *br);
    VkDeviceSize total = 0;
    uint64_t up_bytes = 0;
    uint32_t ri = 0;
    for (uint32_t b = 0; b < nb; ) {
        if (!dirty[b]) { b++; continue; }
        uint32_t b0 = b;
        while (b < nb && dirty[b]) b++;
        uint32_t y0 = b0 * band_rows;
        uint32_t ny = (b * band_rows > rows ? rows : b * band_rows) - y0;
        uint32_t oy = y0 * (uint32_t)bh;
        uint32_t eh = h - oy < ny * (uint32_t)bh ? h - oy : ny * (uint32_t)bh;
        if (linear) {
            uint64_t sz = (uint64_t)ny * rb;
            rg[ri++] = (VkBufferImageCopy){
                .bufferOffset = total,
                .bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw,
                .bufferImageHeight = ny * (uint32_t)bh,
                .imageSubresource = {e->pub.aspect, 0, 0, 1},
                .imageOffset = {0, (int32_t)oy, 0},
                .imageExtent = {w, eh, 1},
            };
            up_bytes += sz;
            total += (sz + 15) & ~15ull;
        } else {
            uint64_t bl_sz = (uint64_t)(b - b0) * row_blk_bytes;
            if (bl_sz > (32ull << 20) || (uint64_t)ny * rb > (32ull << 20)) {
                free(rg), free(br);
                if (dirty != dirty_stk) free(dirty);
                return -1;
            }
            br[ri++] = (VkBlDeswRegion){
                .src = total, .bl_size = bl_sz, .rb = rb, .rows = ny, .slices = 1,
                .w = w, .h = eh, .bw = (uint32_t)bw, .bh = (uint32_t)bh, .bpb = (uint32_t)bpb,
                .level = 0, .layer = 0, .offset_y = (int32_t)oy, .th = (uint8_t)th, .td = 0,
            };
            up_bytes += (uint64_t)ny * rb;
            total += (bl_sz + 15) & ~15ull;
        }
    }

    VkBuffer sb;
    uint8_t *stg;
    VkCommandBuffer cb = vk_upload_begin_staging(total, &sb, (void **)&stg);
    UB_PHASE(UPH_STAGING);
    if (!cb) {
        free(rg), free(br);
        if (dirty != dirty_stk) free(dirty);
        e->up_pending = 1, e->up_fail_frame = frame_now;
        perf_end(PERF_TEX, p0);
        return 0;
    }

    ri = 0;
    for (uint32_t b = 0; b < nb; ) {
        if (!dirty[b]) { b++; continue; }
        uint32_t b0 = b;
        while (b < nb && dirty[b]) b++;
        uint32_t y0 = b0 * band_rows;
        uint32_t ny = (b * band_rows > rows ? rows : b * band_rows) - y0;
        if (linear) {
            uint8_t *dst = stg + rg[ri].bufferOffset;
            const uint8_t *src = (const uint8_t *)t->mem + (uint64_t)y0 * stride;
            if (stride == rb) memcpy(dst, src, (size_t)ny * rb);
            else for (uint32_t r = 0; r < ny; r++) memcpy(dst + (size_t)r * rb, src + (size_t)r * stride, rb);
        } else {
            uint8_t *dst = stg + br[ri].src;
            const uint8_t *src = (const uint8_t *)t->mem + lv_off + (uint64_t)b0 * row_blk_bytes;
            memcpy(dst, src, (size_t)br[ri].bl_size);
        }
        ri++;
    }
    if (dirty != dirty_stk) free(dirty);
    UB_PHASE(UPH_DESWIZZLE);

    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int checked = 1;
    if (linear) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, nruns, rg);
    else checked = vk_bl_deswizzle_record(cb, sb, br, (int)nruns, e->pub.image, e->pub.aspect);
    UB_PHASE(UPH_RECORD);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_upload_end();
    UB_PHASE(UPH_END);
    free(rg), free(br);
    if (!checked) {
        e->up_pending = 1, e->up_fail_frame = frame_now;
        perf_end(PERF_TEX, p0);
        return 0;
    }

    uint64_t dt = kern_now_ns() - u0;
    if (rt_trace & RT_TRACE_PERF) {
        up_ph_n++;
        for (int i = 0; i < UPH_N; i++) up_ph_ns[i] += ph[i];
        if (dt > up_slow.total_ns) {
            up_slow = (UpSlow){.addr = e->pub.addr, .bytes = up_bytes, .total_ns = dt, .w = d->width, .h = d->height,
                               .levels = 1, .layers = 1, .fmt = (uint32_t)d->format, .again = 1};
            memcpy(up_slow.ph, ph, sizeof ph);
        }
    }
    n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
    perf_ledger(PERF_LG_UPLOAD, dt);
    ru_n++, ru_bytes += up_bytes, ru_ns += dt;
    uint64_t full_raw = (uint64_t)rb * rows;
    if (full_raw > up_bytes) ru_skip_sub += nb - nd, ru_skip_bytes += full_raw - up_bytes;
    e->wt_n = 0, e->up_pending = 0, e->n_up++;
    perf_end(PERF_TEX, p0);
#undef UB_PHASE
    return 1;
}

// 1 uploaded (or nothing to upload); 0 deferred for want of staging memory (e->up_pending: tried again next frame).
// partial_since != 0 on a write-watched re-upload: only subresources whose guest pages carry a generation newer than
// partial_since are deswizzled and copied (MKW_TEX_REUP_PARTIAL, default 1, supports "ab").
static int upload(Tex *e, uint32_t partial_since) {
    static volatile int32_t n_bad, n_undecoded;
    uint64_t frame_now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (e->up_pending && e->up_fail_frame == frame_now) return 0;   // failed this frame already: no new attempt
    if (tracing(e->pub.addr))
        rt_log("vk: trace %#llx: upload %u of entry %p (%ux%u target %d), submit epoch %u", (unsigned long long)e->pub.addr,
               e->n_up + 1, (void *)e, e->t.d.width, e->t.d.height, e->t.d.target, submit_epoch);
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || !t->mem) return 1;
    uint64_t p0 = perf_begin(), u0 = kern_now_ns();
    uint64_t ph[UPH_N] = {0}, ph_t = u0;   // phase times (MKW_TRACE=perf)
#define UP_PHASE(i) do { uint64_t now_ = kern_now_ns(); ph[i] += now_ - ph_t; ph_t = now_; } while (0)
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear ? 1 : e->pub.layers;
    int astc = e->f.astc_bw != 0, rgb = e->f.expand_rgb;
    // MKW_ASTC=check stages a CPU decode beside the GPU one: textures over 16 MiB stay unchecked (a 357 MB staging
    // buffer failed at a course load, vis22; 64 MiB still failed with the mip levels, vis24).
    int agpu = astc && vk_astc_gpu(),
        acheck = agpu && vk_astc_mode() == VK_ASTC_CHECK && (uint64_t)e->pub.width * e->pub.height * e->pub.layers * 4 <= (16ull << 20);
    static int bcc = -1;   // MKW_ASTC_BC_CHECK (bc_check)
    if (bcc < 0) bcc = cfg_get("MKW_ASTC_BC_CHECK") ? atoi(cfg_get("MKW_ASTC_BC_CHECK")) : 0;
    int bcheck = bcc > 0 && agpu && e->f.astc_bc7 && !acheck && !linear;
    static int reup_bands = -1;     // MKW_TEX_REUP_BANDS (default 1; "ab"): partial row-band re-upload on large 2D single-level textures
    if (reup_bands < 0) reup_bands = rt_knob("TEX_REUP_BANDS", "1");
    if (partial_since && e->uploaded && levels == 1 && layers == 1 && !is3d && !one_d && !astc && !rgb && !acheck &&
        !bcheck && e->pub.aspect == VK_IMAGE_ASPECT_COLOR_BIT && d->target != 5 && d->target != 6 && d->target != 10 &&
        d->samples <= 1 && t->L.size >= (256u << 10) && RT_KNOB_ON(reup_bands)) {
        int br_res = upload_bands(e, partial_since, bw, bh, bpb, linear, p0, u0, frame_now);
        if (br_res >= 0) return br_res;
    }
    static int reup_partial = -1;   // MKW_TEX_REUP_PARTIAL (default 1; "ab"): skip unchanged mip/layer subresources on re-upload
    if (reup_partial < 0) reup_partial = rt_knob("TEX_REUP_PARTIAL", "1");
    uint32_t n = levels * layers, nd = n, k = 0;
    uint32_t drop = e->pub.drop;   // NVN level = image level + drop
    int do_partial = partial_since && e->uploaded && !linear && n > 1 && !acheck && !bcheck &&
                     d->target != 5 && d->target != 6 && d->target != 10 && d->samples <= 1 && RT_KNOB_ON(reup_partial);
    uint64_t up_raw = 0, skip_raw = 0;
    uint8_t dirty_stk[256], *dirty = NULL;
    if (do_partial) {
        uint64_t offs_stk[256], spans_stk[256];
        uint64_t *offs = n <= 256 ? offs_stk : malloc((size_t)n * sizeof *offs);
        uint64_t *spans = n <= 256 ? spans_stk : malloc((size_t)n * sizeof *spans);
        dirty = n <= 256 ? dirty_stk : malloc((size_t)n);
        if (!offs || !spans || !dirty) {
            if (offs != offs_stk) free(offs);
            if (spans != spans_stk) free(spans);
            if (dirty != dirty_stk) free(dirty);
            dirty = NULL;
            do_partial = 0;
        } else {
            for (uint32_t lv = 0; lv < levels; lv++) {
                uint32_t w = mip(d->width, lv + drop), h = one_d ? 1 : mip(d->height, lv + drop),
                         sl = is3d ? mip(d->depth, lv + drop) : 1;
                uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
                unsigned th = 0, td = 0;
                nvn_tex_level_tiles(d, lv + drop, &th, &td);
                uint64_t lv_off = nvn_tex_level_offset(d, lv + drop);
                uint64_t gobs_x = (rb + 63) / 64, blocks_y = (rows + (8u << th) - 1) / (8u << th),
                         blocks_z = (sl + (1u << td) - 1) / (1u << td);
                uint64_t lv_span = gobs_x * blocks_y * blocks_z * (512ull << th << td);
                for (uint32_t ly = 0; ly < layers; ly++) {
                    uint32_t idx = lv * layers + ly;
                    offs[idx] = (uint64_t)ly * t->L.layer_size + lv_off;
                    spans[idx] = lv_span;
                }
            }
            vk_mem_ranges_written_since(e->pub.addr, offs, spans, dirty, n, partial_since);
            if (offs != offs_stk) free(offs);
            if (spans != spans_stk) free(spans);
            nd = 0;
            for (uint32_t i = 0; i < n; i++) nd += dirty[i] != 0;
            if (!nd) {
                if (dirty != dirty_stk) free(dirty);
                ru_skip_sub += n, ru_skip_bytes += t->L.size;
                e->wt_n = 0, e->up_pending = 0;
                perf_end(PERF_TEX, p0);
                return 1;
            }
        }
    }
    if (!do_partial && agpu && e->f.astc_bc7 == 1 && !acheck && !linear && !drop && vk_texpre_on() && upload_texpre(e, levels, layers)) {
        UP_PHASE(UPH_TEXPRE);
        goto counted;
    }
    UP_PHASE(UPH_TEXPRE);
    static int astc_gpu_desw = -1;   // MKW_ASTC_GPU_DESW (default 1; "ab"): deswizzle block-linear ASTC in astc_decode.comp
    if (astc_gpu_desw < 0) astc_gpu_desw = rt_knob("ASTC_GPU_DESW", "1");
    static int tex_gpu_desw = -1;    // MKW_TEX_GPU_DESW (default 1; "ab"): deswizzle non-ASTC block-linear textures on GPU
    if (tex_gpu_desw < 0) tex_gpu_desw = rt_knob("TEX_GPU_DESW", "1");
    int bl_ok = !linear && d->target != 5 && d->target != 6 && d->target != 10 && d->samples <= 1;
    int do_gpu_desw = agpu && bl_ok && !acheck && !bcheck && bpb == 16 && RT_KNOB_ON(astc_gpu_desw);
    int do_tex_gdesw = !astc && !rgb && bl_ok && e->pub.aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
                       RT_KNOB_ON(tex_gpu_desw) && vk_bl_deswizzle_ready();
    VkBufferImageCopy *rg = calloc(nd, sizeof *rg);
    VkAstcRegion *ar = agpu ? calloc(nd, sizeof *ar) : NULL;
    VkBlDeswRegion *br = do_tex_gdesw ? calloc(nd, sizeof *br) : NULL;
    VkDeviceSize total = 0, decoded = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            uint32_t w = mip(d->width, lv + drop), h = one_d ? 1 : mip(d->height, lv + drop),
                     sl = is3d ? mip(d->depth, lv + drop) : 1;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)rb * rows * sl;
            if (do_partial && !dirty[lv * layers + ly]) { skip_raw += raw; continue; }
            up_raw += raw;
            uint64_t out = agpu ? raw : astc ? texels * 4 : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            if (do_gpu_desw && !(is3d && texels * 4 > (32ull << 20))) {
                unsigned th = 0, td = 0;
                nvn_tex_level_tiles(d, lv + drop, &th, &td);
                uint64_t lv_off = nvn_tex_level_offset(d, lv + drop);
                uint64_t gobs_x = (rb + 63) / 64, blocks_y = (rows + (8u << th) - 1) / (8u << th),
                         blocks_z = (sl + (1u << td) - 1) / (1u << td);
                uint64_t lv_span = gobs_x * blocks_y * blocks_z * (512ull << th << td);
                uint64_t base = (uint64_t)ly * t->L.layer_size + lv_off;
                if (lv_span && base + lv_span <= t->L.size && (!is3d || lv_span <= (16ull << 20))) {
                    ar[k] = (VkAstcRegion){.src = total, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly,
                                           .bl_size = lv_span, .bl_rows = rows, .bl_th = (uint8_t)th, .bl_td = (uint8_t)td};
                    out = lv_span;
                }
            } else if (br && (rb & 3u) == 0 && raw >= 4096 && raw <= (32ull << 20)) {
                // Skip the 1024x1024 RGBA8 button glyph sheet so glyphs_apply can redraw PlayStation icons on CPU.
                int maybe_glyph = lv == 0 && ly == 0 && sl == 1 && w == 1024 && h == 1024 &&
                                  (d->format == 0x25 || d->format == 0x38);
                if (!maybe_glyph) {
                    unsigned th = 0, td = 0;
                    nvn_tex_level_tiles(d, lv + drop, &th, &td);
                    uint64_t lv_off = nvn_tex_level_offset(d, lv + drop);
                    uint64_t gobs_x = (rb + 63) / 64, blocks_y = (rows + (8u << th) - 1) / (8u << th),
                             blocks_z = (sl + (1u << td) - 1) / (1u << td);
                    uint64_t lv_span = gobs_x * blocks_y * blocks_z * (512ull << th << td);
                    uint64_t base = (uint64_t)ly * t->L.layer_size + lv_off;
                    if (lv_span && lv_span <= (32ull << 20) && base + lv_span <= t->L.size) {
                        br[k] = (VkBlDeswRegion){.src = total, .bl_size = lv_span, .rb = rb, .rows = rows, .slices = sl,
                                                 .w = w, .h = h, .bw = (uint32_t)bw, .bh = (uint32_t)bh,
                                                 .bpb = (uint32_t)bpb, .level = lv, .layer = ly,
                                                 .th = (uint8_t)th, .td = (uint8_t)td};
                        out = lv_span;
                    }
                }
            }
            rg[k++] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, lv, ly, 1},
                                          .imageExtent = {w, h, sl}};
            total += (out + 15) & ~15ull;
            decoded += texels * 4;
        }
    if (dirty && dirty != dirty_stk) free(dirty);
    vk_upload_flush_over(64ull << 20);   // as vk_tex_texel.c: the load of a capture sent its textures in one batch
    UP_PHASE(UPH_FLUSH);
    VkBuffer sb, rbuf = VK_NULL_HANDLE;
    VmaAllocation ralloc = NULL;
    uint8_t *stg;
    VkCommandBuffer cb = vk_upload_begin_staging(total, &sb, (void **)&stg);
    UP_PHASE(UPH_STAGING);
    if (!cb) {   // out of staging memory: the image keeps what it had, and the upload comes again next frame
        free(br);
        free(ar);
        free(rg);
        e->up_pending = 1, e->up_fail_frame = frame_now;
        perf_end(PERF_TEX, p0);
        return 0;
    }
    uint8_t *rmap = acheck ? vk_staging(decoded, &rbuf, &ralloc) : NULL;
    uint32_t nc = 0, bi = 0;
    uint64_t raw_all = 0;
    for (uint32_t i = 0; i < nd; i++) {
        uint32_t lv = rg[i].imageSubresource.mipLevel, ly = rg[i].imageSubresource.baseArrayLayer;
        uint32_t w = rg[i].imageExtent.width, h = rg[i].imageExtent.height, sl = rg[i].imageExtent.depth;
        uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
        size_t raw = (size_t)rb * rows * sl;
        raw_all += raw;
        uint8_t *dst = stg + rg[i].bufferOffset;
        if (agpu && ar[i].bl_size) {
            uint64_t base = (uint64_t)ly * t->L.layer_size + nvn_tex_level_offset(d, lv + drop);
            memcpy(dst, (const uint8_t *)t->mem + base, (size_t)ar[i].bl_size);
            continue;
        }
        if (br && br[i].bl_size) {
            uint64_t base = (uint64_t)ly * t->L.layer_size + nvn_tex_level_offset(d, lv + drop);
            memcpy(dst, (const uint8_t *)t->mem + base, (size_t)br[i].bl_size);
            if (bi != i) br[bi] = br[i];
            bi++;
            continue;
        }
        uint8_t *tmp = (astc && !agpu) || rgb ? malloc(raw) : NULL;
        if (nvn_deswizzle(d, lv + drop, ly, t->mem, tmp ? tmp : dst, rb)) {
            log_limited(&n_bad, "texture %#llx fmt %#x target %d: level %u layer %u has no CPU layout; uploaded as zeros",
                        (unsigned long long)e->pub.addr, d->format, d->target, lv, ly);
            memset(tmp ? tmp : dst, 0, raw);
        }
        if (agpu) {
            ar[i] = (VkAstcRegion){.src = rg[i].bufferOffset, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly};
        } else if (astc) {
            uint64_t a0 = perf_begin();
            for (uint32_t z = 0; z < sl; z++)
                astc_decode_ldr_mt(tmp + (size_t)z * rb * rows, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                   dst + (size_t)z * w * h * 4);
            perf_end(PERF_ASTC, a0);
            rg[i].bufferRowLength = w, rg[i].bufferImageHeight = h;
            if (nc != i) rg[nc] = rg[i];
            nc++;
        } else if (rgb) {
            expand_rgb(d->format, tmp, dst, (size_t)w * h * sl);
            rg[i].bufferRowLength = w, rg[i].bufferImageHeight = h;
            if (nc != i) rg[nc] = rg[i];
            nc++;
        } else {
            rg[i].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[i].bufferImageHeight = rows * (uint32_t)bh;
            if (lv == 0 && ly == 0 && sl == 1 && glyphs_apply(e->pub.addr, d->format, w, h, dst, rb)) {
                g_glyph_addr = e->pub.addr;   // the button icon sheet (vk_glyphs_due)
                g_glyph_gen = glyphs_generation();
                __atomic_store_n(&g_glyph_known, 1, __ATOMIC_RELEASE);
            }
            if (nc != i) rg[nc] = rg[i];
            nc++;
        }
        free(tmp);
    }
    if (vk_pool_diag_on()) {   // MKW_MEM_POOL_LABELS: read from a CPU_NO_ACCESS pool's game pages?
        const pool_t *p = nvn_pool_find(e->pub.addr, 1);
        if (p) vk_pool_diag_note(p->flags, 1, raw_all);
        if (p && astc) vk_pool_diag_note(p->flags, 3, raw_all);
    }
    UP_PHASE(UPH_DESWIZZLE);
    uint8_t *raw_copy = acheck || bcheck ? malloc(total) : NULL;
    if (raw_copy) memcpy(raw_copy, stg, total);
    int astc_batched = agpu && !acheck && !bcheck && vk_astc_batch_active();
    if (!astc_batched) barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int checked = 1;
    if (agpu) {
        uint64_t a0 = perf_begin();
        vk_astc_tag(e->pub.addr, 1);
        if (e->f.astc_bc7 == 2) vk_astc_bc1_alpha_note(e->pub.addr, stg, total, e->f.astc_bw, e->f.astc_bh, "upload");
        checked = vk_astc_record(cb, sb, total, ar, (int)nd, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->pub.image,
                                 e->pub.aspect, e->f.astc_bc7, rbuf);
        perf_end(PERF_ASTC, a0);
        if (!checked)
            log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: left as it was, tried "
                        "again next frame", (unsigned long long)e->pub.addr, d->format, d->width, d->height);
    } else {
        if (bi) checked = vk_bl_deswizzle_record(cb, sb, br, (int)bi, e->pub.image, e->pub.aspect);
        if (nc) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, nc, rg);
    }
    UP_PHASE(UPH_RECORD);
    if (!astc_batched) barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_upload_end();
    UP_PHASE(UPH_END);
    if (acheck) {   // waits for the GPU
        vk_upload_flush(1);
        if (checked) {
            vmaInvalidateAllocation(vk_ctx()->vma, ralloc, 0, VK_WHOLE_SIZE);
            astc_check(e, raw_copy, ar, (int)nd, rmap);
        }
        vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
    }
    if (bcheck && checked && raw_copy) bc_check(e, raw_copy, ar, (int)nd, bcc);
    free(raw_copy);
    free(br);
    free(ar);
    free(rg);
    // A decode that was not recorded (its scratch buffer not made, memory short) wrote nothing: the image still holds
    // what it held, for a first upload whatever its recycled memory held (other textures' texels, or zero blocks the
    // samplers see as the error colour). Counted as uploaded, it was never tried again while the storage stayed clean.
    // Deferred as a staging failure is: a first upload binds the zero stand-in meanwhile (tex_unready), a re-upload
    // keeps the older image, and both come again next frame.
    if ((agpu || bi) && !checked) {
        e->up_pending = 1, e->up_fail_frame = frame_now;
        perf_end(PERF_TEX, p0);
        return 0;
    }
counted:;
#undef UP_PHASE
    uint64_t dt = kern_now_ns() - u0;
    perf_ledger(PERF_LG_UPLOAD, dt);
    if (rt_trace & RT_TRACE_PERF) {
        up_ph_n++;
        for (int i = 0; i < UPH_N; i++) up_ph_ns[i] += ph[i];
        if (dt > up_slow.total_ns) {
            up_slow = (UpSlow){.addr = e->pub.addr, .bytes = t->L.size, .total_ns = dt, .w = d->width, .h = d->height,
                               .levels = levels, .layers = layers, .fmt = (uint32_t)d->format, .again = e->uploaded};
            memcpy(up_slow.ph, ph, sizeof ph);
        }
    }
    if (e->uploaded) {
        uint64_t counted_bytes = do_partial ? up_raw : t->L.size;
        n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
        ru_n++, ru_bytes += counted_bytes, ru_ns += dt;
        if (do_partial && nd < n) ru_skip_sub += n - nd, ru_skip_bytes += skip_raw;
        if (e->wt_n) {
            ru_wt_n++, ru_wt_bytes += counted_bytes, ru_wt_ns += dt;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("perf: frame %llu: re-upload after %u WriteTexels: texture %#llx fmt %#x %ux%u, %.1f KiB, %.3f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), e->wt_n,
                       (unsigned long long)e->pub.addr, d->format, d->width, d->height, t->L.size / 1024.0, dt / 1e6);
        }
    } else {
        n_uploads++, perf_count(PERF_CNT_TEX_UP);
    }
    // MKW_TEX_UP_LOG=N (diagnosis): the first N uploads from frame MKW_TEX_UP_LOG_AT (2300) on, with how many submits
    // the game had made past the one being recorded: their storage is read now, from guest memory (Hookshot: MKW's
    // terrain height tiles, 68x68 RGBA16 at new addresses every frame, rewritten right after their submit).
    static int up_log = -1, up_at;
    if (up_log < 0) {
        up_log = cfg_get("MKW_TEX_UP_LOG") ? atoi(cfg_get("MKW_TEX_UP_LOG")) : 0;
        up_at = cfg_get("MKW_TEX_UP_LOG_AT") ? atoi(cfg_get("MKW_TEX_UP_LOG_AT")) : 2300;
    }
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (up_log > 0 && fr >= (uint64_t)up_at) {
        up_log--;
        uint64_t ztail = zero_tail((uintptr_t)t->mem, t->L.size);
        rt_log("vk: upload at frame %llu: %#llx %ux%ux%u fmt %#x levels %u, %.1f KiB%s, game %lld submits ahead; "
               "zero tail from %#llx (%.0f%%)",
               (unsigned long long)fr, (unsigned long long)e->pub.addr, d->width, d->height, d->depth, d->format,
               e->pub.levels, t->L.size / 1024.0, e->uploaded ? " (again)" : "",
               (long long)(nvn_submit_count() - __atomic_load_n(&vk_replay_submit, __ATOMIC_RELAXED)),
               (unsigned long long)ztail, t->L.size ? 100.0 * (t->L.size - ztail) / t->L.size : 0.0);
        if (astc && t->L.size >= (64u << 10)) zero_tail_note((uintptr_t)t->mem, t->L.size, ztail, fr, e->n_up + 1);
        uint64_t rlo, rhi, rfreed, rframe;
        uint32_t rflags;
        if (vk_pool_release_find((uintptr_t)t->mem, t->L.size, &rlo, &rhi, &rfreed, &rframe, &rflags))
            rt_log("vk: upload %#llx (%ux%u fmt %#x) lies over pool-finalize release [%#llx,%#llx) flags %#x of frame "
                   "%llu (%.1f KiB released); its storage %.1f of %.1f KiB backed now",
                   (unsigned long long)(uintptr_t)t->mem, d->width, d->height, d->format, (unsigned long long)rlo,
                   (unsigned long long)rhi, rflags, (unsigned long long)rframe, rfreed / 1024.0,
                   plat_mem_backed_bytes((uintptr_t)t->mem, t->L.size) / 1024.0, t->L.size / 1024.0);
        // MKW_TEX_UP_DUMP=<w>x<h> (diagnosis, task 32): the raw guest bytes of the first 4 such uploads go to
        // MKW_TEX_UP_DUMP_DIR (/app0/logs), tex_<addr>_<w>x<h>_<fmt>.bin, to decode on the PC: a fault that is already in
        // guest memory past the cards' first 64 KiB boundary shows there.
        static unsigned dump_w, dump_h, dump_n;
        static int dump_init;
        if (!dump_init) {
            dump_init = 1;
            const char *dw = cfg_get("MKW_TEX_UP_DUMP");
            if (dw && sscanf(dw, "%ux%u", &dump_w, &dump_h) != 2) dump_w = dump_h = 0;
        }
        if (dump_w && d->width == dump_w && d->height == dump_h && dump_n < 4) {
            dump_n++;
            const char *dir = cfg_get("MKW_TEX_UP_DUMP_DIR");
            char path[512];
            snprintf(path, sizeof path, "%s/tex_%llx_%ux%u_%x.bin", dir && *dir ? dir : "/app0/logs",
                     (unsigned long long)(uintptr_t)t->mem, d->width, d->height, d->format);
            FILE *f = fopen(path, "wb");
            size_t wrote = f ? fwrite(t->mem, 1, t->L.size, f) : 0;
            if (f) fclose(f);
            rt_log("vk: upload dump %s: %zu of %llu bytes", path, wrote, (unsigned long long)t->L.size);
        }
    }
    // MKW_TEX_ZERO_LOG=N (diagnosis, 16/11/2026: magenta on scenery, karts and a road edge in some runs, at random
    // moments): the first N ASTC uploads with a whole 64 KiB unit of zero guest bytes. A zero 16-byte ASTC block is not
    // a valid block and decodes as the error colour, magenta, so real texture data never holds a zero unit; each check
    // stops at the first non-zero byte. The line says whether that unit is backed and frozen now and whether a
    // pool-finalize release covered it, to tell which path left the zeros.
    static int zlog = -1;
    if (zlog < 0) zlog = cfg_get("MKW_TEX_ZERO_LOG") ? atoi(cfg_get("MKW_TEX_ZERO_LOG")) : 0;
    if (zlog > 0 && astc && t->L.size >= (64u << 10)) {
        uint64_t lo = ((uintptr_t)t->mem + 0xffffu) & ~0xffffull, hi = ((uintptr_t)t->mem + t->L.size) & ~0xffffull;
        uint64_t first = 0, nz = 0, units = 0;
        for (uint64_t u = lo; u < hi; u += 0x10000, units++) {
            const uint64_t *q = (const uint64_t *)(uintptr_t)u;
            size_t k = 0;
            while (k < 0x10000 / 8 && !q[k]) k++;
            if (k == 0x10000 / 8) nz++, first = first ? first : u;
        }
        if (nz && __atomic_fetch_sub(&zlog, 1, __ATOMIC_RELAXED) > 0) {
            uint64_t rlo = 0, rhi = 0, rfreed = 0, rframe = 0;
            uint32_t rflags = 0;
            int rel = vk_pool_release_find(first, 0x10000, &rlo, &rhi, &rfreed, &rframe, &rflags);
            rt_log("vk: zero upload at frame %llu: texture %#llx %ux%ux%u fmt %#x levels %u, %.1f KiB, upload %u%s: %llu of "
                   "%llu units all zero, first %#llx (backed %.0f KiB, frozen %d)%s",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                   (unsigned long long)e->pub.addr, d->width, d->height, d->depth, d->format, e->pub.levels,
                   t->L.size / 1024.0, e->n_up + 1, e->uploaded ? " (again)" : "", (unsigned long long)nz,
                   (unsigned long long)units, (unsigned long long)first, plat_mem_backed_bytes(first, 0x10000) / 1024.0,
                   mem_frozen(first) != 0, rel ? "; under a pool-finalize release" : "");
            if (rel)
                rt_log("vk: zero upload %#llx: pool-finalize release [%#llx,%#llx) flags %#x of frame %llu (%.1f KiB)",
                       (unsigned long long)first, (unsigned long long)rlo, (unsigned long long)rhi, rflags,
                       (unsigned long long)rframe, rfreed / 1024.0);
        }
    }
    e->wt_n = 0;
    // A first upload deferred earlier: the draws bound the zero stand-in for it (tex_unready), and the slots and the
    // replay's descriptor memos keep what they resolved while the address generation stays: moved, they take the image.
    if (!e->uploaded && e->up_pending) tex_gen_bump(e->key.addr);
    e->uploaded = 1;
    e->up_pending = 0;
    e->n_up++;
    perf_end(PERF_TEX, p0);
    return 1;
}

// A mirrored texture whose storage holds its content: no entry at its address holds GPU-written content the storage
// lacks (tex_has_gpu_content, the alias copy's rule), so a full upload would make the image exactly the storage's
// bytes. Refusing any entry that was GPU-owned or dirty (ce280b0) also refused textures the GPU wrote once and the CPU
// rewrote since, or whose render was written back, whose storage is their content: the copy was skipped and its
// destination kept what it held (from review; not seen in a run). *uploads: the uploads of t's own entry so far (0
// without one). Texture lock held.
static int cpu_content(const tex_t *t, uint32_t *uploads) {
    *uploads = 0;
    if (!tex_mirrored(t)) return 0;
    TexKey k;
    make_key(t, &k);
    const Tex *own = nvn_hm_get(&by_key, &k);
    if (own) *uploads = own->n_up;
    for (const Tex *p = entries_at(k.addr); p; p = p->next_at)
        if (!p->released && tex_has_gpu_content(p)) return 0;
    return 1;
}

// The zero-block census of an ASTC copy (task G, a measurement for a console A/B with MKW_QUEUE_WAITS=1; nothing is
// skipped or deferred on it):
// an ASTC copy whose staged source region holds over 1% all-zero 16-byte blocks, which decode as the error colour. The
// PS5's logs had the 8192x8192 ASTC 8x8 terrain staging texture decode as all error blocks through these copies at
// loading (frames ~70-80, ~300 copies, every layer's level chain), while the same storage holds valid blocks later. The
// hypothesis: on the console the GPU waits on an event the CPU signals once it wrote the staging texture
// (nvnCommandBufferWaitEvent), and the replay, in submit order, reads it before. The line gives the replay's submission
// against the game's newest and whether that submission had a wait its submit order did not meet (nvn_qorder.c). The
// first 64 copies a line each, then the count in vk_astc.c's "so far" line. ZERO_SAMPLES blocks spread over the level's
// storage in guest memory decide whether to count at all: the staging may be write-combined on the PS5, where reading
// back a 1/64 sample of an 8192x8192 copy (16,384 blocks, a cache line each) may cost milliseconds (unmeasured); guest
// memory is cached. The whole level of each layer is sampled, not only the region (block-linear rows are not contiguous): a zero
// block outside the region costs a count that finds too few, a region whose zeros the samples miss goes uncounted.
#define ZERO_SAMPLES 512u
static int level_has_zero(const tex_t *src, uint32_t level, uint32_t layer, uint32_t layers) {
    nvn_texlayout L;
    nvn_tex_layout(&src->d, &L);
    if (level >= src->d.levels || layer + layers > L.layers) return 1;   // unsure: count
    uint64_t lo = nvn_tex_level_offset(&src->d, level);
    uint64_t hi = level + 1 < src->d.levels ? nvn_tex_level_offset(&src->d, level + 1) : L.layer_size;
    uint64_t blocks = hi > lo ? (hi - lo) / 16 : 0;
    if (!blocks) return 0;
    uint32_t per = (ZERO_SAMPLES + layers - 1) / layers;
    if (per > blocks) per = (uint32_t)blocks;
    for (uint32_t l = 0; l < layers; l++) {
        const uint8_t *base = (const uint8_t *)src->mem + (uint64_t)(layer + l) * L.layer_size + lo;
        for (uint32_t k = 0; k < per; k++) {
            const uint64_t *q = (const uint64_t *)(const void *)(base + (blocks * k / per) * 16);
            if (!(q[0] | q[1])) return 1;
        }
    }
    return 0;
}

static void zero_census(const tex_t *src, const uint8_t *stg, size_t bytes, uint32_t level, uint32_t layer,
                        uint32_t layers, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    if (!level_has_zero(src, level, layer, layers)) return;
    const uint64_t *q = (const uint64_t *)(const void *)stg;
    size_t n = bytes / 16;
    if (!n) return;
    uint64_t zero = 0;
    for (size_t k = 0; k < n; k++) zero += !(q[2 * k] | q[2 * k + 1]);
    if (zero * 100 <= (uint64_t)n) return;
    vk_astc_zero_note(zero);
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) > 64) return;
    uint64_t sub = __atomic_load_n(&vk_replay_submit, __ATOMIC_RELAXED), game = nvn_submit_count();
    int unmet = nvn_qorder_unmet(sub);
    rt_log("vk: ASTC copy zero blocks at frame %llu: source %#llx %ux%u fmt %#x level %u layers %u+%u region %u,%u "
           "%ux%u: %llu of %zu blocks zero (%.1f%%); replay submit %llu, game %llu (%lld ahead), wait not met in submit "
           "order: %s", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
           (unsigned long long)(uintptr_t)src->mem, src->d.width, src->d.height, src->d.format, level, layer, layers, x,
           y, w, h, (unsigned long long)zero, n, 100.0 * (double)zero / (double)n, (unsigned long long)sub,
           (unsigned long long)game, (long long)(game - sub), unmet > 0 ? "yes" : unmet == 0 ? "no" : "unknown");
}

typedef struct InlineStg { VkBuffer b; VmaAllocation a; struct InlineStg *next; } InlineStg;
static void free_inline_stg(void *arg) {   // completion thread, once the submission that read it is done
    InlineStg *t = arg;
    if (t->b) vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a);   // VK_NULL_HANDLE: inline_stg_room freed it already
    free(t);
}

// The inline copies' staging lives until the submission is done, and only its last part signals that: MKW makes ~300
// ASTC copies of up to 16 MiB each in one loading submission (8192x8192 ASTC 8x8 levels into the terrain arrays, the
// PS5's 16/11 logs), several GiB of staging at once on a console with ~1 GiB free. Past INLINE_STG_CAP of staging in
// the submission, what was recorded so far goes to the GPU as a part (vk_exec_split_part), the replay thread waits for
// it and frees the staging the part read. A wait on the replay thread, at loading only: ~64 MiB of decodes. A
// MKW_REPLAY_PAR chunk cannot send a part of its own: its staging waits for the submission, as before.
// Per thread: the replay's own recording, or a chunk's (its list is reset at its next submission, before which the
// completion thread may free the entries, untouched since).
#define INLINE_STG_CAP (64ull << 20)
static _Thread_local struct { uint64_t fence, bytes; InlineStg *held; } istg;
uint64_t vk_astc_inline_cap_test;
static uint64_t astc_counts[VK_ASTC_COPY_N];   // atomics

void vk_texture_astc_copy_counts(uint64_t out[VK_ASTC_COPY_N]) {
    for (int i = 0; i < VK_ASTC_COPY_N; i++) out[i] = __atomic_load_n(&astc_counts[i], __ATOMIC_RELAXED);
}

static void inline_stg_room(VkDeviceSize need) {
    uint64_t fence = vk_exec_recording_fence();
    if (istg.fence != fence) istg.fence = fence, istg.bytes = 0, istg.held = NULL;
    uint64_t cap = vk_astc_inline_cap_test ? vk_astc_inline_cap_test : INLINE_STG_CAP;
    if (!istg.held || istg.bytes + need <= cap) return;
    uint64_t part, t0 = kern_now_ns();
    if (!vk_exec_split_part(&part)) {
        __atomic_fetch_add(&astc_counts[VK_ASTC_COPY_CHUNK_HELD], 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_fetch_add(&astc_counts[VK_ASTC_COPY_PART_WAITS], 1, __ATOMIC_RELAXED);
    vk_queue_lock();   // a part the submit thread holds (MKW_SUBMIT_THREAD) goes to the queue first
    vk_queue_unlock();
    vk_sync_part_wait(part);
    uint64_t freed = istg.bytes;
    for (InlineStg *t = istg.held; t; t = t->next) vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a), t->b = VK_NULL_HANDLE;
    istg.bytes = 0, istg.held = NULL;
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) <= 16)
        rt_log("vk: ASTC copies in the replay: %.1f MiB of staging in submission %llu, sent as part %llu and waited for "
               "(%.2f ms) to free it", freed / 1048576.0, (unsigned long long)fence, (unsigned long long)part,
               (kern_now_ns() - t0) / 1e6);
}

// A texture copy between ASTC textures that went to different Vulkan formats (one kept as RGBA8 under 256x256 into one
// re-encoded as BC7, BC1 against BC7): the source's own ASTC blocks, read from its storage, decoded and encoded
// straight into dst as bc7 says (TexVkFormat.astc_bc7: 0 RGBA8, 1 BC7, 2 BC1). MKW fills the small levels (5-11) of its sparse
// terrain arrays this way, from small ASTC textures; skipped, those levels read zero and the race's bushes and cacti,
// which take their colour from level 11, came out black (m16, 01/11/2026). 0 when it cannot (the caller copies the
// images, or skips).
// Only a source whose storage is its content (cpu_content, as vk_texture_region_copy): one the GPU wrote (a copy into
// it, a sparse array, a GPU-only pool) has stale or no blocks in its storage, which decoded as old texels or the error
// colour. The texture lock covers that check only on the replay's path: the storage is read after it as the game's own
// writes are, and holding the lock over the deswizzle, the census and the staging (whose last resort waits for the GPU)
// blocked the game threads that register textures (16 MiB copies at loading). Probes and the readback keep it.
// A BC destination takes whole 4x4 blocks: dx and dy multiples of 4, and w and h too unless the region reaches the
// level's edge; a region that ends inside a block would write the whole block, clamped texels over its neighbours.
// A region that reaches the source level's own edge is first widened to whole source blocks, as the console's copy
// of a compressed texture moves whole blocks: MKW copies whole 509x1020, 1024x1023, 512x510 ASTC 8x8 textures into its
// 8192x8192 terrain arrays, and refused for their last partial block those copies were skipped (42 in the first
// minute of a race, texcopy 17/11/2026) and the tiles kept what they held, the error colour of an earlier decode.
// Recorded in the replay's command buffer at the copy's place (vk_exec_inline_begin), not in the upload batch: the
// batch runs before the whole part being recorded, so a decode there went ahead of the commands recorded before it (a
// copy into the same destination through vkCmdCopyImage, a draw that sampled it) and was undone by an earlier copy or
// seen by an earlier draw. The decoder's scratch is shared with the upload batches under the upload lock (vk_astc.c,
// FIXED_SCRATCH: one queue, a barrier at each use); the staging is this call's own, freed when the submission is done.
// Outside a replay (probes), and for MKW_ASTC_PROBE_LEVEL's readback, into the upload batch as before.
int vk_texture_astc_copy(const tex_t *src, uint32_t level, uint32_t layer, uint32_t layers, uint32_t x, uint32_t y,
                         uint32_t w, uint32_t h, const VkTexImage *dsti, VkImageAspectFlags aspect, uint32_t dlevel,
                         uint32_t dlayer, int32_t dx, int32_t dy, int bc7) {
    TexVkFormat f;
    if (!dsti || !tex_vk_format(src->d.format, &f) || !f.astc_bw || !src->mem || x % f.astc_bw || y % f.astc_bh)
        return 0;
    if (bc7) {   // a BC destination: whole blocks
        uint32_t lw = mip(dsti->width, dlevel), lh = mip(dsti->height, dlevel);
        uint32_t sw = mip(src->d.width, level), sh = mip(src->d.height, level);
        if (dx >= 0 && dy >= 0 && (uint32_t)dx < lw && (uint32_t)dy < lh) {   // the source's last partial block, whole
            if (x + w == sw && w % f.astc_bw) w = (w + f.astc_bw - 1) / f.astc_bw * f.astc_bw;
            if (y + h == sh && h % f.astc_bh) h = (h + f.astc_bh - 1) / f.astc_bh * f.astc_bh;
            if (w > lw - (uint32_t)dx) w = lw - (uint32_t)dx;
            if (h > lh - (uint32_t)dy) h = lh - (uint32_t)dy;
        }
        if (dx < 0 || dy < 0 || dx % 4 || dy % 4 || (w % 4 && (uint32_t)dx + w != lw) || (h % 4 && (uint32_t)dy + h != lh))
            return 0;
    }
    VkImage dst = dsti->image;
    uint32_t rb = (w + f.astc_bw - 1) / f.astc_bw * 16, rows = (h + f.astc_bh - 1) / f.astc_bh;
    size_t per = (size_t)rb * rows;
    VkAstcRegion *ar = calloc(layers, sizeof *ar);
    if (!ar) return 0;
    lock();
    int locked = 1;
    uint32_t ups;
    if (!cpu_content(src, &ups)) {
        static volatile int32_t n_gpu;
        log_limited(&n_gpu, "ASTC copy out of texture %#llx (fmt %#x %ux%u): its content is on the GPU, not in its storage: "
                    "not decoded from the storage (the images are copied when their formats match)", (unsigned long long)(uintptr_t)src->mem, src->d.format,
                    src->d.width, src->d.height);
        unlock();
        free(ar);
        return 0;
    }
    // MKW_ASTC_PROBE_LEVEL=<n> (diagnosis): the first copies into level n read back the decoded texels and log them with
    // the source's first block (the bushes' level 11 of the terrain arrays sampled black, 01/11/2026).
    static int probe = -2;
    static volatile int32_t n_probed;
    if (probe == -2) probe = cfg_get("MKW_ASTC_PROBE_LEVEL") ? atoi(cfg_get("MKW_ASTC_PROBE_LEVEL")) : -1;
    int want_probe = probe == (int)dlevel && layers == 1 && w * h <= 64 && plat_inc32(&n_probed) <= 6;
    static int copy_gdesw = -1;
    if (copy_gdesw < 0) copy_gdesw = rt_knob("ASTC_GPU_DESW", "1");
    int use_bl = 0;
    unsigned bl_th = 0, bl_td = 0;
    uint64_t bl_off = 0, bl_span = 0;
    if (!want_probe && RT_KNOB_ON(copy_gdesw) && x == 0 && y == 0 && !nvn_tex_is_linear(&src->d) &&
        src->d.target != 2 && src->d.target != 5 && src->d.target != 6 && src->d.target != 10 && src->d.samples <= 1) {
        uint32_t sw = mip(src->d.width, level), sh = mip(src->d.height, level);
        uint32_t srb = (sw + f.astc_bw - 1) / f.astc_bw * 16, srows = (sh + f.astc_bh - 1) / f.astc_bh;
        if (rb == srb && rows == srows) {
            nvn_tex_level_tiles(&src->d, level, &bl_th, &bl_td);
            bl_off = nvn_tex_level_offset(&src->d, level);
            uint64_t gobs_x = (rb + 63) / 64, blocks_y = (rows + (8u << bl_th) - 1) / (8u << bl_th);
            bl_span = gobs_x * blocks_y * (512ull << bl_th << bl_td);
            if (bl_td == 0 && bl_span &&
                (uint64_t)(layer + layers - 1) * src->L.layer_size + bl_off + bl_span <= src->L.size) {
                use_bl = 1;
                per = (size_t)bl_span;
            }
        }
    }
    int reopen = 0;
    VkCommandBuffer icb = VK_NULL_HANDLE;
    if (!want_probe && vk_exec_recording_fence()) {
        unlock(), locked = 0;
        inline_stg_room(per * layers);   // before vk_exec_inline_begin: a part sent there takes the command buffer
        icb = vk_exec_inline_begin(&reopen);
        if (!icb) lock(), locked = 1;
    }
    VkBuffer sb;
    VmaAllocation sa = NULL;
    uint8_t *stg;
    VkCommandBuffer cb;
    if (icb) {
        cb = icb;
        stg = vk_staging_try(per * layers, &sb, &sa);
    } else {
        vk_upload_flush_over(64ull << 20);
        cb = vk_upload_begin_staging(per * layers, &sb, (void **)&stg);
        if (!cb) stg = NULL;
    }
    if (!stg) {   // no staging memory: the caller copies the images or skips (logged by the staging)
        if (icb) vk_exec_inline_end(reopen);
        if (locked) unlock();
        free(ar);
        return 0;
    }
    int ok = 1;
    for (uint32_t l = 0; l < layers && ok; l++) {
        if (use_bl) {
            uint64_t base = (uint64_t)(layer + l) * src->L.layer_size + bl_off;
            memcpy(stg + l * per, (const uint8_t *)src->mem + base, (size_t)bl_span);
            ar[l] = (VkAstcRegion){.src = l * per, .w = w, .h = h, .slices = 1, .level = dlevel, .layer = dlayer + l,
                                   .offset = {dx, dy, 0}, .lw = mip(dsti->width, dlevel), .lh = mip(dsti->height, dlevel),
                                   .bl_size = bl_span, .bl_rows = rows, .bl_th = (uint8_t)bl_th, .bl_td = 0};
        } else {
            ok = !nvn_texels_box(&src->d, level, layer + l, x / f.astc_bw * 16, y / f.astc_bh, 0, rb, rows, src->mem,
                                 stg + l * per, rb, 1);
            ar[l] = (VkAstcRegion){.src = l * per, .w = w, .h = h, .slices = 1, .level = dlevel, .layer = dlayer + l,
                                   .offset = {dx, dy, 0}, .lw = mip(dsti->width, dlevel), .lh = mip(dsti->height, dlevel)};
        }
    }
    if (ok) zero_census(src, stg, per * layers, level, layer, layers, x, y, w, h);
    VkBuffer rbuf = VK_NULL_HANDLE;
    VmaAllocation ralloc = NULL;
    uint8_t *rmap = NULL;
    if (ok && want_probe) rmap = vk_staging((VkDeviceSize)w * h * 4, &rbuf, &ralloc);
    uint8_t block[16];
    memcpy(block, stg, 16);
    if (icb) {
        vmaFlushAllocation(vk_ctx()->vma, sa, 0, VK_WHOLE_SIZE);
        uint64_t fence = vk_exec_recording_fence();
        if (ok) {
            // a scratch buffer of its own (over FIXED_SCRATCH) waits for this submission, not the upload batches
            uint64_t prev = vk_mem_defer_fence(fence);
            vk_upload_lock();
            barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            vk_astc_tag((uintptr_t)src->mem, 2);
            if (bc7 == 2) vk_astc_bc1_alpha_note(dsti->addr, stg, per * layers, f.astc_bw, f.astc_bh, "copy");
            ok = vk_astc_record(cb, sb, per * layers, ar, (int)layers, f.astc_bw, f.astc_bh, f.astc_srgb, dst, aspect,
                                bc7, VK_NULL_HANDLE);
            barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
            vk_upload_unlock();
            vk_mem_defer_fence(prev);
        }
        vk_exec_inline_end_bound(reopen);
        InlineStg *t = malloc(sizeof *t);
        if (!t) rt_fatal("vk: out of memory (ASTC copy staging)");
        *t = (InlineStg){sb, sa, istg.held};
        istg.held = t, istg.bytes += per * layers;
        vk_completion_add(fence, free_inline_stg, t);
    } else {
        if (ok) {
            barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            vk_astc_tag((uintptr_t)src->mem, 2);
            if (bc7 == 2) vk_astc_bc1_alpha_note(dsti->addr, stg, per * layers, f.astc_bw, f.astc_bh, "copy");
            ok = vk_astc_record(cb, sb, per * layers, ar, (int)layers, f.astc_bw, f.astc_bh, f.astc_srgb, dst, aspect,
                                bc7, rbuf);
            barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        }
        vk_upload_end();
    }
    if (locked) unlock();
    if (rmap) {
        vk_upload_flush(1);
        vmaInvalidateAllocation(vk_ctx()->vma, ralloc, 0, VK_WHOLE_SIZE);
        uint32_t sum[4] = {0};
        for (uint32_t i = 0; i < w * h; i++)
            for (int k = 0; k < 4; k++) sum[k] += rmap[i * 4 + k];
        char hex[33];
        for (int k = 0; k < 16; k++) snprintf(hex + 2 * k, 3, "%02x", block[k]);
        rt_log("vk: ASTC probe: level %u -> %u layer %u %ux%u: block %s, mean RGBA %u %u %u %u", level, dlevel, dlayer,
               w, h, hex, sum[0] / (w * h), sum[1] / (w * h), sum[2] / (w * h), sum[3] / (w * h));
        vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
    }
    free(ar);
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) <= 8)
        rt_log("vk: ASTC %ux%u (fmt %#x) level %u layer %u x%u -> %s level %u layer %u at %d,%d: %s%s", w, h,
               src->d.format, level, layer, layers, bc7 == 2 ? "BC1" : bc7 ? "BC7" : "RGBA8", dlevel, dlayer, dx, dy,
               ok ? "decoded" : "failed", icb ? " (in the replay's command buffer)" : " (upload batch)");
    if (ok) __atomic_fetch_add(&astc_counts[icb ? VK_ASTC_COPY_INLINE : VK_ASTC_COPY_BATCH], 1, __ATOMIC_RELAXED);
    return !ok ? 0 : icb ? VK_TEX_COPY_INLINE : VK_TEX_COPY_BATCH;
}

int vk_texture_cpu_source(const tex_t *t, uint32_t *uploads) {
    *uploads = 0;
    if (!t || !t->mem) return 0;
    lock();
    int r = cpu_content(t, uploads);
    unlock();
    return r;
}

// A texture copy out of a mirrored texture the CPU wrote, into an image of the same VkFormat, from the region of the
// source's storage instead of its image. MKW fills its 8192x8192x68 sparse terrain arrays a level of a layer at a time
// through 8192x8192 staging textures it writes in its 2.4 GiB CPU_CACHED pool (ASTC ones, astc_into_sparse, and BC6H
// ones): each copy found the staging texture's pages written since the last one and uploaded all of it again first,
// 87 MiB for a BC6H one, deswizzled on the replay thread (titleup, 16/11/2026: 242 uploads of one texture in frame 77;
// titlepause: 903 re-uploads, 53.6 GB in 13.8 s, the 3-16 s pause behind the title). Only the copied region is read
// here, from the storage at this point of the replay, where make_fresh would have read it: the same bytes the image
// copy gave, since an upload of such a texture is its storage's raw blocks (no ASTC decode, no RGB expansion, not the
// button icon sheet). The source's entry is not made fresh and its page check is left as it was: a later draw that
// samples it still finds the pages written and uploads it then, once. Recorded into the upload batch, as
// vk_texture_astc_copy: it runs before the submission being recorded, so the caller takes this path only for a
// destination nothing earlier in that submission writes or reads in the region (sparse arrays filled by copies).
// 0 when it cannot: the caller copies as before. *why: what refused it (diagnosis).
int vk_texture_region_copy(const tex_t *src, uint32_t level, uint32_t layer, uint32_t layers, uint32_t x, uint32_t y,
                           uint32_t w, uint32_t h, const VkTexImage *dst, VkImageAspectFlags aspect, uint32_t dlevel,
                           uint32_t dlayer, int32_t dx, int32_t dy, const char **why) {
    static int on = -1;   // MKW_TEX_REGION_COPY=0 (diagnosis): every copy through the source's image, as before
    if (on < 0) on = !cfg_get("MKW_TEX_REGION_COPY") || atoi(cfg_get("MKW_TEX_REGION_COPY")) != 0;
    TexVkFormat f;
    int bw, bh, bpb;
    *why = "off";
    if (!on || !src || !src->mem || !dst) return 0;
    *why = "format";
    if (!tex_vk_format(src->d.format, &f) || f.astc_bw || f.expand_rgb || f.depth || f.stencil ||
        f.vk != dst->format || !(bpb = nvn_fmt_bpb(src->d.format, &bw, &bh)) || aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        src->d.target == NVN_TEXTURE_TARGET_3D || src->d.samples > 1 || dst->samples > 1)
        return 0;
    *why = "alignment";
    if (dx % bw || dy % bh || dx < 0 || dy < 0) return 0;
    uint32_t rb = (w + (uint32_t)bw - 1) / (uint32_t)bw * (uint32_t)bpb, rows = (h + (uint32_t)bh - 1) / (uint32_t)bh;
    size_t per = ((size_t)rb * rows + 15) & ~(size_t)15;   // bufferOffset: a multiple of the block size (1-16 bytes)
    uint64_t p0 = perf_begin(), t0 = kern_now_ns();   // PERF_TEX, as the uploads it replaces
    lock();   // as make_fresh: WriteTexels into the storage happen under it
    uint32_t ups;
    *why = "GPU content";
    if (!cpu_content(src, &ups) || (g_glyph_addr && (uintptr_t)src->mem == g_glyph_addr)) {
        unlock();
        perf_end(PERF_TEX, p0);
        return 0;
    }
    vk_upload_flush_over(64ull << 20);
    VkBuffer sb;
    uint8_t *stg;
    *why = "staging";
    VkCommandBuffer cb = vk_upload_begin_staging(per * layers, &sb, (void **)&stg);
    if (!cb) {
        unlock();
        perf_end(PERF_TEX, p0);
        return 0;
    }
    VkBufferImageCopy rg[8], *r = layers <= 8 ? rg : calloc(layers, sizeof *r);
    int ok = r != NULL;
    for (uint32_t l = 0; l < layers && ok; l++) {
        ok = !nvn_texels_region(&src->d, level, layer + l, x, y, w, h, src->mem, stg + l * per, rb);
        r[l] = (VkBufferImageCopy){.bufferOffset = l * per, .bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw,
                                   .bufferImageHeight = rows * (uint32_t)bh,
                                   .imageSubresource = {aspect, dlevel, dlayer + l, 1}, .imageOffset = {dx, dy, 0},
                                   .imageExtent = {w, h, 1}};
    }
    *why = "region";
    if (ok) {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(cb, sb, dst->image, VK_IMAGE_LAYOUT_GENERAL, layers, r);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        rc_n++, rc_bytes += (uint64_t)rb * rows * layers, rc_ns += kern_now_ns() - t0;
    }
    vk_upload_end();
    if (r != rg) free(r);
    unlock();
    perf_end(PERF_TEX, p0);
    return ok;
}

// The write-watch pages wholly inside e's storage, [*lo, *hi); 0 when it fills none. A write anywhere in a watch page
// marks all of its 4 KiB pages (prot_watch_written), and on the PS5 that page is 16 KiB (plat_mem_watch_page): rounding
// to 4 KiB let a neighbour's write in a shared edge page count as a write to e.
static int watch_span(const Tex *e, uint64_t *lo, uint64_t *hi) {
    uint64_t pg = plat_mem_watch_page();
    *lo = (e->pub.addr + pg - 1) & ~(pg - 1), *hi = (e->pub.addr + e->t.L.size) & ~(pg - 1);
    return *hi > *lo;
}

// vk_mem_changed over the watch pages wholly inside e's storage (all of its pages when it fills none): a neighbour that
// shares the first or last page does not count as a write to e. Storage that fills no page is judged by its bytes
// instead (rewritten_since_gpu).
int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    uint64_t lo, hi;
    if (!watch_span(e, &lo, &hi)) lo = e->pub.addr, hi = e->pub.addr + e->t.L.size;
    return vk_mem_changed(lo, hi - lo, since, now);
}

// Storage that fills no whole page (1.8.0: a 16x1 R16F pitch target, 128 bytes, two of them rendered on alternate
// frames) shares its page with other data, and the page's write watch cannot tell a write to the texture from a write
// to its neighbours: any CPU write on that page counted as a rewrite of the storage, and the GPU's texels were never
// written back. In three Grand Prix races (drop2, RX 6700 XT, 21/11/2026) the page alone dropped 36 864 renders of two
// such textures, ~0.6 a frame, and in every one the texture's own bytes were still those of the GPU write. For such
// storage the bytes are kept at the GPU write and compared instead.
// The page is the write watch's (watch_span), not 4 KiB: on the PS5 it is 16 KiB, and the booth Yoshi's bone keys on
// Mario Bros. Circuit (1.8.0, a 10x184 RGBA16F pitch texture, 0x5c00 bytes from 4 KiB into a 16 KiB page, rendered once
// at the course's load and read by the CPU pose sampler main+0x42bea0 ever after) fill no 16 KiB page. The game's writes
// to its neighbours on those pages dropped the render in about 6 races of 7: the sampler then read the bytes the
// memory's previous pool left, and the Yoshi's palette exploded (cyan, white and black flashes, giant triangles; texdiag
// wwv1 skipped, wwv2 written back, 27/11/2026). Up to two watch pages are kept: 8 KiB with 4 KiB pages, as before, and
// 32 KiB on the PS5, copied once per GPU write of such a texture.
static int fills_no_page(const Tex *e) {
    uint64_t lo, hi;
    return !watch_span(e, &lo, &hi) && e->t.L.size <= 2 * (uint64_t)plat_mem_watch_page();
}

// MKW_TEX_WB_BYTES (default 1; 0: the page decides, as before 21/11/2026; --probe-vkexec's pitch page part fails then)
static int wb_bytes(void) {
    static int on = -1;
    if (on < 0) {
        const char *c = cfg_get("MKW_TEX_WB_BYTES");
        on = !c || !*c || atoi(c) != 0;
    }
    return on;
}

void gpu_base_note(Tex *e) {
    if (!fills_no_page(e)) return;   // (L.size is fixed for the entry: the buffer is sized once)
    if (!e->gpu_base && !(e->gpu_base = malloc(e->t.L.size))) rt_fatal("vk: out of memory for a texture's storage copy");
    mem_copy_out(e->gpu_base, e->pub.addr, e->t.L.size);
}

int gpu_base_kept(const Tex *e) {
    if (!fills_no_page(e) || !e->gpu_base) return -1;
    uint8_t now[0x1000];
    for (uint64_t at = 0; at < e->t.L.size; at += sizeof now) {
        size_t n = e->t.L.size - at < sizeof now ? (size_t)(e->t.L.size - at) : sizeof now;
        mem_copy_out(now, e->pub.addr + at, n);
        if (memcmp(now, (const uint8_t *)e->gpu_base + at, n)) return 0;
    }
    return 1;
}

int rewritten_since_gpu(const Tex *e) {
    uint32_t now;
    if (storage_rewritten(e, e->gpu_gen, &now) != 1) return 0;
    if (!wb_bytes() || gpu_base_kept(e) != 1) return 1;
    static volatile int32_t n_kept;
    log_limited(&n_kept, "texture %#llx %ux%u format %#x (%llu bytes, its page shared): a neighbour was written, its own "
                "bytes were not, so the GPU render goes back (MKW_TEX_WB_BYTES)", (unsigned long long)e->pub.addr,
                e->t.d.width, e->t.d.height, e->t.d.format, (unsigned long long)e->t.L.size);
    return 0;
}

// Without a write watch (vk_mem_changed -1: a kernel without userfaultfd, as the Odin 2's SteamOS ARM 7.2) every check
// uploaded the texture again: three 8192x8192 ASTC textures 627 times each at the first menus (run7, 10/11/2026), ~100
// ms of every frame. The storage's hash, once per submission, tells a rewrite apart: 1 when it is unchanged since the
// last upload. MKW_TEX_HASH_UNTRACKED=1 on any GPU, 0 never; on by default wherever the write watch is missing: also
// in a container whose seccomp profile refuses userfaultfd (#241: Docker's default, an RX 9070 at ~6 fps on the title,
// ~20 with this, 60 with userfaultfd allowed), not only on Qualcomm's kernels.
// The check runs once per submission: a rewrite of the storage later in the same submission is uploaded at the next
// one. That is the point (the 627 re-uploads were checks within a few submissions), and a texture the game rewrites
// mid-submission without a write watch reads the bytes of that submission's start.
static int hash_untracked(void) {
    static int on = -1;
    if (on < 0) {
        const char *c = cfg_get("MKW_TEX_HASH_UNTRACKED");
        uint64_t page;
        int no_watch = plat_mem_written(mem_layout()->heap.base, 0x1000, &page, 1, 0) == UINT64_MAX;
        on = c && *c ? atoi(c) != 0 : no_watch;
        if (no_watch)
            rt_log("vk: no write watch (on Linux, userfaultfd refused: in a container, allow it in its seccomp profile): "
                   "texture storage %s, much slower than with it (MKW_TEX_HASH_UNTRACKED)",
                   on ? "is hashed once per submission instead" : "is uploaded again at every check");
    }
    return on;
}

// Before a texture's first upload: its storage's hash, so the first check after it compares with what was uploaded
// instead of with 0 (an upload more for every texture). Taken before, a write while uploading only costs one upload
// more; taken after, it would hide that write.
static void note_storage(Tex *e) {
    if (!hash_untracked() || !e->t.mem) return;
    e->shash_epoch = submit_epoch;
    e->shash = vk_texpre_hash(e->t.mem, e->t.L.size);
}

static int same_storage(Tex *e) {
    if (!hash_untracked()) return 0;
    if (e->shash_epoch == submit_epoch) return 1;   // checked (and uploaded if needed) in this submission
    e->shash_epoch = submit_epoch;
    uint64_t h = vk_texpre_hash(e->t.mem, e->t.L.size);
    if (h == e->shash) return 1;
    e->shash = h;
    return 0;
}

int tex_has_gpu_content(const Tex *e) {
    if (!e->pub.gpu_owned) return 0;
    if (!e->pub.mirrored) return 1;
    uint32_t now;
    (void)now;
    return e->gpu_dirty && !rewritten_since_gpu(e);   // a neighbour on a shared page is not a rewrite (#740)
}

void gpu_baseline(Tex *e) {
    storage_rewritten(e, 0, &e->gpu_gen);   // queries the write watch now: writes up to here carry gpu_gen or older
    e->cpu_newer = 0;
}

// A GPU-owned mirrored texture whose storage the CPU wrote after the GPU's last write (gpu_gen), or whose WriteTexels
// reached the storage only (cpu_newer), goes back to the CPU: on the console the storage is the texture, so the next
// sample sees those bytes. Before, gpu_owned was never cleared (make_fresh returned on it, vk_texture_write_back only
// dropped gpu_dirty, and a released entry taken back kept it), so such a texture showed its last render for good. The
// rule is vk_texture_write_back's and tex_has_gpu_content's: a CPU write after the render makes the storage the
// content. Only a write the page check saw (1); untracked storage (-1) says nothing about the order. From the
// submission's write-watch snapshot (vk_mem_changed_snap): render targets come through make_fresh at every pass, and a
// write the snapshot missed shows at the next submission. Only the watch pages wholly inside the storage count, and a
// storage inside a single watch page is never taken back by a write: unlike storage_rewritten, which keeps the content
// a render added, a wrong answer here overwrites a live render with the storage's bytes every pass (flicker), so a
// write that may be a neighbour's says nothing. 1 when the CPU owns it again (the caller uploads).
static int cpu_took_back(Tex *e) {
    static int logged;
    int why = 1;   // 1 WriteTexels reached the storage only, 2 a CPU write the watch saw
    if (!e->cpu_newer) {
        uint64_t lo, hi;
        uint32_t now;
        if (!watch_span(e, &lo, &hi) || vk_mem_changed_snap(lo, hi - lo, e->gpu_gen, submit_epoch, &now) != 1) return 0;
        why = 2;
    }
    if (logged < 32) {
        logged++;
        rt_log("vk: texture %#llx %ux%u (%llu bytes) taken back by the CPU after a GPU write: %s%s",
               (unsigned long long)e->pub.addr, e->pub.width, e->pub.height, (unsigned long long)e->t.L.size,
               why == 1 ? "a WriteTexels reached the storage only" : "the storage was written",
               logged == 32 ? " (no more logged)" : "");
    }
    e->pub.gpu_owned = 0, e->gpu_dirty = 0, e->cpu_newer = 0;
    return 1;
}

// MKW_REUP_SAME (default 0; "ab": in the B blocks): a written texture (write-watch generation newer than its last
// upload) whose bytes hash as they did just before its last upload is not uploaded again: the image already holds
// them. A 1.8.0 race re-uploads ~195 textures (~36 MiB, ~11 ms of the replay's time) per 60 frames, the uploads' share
// of the replay thread that makes the game's fence wait late (hitch1, 25/11/2026). Only for a write-watched change:
// not a deferred upload (the image is older), a take-back from the GPU (the image holds the GPU's render) or an image
// the last hash does not belong to (rhash_nup). The hash is taken before the upload, so a write made during it leaves
// the hash older than the image and the next check uploads again.
// MKW_REUP_CENSUS=1 (diagnosis): every 1,800 frames, the re-uploaded textures by bytes, and how many had the bytes of
// their last upload.
static int reup_mode = -1, reup_census;
static uint64_t reup_skipped, reup_skipped_bytes;   // tex_lock

typedef struct { uint64_t addr, bytes, same_bytes; uint32_t w, h, fmt, n, same, written, untracked, pending, took, unknown; } RuCen;
enum { RUC_N = 2048 };
static RuCen ruc[RUC_N];
static uint64_t ruc_frame;

static int ruc_cmp(const void *a, const void *b) {
    uint64_t x = ((const RuCen *)a)->bytes, y = ((const RuCen *)b)->bytes;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void ruc_log(uint64_t fr) {
    uint64_t n = 0, same = 0, bytes = 0, sbytes = 0;
    int k = 0;
    for (int i = 0; i < RUC_N; i++)
        if (ruc[i].n) n += ruc[i].n, same += ruc[i].same, bytes += ruc[i].bytes, sbytes += ruc[i].same_bytes, ruc[k++] = ruc[i];
    qsort(ruc, (size_t)k, sizeof *ruc, ruc_cmp);
    rt_log("vk: re-upload census frames %llu-%llu: %llu re-uploads of %d textures, %.1f MiB; %llu (%.1f MiB) had the bytes "
           "of their last upload; skipped %llu (%.1f MiB) (MKW_REUP_SAME)", (unsigned long long)ruc_frame,
           (unsigned long long)fr, (unsigned long long)n, k, bytes / 1048576.0, (unsigned long long)same,
           sbytes / 1048576.0, (unsigned long long)reup_skipped, reup_skipped_bytes / 1048576.0);
    for (int i = 0; i < k && i < 16; i++)
        rt_log("vk:   %#llx %ux%u fmt %#x: %u re-uploads %.1f MiB, same bytes %u; by write %u, untracked %u, deferred %u, "
               "taken back %u, hash unknown %u", (unsigned long long)ruc[i].addr, ruc[i].w, ruc[i].h, ruc[i].fmt, ruc[i].n,
               ruc[i].bytes / 1048576.0, ruc[i].same, ruc[i].written, ruc[i].untracked, ruc[i].pending, ruc[i].took,
               ruc[i].unknown);
    memset(ruc, 0, sizeof ruc);
    ruc_frame = fr, reup_skipped = reup_skipped_bytes = 0;
}

static void ruc_note(const Tex *e, int ch, int took, int valid, int same) {
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!ruc_frame) ruc_frame = fr;
    if (fr >= ruc_frame + 1800) ruc_log(fr);
    uint64_t a = e->pub.addr;
    uint32_t i = (uint32_t)((a >> 8) * 0x9E3779B97F4A7C15ull >> 53) & (RUC_N - 1);
    for (int p = 0; p < RUC_N && ruc[i].n && ruc[i].addr != a; p++) i = (i + 1) & (RUC_N - 1);
    RuCen *r = &ruc[i];
    if (r->n && r->addr != a) return;   // full
    r->addr = a, r->w = e->t.d.width, r->h = e->t.d.height, r->fmt = (uint32_t)e->t.d.format;
    r->n++, r->bytes += e->t.L.size;
    if (same) r->same++, r->same_bytes += e->t.L.size;
    if (e->up_pending) r->pending++;
    else if (took) r->took++;
    else if (ch < 0) r->untracked++;
    else r->written++;
    if (!valid) r->unknown++;
}

// Before an upload of e (uploaded or not): 1 when it can be left out (MKW_REUP_SAME). *h gets the storage's hash when
// one was taken (*hv), for reup_after.
static int reup_skip(Tex *e, int ch, int took, uint64_t *h, int *hv) {
    if (reup_mode < 0) {
        reup_mode = rt_knob("REUP_SAME", "0");
        reup_census = rt_knob("REUP_CENSUS", "0") != 0;
        if (reup_mode || reup_census)
            rt_log("vk: re-uploads of unchanged bytes %s%s (MKW_REUP_SAME, MKW_REUP_CENSUS)",
                   reup_mode == 2 ? "left out in the B blocks" : reup_mode ? "left out" : "counted",
                   reup_census ? ", census every 1800 frames" : "");
    }
    *hv = 0;
    int on = RT_KNOB_ON(reup_mode);
    if ((!on && !reup_census) || !e->t.mem || !e->t.L.size) return 0;
    *h = vk_texpre_hash(e->t.mem, e->t.L.size), *hv = 1;
    if (!e->uploaded) return 0;
    int valid = e->n_up && e->rhash_nup == e->n_up;
    int same = valid && *h == e->rhash;
    if (reup_census) ruc_note(e, ch, took, valid, same);
    if (!on || !same || ch != 1 || e->up_pending || took) return 0;
    reup_skipped++, reup_skipped_bytes += e->t.L.size;
    return 1;
}

static void reup_after(Tex *e, uint32_t nup_before, uint64_t h, int hv) {
    if (hv && e->n_up != nup_before) e->rhash = h, e->rhash_nup = e->n_up;
}

// A mirrored texture is uploaded on first use and again when its storage pages carry a newer write generation.
// seen first: staging that runs short trims textures idle by it on this same thread (vk_texture_trim, recursive
// lock), and the caller goes on using e; the pin keeps it for the rest of the upload.
void make_fresh(Tex *e) {
    e->seen = submit_epoch;
    if (!e->pub.mirrored) return;
    int took = e->pub.gpu_owned && cpu_took_back(e);
    if (e->pub.gpu_owned) return;
    tex_pin(e);
    uint64_t size = e->t.L.size;
    uint64_t h = 0;
    int hv = 0;
    uint32_t nup = e->n_up;
    if (!e->uploaded) {
        vk_mem_changed(e->pub.addr, size, 0, &e->wgen);   // resets the write bits before the storage is read
        note_storage(e);
        reup_skip(e, 0, 0, &h, &hv);   // (the hash only)
        upload(e, 0);
        reup_after(e, nup, h, hv);
    } else {
        uint32_t now, prev_wgen = e->wgen;
        int ch = vk_mem_changed(e->pub.addr, size, prev_wgen, &now);
        e->wgen = now;
        e->last_ch = ch;
        // 1 written, -1 not tracked (the storage's hash decides, unless the CPU just took it back): upload again; a
        // deferred one too
        if (ch < 0 && e->t.mem && !e->up_pending && !took && same_storage(e)) {
        } else if ((ch || e->up_pending || took) && !reup_skip(e, ch, took, &h, &hv)) {
            upload(e, ch == 1 && !e->up_pending && !took ? prev_wgen : 0);
            reup_after(e, nup, h, hv);
        }
    }
    tex_unpin(e);
}

// make_fresh for draw bindings: the page check answers from this submission's write-watch snapshot (vk_memory.c
// vk_mem_changed_snap) instead of a system call per texture. A write the snapshot of the texture's region missed (made
// after it, during this replay) keeps its write-watch bit and is caught by the next submission's snapshot.
// MKW_TRACE=perf: what the checks spend (vk_exec_par.c precheck's split): the snapshot answer, and the uploads.
uint64_t vk_pc_snap_ns, vk_pc_up_n, vk_pc_up_ns;

// MKW_SNAP_SCOPED (vk_mem_gen.c vk_mem_changed_scoped; default 0, 1 in the PS5 profile; "ab"): on now.
int tex_snap_scoped(void) {
    static int scoped = -1;
    if (scoped < 0) {
        int m = rt_knob("SNAP_SCOPED", "0");
        if (m) rt_log("vk: texture checks peek their own span, no 2 MiB snapshot%s (MKW_SNAP_SCOPED)",
                      m == 2 ? " in the B blocks" : "");
        scoped = m;
    }
    return scoped && RT_KNOB_ON(scoped);
}

void make_fresh_snap(Tex *e) {
    e->seen = submit_epoch;   // as make_fresh
    if (!e->pub.mirrored) return;
    int took = e->pub.gpu_owned && cpu_took_back(e);
    if (e->pub.gpu_owned) return;
    uint32_t now, prev_wgen = e->uploaded ? e->wgen : 0;
    n_checks++;
    int perf = (rt_trace & RT_TRACE_PERF) != 0;
    uint64_t t0 = perf ? kern_now_ns() : 0;
    int ch = tex_snap_scoped() ? vk_mem_changed_scoped(e->pub.addr, e->t.L.size, prev_wgen, &now)
                               : vk_mem_changed_snap(e->pub.addr, e->t.L.size, prev_wgen, submit_epoch, &now);
    uint64_t t1 = perf ? kern_now_ns() : 0;
    if (perf) __atomic_add_fetch(&vk_pc_snap_ns, t1 - t0, __ATOMIC_RELAXED);
    e->wgen = now;
    if (e->uploaded && !ch && !e->up_pending && !took) {
        e->last_ch = 0;
        return;   // clean fast path: skip tex_pin/unpin
    }
    tex_pin(e);
    int up = 0, hv = 0;
    uint64_t h = 0;
    uint32_t nup = e->n_up;
    if (!e->uploaded) {
        note_storage(e);
        reup_skip(e, 0, 0, &h, &hv);   // (the hash only)
        upload(e, 0), up = 1;
        reup_after(e, nup, h, hv);
    } else {
        e->last_ch = ch;
        // not tracked (-1): the storage's hash decides, unless the CPU just took the texture back from the GPU
        if (ch < 0 && e->t.mem && !e->up_pending && !took && same_storage(e)) {
        } else if ((ch || e->up_pending || took) && !reup_skip(e, ch, took, &h, &hv)) {
            upload(e, ch == 1 && !e->up_pending && !took ? prev_wgen : 0), up = 1;
            reup_after(e, nup, h, hv);
        }
    }
    if (perf && up)
        __atomic_add_fetch(&vk_pc_up_n, 1, __ATOMIC_RELAXED),
            __atomic_add_fetch(&vk_pc_up_ns, kern_now_ns() - t1, __ATOMIC_RELAXED);
    tex_unpin(e);
}

