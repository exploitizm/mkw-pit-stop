// ASTC textures on the GPU: the MKW_ASTC knob, a compute decoder that turns raw blocks into the RGBA8 image the CPU
// decoder (tex_astc.c) would have written, and the native ASTC formats where the device samples them itself.
//
// MKW_ASTC = auto (default): native when the device reports textureCompressionASTC_LDR and the format's features,
// else gpu. gpu: compute decode into RGBA8 (runtime/shaders/texture/astc_decode.comp). native: like auto, but logs when the
// device cannot. cpu: astc_decode_ldr_mt on the uploading thread, as before. check: gpu, and every upload is also
// decoded on the CPU and compared (waits for the GPU; for testing only).
//
// The decode is recorded into the upload batch (vk_memory.c), or a replay part's command buffer at the place of the copy
// it stands for (vk_texture_astc_copy), in chunks of at most 32 MiB of texels (a larger region in
// bands of block rows, split_regions): a chunk's raw blocks go from staging to a device-local scratch buffer, its
// dispatches write RGBA8 texels behind them, and a buffer-to-image copy puts them into the image; the next chunk reuses
// the scratch after a barrier. The scratch is one kept buffer (FIXED_SCRATCH); only a call with a readback takes one of
// its own, freed once the batch completes (vk_defer_buffer).
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "astc_decode_spv.h"
#include "bc1_encode_spv.h"
#include "bc7_encode_spv.h"
#include "bl_deswizzle_spv.h"
#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "tex.h"
#include "vk.h"
#include "platform/platform.h"

// tex_astc.c's unquantization tables, as the shader reads them (Tables in astc_decode.comp): 12 + 16 row offsets in
// bytes, then the bytes. Kept in step with tex_astc.c by the oracle (--probe-astc), which compares every mode.
static const uint8_t wunq_data[] = {
    0,64,
    0,32,64,
    0,21,43,64,
    0,16,32,48,64,
    0,64,12,52,25,39,
    0,9,18,27,37,46,55,64,
    0,64,7,57,14,50,21,43,28,36,
    0,64,17,47,5,59,23,41,11,53,28,36,
    0,4,8,12,17,21,25,29,35,39,43,47,52,56,60,64,
    0,64,16,48,3,61,19,45,6,58,23,41,9,55,26,38,13,51,29,35,
    0,64,8,56,16,48,24,40,2,62,11,53,19,45,27,37,5,59,13,51,22,42,30,34,
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,
};
static const uint16_t wunq_off[12] = {0,2,5,9,14,20,28,38,50,66,86,110};
static const uint8_t cunq_data[] = {
    0,255,51,204,102,153,
    0,36,73,109,146,182,219,255,
    0,255,28,227,56,199,84,171,113,142,
    0,255,69,186,23,232,92,163,46,209,116,139,
    0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,
    0,255,67,188,13,242,80,175,27,228,94,161,40,215,107,148,54,201,121,134,
    0,255,33,222,66,189,99,156,11,244,44,211,77,178,110,145,22,233,55,200,88,167,121,134,
    0,8,16,24,33,41,49,57,66,74,82,90,99,107,115,123,132,140,148,156,165,173,181,189,198,206,214,222,
    231,239,247,255,
    0,255,32,223,65,190,97,158,6,249,39,216,71,184,104,151,13,242,45,210,78,177,110,145,19,236,52,203,
    84,171,117,138,26,229,58,197,91,164,123,132,
    0,255,16,239,32,223,48,207,65,190,81,174,97,158,113,142,5,250,21,234,38,217,54,201,70,185,86,169,
    103,152,119,136,11,244,27,228,43,212,59,196,76,179,92,163,108,147,124,131,
    0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,65,69,73,77,81,85,89,93,97,101,105,109,113,117,121,125,
    130,134,138,142,146,150,154,158,162,166,170,174,178,182,186,190,195,199,203,207,211,215,219,223,227,
    231,235,239,243,247,251,255,
    0,255,16,239,32,223,48,207,64,191,80,175,96,159,112,143,3,252,19,236,35,220,51,204,67,188,83,172,
    100,155,116,139,6,249,22,233,38,217,54,201,71,184,87,168,103,152,119,136,9,246,25,230,42,213,58,197,
    74,181,90,165,106,149,122,133,13,242,29,226,45,210,61,194,77,178,93,162,109,146,125,130,
    0,255,8,247,16,239,24,231,32,223,40,215,48,207,56,199,64,191,72,183,80,175,88,167,96,159,104,151,
    112,143,120,135,2,253,10,245,18,237,26,229,35,220,43,212,51,204,59,196,67,188,75,180,83,172,91,164,
    99,156,107,148,115,140,123,132,5,250,13,242,21,234,29,226,37,218,45,210,53,202,61,194,70,185,78,177,
    86,169,94,161,102,153,110,145,118,137,126,129,
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,
    70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,
    126,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,
    177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,
    227,229,231,233,235,237,239,241,243,245,247,249,251,253,255,
    0,255,8,247,16,239,24,231,32,223,40,215,48,207,56,199,64,191,72,183,80,175,88,167,96,159,104,151,
    112,143,120,135,1,254,9,246,17,238,25,230,33,222,41,214,49,206,57,198,65,190,73,182,81,174,89,166,
    97,158,105,150,113,142,121,134,3,252,11,244,19,236,27,228,35,220,43,212,51,204,59,196,67,188,75,180,
    83,172,91,164,99,156,107,148,115,140,123,132,4,251,12,243,20,235,28,227,36,219,44,211,52,203,60,195,
    68,187,76,179,84,171,92,163,100,155,108,147,116,139,124,131,6,249,14,241,22,233,30,225,38,217,46,
    209,54,201,62,193,70,185,78,177,86,169,94,161,102,153,110,145,118,137,126,129,
    0,255,4,251,8,247,12,243,16,239,20,235,24,231,28,227,32,223,36,219,40,215,44,211,48,207,52,203,56,
    199,60,195,64,191,68,187,72,183,76,179,80,175,84,171,88,167,92,163,96,159,100,155,104,151,108,147,
    112,143,116,139,120,135,124,131,1,254,5,250,9,246,13,242,17,238,21,234,25,230,29,226,33,222,37,218,
    41,214,45,210,49,206,53,202,57,198,61,194,65,190,69,186,73,182,77,178,81,174,85,170,89,166,93,162,
    97,158,101,154,105,150,109,146,113,142,117,138,121,134,125,130,2,253,6,249,10,245,14,241,18,237,22,
    233,26,229,30,225,34,221,38,217,42,213,46,209,50,205,54,201,58,197,62,193,66,189,70,185,74,181,78,
    177,82,173,86,169,90,165,94,161,98,157,102,153,106,149,110,145,114,141,118,137,122,133,126,129,
};
static const uint16_t cunq_off[16] = {0,6,14,24,36,52,72,96,128,168,216,280,360,456,584,744};
_Static_assert(sizeof wunq_data == 142 && sizeof cunq_data == 936, "unquantization table size");

enum { TAB_BYTES = 28 * 4, TAB_WUNQ = TAB_BYTES, TAB_CUNQ = TAB_BYTES + 144, TAB_SIZE = TAB_CUNQ + 936 };

typedef struct {
    uint32_t in_base, out_base, width, height, bw, bh, bx, bslice, srgb, err_slot;
    uint32_t bl_mode, bl_th, bl_td, bl_y0, bl_z0, bl_gobs_x;
} Push;

static PlatMutex g_lock = PLAT_MUTEX_INIT;
static struct {
    VkDevice dev;                 // the device the objects below belong to (probes recreate it)
    int mode;                     // VK_ASTC_*; -1 before the first vk_astc_mode with a device
    int failed;                   // pipeline creation failed on dev: gpu falls back to cpu
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkPipeline enc;               // bc7_encode.comp, same layout (bindings 0 and 1); NULL: no BC7 re-encoding
    VkPipeline enc1;              // bc1_encode.comp, same layout; NULL: no BC1
    VkPipeline bl_pipe;           // bl_deswizzle.comp, same layout (bindings 0 and 1); generic block-linear deswizzle
    int bc7;                      // vk_astc_bc7's answer on dev; -1 not decided
    PFN_vkCmdPushDescriptorSetKHR push;
    VkBuffer tab;                 // device-local copy of the tables
    VmaAllocation tab_alloc;
    int tab_uploaded;
    VkBuffer err;                 // MKW_ASTC_ERRORS' counters (binding 3), host-visible and mapped
    VmaAllocation err_alloc;
    volatile uint32_t *err_map;
} G = {.mode = -1, .bc7 = -1};

// MKW_ASTC_ERRORS=N (diagnosis, 16/11/2026: flat magenta on scenery, a road edge and karts in some PS5 runs, at random,
// not zero guest bytes: MKW_TEX_ZERO_LOG saw none): every vk_astc_record call gets a counter (a slot of ERR_SLOTS, 0
// unused) that the decoder raises for each block that decodes to the error colour. A call err_lag calls later, or
// when its slot comes round again, its count is read and the first N calls with errors are logged with the caller's
// tag (vk_astc_tag: the texture and which path recorded it).
#define ERR_SLOTS 8192
#define ERR_LEVELS 16   // a counter per mip level (the last one also counts deeper levels) in each slot
static int err_on = -1, err_lag = 256;   // MKW_ASTC_ERRORS_LAG: calls before a count is read (1 for the probe)
static struct {
    uint64_t addr, frame, key, img, call;   // key: the destination (stuck_key); call: err_calls when recorded
    uint32_t w, h, regions, seen, level, layer;
    int32_t x, y;
    uint16_t fp;
    uint8_t path, used;
} err_meta[ERR_SLOTS];
static uint32_t err_next = 1, err_logged, err_calls, err_bad;
static uint64_t err_blocks;
// One line per texture, the first time it has errors (a texture copied level by level made 300 lines in one frame of
// the boot, astcerr-20261116-095851); the rest goes into a summary every 600 frames.
#define ERR_SEEN 1024
static uint64_t err_seen[ERR_SEEN];
static uint32_t err_win_calls, err_win_bad, err_win_new;
static uint64_t err_win_blocks, err_win_frame;
static int err_seen_add(uint64_t a) {   // 1 when a is new
    for (uint32_t h = (uint32_t)(a >> 9) * 2654435761u, i = 0; i < ERR_SEEN; i++) {
        uint64_t *e = &err_seen[(h + i) % ERR_SEEN];
        if (*e == a) return 0;
        if (!*e) return *e = a, 1;
    }
    return 0;   // full: counted in the summary only
}
static _Thread_local struct { uint64_t addr; int path; } tag;
static const char *const err_path[] = {"other", "upload", "astc-copy", "write-texels", "probe"};

void vk_astc_tag(uint64_t addr, int path) { tag.addr = addr, tag.path = path; }

// With MKW_ASTC_ERRORS on, the result of the latest decode into each destination, to tell the magenta that stays on
// screen from the error blocks a later decode replaces (17/11/2026: the same kart textures decoded with errors at the
// same frames in every run, while the magenta seen was random and mostly terrain). A destination is the image and the
// first region's level, layer and offset, with the region count: a texture's uploads all have the same one, and a copy
// into a terrain array has that layer's level and place. Every 600 frames (err_summary) the destinations whose latest
// decode has error blocks are counted, and the ones not listed before are logged ("ASTC stuck"). A destination whose
// image was destroyed since stays in the table: a line's frame and size tell those from the live ones.
#define STUCK_N 16384
#define STUCK_LINES 32   // per summary
static struct {
    uint64_t key, addr, img, frame, call;
    uint32_t err, w, h, level, layer, regions;
    int32_t x, y;
    uint16_t fp;
    uint8_t path, listed;
} stuck[STUCK_N];
static uint32_t stuck_fixed, stuck_full;

static uint64_t stuck_key(uint64_t img, const VkAstcRegion *r, int n) {
    uint64_t k = img * 0x9E3779B97F4A7C15ull;
    uint64_t v[5] = {r->level, r->layer, (uint32_t)r->offset.x, (uint32_t)r->offset.y, (uint64_t)n};
    for (int i = 0; i < 5; i++) k = (k ^ v[i]) * 0x100000001B3ull;
    return k | 1;   // never 0, the empty slot
}

static void stuck_note(uint32_t s, uint32_t n) {   // the upload lock (err_check)
    uint64_t key = err_meta[s].key;
    if (!key) return;
    uint32_t i = (uint32_t)(key >> 20) % STUCK_N, k = 0;
    for (; k < STUCK_N && stuck[i].key && stuck[i].key != key; k++) i = (i + 1) % STUCK_N;
    if (k == STUCK_N) { stuck_full++; return; }
    if (stuck[i].key && stuck[i].call > err_meta[s].call) return;   // a newer decode of it was read already
    if (stuck[i].key && stuck[i].err && !n) stuck_fixed++;
    if (!stuck[i].key || !stuck[i].err || stuck[i].call != err_meta[s].call) stuck[i].listed = 0;
    stuck[i].key = key, stuck[i].call = err_meta[s].call, stuck[i].err = n;
    stuck[i].addr = err_meta[s].addr, stuck[i].img = err_meta[s].img, stuck[i].frame = err_meta[s].frame;
    stuck[i].w = err_meta[s].w, stuck[i].h = err_meta[s].h, stuck[i].level = err_meta[s].level;
    stuck[i].layer = err_meta[s].layer, stuck[i].regions = err_meta[s].regions, stuck[i].x = err_meta[s].x;
    stuck[i].y = err_meta[s].y, stuck[i].fp = err_meta[s].fp, stuck[i].path = err_meta[s].path;
}

static void stuck_summary(uint64_t fr) {
    uint32_t n = 0, listed = 0, shown = 0;
    for (uint32_t i = 0; i < STUCK_N; i++) {
        if (!stuck[i].key || !stuck[i].err) continue;
        n++;
        if (stuck[i].listed) { listed++; continue; }
        stuck[i].listed = 1;
        if (shown++ >= STUCK_LINES) continue;
        rt_log("vk: ASTC stuck: %u error blocks in the latest decode, by %s at frame %llu: source %#llx %ux%u ASTC %ux%u, "
               "%u regions, into image %#llx level %u layer %u at %d,%d",
               stuck[i].err, err_path[stuck[i].path < 5 ? stuck[i].path : 0], (unsigned long long)stuck[i].frame,
               (unsigned long long)stuck[i].addr, stuck[i].w, stuck[i].h, stuck[i].fp / 13, stuck[i].fp % 13,
               stuck[i].regions, (unsigned long long)stuck[i].img, stuck[i].level, stuck[i].layer, stuck[i].x,
               stuck[i].y);
    }
    rt_log("vk: ASTC stuck at frame %llu: %u destinations whose latest decode has error blocks (%u listed before, %u new%s); "
           "%u fixed by a later decode so far%s", (unsigned long long)fr, n, listed, shown,
           shown > STUCK_LINES ? ", the first ones listed" : "", stuck_fixed, stuck_full ? " (table full: some not kept)" : "");
}

// What the blocks of a texture's first level are (every 16 bytes is one block in any layout), to tell why the decoder
// made the error colour: zero (not written, or layout padding), void extents (an HDR one, bit 9, is an error to an LDR
// decoder), reserved block modes, one-partition blocks with an HDR colour endpoint mode (2, 3, 7, 11, 14, 15: also
// errors to an LDR decoder), and the rest. Khronos Data Format Specification 1.3, ASTC (block mode, void extent, CEM).
static void err_classify(uint64_t a, uint32_t w, uint32_t h, uint32_t bw, uint32_t bh, char *out, size_t cap) {
    uint64_t n = (uint64_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh), zero = 0, vldr = 0, vhdr = 0, resv = 0, hdr = 0,
             multi = 0, ldr = 0, vbits = 0, vcoord = 0;
    const uint8_t *p = (const uint8_t *)(uintptr_t)a;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t lo, hi;
        memcpy(&lo, p + i * 16, 8), memcpy(&hi, p + i * 16 + 8, 8);
        if (!lo && !hi) { zero++; continue; }
        unsigned m = (unsigned)(lo & 0x7ff);
        if ((m & 0x1ff) == 0x1fc) {
            if (m & 0x200) { vhdr++; continue; }
            vldr++;
            uint64_t s0 = lo >> 12 & 0x1fff, s1 = lo >> 25 & 0x1fff, t0 = (lo >> 38 & 0x1fff), t1 = (lo >> 51 | hi << 13) & 0x1fff;
            int ones = s0 == 0x1fff && s1 == 0x1fff && t0 == 0x1fff && t1 == 0x1fff;
            if ((lo >> 10 & 3) != 3) vbits++;
            else if ((s0 >= s1 || t0 >= t1) && !ones) vcoord++;
            continue;
        }
        if ((m & 3) == 0 && ((m >> 2) & 3) == 0) { resv++; continue; }
        if ((m & 3) == 0 && ((m >> 7) & 3) == 3 && ((m >> 5) & 3) >= 2) { resv++; continue; }
        if (((lo >> 11) & 3) != 0) { multi++; continue; }
        unsigned cem = (unsigned)((lo >> 13) & 15);
        if (cem == 2 || cem == 3 || cem == 7 || cem == 11 || cem == 14 || cem == 15) hdr++;
        else ldr++;
    }
    snprintf(out, cap, "level 0's %llu blocks: zero %llu, void extent %llu (reserved bits not 11 %llu, coordinates min >= max "
             "%llu) + HDR %llu, reserved mode %llu, HDR endpoints %llu, LDR one partition %llu, several partitions %llu",
             (unsigned long long)n, (unsigned long long)zero, (unsigned long long)vldr, (unsigned long long)vbits,
             (unsigned long long)vcoord, (unsigned long long)vhdr,
             (unsigned long long)resv, (unsigned long long)hdr, (unsigned long long)ldr, (unsigned long long)multi);
}

static void err_check(uint32_t s, const char *when) {   // the caller holds the upload lock (vk_astc_record)
    if (!err_meta[s].used || !G.err_map) return;
    vmaInvalidateAllocation(vk_ctx()->vma, G.err_alloc, (VkDeviceSize)s * ERR_LEVELS * 4, ERR_LEVELS * 4);
    uint32_t n = 0, lv[ERR_LEVELS];
    for (int i = 0; i < ERR_LEVELS; i++) lv[i] = G.err_map[s * ERR_LEVELS + i], n += lv[i];
    stuck_note(s, n);
    if (n <= err_meta[s].seen) return;
    if (!err_meta[s].seen) err_bad++, err_win_bad++;
    err_blocks += n - err_meta[s].seen, err_win_blocks += n - err_meta[s].seen;
    int first = !err_meta[s].seen && err_seen_add(err_meta[s].addr ? err_meta[s].addr : 1);
    err_meta[s].seen = n;
    if (first) err_win_new++;
    if (first && err_logged < (uint32_t)err_on) {
        // the guest bytes now: backed, and whether the first 4 KiB are still zero (not written yet)
        uint64_t a = err_meta[s].addr, backed = a ? plat_mem_backed_bytes(a, 0x10000) : 0;
        int zero = -1;
        if (a && backed) {
            const uint64_t *q = (const uint64_t *)(uintptr_t)a;
            zero = 1;
            for (int k = 0; k < 512 && zero; k++) zero = !q[k];
        }
        err_logged++;
        rt_log("vk: ASTC errors: %u blocks of error colour in texture %#llx %ux%u ASTC %ux%u, %u regions, by %s at frame "
               "%llu (%s; %u of %u calls with errors, %llu blocks)",
               n, (unsigned long long)err_meta[s].addr, err_meta[s].w, err_meta[s].h, err_meta[s].fp / 13,
               err_meta[s].fp % 13, err_meta[s].regions, err_path[err_meta[s].path < 5 ? err_meta[s].path : 0],
               (unsigned long long)err_meta[s].frame, when, err_bad, err_calls, (unsigned long long)err_blocks);
        char lvs[256];
        int o = 0;
        for (int i = 0; i < ERR_LEVELS && o < (int)sizeof lvs - 16; i++)
            if (lv[i]) o += snprintf(lvs + o, sizeof lvs - (size_t)o, " L%d %u", i, lv[i]);
        rt_log("vk: ASTC errors: texture %#llx now: %.0f of 64 KiB backed, first 4 KiB %s; by level:%s", (unsigned long long)a,
               backed / 1024.0, zero < 0 ? "not readable" : zero ? "zero" : "not zero", lvs);
        uint32_t cbw = err_meta[s].fp / 13, cbh = err_meta[s].fp % 13;
        uint64_t l0 = (uint64_t)((err_meta[s].w + cbw - 1) / cbw) * ((err_meta[s].h + cbh - 1) / cbh) * 16;
        if (a && backed && err_meta[s].path == 1 && plat_mem_backed_bytes(a, l0) == l0) {
            char cls[320];
            err_classify(a, err_meta[s].w, err_meta[s].h, err_meta[s].fp / 13, err_meta[s].fp % 13, cls, sizeof cls);
            rt_log("vk: ASTC errors: texture %#llx %s", (unsigned long long)a, cls);
        }
    }
}

static void err_summary(void) {   // every 600 frames, from err_slot (the upload lock)
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (fr / 600 == err_win_frame / 600) return;
    rt_log("vk: ASTC errors at frame %llu: %u calls since frame %llu, %u with errors (%u textures not seen before), %llu "
           "blocks; %u of %u calls with errors in all",
           (unsigned long long)fr, err_win_calls, (unsigned long long)err_win_frame, err_win_bad, err_win_new,
           (unsigned long long)err_win_blocks, err_bad, err_calls);
    err_win_frame = fr, err_win_calls = err_win_bad = err_win_new = 0, err_win_blocks = 0;
    stuck_summary(fr);
}

// A slot for this call (0 when off), after reading the call ERR_LAG back and the slot's last user.
static uint32_t err_slot(const VkAstcRegion *rg, int n, uint32_t bw, uint32_t bh, VkImage img) {
    if (err_on < 0) {
        err_on = cfg_get("MKW_ASTC_ERRORS") ? atoi(cfg_get("MKW_ASTC_ERRORS")) : 0;
        const char *lag = cfg_get("MKW_ASTC_ERRORS_LAG");
        if (lag && atoi(lag) > 0 && atoi(lag) < ERR_SLOTS - 1) err_lag = atoi(lag);
    }
    if (err_on <= 0 || !G.err_map) return 0;
    uint32_t s = err_next;
    err_next = err_next + 1 < ERR_SLOTS ? err_next + 1 : 1;
    uint32_t back = s > (uint32_t)err_lag ? s - (uint32_t)err_lag : s + ERR_SLOTS - 1 - (uint32_t)err_lag;
    err_check(back, "read later");
    err_check(s, "slot reused");
    err_summary();
    err_win_calls++;
    for (int i = 0; i < ERR_LEVELS; i++) G.err_map[s * ERR_LEVELS + i] = 0;
    vmaFlushAllocation(vk_ctx()->vma, G.err_alloc, (VkDeviceSize)s * ERR_LEVELS * 4, ERR_LEVELS * 4);
    err_meta[s].addr = tag.addr, err_meta[s].path = (uint8_t)tag.path;
    err_meta[s].frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    err_meta[s].w = rg[0].w, err_meta[s].h = rg[0].h, err_meta[s].regions = (uint32_t)n;
    err_meta[s].fp = (uint16_t)(bw * 13 + bh), err_meta[s].seen = 0, err_meta[s].used = 1;
    err_meta[s].img = (uint64_t)(uintptr_t)img, err_meta[s].key = stuck_key(err_meta[s].img, rg, n);
    err_meta[s].level = rg[0].level, err_meta[s].layer = rg[0].layer, err_meta[s].x = rg[0].offset.x;
    err_meta[s].y = rg[0].offset.y, err_meta[s].call = ++err_calls;
    tag.addr = 0, tag.path = 0;
    return s;
}
static uint64_t n_dispatch, n_texels;
// vk_texture_astc_copy's census of zero blocks in the staged source (any thread; summed in the "so far" line)
static uint64_t zero_copies, zero_blocks;
void vk_astc_zero_note(uint64_t blocks) {
    __atomic_add_fetch(&zero_copies, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&zero_blocks, blocks, __ATOMIC_RELAXED);
}

// MKW_ASTC_TIME=1 (or vk_astc_time_enable): timestamps around the dispatches of each scratch chunk (the decode
// alone) and around a whole vk_astc_record call (one texture upload or WriteTexels: copies and barriers included),
// read back by later calls. Per footprint: texels and dispatch time; a new largest call is logged with both times.
// Needs hostQueryReset (a slot is reset on the host after its read).
#define TQ 256
static PlatMutex t_lock = PLAT_MUTEX_INIT;
static int t_on = -1;                 // -1: knob not read yet
static VkQueryPool t_pool;
static VkDevice t_dev;
static double t_ns_per_tick;
static uint64_t t_mask;
static struct { uint64_t texels; uint32_t id; uint8_t fp, busy, whole; } t_slot[TQ];   // whole: the call, else a chunk
static int t_next;
static struct { uint64_t texels, chunks; double us; } t_fp[13 * 13];   // [bw * 13 + bh]: dispatch time of the chunks
#define TA 64   // calls in flight, by id % TA
typedef struct { uint32_t id, w, h, regions, chunks, done; int complete; uint64_t texels; double us, total; uint8_t fp; } TCall;
static TCall t_call[TA];
static uint32_t t_id;
static uint64_t t_max_texels;
static double t_max_us, t_max_total;
static uint8_t t_max_fp;

static void time_init(VkCtx *c) {   // t_lock held
    if (t_on < 0) {
        const char *e = cfg_get("MKW_ASTC_TIME");
        t_on = e && *e && *e != '0';
    }
    if (t_on <= 0 || (t_pool && t_dev == c->device)) return;
    t_pool = VK_NULL_HANDLE;
    memset(t_slot, 0, sizeof t_slot);
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * TQ};
    if (!f12.hostQueryReset || !bits || vkCreateQueryPool(c->device, &qi, NULL, &t_pool) != VK_SUCCESS) {
        rt_log("vk: MKW_ASTC_TIME: no hostQueryReset or no timestamps on the queue; ASTC decodes not timed");
        t_pool = VK_NULL_HANDLE;
        t_on = 0;
        return;
    }
    vkResetQueryPool(c->device, t_pool, 0, 2 * TQ);
    t_dev = c->device;
    t_ns_per_tick = c->props.limits.timestampPeriod;
    t_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
}

// Takes in the slots whose timestamps are available; t_lock held.
static void time_poll_locked(VkCtx *c) {
    if (!t_pool || t_dev != c->device) return;
    for (int i = 0; i < TQ; i++) {
        if (!t_slot[i].busy) continue;
        uint64_t r[4];
        if (vkGetQueryPoolResults(c->device, t_pool, 2 * (uint32_t)i, 2, sizeof r, r, 16,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS ||
            !r[1] || !r[3])
            continue;
        double us = (double)((r[2] - r[0]) & t_mask) * t_ns_per_tick / 1e3;
        int fp = t_slot[i].fp;
        if (!t_slot[i].whole) t_fp[fp].texels += t_slot[i].texels, t_fp[fp].chunks++, t_fp[fp].us += us;
        TCall *k = &t_call[t_slot[i].id % TA];
        if (k->id == t_slot[i].id) {
            if (t_slot[i].whole) k->total = us;
            else k->us += us, k->texels += t_slot[i].texels;
            if (++k->done == k->chunks + 1 && k->complete && k->texels > t_max_texels) {
                t_max_texels = k->texels, t_max_us = k->us, t_max_total = k->total, t_max_fp = k->fp;
                rt_log("vk: ASTC time: largest decode so far %.2f M texels (%dx%d; %u regions, the first %ux%u; %u scratch "
                       "chunks): dispatches %.0f us of GPU time (%.0f us per M texel), %.0f us with the copies",
                       (double)k->texels / 1e6, k->fp / 13, k->fp % 13, k->regions, k->w, k->h, k->chunks, k->us,
                       k->us * 1e6 / (double)k->texels, k->total);
            }
        }
        t_slot[i].busy = 0;
        vkResetQueryPool(c->device, t_pool, 2 * (uint32_t)i, 2);
    }
}

// A free slot for a call's id (t_lock held); -1 when the ring is full.
static int time_slot(uint32_t id, uint8_t fp, uint64_t texels, int whole) {
    for (int i = 0; i < TQ; i++, t_next = (t_next + 1) % TQ)
        if (!t_slot[t_next].busy) {
            int k = t_next;
            t_slot[k].busy = 1, t_slot[k].id = id, t_slot[k].fp = fp, t_slot[k].texels = texels;
            t_slot[k].whole = (uint8_t)whole;
            t_next = (k + 1) % TQ;
            return k;
        }
    return -1;
}

void vk_astc_time_enable(int on) {
    plat_mutex_lock(&t_lock);
    t_on = on;
    plat_mutex_unlock(&t_lock);
}

int vk_astc_time_get(uint32_t bw, uint32_t bh, uint64_t *texels, uint64_t *calls, double *us) {
    VkCtx *c = vk_ctx();
    if (!c || bw > 12 || bh > 12) return 0;
    plat_mutex_lock(&t_lock);
    time_poll_locked(c);
    int fp = (int)(bw * 13 + bh);
    *texels = t_fp[fp].texels, *calls = t_fp[fp].chunks, *us = t_fp[fp].us;
    plat_mutex_unlock(&t_lock);
    return 1;
}

void vk_astc_time_summary(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_mutex_lock(&t_lock);
    if (t_on > 0) {
        time_poll_locked(c);
        for (int fp = 0; fp < 13 * 13; fp++)
            if (t_fp[fp].chunks)
                rt_log("vk: ASTC time: %dx%d: %llu dispatch batches, %.2f M texels, %.0f us GPU (%.0f us per M texel)",
                       fp / 13, fp % 13, (unsigned long long)t_fp[fp].chunks, (double)t_fp[fp].texels / 1e6, t_fp[fp].us,
                       t_fp[fp].us * 1e6 / (double)t_fp[fp].texels);
        if (t_max_texels)
            rt_log("vk: ASTC time: largest decode %.2f M texels (%dx%d): dispatches %.0f us GPU, %.0f us with the copies",
                   (double)t_max_texels / 1e6, t_max_fp / 13, t_max_fp % 13, t_max_us, t_max_total);
    }
    plat_mutex_unlock(&t_lock);
}

static void fill_tables(uint8_t *p) {
    uint32_t off[28];
    for (int i = 0; i < 12; i++) off[i] = TAB_WUNQ + wunq_off[i];
    for (int i = 0; i < 16; i++) off[12 + i] = TAB_CUNQ + cunq_off[i];
    memset(p, 0, TAB_SIZE);
    memcpy(p, off, sizeof off);
    memcpy(p + TAB_WUNQ, wunq_data, sizeof wunq_data);
    memcpy(p + TAB_CUNQ, cunq_data, sizeof cunq_data);
}

// Objects of the current device; under g_lock (exclusive).
static void reset_for(VkCtx *c) {
    if (G.dev == c->device) return;
    memset(&G.dsl, 0, sizeof G - offsetof(__typeof__(G), dsl));   // the old device's objects went with it
    G.dev = c->device;
    G.failed = 0;
    G.mode = -1;
    G.bc7 = -1;
}

static int create_pipeline(VkCtx *c) {
    G.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[4];
    for (int i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof astc_decode_spv,
                                   .pCode = astc_decode_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!G.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &G.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &G.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &G.layout) != VK_SUCCESS) return 0;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
    uint64_t t0 = plat_now_ns();
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = G.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &G.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = TAB_SIZE,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &bi, &aci, &G.tab, &G.tab_alloc, NULL) != VK_SUCCESS) return 0;
    VkBufferCreateInfo ebi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ERR_SLOTS * ERR_LEVELS * 4,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo eaci = {.usage = VMA_MEMORY_USAGE_AUTO,
                                    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo einfo = {0};
    if (vmaCreateBuffer(c->vma, &ebi, &eaci, &G.err, &G.err_alloc, &einfo) != VK_SUCCESS) return 0;
    G.err_map = einfo.pMappedData;
    if (G.err_map) memset((void *)G.err_map, 0, ERR_SLOTS * ERR_LEVELS * 4), vmaFlushAllocation(c->vma, G.err_alloc, 0, VK_WHOLE_SIZE);
    rt_log("vk: ASTC compute decoder ready (pipeline built in %.1f ms)", (double)(plat_now_ns() - t0) / 1e6);
    mi.codeSize = sizeof bc7_encode_spv, mi.pCode = bc7_encode_spv;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) == VK_SUCCESS) {
        ci.stage.module = m;
        if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &G.enc) != VK_SUCCESS) G.enc = VK_NULL_HANDLE;
        vkDestroyShaderModule(c->device, m, NULL);
    }
    mi.codeSize = sizeof bc1_encode_spv, mi.pCode = bc1_encode_spv;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) == VK_SUCCESS) {
        ci.stage.module = m;
        if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &G.enc1) != VK_SUCCESS) G.enc1 = VK_NULL_HANDLE;
        vkDestroyShaderModule(c->device, m, NULL);
    }
    mi.codeSize = sizeof bl_deswizzle_spv, mi.pCode = bl_deswizzle_spv;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) == VK_SUCCESS) {
        ci.stage.module = m;
        if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &G.bl_pipe) != VK_SUCCESS) G.bl_pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(c->device, m, NULL);
    }
    return 1;
}

// MKW_ASTC_BC7 (default 1; 0 off): ASTC images of 256x256 or more (an array's or cube map's layers counted: vk_tex_image.c)
// become BC7 images (BC1 when opaque), filled by the decoder and then bc7_encode.comp. The RGBA8
// copies of the textures took 2.3 GB of a 3.4 GB heap on a course (run113) and pushed render targets out of memory.
int vk_astc_bc7(void) {
    VkCtx *c = vk_ctx();
    if (!c || !vk_astc_gpu()) return 0;
    plat_mutex_lock_shared(&g_lock);
    int v = G.dev == c->device ? G.bc7 : -1;
    plat_mutex_unlock_shared(&g_lock);
    if (v >= 0) return v;
    const char *e = cfg_get("MKW_ASTC_BC7");
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(c->phys, VK_FORMAT_BC7_SRGB_BLOCK, &p);
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    plat_mutex_lock(&g_lock);
    G.bc7 = !(e && *e && !atoi(e)) && G.enc && (p.optimalTilingFeatures & need) == need;
    v = G.bc7;
    plat_mutex_unlock(&g_lock);
    rt_log("vk: ASTC textures re-encoded as BC7: %s", v ? "yes" : "no");
    return v;
}

static int native_device(VkCtx *c) { return c->has_astc_ldr; }

static const char *mode_name(int m) {
    return m == VK_ASTC_CPU ? "cpu" : m == VK_ASTC_GPU ? "gpu" : m == VK_ASTC_NATIVE ? "native" : "check";
}

int vk_astc_mode(void) {
    VkCtx *c = vk_ctx();
    if (!c) return VK_ASTC_CPU;
    plat_mutex_lock_shared(&g_lock);
    int m = G.dev == c->device ? G.mode : -1;
    plat_mutex_unlock_shared(&g_lock);
    if (m >= 0) return m;
    plat_mutex_lock(&g_lock);
    reset_for(c);
    if (G.mode < 0) {
        const char *e = cfg_value("MKW_ASTC");
        int want = !strcmp(e, "auto") ? -1 : !strcmp(e, "cpu") ? VK_ASTC_CPU : !strcmp(e, "gpu") ? VK_ASTC_GPU
                 : !strcmp(e, "native") ? VK_ASTC_NATIVE : !strcmp(e, "check") ? VK_ASTC_CHECK : -2;
        if (want == -2) rt_log("vk: MKW_ASTC=%s is not cpu, gpu, native, check or auto: auto", e), want = -1;
        if ((want == -1 || want == VK_ASTC_NATIVE) && native_device(c)) m = VK_ASTC_NATIVE;
        else {
            if (want == VK_ASTC_NATIVE) rt_log("vk: MKW_ASTC=native: the device lacks textureCompressionASTC_LDR; gpu");
            m = want == VK_ASTC_CPU ? VK_ASTC_CPU : want == VK_ASTC_CHECK ? VK_ASTC_CHECK : VK_ASTC_GPU;
        }
        if ((m == VK_ASTC_GPU || m == VK_ASTC_CHECK || m == VK_ASTC_NATIVE) && !G.pipe && !G.failed &&
            !create_pipeline(c)) {
            G.failed = 1;
            rt_log("vk: ASTC compute decoder unavailable on %s: ASTC decodes on the CPU", c->props.deviceName);
        }
        if (G.failed && m != VK_ASTC_NATIVE) m = VK_ASTC_CPU;
        G.mode = m;
        rt_log("vk: ASTC textures: %s (textureCompressionASTC_LDR %d)", mode_name(m), native_device(c));
    }
    m = G.mode;
    plat_mutex_unlock(&g_lock);
    return m;
}

int vk_astc_native(VkFormat astc_format) {
    VkCtx *c = vk_ctx();
    if (!c || vk_astc_mode() != VK_ASTC_NATIVE) return 0;
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(c->phys, astc_format, &p);
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    return (p.optimalTilingFeatures & need) == need;
}

int vk_astc_gpu(void) {
    int m = vk_astc_mode();
    if (m == VK_ASTC_GPU || m == VK_ASTC_CHECK) return 1;
    if (m != VK_ASTC_NATIVE) return 0;
    plat_mutex_lock_shared(&g_lock);   // native mode, a format the device lacks: the decoder when it exists
    int ok = G.pipe != VK_NULL_HANDLE;
    plat_mutex_unlock_shared(&g_lock);
    return ok;
}

static void mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                        VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// Scratch buffers come from an allocator of their own, without MKW_GPU_MEM_LIMIT's heap limits: a course fills the
// limited heaps, and a scratch that did not fit left its texture undecoded, showing whatever the memory held (the noise
// over the world map menu, run-bc7-0 to run-bc7-3: at least 16 textures each). Each is one chunk (CHUNK_OUT) and is
// freed with its upload batch.
static VmaAllocator svma_;

static void svma_init(void *arg) {
    (void)arg;
    const VkCtx *c = vk_ctx();
    VmaAllocatorCreateInfo ai = {.physicalDevice = c->phys, .device = c->device, .instance = c->instance,
                                 .vulkanApiVersion = VK_API_VERSION_1_3};
    if (vmaCreateAllocator(&ai, &svma_) != VK_SUCCESS) svma_ = NULL;
}

static VmaAllocator svma(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, svma_init, NULL);
    return svma_ ? svma_ : vk_ctx()->vma;
}

// One scratch buffer, made once and kept, serves every call whose chunk fits (blocks + texels + BC of a CHUNK_OUT
// chunk fit with room): one buffer per call cost 4.9 ms each in the frame after «VALE», 1 719 of its 2 137 ms on the
// Radeon 660M (vale1, 06/11/2026: 351 allocations), and a buffer per upload batch was worse (wastc4: still new memory,
// held until the batch ends). Every caller records under the upload lock (vk_upload_begin, or vk_upload_lock for a
// replay part's command buffer), and the upload batches and the parts all go to the one graphics queue: a barrier at
// the start of each use waits for every earlier use in submission order, whichever command buffer it was in, as between
// the chunks of one call. Regions over a chunk are cut into bands (split_regions), so every call fits; only MKW_ASTC=check and the
// probes (a readback) take a buffer of their own, freed with the batch.
#define FIXED_SCRATCH (64ull << 20)
static VkBuffer fx_buf;
static VmaAllocation fx_alloc;
static int fx_tried;
// A failed creation is tried again FX_RETRY_FRAMES later: memory short at one moment (a course load) left every later
// call on a buffer of its own for the rest of the run, the 4.9 ms per call the kept buffer was made to save.
#define FX_RETRY_FRAMES 120
static uint64_t fx_retry_frame;
static uint32_t fx_failures;

// Scratch per vk_astc_record: texels of at most this many bytes at a time, so a large texture array (169 M texels, 677 MB
// as RGBA8, on monsters2) or an 8192x8192 level (256 MiB as RGBA8) does not need its whole RGBA8 copy as scratch.
#define CHUNK_OUT (32ull << 20)
#define CHUNK_IN (16ull << 20)

static uint32_t lcm_4(uint32_t bh) { return bh % 4 == 0 ? bh : bh % 2 == 0 ? bh * 2 : bh * 4; }   // lcm(bh, 4)

static VkDeviceSize region_out(const VkAstcRegion *r) { return (VkDeviceSize)r->w * r->h * r->slices * 4; }
static VkDeviceSize region_bc(const VkAstcRegion *r, VkDeviceSize block) {   // BC blocks of a region: per 4x4
    return (VkDeviceSize)((r->w + 3) / 4) * ((r->h + 3) / 4) * r->slices * block;
}
static VkDeviceSize region_in(const VkAstcRegion *r, uint32_t bw, uint32_t bh) {
    return r->bl_size ? r->bl_size : (VkDeviceSize)((r->w + bw - 1) / bw) * ((r->h + bh - 1) / bh) * r->slices * 16;
}

// A region over CHUNK_OUT is cut into pieces that each fit one: groups of whole slices (3D), and a slice over a chunk
// into bands of whole block rows, so the scratch never holds more than a chunk of texels and the one kept buffer
// (FIXED_SCRATCH) serves every call. Before, an 8192x8192 level got a buffer of its own of 278-285 MB, which failed
// when memory was short and left the texture undecoded (PS5 runs nocache and the PGO baseline, 16/11/2026).
// A band starts on a block row of the ASTC block and on a multiple of 4 texel rows (the BC block, and the image
// offset a copy into a BC image needs), so it holds whole blocks of both: the decoder works block by block (the
// partition and the weights only use the texel's place in its block) and the BC encoders clamp reads to the region's
// last row, which only the last band reaches, so the texels and the BC blocks are the ones the whole region gives.
// The blocks are already linear here: the caller deswizzled them into tight rows (vk_tex_upload.c upload,
// vk_texture_astc_copy and vk_tex_texel.c read them out of the block-linear storage), so a band is one byte range.
// Pieces keep the region's order and texels in it, so a readback gets the same bytes. Returns the pieces in *out
// (rg itself when nothing is cut; else malloc'd) and their count.
static int bands_on = 1;   // vk_astc_bands: --probe-astc records with and without them and compares
void vk_astc_bands(int on) { bands_on = on; }

static int split_regions(const VkAstcRegion *rg, int n, uint32_t bw, uint32_t bh, const VkAstcRegion **out) {
    if (!bands_on) return *out = rg, n;
    int need = 0;
    for (int i = 0; i < n; i++) {
        const VkAstcRegion *r = &rg[i];
        VkDeviceSize slice = (VkDeviceSize)r->w * r->h * 4;
        if (slice * r->slices <= CHUNK_OUT) { need++; continue; }
        uint32_t step = (uint32_t)(lcm_4(bh) / bh), rows = (r->h + bh - 1) / bh;   // block rows per aligned step
        uint64_t per = CHUNK_OUT / ((VkDeviceSize)r->w * bh * 4) / step * step;
        if (!per) per = step;
        need += slice <= CHUNK_OUT ? (int)((r->slices + CHUNK_OUT / slice - 1) / (CHUNK_OUT / slice))
                                   : (int)(r->slices * ((rows + per - 1) / per));
    }
    if (need == n) return *out = rg, n;
    VkAstcRegion *p = malloc((size_t)need * sizeof *p);
    if (!p) rt_fatal("vk: out of memory (ASTC bands)");
    int k = 0;
    for (int i = 0; i < n; i++) {
        const VkAstcRegion *r = &rg[i];
        uint32_t bx = (r->w + bw - 1) / bw, rows = (r->h + bh - 1) / bh;
        VkDeviceSize slice = (VkDeviceSize)r->w * r->h * 4, slice_in = (VkDeviceSize)bx * rows * 16;
        if (slice * r->slices <= CHUNK_OUT) { p[k++] = *r; continue; }
        if (slice <= CHUNK_OUT) {   // whole slices, as many as a chunk holds
            uint32_t g = (uint32_t)(CHUNK_OUT / slice);
            for (uint32_t z = 0; z < r->slices; z += g) {
                p[k] = *r;
                p[k].src = r->src + z * slice_in, p[k].slices = r->slices - z < g ? r->slices - z : g;
                p[k++].offset.z = r->offset.z + (int32_t)z;
            }
            continue;
        }
        uint32_t step = (uint32_t)(lcm_4(bh) / bh);
        uint64_t per = CHUNK_OUT / ((VkDeviceSize)r->w * bh * 4) / step * step;
        if (!per) per = step;
        for (uint32_t z = 0; z < r->slices; z++)
            for (uint32_t y = 0; y < rows; y += (uint32_t)per) {
                uint32_t nb = rows - y < per ? rows - y : (uint32_t)per, y0 = y * bh;
                p[k] = *r;
                if (r->bl_size) {
                    // 2D block-linear band (caller keeps 3D > CHUNK_OUT linear): copy only the covering block-rows.
                    uint32_t gobs_x = (bx + 3u) >> 2u, blk_h = 8u << r->bl_th;
                    uint64_t row_blk_bytes = (uint64_t)gobs_x * (512ull << r->bl_th);
                    uint32_t y_blk0 = y >> (3u + r->bl_th);
                    uint32_t y_blk1 = (y + nb + blk_h - 1u) >> (3u + r->bl_th);
                    p[k].src = r->src + (VkDeviceSize)y_blk0 * row_blk_bytes;
                    p[k].bl_size = (VkDeviceSize)(y_blk1 - y_blk0) * row_blk_bytes;
                    p[k].bl_y0 = y - (y_blk0 << (3u + r->bl_th));
                    p[k].bl_z0 = 0;
                    p[k].bl_rows = (y_blk1 - y_blk0) << (3u + r->bl_th);
                } else {
                    p[k].src = r->src + z * slice_in + (VkDeviceSize)y * bx * 16;
                }
                p[k].h = r->h - y0 < nb * bh ? r->h - y0 : nb * bh, p[k].slices = 1;
                p[k].offset.y = r->offset.y + (int32_t)y0, p[k++].offset.z = r->offset.z + (int32_t)z;
            }
    }
    return *out = p, k;
}

// Chunks of consecutive regions: the scratch holds the largest chunk's blocks, then its texels, then its BC blocks.
static void scratch_caps(const VkAstcRegion *rg, int n, uint32_t bw, uint32_t bh, VkDeviceSize blk, VkDeviceSize *cap_in,
                         VkDeviceSize *cap_out, VkDeviceSize *cap_bc, VkDeviceSize *out_total) {
    VkDeviceSize ci = 0, co = 0, cb7 = 0;
    *cap_in = *cap_out = *cap_bc = *out_total = 0;
    for (int i = 0; i < n; i++) {
        VkDeviceSize o = region_out(&rg[i]), b = region_in(&rg[i], bw, bh);
        if (co && (co + o > CHUNK_OUT || ci + b > CHUNK_IN)) ci = co = cb7 = 0;
        ci += b, co += o, cb7 += region_bc(&rg[i], blk), *out_total += o;
        if (ci > *cap_in) *cap_in = ci;
        if (co > *cap_out) *cap_out = co;
        if (cb7 > *cap_bc) *cap_bc = cb7;
    }
}

// The largest scratch so far, and what the call would have needed without the bands (logged when either grows).
static VkDeviceSize max_scratch, max_unsplit;

// MKW_TRACE=perf: vk_astc_record's time in its setup, its dispatch loop and after it (the summary log line), read and
// reset by vk_astc_record_split; the callers hold the upload lock.
static uint64_t ar_ns[3], ar_n;
void vk_astc_record_split(uint64_t out[4]) {
    out[0] = ar_ns[0], out[1] = ar_ns[1], out[2] = ar_ns[2], out[3] = ar_n;
    ar_ns[0] = ar_ns[1] = ar_ns[2] = ar_n = 0;
}

// Multi-texture ASTC batch (MKW_ASTC_BATCH, default 1, supports "ab"): packs multiple ASTC uploads from precheck()
// into one FIXED_SCRATCH pass, cutting per-texture ALL_COMMANDS barriers, pipeline binds, and push descriptors by ~30x.
enum { ASTC_BATCH_MAX_ITEMS = 64, ASTC_BATCH_MAX_REG = 512 };
typedef struct {
    VkBuffer src;
    VkImage img;
    VkImageAspectFlags aspect;
    uint32_t bw, bh, eslot, first, count;
    int srgb, bc7;
} AstcBatchItem;
static int ab_open;
static uint64_t ab_tid;
static VkCommandBuffer ab_cb;
static AstcBatchItem ab_items[ASTC_BATCH_MAX_ITEMS];
static VkAstcRegion ab_rg[ASTC_BATCH_MAX_REG];
static int ab_n, ab_nr;
static VkDeviceSize ab_in, ab_out, ab_bc;

static int astc_batch_knob(void) {
    static int k = -1;
    if (k < 0) k = rt_knob("ASTC_BATCH", "1");
    return RT_KNOB_ON(k);
}

int vk_astc_batch_active(void) { return ab_open && ab_tid == plat_thread_id(); }

void vk_astc_batch_begin(void) {
    if (!astc_batch_knob()) return;
    ab_open = 1, ab_tid = plat_thread_id();
}

void vk_astc_batch_flush(void) {
    if (!ab_n || ab_tid != plat_thread_id()) return;
    uint64_t t0 = plat_now_ns();
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = vk_upload_begin();
    int n_items = ab_n, n_rg = ab_nr;
    VkDeviceSize raw_in = ab_in, raw_out = ab_out, raw_bc = ab_bc;
    ab_n = 0, ab_nr = 0, ab_in = 0, ab_out = 0, ab_bc = 0, ab_cb = VK_NULL_HANDLE;

    VkDeviceSize al = c->props.limits.minStorageBufferOffsetAlignment;
    if (al < 16) al = 16;
    VkDeviceSize cap_in = (raw_in + al - 1) / al * al;
    VkDeviceSize cap_out = (raw_out + al - 1) / al * al;
    VkDeviceSize cap_bc = raw_bc;
    VkBuffer sb = fx_buf;

    plat_mutex_lock_shared(&g_lock);
    VkPipeline pipe = G.pipe, enc7 = G.enc, enc1 = G.enc1;
    VkBuffer tab = G.tab;
    plat_mutex_unlock_shared(&g_lock);

    // One barrier for all target images (ALL_RW -> COPY) + FIXED_SCRATCH reuse.
    mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    VkBufferCopy bcp[ASTC_BATCH_MAX_REG];
    VkBufferImageCopy copies[ASTC_BATCH_MAX_REG];
    VkDeviceSize in_off = 0;
    for (int i = 0; i < n_items; ) {
        VkBuffer src = ab_items[i].src;
        int nb = 0;
        for (; i < n_items && ab_items[i].src == src; i++) {
            const AstcBatchItem *it = &ab_items[i];
            for (uint32_t j = 0; j < it->count; j++) {
                const VkAstcRegion *r = &ab_rg[it->first + j];
                VkDeviceSize sz = region_in(r, it->bw, it->bh);
                bcp[nb++] = (VkBufferCopy){r->src, in_off, sz};
                in_off += (sz + 15) & ~15ull;
            }
        }
        vkCmdCopyBuffer(cb, src, sb, (uint32_t)nb, bcp);
    }
    mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    VkDescriptorBufferInfo dbi[4] = {{sb, 0, cap_in}, {sb, cap_in, cap_out}, {tab, 0, TAB_SIZE}, {G.err, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[4];
    for (int i = 0; i < 4; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &dbi[i]};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, 4, w);

    VkDeviceSize ib = 0, ob = 0;
    int has_bc7 = 0, has_bc1 = 0;
    for (int i = 0; i < n_items; i++) {
        const AstcBatchItem *it = &ab_items[i];
        if (it->bc7 == 1) has_bc7 = 1;
        else if (it->bc7 == 2) has_bc1 = 1;
        uint32_t bpw = it->bw * it->bh <= 64 ? 64 / (it->bw * it->bh) : 1;
        for (uint32_t j = 0; j < it->count; j++) {
            const VkAstcRegion *r = &ab_rg[it->first + j];
            uint32_t bx = (r->w + it->bw - 1) / it->bw, by = (r->h + it->bh - 1) / it->bh;
            uint32_t bl = r->bl_size != 0;
            uint32_t bl_rows = bl ? (r->bl_rows ? r->bl_rows : by) : by;
            uint32_t blocks_y = bl ? (bl_rows + (8u << r->bl_th) - 1u) >> (3u + r->bl_th) : 0;
            Push pc = {.in_base = (uint32_t)(ib / 16), .out_base = (uint32_t)(ob / 4), .width = r->w, .height = r->h,
                       .bw = it->bw, .bh = it->bh, .bx = bx, .bslice = bl ? blocks_y : bx * by,
                       .srgb = (uint32_t)(it->srgb != 0),
                       .err_slot = it->eslot ? it->eslot * ERR_LEVELS + (r->level < ERR_LEVELS ? r->level : ERR_LEVELS - 1) : 0,
                       .bl_mode = bl, .bl_th = r->bl_th, .bl_td = r->bl_td, .bl_y0 = r->bl_y0, .bl_z0 = r->bl_z0,
                       .bl_gobs_x = (bx + 3u) >> 2u};
            vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            vkCmdDispatch(cb, (bx + bpw - 1) / bpw, by, r->slices);
            copies[it->first + j] = (VkBufferImageCopy){.bufferOffset = cap_in + ob, .bufferRowLength = r->w,
                                                        .bufferImageHeight = r->h,
                                                        .imageSubresource = {it->aspect, r->level, r->layer, 1},
                                                        .imageOffset = r->offset,
                                                        .imageExtent = {r->w, r->h, r->slices}};
            ib += (region_in(r, it->bw, it->bh) + 15) & ~15ull;
            ob += region_out(r);
            n_texels += (uint64_t)r->w * r->h * r->slices;
        }
    }

    if (has_bc7 || has_bc1) {
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        VkDescriptorBufferInfo ebi[2] = {{sb, cap_in, cap_out}, {sb, cap_in + cap_out, cap_bc ? cap_bc : 16}};
        VkWriteDescriptorSet ew[2];
        for (int i = 0; i < 2; i++)
            ew[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           .pBufferInfo = &ebi[i]};
        for (int pass = 1; pass <= 2; pass++) {
            if ((pass == 1 && !has_bc7) || (pass == 2 && !has_bc1)) continue;
            VkDeviceSize blk = pass == 2 ? 8 : 16;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pass == 2 ? enc1 : enc7);
            rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, 2, ew);
            VkDeviceSize ob2 = 0, eb = 0;
            for (int i = 0; i < n_items; i++) {
                const AstcBatchItem *it = &ab_items[i];
                for (uint32_t j = 0; j < it->count; j++) {
                    const VkAstcRegion *r = &ab_rg[it->first + j];
                    VkDeviceSize rbc = it->bc7 ? region_bc(r, it->bc7 == 2 ? 8 : 16) : 0;
                    if (it->bc7 == pass) {
                        uint32_t ex = (r->w + 3) / 4, ey = (r->h + 3) / 4;
                        Push pc = {.in_base = (uint32_t)(ob2 / 4), .out_base = (uint32_t)(eb / blk), .width = r->w,
                                   .height = r->h, .bw = ex, .bh = ex * ey};
                        vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
                        vkCmdDispatch(cb, (ex + 7) / 8, (ey + 7) / 8, r->slices);
                        copies[it->first + j].bufferOffset = cap_in + cap_out + eb;
                        copies[it->first + j].bufferRowLength = ex * 4;
                        copies[it->first + j].bufferImageHeight = ey * 4;
                        if (r->lw && r->lh) {
                            uint32_t ew_ = r->lw - (uint32_t)r->offset.x, eh_ = r->lh - (uint32_t)r->offset.y;
                            copies[it->first + j].imageExtent.width = ex * 4 < ew_ ? ex * 4 : ew_;
                            copies[it->first + j].imageExtent.height = ey * 4 < eh_ ? ey * 4 : eh_;
                        }
                    }
                    ob2 += region_out(r);
                    eb += (rbc + 15) & ~15ull;
                }
            }
        }
    }

    mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < n_items; i++)
        vkCmdCopyBufferToImage(cb, sb, ab_items[i].img, VK_IMAGE_LAYOUT_GENERAL, ab_items[i].count,
                               copies + ab_items[i].first);
    mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

    n_dispatch += (uint64_t)n_rg;
    vk_upload_end();
    uint64_t dt = plat_now_ns() - t0;
    ar_ns[1] += dt;
    perf_ledger(PERF_LG_ASTC, dt);
}

void vk_astc_batch_end(void) {
    if (!ab_open || ab_tid != plat_thread_id()) return;
    vk_astc_batch_flush();
    ab_open = 0;
}

int vk_astc_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_size, const VkAstcRegion *rg0, int n0, uint32_t bw,
                   uint32_t bh, int srgb, VkImage img, VkImageAspectFlags aspect, int bc7, VkBuffer readback) {
    uint64_t ar_t0 = plat_now_ns(), ar_t1 = 0, ar_t2 = 0;
    static volatile int32_t n_fail;
    (void)src_size;
    VkCtx *c = vk_ctx();
    if (!c || !vk_astc_gpu() || bw < 4 || bh < 4 || bw > 12 || bh > 12) return 0;
    plat_mutex_lock_shared(&g_lock);
    VkPipeline pipe = G.dev == c->device ? G.pipe : VK_NULL_HANDLE;
    int need_tab_fast = !G.tab_uploaded;
    plat_mutex_unlock_shared(&g_lock);
    if (!pipe) return 0;
    // bc7: 1 the image is BC7, 2 BC1 (TexVkFormat.astc_bc7). MKW_ASTC=check reads back the RGBA8 the decoder wrote,
    // before the re-encoding: turning that off for it put RGBA8 rows into BC7 images (the menus scrambled, vis22).
    if (!vk_astc_bc7()) bc7 = 0;
    VkPipeline enc = bc7 == 2 ? G.enc1 : bc7 ? G.enc : VK_NULL_HANDLE;
    VkDeviceSize blk = bc7 == 2 ? 8 : 16;
    if (bc7 && !enc) return 0;

    // Fast batched path: enqueue into the open precheck() batch when fx_buf + tables are ready and no band split is needed.
    if (ab_open && ab_tid == plat_thread_id() && !readback && !need_tab_fast && fx_buf && n0 > 0 && n0 <= ASTC_BATCH_MAX_REG &&
        t_on <= 0) {
        VkDeviceSize i_in = 0, i_out = 0, i_bc = 0;
        int splittable = 0;
        for (int i = 0; i < n0; i++) {
            VkDeviceSize o = region_out(&rg0[i]);
            if (o > CHUNK_OUT) { splittable = 1; break; }
            i_in += (region_in(&rg0[i], bw, bh) + 15) & ~15ull;
            i_out += o;
            if (bc7) i_bc += (region_bc(&rg0[i], blk) + 15) & ~15ull;
        }
        if (!splittable && i_out && i_in <= CHUNK_IN && i_out <= CHUNK_OUT && i_in + i_out + i_bc <= FIXED_SCRATCH - 4096) {
            if ((ab_cb && ab_cb != cb) || ab_n >= ASTC_BATCH_MAX_ITEMS || ab_nr + n0 > ASTC_BATCH_MAX_REG ||
                ab_in + i_in > CHUNK_IN || ab_out + i_out > CHUNK_OUT ||
                ab_in + i_in + ab_out + i_out + ab_bc + i_bc > FIXED_SCRATCH - 4096)
                vk_astc_batch_flush();
            ab_cb = cb;
            ab_items[ab_n++] = (AstcBatchItem){.src = src, .img = img, .aspect = aspect, .bw = bw, .bh = bh,
                                               .eslot = err_slot(rg0, n0, bw, bh, img), .first = (uint32_t)ab_nr,
                                               .count = (uint32_t)n0, .srgb = srgb, .bc7 = bc7};
            memcpy(ab_rg + ab_nr, rg0, (size_t)n0 * sizeof *rg0);
            ab_nr += n0, ab_in += i_in, ab_out += i_out, ab_bc += i_bc;
            vk_upload_work(cb, i_out / 4);
            uint64_t dt = plat_now_ns() - ar_t0;
            ar_ns[0] += dt, ar_n++;
            perf_ledger(PERF_LG_ASTC, dt);
            return 1;
        }
    }
    if (ab_n && ab_tid == plat_thread_id()) vk_astc_batch_flush();
    const VkAstcRegion *rg;
    int n = split_regions(rg0, n0, bw, bh, &rg);
    VkDeviceSize out_total, cap_in, cap_out, cap_bc;
    scratch_caps(rg, n, bw, bh, blk, &cap_in, &cap_out, &cap_bc, &out_total);
    if (!out_total) {
        if (rg != rg0) free((void *)rg);
        return 1;
    }
    vk_upload_work(cb, out_total / 4);   // RGBA8 texels: bounds the batch's GPU time (vk_upload_flush_over)
    VkDeviceSize al = c->props.limits.minStorageBufferOffsetAlignment;   // the texels bind at cap_in, BC7 at cap_in + cap_out
    if (al < 16) al = 16;   // also the BC7 copy: its buffer offset must be a multiple of the 16-byte block
    cap_in = (cap_in + al - 1) / al * al, cap_out = (cap_out + al - 1) / al * al;
    if (!bc7) cap_bc = 0;
    VkDeviceSize unsplit = cap_in + cap_out + cap_bc;
    if (rg != rg0) {   // the same sizes over the caller's regions: the buffer one call took before the bands
        VkDeviceSize ui, uo, ub, ut;
        scratch_caps(rg0, n0, bw, bh, blk, &ui, &uo, &ub, &ut);
        unsplit = (ui + al - 1) / al * al + (uo + al - 1) / al * al + (bc7 ? ub : 0);
    }
    if (cap_in + cap_out + cap_bc > max_scratch || unsplit > max_unsplit) {   // the upload lock serialises callers
        if (cap_in + cap_out + cap_bc > max_scratch) max_scratch = cap_in + cap_out + cap_bc;
        if (unsplit > max_unsplit) max_unsplit = unsplit;
        rt_log("vk: ASTC scratch: largest %.1f MiB so far (%.1f MiB without the bands; this call %ux%u x%u, %d regions in "
               "%d pieces)", max_scratch / 1048576.0, max_unsplit / 1048576.0, rg0[0].w, rg0[0].h, rg0[0].slices, n0, n);
    }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap_in + cap_out + cap_bc,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkBuffer sb = VK_NULL_HANDLE;
    VmaAllocation sa = NULL;
    static volatile int64_t alloc_ns, alloc_n;   // time in the scratch allocation
    VkResult ar = VK_SUCCESS;
    int fixed = 0;
    if (bi.size <= FIXED_SCRATCH && !readback) {   // the caller holds the upload lock: fx_* need no lock of their own
        uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (!fx_buf && (!fx_tried || fr >= fx_retry_frame)) {
            VkBufferCreateInfo fi = bi;
            fi.size = FIXED_SCRATCH;
            fx_tried = 1;
            if (vmaCreateBuffer(svma(), &fi, &aci, &fx_buf, &fx_alloc, NULL) != VK_SUCCESS) fx_buf = VK_NULL_HANDLE;
            if (!fx_buf) fx_retry_frame = fr + FX_RETRY_FRAMES, fx_failures++;
            if (fx_buf || fx_failures <= 4 || !(fx_failures & (fx_failures - 1)))
                rt_log("vk: ASTC fixed scratch of %llu MiB at frame %llu: %s (%u failed attempts)",
                       (unsigned long long)(FIXED_SCRATCH >> 20), (unsigned long long)fr,
                       fx_buf ? "made" : "failed, one buffer per call until it is tried again", fx_failures);
        }
        if (fx_buf) {   // its previous use (an earlier call or batch) has read and written it
            sb = fx_buf, fixed = 1;
            mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
    }
    if (!fixed) {
        uint64_t t_alloc = plat_now_ns();
        ar = vmaCreateBuffer(svma(), &bi, &aci, &sb, &sa, NULL);
        plat_add64(&alloc_ns, (int64_t)(plat_now_ns() - t_alloc)), plat_inc64(&alloc_n);
    }
    if (ar != VK_SUCCESS) {
        if (plat_inc32(&n_fail) <= 16)
            rt_log("vk: ASTC scratch buffer of %llu bytes failed: texture left undecoded",
                   (unsigned long long)(cap_in + cap_out));
        if (rg != rg0) free((void *)rg);
        return 0;
    }
    plat_mutex_lock(&g_lock);
    int need_tab = !G.tab_uploaded;
    G.tab_uploaded = 1;
    VkBuffer tab = G.tab;
    plat_mutex_unlock(&g_lock);
    if (need_tab) {   // first use on this device: the tables go up in the upload batch, before the dispatch
        // (the batch is cb itself for a caller recording into it; a replay part follows the batch in queue order, and
        // the staging, held by the batch, is freed once the batch is done)
        VkCommandBuffer tcb = vk_upload_begin();
        VkBuffer tb;
        uint8_t *p = vk_staging(TAB_SIZE, &tb, NULL);
        fill_tables(p);
        VkBufferCopy cp = {0, 0, TAB_SIZE};
        vkCmdCopyBuffer(tcb, tb, tab, 1, &cp);
        if (tcb != cb)
            mem_barrier(tcb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        vk_upload_end();
    }
    int timed = 0, whole = -1;
    uint32_t id = 0;
    uint8_t fp = (uint8_t)(bw * 13 + bh);
    plat_mutex_lock(&t_lock);
    time_init(c);
    if (t_on > 0 && t_pool) {
        time_poll_locked(c);
        timed = 1;
        id = ++t_id;
        t_call[id % TA] = (TCall){.id = id, .w = rg0[0].w, .h = rg0[0].h, .regions = (uint32_t)n0, .fp = fp, .complete = 1};
        whole = time_slot(id, fp, out_total / 4, 1);
        if (whole < 0) t_call[id % TA].complete = 0;
    }
    VkQueryPool tpool = t_pool;
    plat_mutex_unlock(&t_lock);
    if (whole >= 0) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, tpool, 2 * (uint32_t)whole);
    uint32_t eslot = err_slot(rg0, n0, bw, bh, img);
    VkDescriptorBufferInfo dbi[4] = {{sb, 0, cap_in}, {sb, cap_in, cap_out}, {tab, 0, TAB_SIZE}, {G.err, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[4];
    for (int i = 0; i < 4; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &dbi[i]};
    VkDescriptorBufferInfo ebi[2] = {{sb, cap_in, cap_out}, {sb, cap_in + cap_out, cap_bc ? cap_bc : 16}};
    VkWriteDescriptorSet ew[2];
    for (int i = 0; i < 2; i++)
        ew[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .pBufferInfo = &ebi[i]};
    VkBufferImageCopy copies_stk[32], *copies = n <= 32 ? copies_stk : calloc((size_t)n, sizeof *copies);
    VkBufferCopy bcp_stk[32], *bcp = n <= 32 ? bcp_stk : calloc((size_t)n, sizeof *bcp);
    if (!copies || !bcp) rt_fatal("vk: out of memory (ASTC regions)");
    VkDeviceSize rb_off = 0;
    int chunks = 0;
    ar_t1 = plat_now_ns();
    for (int i0 = 0; i0 < n; chunks++) {
        int i1 = i0;
        VkDeviceSize in = 0, out = 0;
        for (; i1 < n && (i1 == i0 || (out + region_out(&rg[i1]) <= CHUNK_OUT && in + region_in(&rg[i1], bw, bh) <= CHUNK_IN)); i1++) {
            bcp[i1 - i0] = (VkBufferCopy){rg[i1].src, in, region_in(&rg[i1], bw, bh)};
            in += region_in(&rg[i1], bw, bh), out += region_out(&rg[i1]);
        }
        if (chunks)   // the previous chunk's dispatches and image copy are done with the scratch
            mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdCopyBuffer(cb, src, sb, (uint32_t)(i1 - i0), bcp);
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        int slot = -1;   // the timestamps bracket this chunk's dispatches: the first waits for the blocks copy
        if (timed) {
            plat_mutex_lock(&t_lock);
            slot = time_slot(id, fp, out / 4, 0);
            if (slot < 0) t_call[id % TA].complete = 0;
            plat_mutex_unlock(&t_lock);
        }
        if (slot >= 0) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, tpool, 2 * (uint32_t)slot);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, 4, w);
        VkDeviceSize ib = 0, ob = 0;
        for (int i = i0; i < i1; i++) {
            const VkAstcRegion *r = &rg[i];
            uint32_t bx = (r->w + bw - 1) / bw, by = (r->h + bh - 1) / bh;
            uint32_t bl = r->bl_size != 0;
            uint32_t bl_rows = bl ? (r->bl_rows ? r->bl_rows : by) : by;
            uint32_t blocks_y = bl ? (bl_rows + (8u << r->bl_th) - 1u) >> (3u + r->bl_th) : 0;
            Push pc = {.in_base = (uint32_t)(ib / 16), .out_base = (uint32_t)(ob / 4), .width = r->w, .height = r->h,
                       .bw = bw, .bh = bh, .bx = bx, .bslice = bl ? blocks_y : bx * by, .srgb = (uint32_t)(srgb != 0),
                       .err_slot = eslot ? eslot * ERR_LEVELS + (r->level < ERR_LEVELS ? r->level : ERR_LEVELS - 1) : 0,
                       .bl_mode = bl, .bl_th = r->bl_th, .bl_td = r->bl_td, .bl_y0 = r->bl_y0, .bl_z0 = r->bl_z0,
                       .bl_gobs_x = (bx + 3u) >> 2u};
            vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            uint32_t bpw = bw * bh <= 64 ? 64 / (bw * bh) : 1;   // blocks per workgroup, as astc_decode.comp main
            vkCmdDispatch(cb, (bx + bpw - 1) / bpw, by, r->slices);
            copies[i - i0] = (VkBufferImageCopy){.bufferOffset = cap_in + ob, .bufferRowLength = r->w,
                                                 .bufferImageHeight = r->h, .imageSubresource = {aspect, r->level, r->layer, 1},
                                                 .imageOffset = r->offset, .imageExtent = {r->w, r->h, r->slices}};
            ib += region_in(r, bw, bh), ob += region_out(r);
            n_texels += (uint64_t)r->w * r->h * r->slices;
        }
        if (slot >= 0)
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, tpool, 2 * (uint32_t)slot + 1);
        if (enc) {   // RGBA8 texels -> BC7 or BC1 blocks, copied instead of the texels
            mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, enc);
            rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, 2, ew);
            VkDeviceSize ob2 = 0, eb = 0;
            for (int i = i0; i < i1; i++) {
                const VkAstcRegion *r = &rg[i];
                uint32_t ex = (r->w + 3) / 4, ey = (r->h + 3) / 4;
                Push pc = {.in_base = (uint32_t)(ob2 / 4), .out_base = (uint32_t)(eb / blk), .width = r->w,
                           .height = r->h, .bw = ex, .bh = ex * ey};
                vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
                vkCmdDispatch(cb, (ex + 7) / 8, (ey + 7) / 8, r->slices);
                copies[i - i0].bufferOffset = cap_in + cap_out + eb;
                copies[i - i0].bufferRowLength = ex * 4, copies[i - i0].bufferImageHeight = ey * 4;
                // The extent in whole blocks too, or up to the level's edge (VUID 00207/00208): MKW writes 1017x107
                // texels of its ASTC terrain arrays, kept as BC7, into 1024-wide levels (wval43).
                if (r->lw && r->lh) {
                    uint32_t ew = r->lw - (uint32_t)r->offset.x, eh = r->lh - (uint32_t)r->offset.y;
                    copies[i - i0].imageExtent.width = ex * 4 < ew ? ex * 4 : ew;
                    copies[i - i0].imageExtent.height = ey * 4 < eh ? ey * 4 : eh;
                }
                ob2 += region_out(r), eb += region_bc(r, blk);
            }
        }
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyBufferToImage(cb, sb, img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)(i1 - i0), copies);
        if (readback) {
            VkBufferCopy rc = {cap_in, rb_off, out};
            vkCmdCopyBuffer(cb, sb, readback, 1, &rc);
            rb_off += out;
        }
        i0 = i1;
    }
    ar_t2 = plat_now_ns();
    if (vk_astc_batch_active() && !readback)
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    if (readback)
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                    VK_ACCESS_2_HOST_READ_BIT);
    if (whole >= 0) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, tpool, 2 * (uint32_t)whole + 1);
    if (timed) {
        plat_mutex_lock(&t_lock);
        t_call[id % TA].chunks = (uint32_t)chunks;
        plat_mutex_unlock(&t_lock);
    }
    if ((n_dispatch + (uint64_t)n0) / 1024 != n_dispatch / 1024 || !n_dispatch) {
        rt_log("vk: ASTC: %llu regions, %.1f M texels decoded on the GPU so far; scratch allocations %lld, %.1f ms; copies "
               "from a source region over 1%% zero blocks %llu (%llu blocks)",
               (unsigned long long)(n_dispatch + (uint64_t)n0), (double)n_texels / 1e6, (long long)alloc_n, alloc_ns / 1e6,
               (unsigned long long)__atomic_load_n(&zero_copies, __ATOMIC_RELAXED),
               (unsigned long long)__atomic_load_n(&zero_blocks, __ATOMIC_RELAXED));
        vk_astc_time_summary();
    }
    n_dispatch += (uint64_t)n0;
    if (copies != copies_stk) free(copies);
    if (bcp != bcp_stk) free(bcp);
    if (rg != rg0) free((void *)rg);
    if (!fixed) vk_defer_buffer_in(svma(), sb, sa);
    uint64_t ar_t3 = plat_now_ns();
    ar_ns[0] += ar_t1 - ar_t0, ar_ns[1] += ar_t2 - ar_t1, ar_ns[2] += ar_t3 - ar_t2, ar_n++;
    perf_ledger(PERF_LG_ASTC, ar_t3 - ar_t0);
    return 1;
}

int vk_bl_deswizzle_ready(void) {
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    vk_astc_gpu();   // ensures create_pipeline has run on this device
    plat_mutex_lock_shared(&g_lock);
    int ok = G.dev == c->device && G.bl_pipe != VK_NULL_HANDLE;
    plat_mutex_unlock_shared(&g_lock);
    return ok;
}

int vk_bl_deswizzle_record(VkCommandBuffer cb, VkBuffer src, const VkBlDeswRegion *rg, int n, VkImage img,
                           VkImageAspectFlags aspect) {
    if (!rg || n <= 0) return 1;
    if (ab_n && ab_tid == plat_thread_id()) vk_astc_batch_flush();
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    plat_mutex_lock_shared(&g_lock);
    VkPipeline pipe = G.dev == c->device ? G.bl_pipe : VK_NULL_HANDLE;
    plat_mutex_unlock_shared(&g_lock);
    if (!pipe) return 0;

    VkDeviceSize al = c->props.limits.minStorageBufferOffsetAlignment;
    if (al < 16) al = 16;
    const VkDeviceSize MAX_HALF = 32ull << 20;
    VkDeviceSize cap_in = 0, cap_out = 0, ci = 0, co = 0;
    for (int i = 0; i < n; i++) {
        VkDeviceSize bi = (rg[i].bl_size + 15) & ~15ull;
        VkDeviceSize bo = ((VkDeviceSize)rg[i].rb * rg[i].rows * rg[i].slices + 15) & ~15ull;
        if (bi > MAX_HALF || bo > MAX_HALF) return 0;
        if (co && (ci + bi > MAX_HALF || co + bo > MAX_HALF)) ci = co = 0;
        ci += bi, co += bo;
        if (ci > cap_in) cap_in = ci;
        if (co > cap_out) cap_out = co;
    }
    cap_in = (cap_in + al - 1) / al * al;
    cap_out = (cap_out + al - 1) / al * al;
    if (!cap_in || !cap_out || cap_in + cap_out > FIXED_SCRATCH) return 0;

    // Reuse FIXED_SCRATCH (caller holds the upload lock, same graphics queue as vk_astc_record).
    VkBuffer sb = fx_buf;
    if (!sb) {
        uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (!fx_tried || fr >= fx_retry_frame) {
            VkBufferCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = FIXED_SCRATCH,
                                     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
            fx_tried = 1;
            if (vmaCreateBuffer(svma(), &fi, &aci, &fx_buf, &fx_alloc, NULL) != VK_SUCCESS) fx_buf = VK_NULL_HANDLE;
            if (!fx_buf) fx_retry_frame = fr + FX_RETRY_FRAMES, fx_failures++;
        }
        sb = fx_buf;
    }
    if (!sb) return 0;
    mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    VkDescriptorBufferInfo dbi[2] = {{sb, 0, cap_in}, {sb, cap_in, cap_out}};
    VkWriteDescriptorSet w[2];
    for (int i = 0; i < 2; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &dbi[i]};
    VkBufferImageCopy *copies = calloc((size_t)n, sizeof *copies);
    VkBufferCopy *bcp = calloc((size_t)n, sizeof *bcp);
    if (!copies || !bcp) { free(copies), free(bcp); return 0; }

    int chunks = 0;
    for (int i0 = 0; i0 < n; chunks++) {
        int i1 = i0;
        VkDeviceSize in = 0, out = 0;
        for (; i1 < n; i1++) {
            VkDeviceSize bi = (rg[i1].bl_size + 15) & ~15ull;
            VkDeviceSize bo = ((VkDeviceSize)rg[i1].rb * rg[i1].rows * rg[i1].slices + 15) & ~15ull;
            if (i1 > i0 && (in + bi > MAX_HALF || out + bo > MAX_HALF)) break;
            bcp[i1 - i0] = (VkBufferCopy){rg[i1].src, in, rg[i1].bl_size};
            in += bi, out += bo;
        }
        if (chunks)
            mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdCopyBuffer(cb, src, sb, (uint32_t)(i1 - i0), bcp);
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, 2, w);
        VkDeviceSize ib = 0, ob = 0;
        for (int i = i0; i < i1; i++) {
            const VkBlDeswRegion *r = &rg[i];
            uint32_t row_u32 = r->rb >> 2u;
            uint32_t gobs_x = (row_u32 + 15u) >> 4u;
            uint32_t blocks_y = (r->rows + (8u << r->th) - 1u) >> (3u + r->th);
            Push pc = {.in_base = (uint32_t)(ib >> 2u), .out_base = (uint32_t)(ob >> 2u),
                       .width = row_u32, .height = r->rows, .bw = gobs_x, .bh = blocks_y,
                       .bx = r->th, .bslice = r->td};
            vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            vkCmdDispatch(cb, (row_u32 + 15u) / 16u, (r->rows + 3u) / 4u, r->slices);
            copies[i - i0] = (VkBufferImageCopy){
                .bufferOffset = cap_in + ob,
                .bufferRowLength = r->rb / r->bpb * r->bw,
                .bufferImageHeight = r->rows * r->bh,
                .imageSubresource = {aspect, r->level, r->layer, 1},
                .imageOffset = {0, r->offset_y, 0},
                .imageExtent = {r->w, r->h, r->slices},
            };
            ib += (r->bl_size + 15) & ~15ull;
            ob += ((VkDeviceSize)r->rb * r->rows * r->slices + 15) & ~15ull;
        }
        mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyBufferToImage(cb, sb, img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)(i1 - i0), copies);
        i0 = i1;
    }
    free(copies), free(bcp);
    return 1;
}

void vk_astc_release(void) {
    VkCtx *c = vk_ctx();
    vk_astc_time_summary();
    if (err_on > 0 && G.err_map) {   // the device is idle: every decode's count, and what stays stuck at the end
        for (uint32_t s = 1; s < ERR_SLOTS; s++) err_check(s, "at the end");
        stuck_summary(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    }
    plat_mutex_lock(&t_lock);
    if (c && t_pool && t_dev == c->device) vkDestroyQueryPool(c->device, t_pool, NULL);
    t_pool = VK_NULL_HANDLE, t_dev = VK_NULL_HANDLE;
    plat_mutex_unlock(&t_lock);
    plat_mutex_lock(&g_lock);
    if (c && G.dev == c->device) {
        if (G.pipe) vkDestroyPipeline(c->device, G.pipe, NULL);
        if (G.enc) vkDestroyPipeline(c->device, G.enc, NULL);
        if (G.enc1) vkDestroyPipeline(c->device, G.enc1, NULL);
        if (G.bl_pipe) vkDestroyPipeline(c->device, G.bl_pipe, NULL);
        if (G.layout) vkDestroyPipelineLayout(c->device, G.layout, NULL);
        if (G.dsl) vkDestroyDescriptorSetLayout(c->device, G.dsl, NULL);
        if (G.tab) vmaDestroyBuffer(c->vma, G.tab, G.tab_alloc);
        if (G.err) vmaDestroyBuffer(c->vma, G.err, G.err_alloc);
        if (fx_buf) vmaDestroyBuffer(svma(), fx_buf, fx_alloc);
    }
    fx_buf = VK_NULL_HANDLE, fx_alloc = NULL, fx_tried = 0, fx_retry_frame = 0, fx_failures = 0;
    memset(&G.dsl, 0, sizeof G - offsetof(__typeof__(G), dsl));
    G.dev = VK_NULL_HANDLE;
    G.mode = -1;
    G.bc7 = -1;
    G.failed = 0;
    plat_mutex_unlock(&g_lock);
}

