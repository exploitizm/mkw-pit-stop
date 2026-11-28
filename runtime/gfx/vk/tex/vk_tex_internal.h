// Internal interface of the texture cache (vk_texture.c and vk_tex_*.c): the entry and slot records and the state the
// parts share. Everything else goes through vk.h.
#pragma once

#include <stdint.h>

#include "gfx.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "platform/platform.h"

typedef struct {
    uint64_t addr;
    nvn_texdesc d;
    uint8_t has_layout, layout[8], pad[7];
} TexKey;

typedef struct Tex {
    VkTexImage pub;               // first: vk_texture_get hands out &pub
    TexKey key;
    tex_t t;                      // the newest texture object seen with this key
    TexVkFormat f;
    VmaAllocation alloc;          // own memory; NULL when bound into pool memory
    VmaAllocation bound_to;       // the pool memory it is bound into
    // An alias copy deferred by the replay's walk (vk_texture_alias_defer): from alias_src at replay position
    // alias_pos of the submission of alias_epoch; NULL none
    struct Tex *alias_src;
    uint64_t alias_pos;
    uint32_t alias_epoch;
    uint32_t wgen;                // page generation of the last upload check
    uint32_t checked;             // submit_epoch of the last check from a draw binding
    uint32_t seen;                // submit_epoch of the last lookup of any kind (evict_idle: idle since then)
    int uploaded, released;
    uint32_t n_up;                // uploads of this entry (vk_texture_debug)
    int last_ch;                  // last vk_mem_changed result of make_fresh: 1 written, 0 clean, -1 not tracked
    struct Tex *next_at;          // other entries at the same storage address
    uint8_t gpu_dirty;            // mirrored, and the GPU wrote the image after the last write-back to the storage
    uint32_t gpu_gen;             // page generation when the GPU write that set gpu_dirty was recorded
    uint8_t *gpu_base;            // storage that fills no whole page: its bytes at that GPU write (gpu_base_note)
    uint64_t gpu_seq;             // order of that GPU write among all of them (gpu_writes)
    uint64_t gpu_fence;           // fence of the submission that made that write (0: not recorded in one)
    uint64_t wr_fence;            // any entry: newest submission recorded to write its image (render, copy, clear,
                                  // storage binding); ReadTexels waits for it instead of draining the replay queue
    uint8_t listed;               // in owned[]
    uint32_t wt_n;                // WriteTexels into the storage since the last upload
    uint8_t up_pending;           // an upload deferred for want of staging memory: the storage counts as changed
    uint64_t up_fail_frame;       // the frame of that failure: tried again from the next frame on
    struct Tex *rel_prev, *rel_next;   // while released: the list of released entries, oldest first
    uint64_t gpu_frame;           // the frame the GPU last wrote it (vk_tex_evict.c rel_keep)
    uint64_t rel_bytes;           // while released: its own memory, counted in released_bytes
    uint64_t shash;               // XXH3 of the storage at the last upload (same_storage: no write watch)
    uint64_t rhash;               // MKW_REUP_SAME / MKW_REUP_CENSUS: XXH3 of the storage just before upload n_up
    uint32_t rhash_nup;           // the n_up rhash belongs to; any other n_up (an upload since, a new image): unknown
    uint32_t shash_epoch;         // submit_epoch of the last hash check
    uint8_t sparse;               // a Vulkan sparse image whose tiles follow the game's MapVirtual (vk_sparse.c)
    VkSparseTail sparse_tail;      // privately backed small mips; zeroed at image initialization, without touching shared tiles
    uint64_t use;                 // MKW_GPU_GUARD: newest submission part that looked it up (vk_guard_note)
    uint16_t pin;                 // held by a caller inside an upload or texel copy (tex_pin): eviction and the release
                                  // sweeps skip it, since staging that runs short trims textures on this same thread
                                  // (the lock is recursive) and the caller goes on using the entry afterwards
    uint8_t cpu_newer;            // GPU-owned and mirrored: a WriteTexels reached the storage but not the image (no
                                  // staging), so the storage is the newer content (make_fresh takes it back)
    VkBuffer rb;                  // readback of a compute-written pitch texture (readback_record_locked): host-visible
    VmaAllocation rba;
    uint8_t *rbp;                 // its mapping
    size_t rb_size;
    uint64_t rb_fence;            // the submission whose last write of the image the buffer's copy follows (0: none)
    uint64_t rb_other;            // the newest submission with another GPU write of the image (not a dispatch)
} Tex;

// A slot keeps what vk_texture_by_id resolved for it, so a draw binding skips the key lookup and the address chain: the
// entry (ce), valid while the generation of its address (agen) is still cgen, the view and has_view handed out with it,
// and the submit_epoch of the last freshness check through the slot (cepoch). Hot fields first: a lookup that hits
// touches one or two cache lines.
typedef struct {
    struct Tex *ce;               // NULL: nothing cached (registered since, or never resolved)
    uint32_t cgen;                // agen of the slot's address when ce was resolved
    uint32_t cepoch;
    view_t cview;
    uint8_t chas_view, used, has_view;
    view_t v;
    tex_t t;
} TexSlot;
typedef struct { samp_par p; VkSampler s, sp; uint8_t used; } SampSlot;   // s: vk_sampler_for(&p) once looked up; sp:
                                                                           // its point-sampled form (vk_sampler_point_by_id_locked)

// vk_texture.c: the cache state, under tex_lock unless noted
#define AGEN_N 4096
extern PlatRMutex tex_lock;
extern PlatOnce tex_once;
extern volatile int32_t tex_ready;
extern nvn_hmap by_key, by_addr, samplers;
extern TexSlot *tslots;
extern int ntslots;
extern SampSlot *sslots;
extern int nsslots, nreleased;
extern uint64_t released_bytes;   // own memory of the released entries (vk_tex_evict.c sweep_released)
extern Tex *rel_head, *rel_tail;
extern uint64_t n_uploads, n_reuploads, ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns, ru_skip_sub, ru_skip_bytes;
// MKW_TRACE=perf: where upload() spends its time (tex_lock), per perf line: the precomputed-texture path (hash and
// copy), flushing the open batch, getting staging memory (may wait for the GPU), deswizzle/decode on the CPU, recording
// the copy or ASTC decode; and the slowest single upload with its own split.
enum { UPH_TEXPRE, UPH_FLUSH, UPH_STAGING, UPH_DESWIZZLE, UPH_RECORD, UPH_END, UPH_N };   // RECORD: the ASTC decode or copy call; END: barriers and closing the batch
extern uint64_t up_ph_ns[UPH_N], up_ph_n;
typedef struct { uint64_t addr, bytes, total_ns, ph[UPH_N]; uint32_t w, h, levels, layers, fmt; int again; } UpSlow;
extern UpSlow up_slow;
extern uint64_t rc_n, rc_bytes, rc_ns;   // vk_texture_region_copy (vk_tex_upload.c)
extern int tex_alias;
extern uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
extern uint64_t n_alias_deferred, n_alias_placed, n_alias_dropped;   // vk_tex_alias.c (MKW_REPLAY_PAR's walk)
extern uint32_t submit_epoch, tex_gen, slots_gen, samp_gen, reg_serial;
extern PlatMutex owned_lock;
extern Tex **owned;
extern int nowned, cowned;
extern uint64_t max_owned, gpu_writes, n_checks;
extern uint64_t n_entries_made;   // entries entry_of created (tex_lock), for the precheck slow-pass line
int tex_snap_scoped(void);   // MKW_SNAP_SCOPED on now (vk_tex_upload.c)
extern uint32_t agen[AGEN_N];
extern uint64_t *sinfo, *schk;
void tex_init(void *p);
uint32_t agen_idx(uint64_t addr);
void tex_gen_bump(uint64_t addr);
void make_key(const tex_t *t, TexKey *k);
int same_but_swizzle(const TexKey *a, const TexKey *b);
uint32_t tex_evict_idle(void);
int tex_mirrored(const tex_t *t);
void tex_zero_tail_recheck(uint64_t frame);   // MKW_TEX_UP_LOG's zero tails read again (vk_tex_upload.c)
void log_limited(volatile int32_t *n, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
             VkAccessFlags2 da);
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT

// With MKW_TRACE=perf, a lock() that has to wait is timed (tl_*, under tex_lock once taken): the PS5's race hitches
// (run hang05, 16/11/2026) had the game's texture registration and the chunk threads waiting on tex_lock while the
// replay re-uploaded whole textures under it.
extern uint64_t tl_n, tl_ns, tl_max;
// Also with MKW_TRACE=perf: who holds it. The outermost take on each thread notes the thread and its call site
// (tl_took: noinline, so its return address is the site where lock() was inlined), the last release clears it, and a
// wait of TL_LONG_NS or more logs the waiter, the holder, how long it had held it and where it took it (vk_texture.c
// tl_long_wait). Bench of 28/11/2026: "replay > precheck > texlock" was ~30 % of the 1 % low's frames, and the frames
// that ended a memory-governor pass had the precheck at 11-16 ms and lock waits at 6-8 ms.
#define TL_LONG_NS 2000000ull
void tl_took(void);
void tl_site(uint64_t ra);   // vk_texture_lock: its own caller as the site
void tl_released(void);
void tl_long_wait(uint64_t dt, uint32_t htid, uint64_t hra, uint64_t ht0, const char *hname);
extern volatile uint32_t tl_htid;
extern volatile uint64_t tl_hra, tl_ht0;
extern const char *volatile tl_hname;
// MKW_TEX_LOCK_SPIN (default 1; "ab", applied per present by vk_texture_spin_update): a lock() that finds tex_lock
// taken tries again for up to TEX_SPIN_NS before it sleeps. Its holds are ~1 us (the chunk threads' descriptor
// lookups, 1,100-1,500 a frame) and a sleep costs a wake of ~15-20 us: in a course's streaming bursts the chunk threads
// took it twice as often and waited 3.8 ms a frame on it while their holds came to 0.9 (PS5, 20/11/2026). In-race
// A/B, three races: its waits 138-154 -> 41-50 ms per 60 frames, the replay -0.4 to -0.9 ms a frame, the frame
// -0.05 to -0.41 ms (medians -0.07 to -0.15); the frames over 20 ms no fewer.
#define TEX_SPIN_NS 30000
extern volatile int tex_spin_on;
static inline int tex_spin(void) {
    if (!__atomic_load_n(&tex_spin_on, __ATOMIC_RELAXED)) return 0;
    uint64_t t0 = plat_now_ns();
    do {
        for (int i = 0; i < 16; i++) plat_pause();   // (x86 pause, AArch64 yield: platform.h)
        if (plat_rmutex_trylock(&tex_lock)) return 1;
    } while (plat_now_ns() - t0 < TEX_SPIN_NS);
    return 0;
}
static inline void lock(void) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    if (!(rt_trace & RT_TRACE_PERF)) {
        if (!plat_rmutex_trylock(&tex_lock) && !tex_spin()) plat_rmutex_lock(&tex_lock);
        return;
    }
    if (plat_rmutex_trylock(&tex_lock)) {
        tl_took();
        return;
    }
    uint32_t htid = tl_htid;   // the holder as the wait begins
    uint64_t hra = tl_hra, ht0 = tl_ht0;
    const char *hname = tl_hname;
    uint64_t t0 = plat_now_ns();
    if (!tex_spin()) plat_rmutex_lock(&tex_lock);
    uint64_t dt = plat_now_ns() - t0;
    tl_n++, tl_ns += dt;
    perf_ledger(PERF_LG_TEXLOCK, dt);
    if (dt > tl_max) tl_max = dt;
    if (dt >= TL_LONG_NS) tl_long_wait(dt, htid, hra, ht0, hname);
    tl_took();
}
static inline void unlock(void) {
    if (rt_trace & RT_TRACE_PERF) tl_released();
    plat_rmutex_unlock(&tex_lock);
}
#define TL_TOOK() ((rt_trace & RT_TRACE_PERF) ? tl_took() : (void)0)   // after a direct take of tex_lock

// An entry a caller keeps using across a call that may run short of staging (vk_texture_trim, evict_idle and the
// release sweeps then free textures on this thread): those skip a pinned one.
static inline void tex_pin(Tex *e) { e->pin++; }
static inline void tex_unpin(Tex *e) { e->pin--; }

// A mirrored image whose first upload was deferred for want of staging memory: never written, it holds whatever the
// recycled memory held (other textures' texels), so a draw binds the zero stand-in instead until the upload is made.
// Not once the GPU owns it (a render wrote it) or an alias copy filled it (uploaded).
static inline int tex_unready(const Tex *e) {
    return e->pub.mirrored && !e->pub.gpu_owned && !e->uploaded && e->up_pending;
}

static inline uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static inline uint32_t mip(uint32_t v, unsigned lv) { return umax(v >> lv, 1); }

// vk_tex_image.c: images
Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init);
void init_image(Tex *e, VkCommandBuffer cb);
VkResult vma_image(VkCtx *c, const VkImageCreateInfo *ci, const VmaAllocationCreateInfo *ai, VkImage *img,
                   VmaAllocation *alloc);   // in the texture pool when it fits

// vk_tex_evict.c: released entries, eviction and the owned list
void destroy_tex(Tex *e);
void destroy_idle(Tex *e);   // one the submission being recorded does not use (released or idle): freed sooner
VkResult evict_idle(VkCtx *c, const VkImageCreateInfo *ci, const VmaAllocationCreateInfo *ai, VkImage *img,
                    VmaAllocation *alloc);
void own_add(Tex *e);
int owned_lower(uint64_t a);
void rel_append(Tex *e);
void rel_unlink(Tex *e);
Tex *rel_oldest(void);   // the oldest released entry not pinned; NULL when none
void sweep_released(void);
int drop_overwritten_live(void);   // issue 11: live GPU-owned images written over and idle go; how many
void tex_gpu_made(uint64_t bytes);   // a GPU-owned image of bytes made: drop_overwritten_live every 64 MiB
void tex_evict_shutdown(void);

// vk_tex_pressure.c: idle recreatable textures out when VMA's blocks near the live VRAM budget (MKW_TEX_PRESSURE)
void tex_pressure(void);   // each submit, the texture lock held

// vk_tex_upload.c: uploads and the freshness checks
void make_fresh(Tex *e);
// MKW_ASTC_BC1_ALPHA: ASTC bytes going into a BC1 image checked for alpha below 1 (vk_tex_image.c)
void vk_astc_bc1_alpha_note(uint64_t addr, const uint8_t *src, size_t size, uint32_t bw, uint32_t bh, const char *path);
void make_fresh_snap(Tex *e);
void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n);
void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n);
int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now);
// e holds content its storage bytes do not: the GPU wrote it and, for a mirrored texture, it was not written back and
// the CPU has not rewritten the storage since (an upload then reads the newer bytes). The alias copy's rule and the
// ASTC/region copies' (cpu_content).
int tex_has_gpu_content(const Tex *e);
void gpu_baseline(Tex *e);   // CPU writes to the storage from here on are newer than the image (cpu_took_back)
void gpu_base_note(Tex *e);          // at a GPU write: the bytes of storage that fills no whole page
int gpu_base_kept(const Tex *e);     // those bytes unchanged since: 1; changed: 0; storage with whole pages: -1
int rewritten_since_gpu(const Tex *e);   // the CPU rewrote e's storage after its GPU write: that render is not written back
extern uint64_t trace_addr;
int tracing(uint64_t addr);

// vk_tex_alias.c: aliases (MKW_TEX_ALIAS) and GPU writes into mirrored storage
enum { WB_SMALL = 64 << 10 };
// What goes back to its storage when the CPU waits for a fence (vk_texture_write_back_small): targets up to WB_SMALL
// bytes, and pitch (LINEAR, LINEAR_RENDER_TARGET) textures of any size: that layout is asked for so that the CPU reads
// the texels in place.
static inline int wb_at_fence(const nvn_texdesc *d, uint64_t size) { return size <= WB_SMALL || nvn_tex_is_linear(d); }
extern int small_pending;
Tex *entry_of(const tex_t *t);
Tex *entries_at(uint64_t addr);   // the first entry at that storage address; the rest by next_at

// vk_tex_texel.c: texel reads and writes
typedef struct { unsigned level, layer0; int nz, is3d, bw, bh, bpb; size_t wb, hb, row, img, x0, y0; } Region;
void readback_record_locked(Tex *e, VkCommandBuffer cb);   // after a dispatch wrote e (texture lock held)
int region_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img, Region *g);
int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write);
int storage_texels(const tex_t *t, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write);
