#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "heap_census.h"
#include "gfx.h"
#include "hangwatch.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_mem_internal.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- upload batches and deferred destruction
typedef struct { VkBuffer buf; VmaAllocation alloc; } Staging;
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg, cap;
    VkDeviceSize stg_bytes;   // staging the batch holds (vk_upload_flush_over)
    uint64_t texels;          // ASTC texels its dispatches decode (vk_upload_work)
    uint32_t calls;           // vk_upload_begin calls that recorded into it
    int ut;                   // MKW_UPLOAD_TIME slot, -1: untimed
    uint32_t crumb;           // its start mark (diag/vk_crumbs.c), 0 none
} Batch;

enum { DEAD_IMAGE = 1, DEAD_IMAGE_VMA, DEAD_VIEW, DEAD_BUFFER_VMA, DEAD_CALL };
// seq: the upload batch, tl: the submission part (vk_sync_part_done) and fence: the submission (timeline value) that may
// still use it; vma: NULL = the runtime's; use: MKW_GPU_GUARD, the newest part that named it (vk_mem_defer_use)
typedef struct { int kind; uint64_t handle; VmaAllocation alloc; uint64_t seq, tl, fence; VmaAllocator vma; uint64_t use; } Dead;

// Lock order: the queue lock, then sp_lock (vk_sparse.c: vk_sparse_order takes it under the queue lock), then cmd_lock
// (slot_get records each new slot's zero fill here under sp_lock). vk_upload_flush, the one place that submits under
// cmd_lock, takes the queue lock first. Taking it under cmd_lock made a cycle, the likeliest cause of the PS5's stop
// after "sparse bind 2" (run 5, 13/11/2026; MKW_HANG_WATCH names the threads if it comes back): the replay thread
// holds the queue lock and waits for sp_lock (vk_sparse_order, after its memory census, which on the PS5 makes several
// kernel calls with the queue lock held), a game thread holds sp_lock and waits for cmd_lock (vk_sparse_remap ->
// slot_get -> vk_upload_begin), and a thread sending an upload batch holds cmd_lock and waits for the queue lock.
static PlatRMutex cmd_lock;
static uint32_t cmd_owner;               // the thread holding cmd_lock, for the hang watch (read with no lock)
static _Thread_local int cmd_depth;
// MKW_TRACE=perf: cmd_lock takes that had to wait, and wait_and_collect's GPU waits (read and reset by
// vk_upload_wait_stats): an ASTC upload's recording took up to 1.9 s under the texture lock (run uptime, 16/11/2026).
static uint64_t ct_n, ct_ns, ct_max, wc_n, wc_ns, wc_max;
static inline __attribute__((always_inline)) void cmd_take(void) {
    const char *prev = hang_mark("upload: cmd_lock waiting");
    if ((rt_trace & RT_TRACE_PERF) && !plat_rmutex_trylock(&cmd_lock)) {
        uint64_t t0 = plat_now_ns();
        plat_rmutex_lock(&cmd_lock);
        uint64_t dt = plat_now_ns() - t0;
        ct_n++, ct_ns += dt;
        if (dt > ct_max) ct_max = dt;
    } else if (!(rt_trace & RT_TRACE_PERF)) {
        plat_rmutex_lock(&cmd_lock);
    }
    hang_mark(prev);
    if (!cmd_depth++) __atomic_store_n(&cmd_owner, hang_tid(), __ATOMIC_RELAXED);
}
static inline __attribute__((always_inline)) void cmd_give(void) {
    if (!--cmd_depth) __atomic_store_n(&cmd_owner, 0, __ATOMIC_RELAXED);
    plat_rmutex_unlock(&cmd_lock);
}
uint32_t vk_upload_owner(void) { return __atomic_load_n(&cmd_owner, __ATOMIC_RELAXED); }
static int depth;                     // vk_upload_begin calls not yet ended: a flush inside one would end its cb
static PlatOnce cmd_once = PLAT_ONCE_INIT;
static VkCommandPool cmd_pool;
static Batch open_b;
static int open_active;
static Batch *inflight;
static int ninflight, cap_inflight;
static Batch *spare;                  // completed batches whose command buffer and fence are reused
static int nspare, cap_spare;
static uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
static Dead *dead;
static int ndead, cap_dead;
static uint64_t tl_noted;             // newest part value vk_sync.c reported submitted

static void cmd_init(void *p) {
    (void)p;
    return;
}

void vk_mem_grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 16;
    void *n = realloc(*arr, (size_t)*cap * elem);
    if (!n) rt_fatal("vk: out of memory (upload bookkeeping)");
    *arr = n;
}

static void free_staging(VkCtx *c, Batch *b) {
    for (int i = 0; i < b->nstg; i++) vmaDestroyBuffer(c->vma, b->stg[i].buf, b->stg[i].alloc);
    b->nstg = 0;
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
static void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) {
            const char *prev = hang_mark_val("upload retire: vkWaitForFences", b->seq);
            vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
            hang_mark(prev);
        }
        else if (vkGetFenceStatus(c->device, b->fence) != VK_SUCCESS) break;
        free_staging(c, b);
        vk_uptime_done(b->ut, b->seq, b->stg_bytes, b->texels, b->calls);
        done_seq = b->seq;
        if (nspare == cap_spare) vk_mem_grow((void **)&spare, &cap_spare, sizeof *spare);
        spare[nspare++] = *b;
    }
    memmove(inflight, inflight + k, (size_t)(ninflight - k) * sizeof *inflight);
    ninflight -= k;
}

static void collect_locked(VkCtx *c) {
    retire(c, 0);
    uint64_t tl = vk_sync_part_done(), fdone = 0;
    if (c->timeline) vkGetSemaphoreCounterValue(c->device, c->timeline, &fdone);
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        Dead *d = &dead[i];
        if (d->seq > done_seq || d->tl > tl || d->fence > fdone) { dead[k++] = *d; continue; }
        if (d->use && d->kind != DEAD_CALL)
            vk_guard_dead(d->kind == DEAD_VIEW ? "image view" : d->kind == DEAD_BUFFER_VMA ? "buffer" : "image",
                          d->handle, 0, d->use);
        switch (d->kind) {
        case DEAD_IMAGE: vkDestroyImage(c->device, (VkImage)(uintptr_t)d->handle, NULL); break;
        case DEAD_IMAGE_VMA: vmaDestroyImage(c->vma, (VkImage)(uintptr_t)d->handle, d->alloc); break;
        case DEAD_VIEW: vkDestroyImageView(c->device, (VkImageView)(uintptr_t)d->handle, NULL); break;
        case DEAD_BUFFER_VMA: vmaDestroyBuffer(d->vma ? d->vma : c->vma, (VkBuffer)(uintptr_t)d->handle, d->alloc); break;
        case DEAD_CALL: ((void (*)(void *))(uintptr_t)d->handle)((void *)d->alloc); break;
        }
    }
    ndead = k;
}

VkCommandBuffer vk_upload_begin(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    VkCtx *c = vk_ctx();
    if (!c) rt_fatal("vk: upload before vk_init");
    depth++;
    if (open_active) {
        open_b.calls++;
        vk_uptime_call(open_b.ut, __builtin_return_address(0));
        return open_b.cb;
    }
    if (!cmd_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: upload command pool");
    }
    retire(c, 0);
    memset(&open_b, 0, sizeof open_b);
    if (nspare) {
        Batch s = spare[--nspare];
        open_b.cb = s.cb, open_b.fence = s.fence, open_b.stg = s.stg, open_b.cap = s.cap;
        vkResetCommandBuffer(open_b.cb, 0);
        vkResetFences(c->device, 1, &open_b.fence);
    } else {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, &open_b.cb) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, NULL, &open_b.fence) != VK_SUCCESS)
            rt_fatal("vk: upload command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(open_b.cb, &bi);
    open_b.seq = ++next_seq;
    open_b.crumb = vk_crumb_begin(open_b.cb, VK_CRUMB_UPLOAD, (uint32_t)open_b.seq, 0, 0, 0, NULL, 0);
    open_b.calls = 1;
    open_b.ut = vk_uptime_begin(open_b.cb);
    vk_uptime_call(open_b.ut, __builtin_return_address(0));
    open_active = 1;
    return open_b.cb;
}

void vk_upload_end(void) {
    depth--;
    cmd_give();
}

void vk_upload_lock(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
}
void vk_upload_unlock(void) { cmd_give(); }

void vk_upload_work(VkCommandBuffer cb, uint64_t texels) {
    cmd_take();
    if (open_active && open_b.cb == cb) open_b.texels += texels;
    cmd_give();
}

// Staging memory comes from a pool of its own that keeps STG_MIN_BLOCKS blocks from its first use (at boot). In the
// default pools VMA creates a block under the lock of its memory type (VmaBlockVector::Allocate), so a block made by
// the headroom thread (vk_headroom.c) or a burst of uploads held every staging allocation of the render thread: one
// took 47-50 ms, for a few bytes, in the first frame after a save load. Larger buffers than a block go to the default
// pool.
#ifdef MKW_PS5_NATIVE
#define STG_BLOCK (32ull << 20)
#define STG_MIN_BLOCKS 1
#else
#define STG_BLOCK (64ull << 20)
#define STG_MIN_BLOCKS 2
#endif
static VmaPool stg_pool;
static int stg_pool_tried;

// Waits for the GPU up to the newest submission, its parts already sent (the one being recorded) and every upload batch
// in flight, then frees what they held.
void vk_upload_wait_stats(uint64_t out[6]) {   // cmd_lock waits (n, ns, max), wait_and_collect (n, ns, max); resets
    out[0] = ct_n, out[1] = ct_ns, out[2] = ct_max, out[3] = wc_n, out[4] = wc_ns, out[5] = wc_max;
    ct_n = ct_ns = ct_max = wc_n = wc_ns = wc_max = 0;
}
static void wait_and_collect_body(VkCtx *c);
static void wait_and_collect(VkCtx *c) {
    uint64_t t0 = plat_now_ns();
    wait_and_collect_body(c);
    uint64_t dt = plat_now_ns() - t0;
    wc_n++, wc_ns += dt;
    if (dt > wc_max) wc_max = dt;
}
static void wait_and_collect_body(VkCtx *c) {
    vk_sync_fence_wait(vk_sync_issued(), UINT64_MAX);
    cmd_take();
    uint64_t parts = tl_noted;
    cmd_give();
    vk_sync_part_wait(parts);
    cmd_take();
    retire(c, 1);
    collect_locked(c);
    cmd_give();
}

// --probe-vkmem: the next vk_staging_fail_test staging buffers fail as out of memory, so the relief paths (texture trim,
// waits) run as on a console low on memory; 0 in a game run.
int vk_staging_fail_test;
static VkResult create_staging(VkCtx *c, const VkBufferCreateInfo *bi, const VmaAllocationCreateInfo *aci, VkBuffer *b,
                               VmaAllocation *a, VmaAllocationInfo *info) {
    if (__atomic_load_n(&vk_staging_fail_test, __ATOMIC_RELAXED) > 0) {
        __atomic_sub_fetch(&vk_staging_fail_test, 1, __ATOMIC_RELAXED);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return vmaCreateBuffer(c->vma, bi, aci, b, a, info);
}

// NULL when out of memory and fatal is 0.
static void *staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc, int fatal) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (!__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        cmd_take();
        if (!stg_pool_tried) {
            uint32_t type;
            VmaPoolCreateInfo pi = {.blockSize = STG_BLOCK, .minBlockCount = STG_MIN_BLOCKS};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, &bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                if (vmaCreatePool(c->vma, &pi, &stg_pool) != VK_SUCCESS) stg_pool = NULL;
            }
            rt_log("vk: staging pool %s (%d blocks of %llu MiB kept)", stg_pool ? "created" : "unavailable: default pool",
                   STG_MIN_BLOCKS, (unsigned long long)(STG_BLOCK >> 20));
            __atomic_store_n(&stg_pool_tried, 1, __ATOMIC_RELEASE);
        }
        cmd_give();
    }
    if (stg_pool && bi.size <= STG_BLOCK / 2) aci.pool = stg_pool;
    VmaAllocationInfo info;
    VkBuffer b;
    VmaAllocation a;
    VkResult r = create_staging(c, &bi, &aci, &b, &a, &info);
    // Out of memory: the default pool (the staging pool's blocks may all be in use), then idle textures given back
    // (run-race6 stopped on a 341 KiB staging buffer with the heap full of them).
    if (r != VK_SUCCESS && aci.pool) aci.pool = NULL, r = create_staging(c, &bi, &aci, &b, &a, &info);
    for (int round = 0; r != VK_SUCCESS && round < 4 && vk_texture_trim(bi.size * 2 + (8ull << 20)); round++) {
        vk_mem_collect();
        r = create_staging(c, &bi, &aci, &b, &a, &info);
    }
    // Last: wait for the GPU up to the newest submission, its parts already sent (the one being recorded) and every
    // upload batch in flight, so finalized pools, released textures waiting for their fence (cap-race1 died with 454
    // of them and heap0 at its cap; img-par4 with 2 630) and the staging of finished batches are freed, then try once
    // more.
    if (r != VK_SUCCESS && (retiring_count() || dead_pending() || ninflight)) {
        wait_and_collect(c);
        r = create_staging(c, &bi, &aci, &b, &a, &info);
        rt_log("vk: staging of %llu bytes after waiting for the GPU and the upload batches: %s",
               (unsigned long long)size, r == VK_SUCCESS ? "made" : "still out of memory");
    }
    // No retry in device-local host-visible memory: outside a pool VMA already tries every memory type that fits once
    // the preferred one fails (heap1's host-visible types on the Radeon 660M; h0b, 07/11/2026: heap0 at 200 MiB
    // replays the race without failing). cap-race1, cap-race6 and img-par4 died with both heaps at their caps.
    if (r != VK_SUCCESS) {
        vk_oom_census("staging buffer");
        vram_log(0);
        if (!fatal) return NULL;
        rt_fatal("vk: staging buffer of %llu bytes failed (%d)", (unsigned long long)size, (int)r);
    }
    *buf = b;
    if (alloc) *alloc = a;   // the caller destroys it (vmaDestroyBuffer) after vk_upload_flush(1)
    else {
        if (open_b.nstg == open_b.cap) vk_mem_grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
        open_b.stg[open_b.nstg++] = (Staging){b, a};
        open_b.stg_bytes += size;
    }
    return info.pMappedData;
}

void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) { return staging(size, buf, alloc, 1); }
// The same, NULL when out of memory even after waiting for the GPU (the caller gives up on what it was doing).
void *vk_staging_try(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) { return staging(size, buf, alloc, 0); }

// Out of memory: released and idle textures go (vk_texture_trim), and, with no vk_upload_begin open (this thread's or
// another's: the lock waits for it), the open batch is sent and waited for; then the GPU and the batches in flight are
// waited for and what they held is freed. What was released or evicted while the batch was open waits for it (defer),
// and nothing inside the batch can send it (tt4, 09/11/2026: 978 releases still pending after waiting for the GPU).
#define DMEM_SOFT_BYTES (1536ull << 20)
#define DMEM_HARD_BYTES (1024ull << 20)
#define GOV_REL_SOFT    (64ull << 20)
#define GOV_TRIM_HARD   (512ull << 20)
// MKW_GOV_TOTAL (default 1, "ab" to compare): the levels read all free direct memory (plat_mem_dmem_free_total), not
// only its largest free area, with DMEM_FLOOR_BYTES as a floor for that area so a large allocation still finds room;
// and the hard level's texture trim is the gentler vk_texture_trim_gov. On the largest area alone the governor sat at
// 602 MiB in a race while 1,262-1,503 MiB were free (tls-new4, 19/11/2026): hard every pass, 10 GiB of textures evicted.
#define DMEM_FLOOR_BYTES (128ull << 20)

uint64_t vk_upload_stg_bytes(void) {
    VkCtx *c = vk_ctx();
    VmaPool p = __atomic_load_n(&stg_pool, __ATOMIC_ACQUIRE);
    if (!c || !p) return 0;
    VmaStatistics st = {0};
    vmaGetPoolStatistics(c->vma, p, &st);
    return st.blockBytes;
}

// MKW_PACK=1 (default 0, 1 in the PS5 profile; cfg.c): cold guest memory compressed and its direct memory given
// back (plat_mem_pack, ps5_guest_memory.c), before textures are evicted. Run memgrow2 (16/11/2026): ~1.2 GiB of one pool
// is data the game no longer reads but never returns. Only on the governor's own thread (vk_headroom.c, force 0): the
// others that call it (vk_mem_relieve) are the game's and the replay's, on the frame's critical path. It packs toward
// DMEM_SOFT_BYTES + PACK_MARGIN free, at most PACK_BUDGET_US a tick (~750 units at ~130 us each, below-normal priority,
// one tick per 500 ms). MKW_PACK_COLD_S (default 30; EglantinePS5 used 20-60 s): frozen for at least that long.
// MKW_PACK_STORE_MIB (default 512, 1024 in the PS5 profile): the store's cap, in 1 MiB slabs of direct memory. MKW_PACK_VERIFY=1: each unit's
// bytes kept too and compared when it unpacks (the copies count toward the cap).
#define PACK_MARGIN (256ull << 20)
#define PACK_BUDGET_US 100000u

// MKW_PACK_TEXFREE (default 1 in the PS5 profile, with MKW_PACK): the free blocks of agl::TextureDataMgr's texture heap
// are packed first. A Knockout Tour's guest growth is that heap (2,372 MiB), and its free blocks held 300-650 MiB of
// backed memory while racing and 1,349 MiB at a rally's change (heap census, koTEX2 27/11/2026): texture data the heap
// has evicted or moved, still backed. The 2 MiB regions that lie wholly inside a free block are frozen (the cold
// feed's vk_mem_cold_freeze: a write thaws one) and packed without the cold wait, TEXFREE_COLD_MS only; then the usual
// pass. Lossless as all packing: a block the game allocates meanwhile unpacks on its first access with its bytes.
// Only on the governor's own thread (pack_step, force 0).
#define TEXFREE_MAX 64
#define TEXFREE_COLD_MS 2000u
#define TEXFREE_REGION (1ull << REG_SHIFT)
static uint64_t texfree_regions, texfree_packed;   // regions that gave memory back, and how much, since the last line
static uint64_t texfree_pack(const PlatPackParams *base) {
    static int on = -1;
    if (on < 0) on = cfg_bool("MKW_PACK_TEXFREE");
    if (!on) return 0;
    uint64_t r[TEXFREE_MAX][2], freed = 0, t0 = plat_now_ns(), budget_ns = (uint64_t)base->budget_us * 1000 / 2;
    int n = texheap_free_ranges(r, TEXFREE_MAX);
    PlatPackParams p = *base;
    p.cold_ms = TEXFREE_COLD_MS;
    for (int i = 0; i < n && freed < base->target_bytes && plat_now_ns() - t0 < budget_ns; i++) {
        uint64_t lo = (r[i][0] + TEXFREE_REGION - 1) & ~(TEXFREE_REGION - 1), hi = r[i][1] & ~(TEXFREE_REGION - 1);
        for (uint64_t a = lo; a < hi && freed < base->target_bytes && plat_now_ns() - t0 < budget_ns; a += TEXFREE_REGION) {
            if (!vk_mem_cold_freeze(a)) continue;   // (1 also for a region frozen by an earlier pass)
            p.target_bytes = base->target_bytes - freed;
            uint64_t f = plat_mem_pack_range(a, a + TEXFREE_REGION, &p);
            freed += f, texfree_packed += f, texfree_regions += f != 0;
        }
    }
    static uint64_t last_log;
    if (texfree_packed && plat_now_ns() - last_log > 30000000000ull) {
        rt_log("vk: memory governor: texture heap free blocks (MKW_PACK_TEXFREE): %d blocks now, %.1f MiB packed from %llu "
               "regions since the last line", n, texfree_packed / 1048576.0, (unsigned long long)texfree_regions);
        texfree_regions = texfree_packed = 0, last_log = plat_now_ns();
    }
    return freed;
}

static uint64_t pack_step(uint64_t avail) {
    static int on = -1, verify, backoff;
    static uint32_t cold_ms;
    static uint64_t cap;
    if (on < 0) {
        const char *e = cfg_get("MKW_PACK_COLD_S");
        long s = e && *e ? atol(e) : 30;
        cold_ms = (uint32_t)(s > 0 ? s : 0) * 1000u;
        e = cfg_get("MKW_PACK_STORE_MIB");
        long mib = e && *e ? atol(e) : 512;
        cap = (uint64_t)(mib > 0 ? mib : 0) << 20;
        verify = (e = cfg_get("MKW_PACK_VERIFY")) && atoi(e);
        backoff = cfg_int("MKW_PACK_BACKOFF");   // (ps5_guest_memory.c PK_BOUNCE_MAX)
        on = (e = cfg_get("MKW_PACK")) && atoi(e);
        if (on)
            rt_log("vk: memory governor: packing cold guest memory (frozen %u s, store up to %llu MiB%s)", cold_ms / 1000,
                   (unsigned long long)(cap >> 20), verify ? ", verified" : "");
    }
    if (!on || avail >= DMEM_SOFT_BYTES + PACK_MARGIN) return 0;
    PlatPackParams p = {.target_bytes = DMEM_SOFT_BYTES + PACK_MARGIN - avail, .store_cap = cap, .cold_ms = cold_ms,
                        .budget_us = PACK_BUDGET_US, .verify = verify, .backoff = backoff};
    uint64_t t0 = plat_now_ns(), freed = texfree_pack(&p);
    if (freed >= p.target_bytes) return freed;
    p.target_bytes -= freed;
    uint64_t used_us = (plat_now_ns() - t0) / 1000;
    p.budget_us = used_us < PACK_BUDGET_US ? PACK_BUDGET_US - (uint32_t)used_us : 1;
    return freed + plat_mem_pack(&p);
}

// MKW_BENCH's memory report (core/diag/bench_mem.c): governor passes by level, what they freed and out-of-memory
// reliefs since start (vk_mem_pressure, with the staging uploads deferred below).
static volatile uint64_t gov_soft_n, gov_hard_n, gov_freed_bytes, relieve_n;

void vk_mem_governor(int force) {
    static uint32_t idle_logs;
    if (!force) plat_mem_pack_collect();   // the blobs of units unpacked since: freed here, not in the fault handler
    // The cold tracker (ps5_guest_memory.c plat_mem_cold_census_step): runs at 20 ms/tick so cold units are already
    // frozen before dmem hits DMEM_SOFT_BYTES; when MKW_COLD_CENSUS=0 (only MKW_PACK on), line is NULL so cc_unit
    // skips the ZSTD ratio sampling overhead.
    static int census_diag = -1, feed = 0;
    static uint32_t feed_s;
    if (census_diag < 0) {
        const char *e = cfg_get("MKW_PACK");
        feed = e && atoi(e) > 0;
        census_diag = rt_knob("COLD_CENSUS", "0") > 0;
        e = cfg_get("MKW_PACK_FEED_S");
        long v = e && *e ? atol(e) : 60;
        feed_s = (uint32_t)(v > 0 ? v : 60);
    }
    if (!force && (census_diag || feed)) {
        char line[1024];
        if (plat_mem_cold_census_step(20000, feed_s, feed ? vk_mem_cold_freeze : NULL,
                                      census_diag ? line : NULL, census_diag ? sizeof line : 0))
            rt_log("mem: cold census: %s", line);
    }
    // MKW_SHADOW_TRIM_TEST=1 (a test; with MKW_SHADOW_TRIM=1): every tick trims the cold shadows whatever the memory, so
    // Linux, whose dmem never runs low, drops and rebuilds them as the PS5's governor does (task 28's magenta check).
    static int trim_test = -1;
    if (trim_test < 0) trim_test = cfg_get("MKW_SHADOW_TRIM_TEST") != NULL;
    if (trim_test) {
        static uint64_t trimmed;
        static uint32_t n_trims;
        uint64_t f = shadow_trim(512ull << 20);
        trimmed += f;
        if (f && (++n_trims <= 8 || !(n_trims & (n_trims - 1))))
            rt_log("vk: shadow trim test: %.1f MiB dropped (%.1f MiB in %u trims)", f / 1048576.0, trimmed / 1048576.0, n_trims);
    }
    uint64_t avail = plat_mem_dmem_avail();
    static int gov_total = -1;
    if (gov_total < 0) gov_total = rt_knob("GOV_TOTAL", "1");
    int by_total = RT_KNOB_ON(gov_total);
    uint64_t total = avail;
    if (by_total && avail < DMEM_SOFT_BYTES) {
        uint64_t t = plat_mem_dmem_free_total();
        if (t != UINT64_MAX && t > avail) total = t;
    }
    int hard = by_total ? total < DMEM_HARD_BYTES || avail < DMEM_FLOOR_BYTES : avail < DMEM_HARD_BYTES;
    if (!hard && total >= DMEM_SOFT_BYTES) {
        vk_mem_shadow_shed(0);
        return;
    }
    uint64_t rel_freed = 0, hr_freed = 0, trim_freed = 0, sp_freed = 0, sh_freed = 0, h0_freed = 0, zero_freed = 0,
             zero_scanned = 0, pk_freed = 0;
    const char *level;
    uint64_t t0 = plat_now_ns(), zero_ns = 0;
    int ab_side = rt_ab();
    if (hard) {
        level = "hard";
        vk_mem_shadow_shed(1);
        rel_freed = vk_texture_trim_released(0);
        hr_freed = vk_headroom_shed();
        if (!force) pk_freed = pack_step(plat_mem_dmem_avail());
        trim_freed = by_total ? vk_texture_trim_gov(GOV_TRIM_HARD) : vk_texture_trim(GOV_TRIM_HARD);
        sp_freed = vk_sparse_trim(GOV_TRIM_HARD);
        sh_freed = shadow_trim(GOV_TRIM_HARD);
        h0_freed = vk_usnap_trim(8ull << 20) + (uint64_t)vk_exec_cb_trim_spare(2) * (4ull << 20);
        vk_mem_collect();
        // MKW_ZERO_SCAN_CACHE (default 1, "ab" to compare): skip the units found not all zero earlier in the same
        // freeze. Without it a race's hard level, every ~4 frames, read ~1.4 GB of frozen units and freed 0 MiB
        // (wake-20261115-185222: 2,221 passes), memory bandwidth and L3 the game's threads share.
        static int zcache = -1;
        if (zcache < 0) zcache = rt_knob("ZERO_SCAN_CACHE", "1");
        uint64_t z0 = plat_now_ns();
        zero_freed = plat_mem_reclaim_zeros(256ull << 20, &zero_scanned, RT_KNOB_ON(zcache));
        zero_ns = plat_now_ns() - z0;
    } else {
        level = "soft";
        rel_freed = vk_texture_trim_released(GOV_REL_SOFT);
        hr_freed = vk_headroom_shed();
        if (!force) pk_freed = pack_step(plat_mem_dmem_avail());
        sh_freed = shadow_trim(128ull << 20);
    }
    uint64_t after = plat_mem_dmem_avail();
    perf_ledger(PERF_LG_GOV, plat_now_ns() - t0);
    __atomic_add_fetch(hard ? &gov_hard_n : &gov_soft_n, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&gov_freed_bytes,
                       rel_freed + hr_freed + trim_freed + sp_freed + sh_freed + h0_freed + zero_freed + pk_freed,
                       __ATOMIC_RELAXED);
    if (rel_freed || hr_freed || trim_freed || sp_freed || sh_freed || h0_freed || zero_freed || pk_freed ||
        (__atomic_add_fetch(&idle_logs, 1, __ATOMIC_RELAXED) % 60) == 1)
        rt_log("vk: memory governor (%s): dmem=%llu->%llu MiB (all free %llu), freed rel=%.1f headroom=%.1f pack=%.1f trim=%.1f sparse=%.1f shadow=%.1f heap0=%.1f zero=%.1f (scanned=%.1f) MiB; %.2f ms (zero scan %.2f) ab=%c",
               level, (unsigned long long)(avail >> 20), (unsigned long long)(after >> 20), (unsigned long long)(total >> 20),
               rel_freed / 1048576.0, hr_freed / 1048576.0, pk_freed / 1048576.0, trim_freed / 1048576.0,
               sp_freed / 1048576.0, sh_freed / 1048576.0, h0_freed / 1048576.0,
               zero_freed / 1048576.0, zero_scanned / 1048576.0, (plat_now_ns() - t0) / 1e6, zero_ns / 1e6,
               ab_side ? 'B' : 'A');
}

static void binds_before_flush(void);

void vk_mem_relieve(uint64_t bytes, const char *what) {
    static volatile int32_t n_logged;
    VkCtx *c = vk_ctx();
    if (!c) return;
    uint64_t t0 = plat_now_ns();
    __atomic_add_fetch(&relieve_n, 1, __ATOMIC_RELAXED);
    vk_mem_governor(1);
    vk_texture_trim(bytes);
    plat_once(&cmd_once, cmd_init, NULL);
    binds_before_flush();   // the flush below runs under cmd_lock, where it cannot issue them
    cmd_take();
    int outside = !depth;
    if (outside) vk_upload_flush(1);
    cmd_give();
    wait_and_collect(c);
    if (plat_inc32(&n_logged) <= 32)
        rt_log("vk: out of memory for %s of %llu bytes: textures trimmed, %s, GPU waited for (%.1f ms)", what,
               (unsigned long long)bytes, outside ? "open upload batch sent" : "inside an upload batch",
               (plat_now_ns() - t0) / 1e6);
}

// vk_upload_begin and the caller's staging buffer, before it records anything; out of memory with no other
// vk_upload_begin of this thread open, vk_mem_relieve outside the batch and both are tried again. Still out of memory:
// VK_NULL_HANDLE with *map NULL and no vk_upload_begin held (the caller does not call vk_upload_end), so the caller
// defers its upload to a later frame. It was fatal: run 31 ended on a 22 MB staging buffer with dmem at 292 MiB
// (14/11/2026), where the texture could have streamed in a frame later.
static volatile int32_t n_stg_deferred;
static VkCommandBuffer staging_failed(VkDeviceSize size) {
    vk_upload_end();
    long n = plat_inc32(&n_stg_deferred);
    if (n <= 16 || !(n & (n - 1)))
        rt_log("vk: no staging for %llu bytes even after relieving memory: the upload is deferred (%ld so far)",
               (unsigned long long)size, n);
    return VK_NULL_HANDLE;
}

void vk_mem_pressure(VkMemPressure *p) {
    p->gov_soft = __atomic_load_n(&gov_soft_n, __ATOMIC_RELAXED);
    p->gov_hard = __atomic_load_n(&gov_hard_n, __ATOMIC_RELAXED);
    p->gov_freed = __atomic_load_n(&gov_freed_bytes, __ATOMIC_RELAXED);
    p->relieves = __atomic_load_n(&relieve_n, __ATOMIC_RELAXED);
    p->stg_deferred = (uint64_t)__atomic_load_n(&n_stg_deferred, __ATOMIC_RELAXED);
}

VkCommandBuffer vk_upload_begin_staging(VkDeviceSize size, VkBuffer *buf, void **map) {
    VkCommandBuffer cb = vk_upload_begin();
    if (depth > 1) {   // inside another batch of this thread: nothing can be relieved here
        if ((*map = staging(size, buf, NULL, 0))) return cb;
        return staging_failed(size);
    }
    if ((*map = staging(size, buf, NULL, 0))) return cb;
    vk_upload_end();
    vk_mem_relieve(size * 2 + (8ull << 20), "staging");
    cb = vk_upload_begin();
    if ((*map = staging(size, buf, NULL, 0))) return cb;
    return staging_failed(size);
}

// A caller that adds to the batch without waiting (vk_tex_texel.c writes, vk_tex_upload.c) sends it once it holds
// `limit` bytes of staging, UPLOAD_TEXELS ASTC texels to decode or UPLOAD_CALLS recordings: the batch is only sent with
// the next submission, and the title capture's 8 155 writes in one batch ran the staging out of memory (wt1). The other
// two bound its GPU time under amdgpu's 2 s job timeout, which the race capture's load passed (xb-*, hang1): its
// batches took 170-430 ms per ~67 M texels, and 752-1 112 ms with 642 textures (MKW_UPLOAD_TIME up1, up2; each a full
// barrier, the GPU-only ones a clear of the whole image). Not inside a vk_upload_begin.
#define UPLOAD_TEXELS (64ull << 20)
#define UPLOAD_CALLS 128
void vk_upload_flush_over(VkDeviceSize limit) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    int over = open_active && !depth &&
               (open_b.stg_bytes >= limit || open_b.texels >= UPLOAD_TEXELS || open_b.calls >= UPLOAD_CALLS);
    cmd_give();
    if (over) vk_upload_flush(0);
}

// The tiles MapVirtual mapped since the last part are bound only at a part's split or submit (vk_sync.c, binds 2), and
// vk_upload_flush orders its batch behind the binds already issued (vk_sparse_order(0, 0)), not the queued ones. A batch
// sent in the middle of a recording (vk_upload_flush_over: 64 MiB of staging, 64 M ASTC texels or 128 calls; a
// readback) carried copies into tiles mapped since the last part to the GPU before those tiles were bound: the writes
// went nowhere and the tiles read zero (MKW fills its sparse terrain arrays a level of a layer at a time, ~300 copies
// at a load, right after mapping their pages). So the queued binds go first, as vk_sync.c does before its flush.
// Only when this thread does not hold the upload lock (the queue lock and sp_lock come before it) and there is a batch
// to send; nothing beyond two atomic loads when no bind is queued. vk_mem_relieve flushes under cmd_lock, so it calls
// this before taking it: a bind queued between the two (another thread mapping) is not for a copy already in the
// batch, since the game maps a submission's tiles before it submits and the replay records it after. A flush with
// this thread inside cmd_lock (vk_mem_relieve from a nested batch, which does not flush) issues none.
// On the PS5 (MKW_SPARSE_CPU_WAIT=1, vk_sparse.c cpu_wait_mode) vk_sparse_order(..., 2) waits on this thread, under the
// queue lock, for the GPU to finish every earlier submission before it binds: a mid-recording flush with binds queued
// costs that wait here instead of at the part's split. fence: what frees the tiles unbound by these binds: the
// submission being recorded, as vk_sync.c passes, or outside a recording two past the newest issued (the one being
// recorded may have taken its wait already, as tail_bind_now).
static void binds_before_flush(void) {
    if (cmd_depth || !vk_sparse_binds_queued()) return;
    cmd_take();
    int open = open_active;
    cmd_give();
    if (!open) return;
    uint64_t fence = vk_exec_recording_fence();
    if (!fence) fence = vk_sync_issued() + 2;
    vk_queue_lock();
    vk_sparse_order(fence, 2);
    vk_queue_unlock();
}

uint64_t vk_upload_flush(int wait) {
    plat_once(&cmd_once, cmd_init, NULL);
    vk_astc_batch_flush();
    binds_before_flush();
    cmd_take();
    VkCtx *c = vk_ctx();
    // The queue lock comes before cmd_lock (see cmd_lock): when another thread holds it, cmd_lock is let go while this
    // one waits for it. Only one batch is ever open and it is sent under both locks, so batches keep their order; a
    // caller already inside cmd_lock (vk_mem_relieve) still holds it through the wait, as before.
    int ql = 0;
    if (c && open_active && !(ql = vk_queue_trylock())) {
        cmd_give();
        vk_queue_lock();
        cmd_take();
        ql = 1;
    }
    uint64_t seq = next_seq;
    if (c && open_active) {
        for (int i = 0; i < open_b.nstg; i++) vmaFlushAllocation(c->vma, open_b.stg[i].alloc, 0, VK_WHOLE_SIZE);
        vk_uptime_end(open_b.cb, open_b.ut);
        vk_crumb_end(open_b.cb, VK_CRUMB_UPLOAD, open_b.crumb);
        if (vkEndCommandBuffer(open_b.cb) != VK_SUCCESS) {
            static int warned;
            if (!warned++) rt_log("vk: upload command buffer recording failed: batch skipped");
            for (int i = 0; i < open_b.nstg; i++) vmaDestroyBuffer(c->vma, open_b.stg[i].buf, open_b.stg[i].alloc);
            if (open_b.fence) vkDestroyFence(c->device, open_b.fence, NULL);
            free(open_b.stg);
            memset(&open_b, 0, sizeof open_b);
            open_active = 0;
        } else {
            VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &open_b.cb};
            VkSparseOrder so = vk_sparse_order(0, 0);   // ordered with the sparse binds like every graphics-queue submission
            VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkTimelineSemaphoreSubmitInfo tl = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                                .waitSemaphoreValueCount = so.wait ? 1u : 0u, .pWaitSemaphoreValues = &so.wait,
                                                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &so.signal};
            if (so.ssem) si.pNext = &tl, si.signalSemaphoreCount = 1, si.pSignalSemaphores = &so.ssem;
            if (so.ssem && so.wait) si.waitSemaphoreCount = 1, si.pWaitSemaphores = &so.wsem, si.pWaitDstStageMask = &ws;
            const char *prev = hang_mark_val("upload flush: vkQueueSubmit", seq);
            VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, open_b.fence);
            hang_mark(prev);
            if (r == VK_SUCCESS && so.ssem) vk_sparse_order_submitted(so.signal);
            if (r == VK_ERROR_DEVICE_LOST) vk_exec_recent_log();
            if (r != VK_SUCCESS) rt_fatal("vk: upload submit failed (%d)", (int)r);
            if (ninflight == cap_inflight) vk_mem_grow((void **)&inflight, &cap_inflight, sizeof *inflight);
            inflight[ninflight++] = open_b;
            open_active = 0;
        }
    }
    if (ql) vk_queue_unlock();   // before the fence waits: they need no order, and the present thread needs the lock
    if (c && wait) {
        while (ninflight && done_seq < seq) {
            const char *prev = hang_mark_val("upload flush: vkWaitForFences", inflight[0].seq);
            vkWaitForFences(c->device, 1, &inflight[0].fence, VK_TRUE, UINT64_MAX);
            hang_mark(prev);
            retire(c, 0);
        }
    }
    if (c) collect_locked(c);
    cmd_give();
    return seq;
}

static _Thread_local VmaAllocator defer_vma;   // vk_defer_buffer_in: the allocator of the buffer being deferred
// What a recording thread destroys may be named by the part it has not sent yet, or by a chunk not sent yet: it waits
// for the whole submission. A texture released or idle in it (vk_mem_defer_idle) waits only for the parts already sent:
// 07/11/2026, tying those to the submission too left the race's load (ht1) out of memory with 852-5 250 of them pending,
// and tying everything to the parts sent hung the GPU once in a capture replay (c-ser1, a page fault in 0x8001...).
static _Thread_local uint64_t rec_fence;
static _Thread_local int idle_defer;
static _Thread_local uint64_t defer_use;
static _Thread_local uint64_t defer_fence;   // vk_mem_defer_fence: the submission the object waits for, whatever the rest

void vk_mem_recording(uint64_t fence) { rec_fence = fence; }
void vk_mem_defer_idle(int on) { idle_defer += on ? 1 : -1; }
void vk_mem_defer_use(uint64_t use) { defer_use = vk_guard_on() ? use : 0; }
uint64_t vk_mem_defer_fence(uint64_t fence) {
    uint64_t prev = defer_fence;
    defer_fence = fence;
    return prev;
}

static void defer(int kind, uint64_t handle, VmaAllocation alloc) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    // MKW_GPU_GUARD: tied to the submission being recorded, every part of it is covered; otherwise only the parts sent.
    uint64_t fence = defer_fence ? defer_fence : idle_defer ? 0 : rec_fence;
    if (defer_use && !fence)
        vk_guard_tie(kind == DEAD_VIEW ? "image view" : kind == DEAD_BUFFER_VMA ? "buffer" : "image", handle, 0,
                     defer_use, tl_noted);
    if (ndead == cap_dead) vk_mem_grow((void **)&dead, &cap_dead, sizeof *dead);
    // The open batch may record into the object: it counts as the newest user.
    dead[ndead++] = (Dead){kind, handle, alloc, open_active ? open_b.seq : next_seq, tl_noted,
                           fence, defer_vma, defer_use};
    cmd_give();
}

int dead_pending(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    int n = ndead;
    cmd_give();
    return n;
}

void vk_defer_image(VkImage img, VmaAllocation alloc) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc);
}
void vk_defer_view(VkImageView view) { if (view) defer(DEAD_VIEW, (uint64_t)(uintptr_t)view, NULL); }
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc);
}
// fn(arg) once the upload batches recorded so far are done (a buffer they use goes back to a pool, vk_astc.c).
void vk_defer_call(void (*fn)(void *), void *arg) { defer(DEAD_CALL, (uint64_t)(uintptr_t)fn, (VmaAllocation)arg); }

void vk_defer_buffer_in(VmaAllocator vma, VkBuffer buf, VmaAllocation alloc) {
    defer_vma = vma;
    vk_defer_buffer(buf, alloc);
    defer_vma = NULL;
}

void vk_mem_note_submit(uint64_t part_value) {
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    if (part_value > tl_noted) tl_noted = part_value;
    cmd_give();
}

void vk_mem_collect(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_once(&cmd_once, cmd_init, NULL);
    cmd_take();
    collect_locked(c);
    cmd_give();
}

uint64_t upload_batches(void) { return next_seq; }

// vk_mem_shutdown, the device idle: every batch and dead object goes, and the staging pool.
void upload_shutdown(VkCtx *c) {
    cmd_take();
    retire(c, 1);
    uint64_t keep = tl_noted;
    tl_noted = 0;
    for (int i = 0; i < ndead; i++) dead[i].seq = dead[i].tl = dead[i].fence = 0;   // the device is idle
    collect_locked(c);
    tl_noted = keep;
    for (int i = 0; i < nspare; i++) {
        vkDestroyFence(c->device, spare[i].fence, NULL);
        free(spare[i].stg);
    }
    nspare = 0;
    if (cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
    cmd_pool = VK_NULL_HANDLE;
    if (stg_pool) vmaDestroyPool(c->vma, stg_pool);   // after retire: no staging buffer is left in it
    stg_pool = NULL;
    stg_pool_tried = 0;
    cmd_give();
}
