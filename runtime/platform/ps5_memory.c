// CPU backing for the PS5 platform. PS5's direct-memory unit is larger than
// its protection page, and both are larger than a guest page.
#include "ps5_memory.h"
#include "prot_watch.h"
#include "platform.h"

#include <ps5platform/kernel.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(PS5_MEM_PAGE == PS5_KERNEL_PAGE_SIZE, "unexpected PS5 protection page");
_Static_assert(PS5_MEM_UNIT == PS5_KERNEL_DIRECT_ALIGNMENT, "unexpected PS5 direct-memory unit");

static const int cpu_rw = PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE;

static int discard_relocation(Ps5MemBlock *block, void *placed) {
    block->relocated = placed;
    int result = sceKernelMunmap(placed, PS5_MEM_UNIT);
    if (!result) block->relocated = NULL;
    return result ? result : PS5_MEM_RELOCATED;
}

static int reserve_exact(Ps5MemBlock *block, uintptr_t address) {
    void *placed = (void *)address;
    int result = sceKernelReserveVirtualRange(&placed, PS5_MEM_UNIT, 0, PS5_MEM_UNIT);
    if (result || placed == (void *)address) return result;
    return discard_relocation(block, placed);
}

static int release_physical(Ps5MemBlock *block) {
    if (!block->allocated) return 0;
    int result = sceKernelReleaseDirectMemory(block->physical, PS5_MEM_UNIT);
    if (!result) block->allocated = false;
    return result;
}

int ps5_mem_reserve(Ps5MemBlock *block, uintptr_t address) {
    if (block->address || block->relocated || !address || address % PS5_MEM_UNIT || address > UINTPTR_MAX - PS5_MEM_UNIT)
        return PS5_MEM_BAD_REQUEST;
    int result = reserve_exact(block, address);
    if (!result) {
        block->address = address;
        block->reserved = true;
    }
    return result;
}

static int probe_any_mapping(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    void *address = NULL;
    report->any_attempted = true;
    report->any_status = sceKernelMapDirectMemory(&address, PS5_MEM_UNIT, cpu_rw, 0,
                                                 block->physical, PS5_MEM_UNIT);
    report->any_address = address;
    if (report->any_status) return 0;
    if (!address) return PS5_MEM_RELOCATED;

    // Keep the physical allocation live until its temporary map is gone.
    report->any_unmap_status = sceKernelMunmap(address, PS5_MEM_UNIT);
    if (report->any_unmap_status && address != (void *)block->address) block->relocated = address;
    if (address != (void *)block->address) return report->any_unmap_status;
    // An OS-chosen map should never overlap the reservation. A successful
    // unmap in that case would leave no owned range for the fixed test.
    if (!report->any_unmap_status) block->reserved = false;
    return report->any_unmap_status ? report->any_unmap_status : PS5_MEM_RELOCATED;
}

// The unit's first contents (zeros, or a packed unit's bytes) are written through a private map of it, and only then is
// it mapped at its guest address. Zeroing after the fixed map let another thread write the unit in between (it no
// longer faults, so it never waits for the unit's lock) and lost that write: the title's MetaDataBase::setData crash
// (19/11/2026), a SharedBuffer header the game had just written read back as 0. Two more calls per unit backed.
// Step times (ps5_mem_commit_times): bench of 28/11/2026, an 83.6 ms frame carried 83.8 ms of unpacks, and the fault
// storms of streamed zstd output were all fresh units committed, with no duration accounted for them at all.
static uint64_t commit_ns[PS5_COMMIT_N];
#define COMMIT_ADD(k, v) __atomic_add_fetch(&commit_ns[k], (v), __ATOMIC_RELAXED)
void ps5_mem_commit_times(uint64_t out[PS5_COMMIT_N]) {
    for (int i = 0; i < PS5_COMMIT_N; i++) out[i] = __atomic_load_n(&commit_ns[i], __ATOMIC_RELAXED);
}

static int fill_private(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx) {
    void *at = NULL;
    uint64_t t0 = plat_now_ns();
    int result = sceKernelMapDirectMemory(&at, PS5_MEM_UNIT, cpu_rw, 0, block->physical, PS5_MEM_UNIT);
    uint64_t t1 = plat_now_ns();
    if (report) report->private_status = result;
    if (result) return result;
    if (!at) return PS5_MEM_RELOCATED;
    if (fill) result = fill(ctx, at);
    else memset(at, 0, PS5_MEM_UNIT);   // the direct-memory allocator can return a previously used unit
    uint64_t t2 = plat_now_ns();
    int unmapped = sceKernelMunmap(at, PS5_MEM_UNIT);
    COMMIT_ADD(PS5_COMMIT_PRIVMAP, (t1 - t0) + (plat_now_ns() - t2));
    COMMIT_ADD(PS5_COMMIT_FILL, t2 - t1);
    if (report) report->private_unmap_status = unmapped;
    if (unmapped) block->relocated = at;   // still mapped: ps5_mem_release unmaps it, the physical unit is kept
    return result ? result : unmapped;
}

static int commit(Ps5MemBlock *block, Ps5MemCommitReport *report, bool probe_any, int (*fill)(void *ctx, void *unit),
                  void *ctx) {
    if (report) *report = (Ps5MemCommitReport){0};
    if (!block->reserved || block->committed || block->allocated || block->relocated) return PS5_MEM_BAD_REQUEST;
    uint64_t c0 = plat_now_ns();
    int result = sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), PS5_MEM_UNIT,
                                               PS5_MEM_UNIT, PS5_KERNEL_DIRECT_TYPE_CPU, &block->physical);
    COMMIT_ADD(PS5_COMMIT_ALLOC, plat_now_ns() - c0);
    if (report) *report = (Ps5MemCommitReport){.attempted = true, .allocation_status = result};
    if (result) return result;
    block->allocated = true;
    if (report) report->physical = block->physical;
    if (probe_any) {
        result = probe_any_mapping(block, report);
        if (result) return result;
    }

    result = fill_private(block, report, fill, ctx);
    if (result) {
        int cleanup = block->relocated ? 0 : release_physical(block);
        return cleanup ? cleanup : result;
    }

    // MAP_FIXED replaces the single reservation this block already owns. Use
    // the direct-unit alignment seen in Hookshot a3551c6's guest mappings.
    void *mapped = (void *)block->address;
    uint64_t f0 = plat_now_ns();
    result = sceKernelMapDirectMemory(&mapped, PS5_MEM_UNIT, cpu_rw, PS5_KERNEL_MAP_FIXED,
                                      block->physical, PS5_MEM_UNIT);
    uint64_t f1 = plat_now_ns();
    COMMIT_ADD(PS5_COMMIT_FIXED, f1 - f0);
    if (report) {
        report->fixed_attempted = true;
        report->fixed_address = mapped;
        report->mapping_status = result;
    }
    if (!result && mapped != (void *)block->address) result = discard_relocation(block, mapped);
    if (result) {
        int cleanup = block->relocated ? 0 : release_physical(block);
        return cleanup ? cleanup : result;
    }
    block->committed = true;
    if (!fill) COMMIT_ADD(PS5_COMMIT_FRESH_NS, f1 - c0), COMMIT_ADD(PS5_COMMIT_FRESH_N, 1);
    return 0;
}

int ps5_mem_commit(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    return commit(block, report, false, NULL, NULL);
}

int ps5_mem_commit_fill(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx) {
    return commit(block, report, false, fill, ctx);
}

int ps5_mem_commit_probe(Ps5MemBlock *block, Ps5MemCommitReport *report) {
    if (!report) return PS5_MEM_BAD_REQUEST;
    return commit(block, report, true, NULL, NULL);
}

static _Thread_local Ps5ProtBatch *tl_batch;   // ps5_mem_batch_begin
static void batch_flush(Ps5ProtBatch *b);

int ps5_mem_protect_native(Ps5MemBlock *block, size_t offset, size_t bytes, Ps5MemProt protection) {
    const int flags[] = {0, PS5_KERNEL_PROT_CPU_READ, PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE};
    if (!block->committed || !bytes || offset >= PS5_MEM_UNIT || bytes > PS5_MEM_UNIT - offset ||
        offset % PS5_MEM_PAGE || bytes % PS5_MEM_PAGE || (unsigned)protection >= sizeof flags / sizeof *flags)
        return PS5_MEM_BAD_REQUEST;
    int prot = flags[protection];
    if (protection > 0 && prot_watch_has_gpu(block->address + offset, bytes))
        prot |= PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
    Ps5ProtBatch *b = tl_batch;
    if (!b) return sceKernelMprotect((void *)(block->address + offset), bytes, prot);
    for (uintptr_t p = block->address + offset, end = p + bytes; p < end; p += PS5_MEM_PAGE) {
        if (b->n && (p < b->base || (p - b->base) / PS5_MEM_PAGE >= PS5_PROT_BATCH_PAGES)) batch_flush(b);
        if (!b->n) b->base = p;
        int i = (int)((p - b->base) / PS5_MEM_PAGE);
        while (b->n <= i) b->prot[b->n++] = -1;   // pages between that change nothing
        b->prot[i] = (int16_t)prot;
    }
    return 0;
}

static void batch_flush(Ps5ProtBatch *b) {
    for (int i = 0; i < b->n;) {
        int j = i + 1;
        while (j < b->n && b->prot[j] == b->prot[i]) j++;
        if (b->prot[i] >= 0) {
            int r = sceKernelMprotect((void *)(b->base + (uintptr_t)i * PS5_MEM_PAGE), (size_t)(j - i) * PS5_MEM_PAGE,
                                      b->prot[i]);
            if (r && !b->err) b->err = r;
        }
        i = j;
    }
    b->n = 0;
}

void ps5_mem_batch_begin(Ps5ProtBatch *batch) {
    batch->n = 0, batch->err = 0, batch->base = 0;
    tl_batch = batch;
}

int ps5_mem_batch_flush(void) {
    Ps5ProtBatch *b = tl_batch;
    if (!b) return 0;
    batch_flush(b);
    int e = b->err;
    b->err = 0;
    return e;
}

Ps5ProtBatch *ps5_mem_batch_swap(Ps5ProtBatch *batch) {
    Ps5ProtBatch *prev = tl_batch;
    tl_batch = batch;
    return prev;
}

int ps5_mem_batch_end(void) {
    int e = ps5_mem_batch_flush();
    tl_batch = NULL;
    return e;
}

int ps5_mem_protect(Ps5MemBlock *block, size_t offset, size_t bytes, bool writable) {
    return ps5_mem_protect_native(block, offset, bytes, writable ? PS5_MEM_RW : PS5_MEM_READ);
}

int ps5_mem_decommit(Ps5MemBlock *block) {
    if (!block->committed) return PS5_MEM_BAD_REQUEST;
    int result = sceKernelMunmap((void *)block->address, PS5_MEM_UNIT);
    if (result) return result;
    block->reserved = false;
    block->committed = false;

    // Reacquire without replacement. If someone took the hole, release only
    // our physical unit and leave the handle in a releasable state.
    result = reserve_exact(block, block->address);
    if (!result) block->reserved = true;
    int cleanup = release_physical(block);
    return cleanup ? cleanup : result;
}

int ps5_mem_release(Ps5MemBlock *block) {
    if (block->relocated) {
        int result = sceKernelMunmap(block->relocated, PS5_MEM_UNIT);
        if (result) return result;
        block->relocated = NULL;
    }
    if (block->reserved) {
        int result = sceKernelMunmap((void *)block->address, PS5_MEM_UNIT);
        if (result) return result;
        block->reserved = false;
        block->committed = false;
    }
    int result = release_physical(block);
    if (!result) *block = (Ps5MemBlock){0};
    return result;
}

void *ps5_mem_map_host(size_t bytes, size_t align, int64_t *physical) {
    if (!bytes || bytes % PS5_MEM_UNIT || !align || align % PS5_MEM_UNIT) return NULL;
#ifdef __PROSPERO__
    int64_t phys = 0;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, PS5_MEM_UNIT,
                                      PS5_KERNEL_DIRECT_TYPE_CPU, &phys))
        return NULL;
    void *at = NULL;
    if (sceKernelMapDirectMemory(&at, bytes, cpu_rw, 0, phys, align) || !at) {
        sceKernelReleaseDirectMemory(phys, bytes);
        return NULL;
    }
    *physical = phys;
    return at;
#else
    *physical = -1;
    return aligned_alloc(align, bytes);
#endif
}

int ps5_mem_unmap_host(void *at, size_t bytes, int64_t physical) {
#ifdef __PROSPERO__
    int result = sceKernelMunmap(at, bytes);
    if (result) return result;   // still mapped: the memory stays ours
    return sceKernelReleaseDirectMemory(physical, bytes);
#else
    (void)bytes, (void)physical;
    free(at);
    return 0;
#endif
}
