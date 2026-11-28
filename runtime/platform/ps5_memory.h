#ifndef MKW_PS5_MEMORY_H
#define MKW_PS5_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// One CPU-only direct-memory unit. Initialize to zero and serialize calls and
// accesses to the mapping. SDK failures are returned unchanged.
typedef struct {
    uintptr_t address;
    void *relocated;
    int64_t physical;
    bool reserved;
    bool committed;
    bool allocated;
} Ps5MemBlock;

enum { PS5_MEM_PAGE = 0x4000, PS5_MEM_UNIT = 0x10000 };
enum { PS5_MEM_BAD_REQUEST = -1, PS5_MEM_RELOCATED = -2 };

typedef struct {
    bool attempted;
    int allocation_status;
    int64_t physical;
    bool any_attempted;
    void *any_address;
    int any_status;
    int any_unmap_status;
    int private_status;         // the private map that writes the unit's first contents (fill_private)
    int private_unmap_status;
    bool fixed_attempted;
    void *fixed_address;
    int mapping_status;
} Ps5MemCommitReport;

int ps5_mem_reserve(Ps5MemBlock *block, uintptr_t address);
// An optional report preserves raw SDK results even if cleanup also fails.
int ps5_mem_commit(Ps5MemBlock *block, Ps5MemCommitReport *report);
// The same with the unit's first contents written by fill(ctx, unit) instead of zeros, before any thread can see them.
// A fill's nonzero result fails the commit and is returned.
int ps5_mem_commit_fill(Ps5MemBlock *block, Ps5MemCommitReport *report, int (*fill)(void *ctx, void *unit), void *ctx);
// First try an OS-chosen map of the same unit and discard it before mapping at
// the owned reservation. A failed OS-chosen map does not suppress the fixed test.
int ps5_mem_commit_probe(Ps5MemBlock *block, Ps5MemCommitReport *report);
// Running totals in ns of the commits' steps, every thread (never reset): direct-memory allocation, the private map and
// its unmap, the fill (zeros for a fresh unit, the bytes of a packed one), the fixed map at the guest address; then the
// same commits' whole time and count for fresh units (no fill function) apart. plat_mem_cost_ns reads them.
enum { PS5_COMMIT_ALLOC, PS5_COMMIT_PRIVMAP, PS5_COMMIT_FILL, PS5_COMMIT_FIXED, PS5_COMMIT_FRESH_NS, PS5_COMMIT_FRESH_N,
       PS5_COMMIT_N };
void ps5_mem_commit_times(uint64_t out[PS5_COMMIT_N]);
typedef enum { PS5_MEM_NONE, PS5_MEM_READ, PS5_MEM_RW } Ps5MemProt;
int ps5_mem_protect_native(Ps5MemBlock *block, size_t offset, size_t bytes, Ps5MemProt protection);
// A protect batch (one per thread): while one is open, ps5_mem_protect_native records each native page's final mode and
// the flush makes one sceKernelMprotect per run of equal mode. A frozen 2 MiB region's thaw was ~128 calls of ~44 us
// each, 9-10 ms on the game thread that wrote it (15/11/2026). The caller keeps the pages' units locked until the
// flush, so nobody sees the bookkeeping ahead of the protection.
enum { PS5_PROT_BATCH_PAGES = 512 };   // 8 MiB of native pages: a flush comes first when a page falls outside
typedef struct {
    uintptr_t base;   // first native page of the batch
    int n;            // pages from base with a slot (-1: no change)
    int err;          // the first failed call's result
    int16_t prot[PS5_PROT_BATCH_PAGES];
} Ps5ProtBatch;
void ps5_mem_batch_begin(Ps5ProtBatch *batch);
int ps5_mem_batch_flush(void);   // the first error since the last flush, 0 if none
int ps5_mem_batch_end(void);     // flushes and closes the thread's batch
Ps5ProtBatch *ps5_mem_batch_swap(Ps5ProtBatch *batch);   // the fault handler: protections it makes take effect now
int ps5_mem_protect(Ps5MemBlock *block, size_t offset, size_t bytes, bool writable);
int ps5_mem_decommit(Ps5MemBlock *block);
int ps5_mem_release(Ps5MemBlock *block);
// Direct memory for the runtime's own use (MKW_PACK's store) at an OS-chosen address aligned to align: bytes and align
// multiples of PS5_MEM_UNIT. NULL when the memory or the map fails (nothing kept). The host model (not __PROSPERO__)
// takes it from the C heap, so its tests' direct-memory counts stay those of the guest.
void *ps5_mem_map_host(size_t bytes, size_t align, int64_t *physical);
int ps5_mem_unmap_host(void *at, size_t bytes, int64_t physical);

#endif
