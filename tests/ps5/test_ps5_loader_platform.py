"""Build the PS5 loader platform against a Linux model of its SDK calls."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recomp"))
from layout import placement_defines  # noqa: E402

KERNEL_MODEL = """
#include <stddef.h>
#include <stdint.h>
#define PS5_KERNEL_PAGE_SIZE 0x4000
#define PS5_KERNEL_DIRECT_ALIGNMENT 0x10000
#define PS5_KERNEL_DIRECT_TYPE_CPU 12
#define PS5_KERNEL_MAP_FIXED 0x10
#define PS5_KERNEL_PROT_CPU_READ 1
#define PS5_KERNEL_PROT_CPU_WRITE 2
#define PS5_KERNEL_PROT_GPU_READ 0x10
#define PS5_KERNEL_PROT_GPU_WRITE 0x20
int sceKernelGetPageTableStats(int *, int *, int *, int *);
int64_t sceKernelGetDirectMemorySize(void);
int32_t sceKernelAvailableDirectMemorySize(int64_t, int64_t, size_t, int64_t *, size_t *);
int32_t sceKernelAvailableFlexibleMemorySize(size_t *);
int32_t sceKernelConfiguredFlexibleMemorySize(size_t *);
int32_t sceKernelReserveVirtualRange(void **, size_t, int, size_t);
int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int, int64_t *);
int32_t sceKernelReleaseDirectMemory(int64_t, size_t);
int32_t sceKernelMapDirectMemory(void **, size_t, int, int, int64_t, size_t);
int32_t sceKernelMunmap(void *, size_t);
int32_t sceKernelMprotect(const void *, size_t, int);
int32_t sceKernelQueryMemoryProtection(void *, void **, void **, uint32_t *);
"""


@pytest.mark.skipif(sys.platform != "linux" or sys.maxsize < 2**39, reason="requires 64-bit Linux mmap")
def test_ps5_loader_platform(tmp_path: Path) -> None:
    cc = shutil.which("clang")
    if not cc:
        pytest.skip("requires Clang")
    sdk = tmp_path / "ps5platform"
    sdk.mkdir()
    (sdk / "kernel.h").write_text(KERNEL_MODEL)
    (tmp_path / "guest_layout.h").write_text(placement_defines("ps5-high144"))
    exe = tmp_path / "loader-platform"
    platform = ROOT / "runtime/platform"
    sources = (
        "ps5_guest_memory.c", "ps5_memory.c", "ps5_mixed.c", "ps5_loader_io.c",
        "ps5_mutex.c", "prot_watch.c", "fault_guard.c",
    )
    subprocess.run(
        [
            cc,
            "-std=gnu11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-pthread",
            "-DMKW_PS5_LOADER_ONLY",
            "-DZP_TRIM_AVAIL=0",   # MKW_ZERO_POOL: the model's 1 MiB of direct memory neither trims the pool
            "-DZP_TOP_AVAIL=UINT64_MAX",   # nor has units allocated ahead (ps5_memory.c)
            f"-I{platform}",
            *(f"-I{ROOT / 'runtime' / d}" for d in ("", "core/memory", "core/base", "core/kernel", "core/diag", "cpu")),
            f"-I{tmp_path}",
            str(ROOT / "tests/ps5/ps5_loader_platform_host.c"),
            str(ROOT / "runtime/core/memory/memory_early.c"),
            *(str(platform / name) for name in sources),
            "-lzstd",   # MKW_PACK's compression (the PS5 build links the same library)
            "-o",
            str(exe),
        ],
        check=True,
    )
    (tmp_path / "binary").write_bytes(b"mkw\0test")
    (tmp_path / "empty").touch()
    subprocess.run([str(exe)], cwd=tmp_path, check=True, timeout=15)
