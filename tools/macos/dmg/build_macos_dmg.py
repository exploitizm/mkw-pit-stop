#!/usr/bin/env python3
"""Build a self-contained Universal macOS (arm64 + x86_64) PS5 Builder .app and .dmg.

Why: upstream and the PS5 fork ship builders only for Linux and Windows. This script stages
dual-arch (arm64 + x86_64) Python 3.14, numpy/lz4/fonttools, conda-forge glslc + libshaderc
(relocated with @loader_path), universal zig, universal blink + minimal Debian 11 glibc sysroot
(for sm86/spec/spec.py), universal ps5-native-tool, the game-free ps5-bundle/, and the SwiftUI
GUI app ("Mario Kart World Pit Stop.app"), then packages everything into a compressed .dmg.

Prerequisites in build/macos-cache/ (built once):
  - ps5-bundle/         extracted from ps5-0.3.0 Linux builder
  - ps5-native-tool     universal Mach-O binary compiled from PS5_Vulkan/tooling/native
  - zig-universal/      universal Zig 0.17.1 binary + shared lib/
  - blink               universal blink 1.1.0 binary (--disable-strace) + blink-src/LICENSE
  - linux-sysroot/      minimal Debian 11 amd64 glibc 2.31 lib64/*.so

Usage: uv run tools/macos/dmg/build_macos_dmg.py [--version 0.3.0] [--stage-only]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from compression import zstd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "build/macos-cache"
DL = ROOT / "build/dist/dl"
APP_OUT = ROOT / "build/macos-app"
DMG_OUT = ROOT / "build/release/macos"

for _p in (ROOT / "tools/build", ROOT / "tools/ps5", ROOT / "tools/prepare", ROOT / "recomp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_ps5  # noqa: E402

CONDA = "https://conda.anaconda.org/conda-forge/"
PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
PYPI = "https://files.pythonhosted.org/packages/"

MACOS_PINS = {
    "arm64": {
        "python": (
            PBS + "cpython-3.14.10%2B20261026-aarch64-apple-darwin-install_only_stripped.tar.gz",
            "95eb8c87d16af37c54e9765e31d8c56c0d8ffa9968ea0f704f73a9b83f84e9f9",
        ),
        "numpy": (
            PYPI + "f3/49/fee7587c33ee35f7977f9051d7f2023d4e7246d62710c80f20c2361ea232/"
            "numpy-2.5.4-cp314-cp314-macosx_11_0_arm64.whl",
            "ffa6ce09a1c6a08e9667dd9c97aa0b14184e8d18f2a14b78b2a2328c9147f076",
        ),
        "lz4": (
            PYPI + "b6/cb/bfead8f437741ce51e14b3c7d404e3a1f6b409c440bad9b8f3945d4c40a7/"
            "lz4-4.4.5-cp314-cp314-macosx_11_0_arm64.whl",
            "c8e71b14938082ebaf78144f3b3917ac715f72d14c076f384a4c062df96f9df6",
        ),
        "fonttools": (
            PYPI + "35/74/4ea331ecc65ba05a58b1e4e8ccd604c8a15565f876dfbf93fc2a2ea2e22b/"
            "fonttools-4.68.0-py3-none-any.whl",
            "d76f8fdcb6936f5e1045a0c9dd721a4bf11112e60d1e243bfa1d39f06a8b3104",
        ),
        "shaderc": (
            CONDA + "osx-arm64/shaderc-2026.4-hfd4ff19_1.conda",
            "db004be85f8a777a0a4d62c465f4141be543a1a6af8b9c48d7d8fe7dcb02e8e8",
        ),
        "glslang": (
            CONDA + "osx-arm64/glslang-16.6.0-h1a33c25_0.conda",
            "4d485440f96b81337b8f810488e695a0fda0aea540025d02c021ccce51837720",
        ),
        "spirv-tools": (
            CONDA + "osx-arm64/spirv-tools-2026.4-hc0b298b_2.conda",
            "e3529ae4406adff290ae541501e3349b7d08d057720ae46c56f4a41a97922ad1",
        ),
        "spirv-cross": (
            CONDA + "osx-arm64/spirv-cross-1.4.363.0-he80e296_1.conda",
            "7a51fa836f1319e93b758039cacc99059b8397063e154f6cb83c47c2e22fd2b7",
        ),
        "libcxx": (
            CONDA + "osx-arm64/libcxx-23.1.6-hc645fd0_0.conda",
            "822b2a88313977594230b7be745324e4c50293a35cf9244bb19ecc9c2c385288",
        ),
    },
    "x86_64": {
        "python": (
            PBS + "cpython-3.14.10%2B20261026-x86_64-apple-darwin-install_only_stripped.tar.gz",
            "2fd8becd48cf0ff5f0a15df3c0db7cae7aa708dcaacf901bd30d6cdd6229370b",
        ),
        "numpy": (
            PYPI + "99/ba/005cb5edd580d2f84d7ca3206b92dc17d4388e56e6f87ffe8f2762f83139/"
            "numpy-2.5.4-cp314-cp314-macosx_10_15_x86_64.whl",
            "c668b2f0d651605b58892644b0e302c7157f7159544227758c896982ef384b18",
        ),
        "lz4": (
            PYPI + "63/9c/70bdbdb9f54053a308b200b4678afd13efd0eafb6ddcbb7f00077213c2e5/"
            "lz4-4.4.5-cp314-cp314-macosx_10_15_x86_64.whl",
            "c216b6d5275fc060c6280936bb3bb0e0be6126afb08abccde27eed23dead135f",
        ),
        "fonttools": (
            PYPI + "35/74/4ea331ecc65ba05a58b1e4e8ccd604c8a15565f876dfbf93fc2a2ea2e22b/"
            "fonttools-4.68.0-py3-none-any.whl",
            "d76f8fdcb6936f5e1045a0c9dd721a4bf11112e60d1e243bfa1d39f06a8b3104",
        ),
        "shaderc": (
            CONDA + "osx-64/shaderc-2026.4-hace2177_1.conda",
            "08453cc73096f79b2c0e3c0a7332d62d73b631c3b22ddb6ea5c91de546418efd",
        ),
        "glslang": (
            CONDA + "osx-64/glslang-16.6.0-he9b0248_0.conda",
            "857b86e48a1f88c42c79cc3fc44ece99f1077f82de996d8575efa385107bec12",
        ),
        "spirv-tools": (
            CONDA + "osx-64/spirv-tools-2026.4-hf9f308f_2.conda",
            "3014c4d49e708741a8da460ad459b4cc88583cd6d4497037eb2451f8103d6d67",
        ),
        "spirv-cross": (
            CONDA + "osx-64/spirv-cross-1.4.363.0-h41675c4_1.conda",
            "d881243fffdaf33f5ef96cfd9c3ae46508d1255f8861e925e5358f42e44f9733",
        ),
        "libcxx": (
            CONDA + "osx-64/libcxx-23.1.6-h88b51c2_0.conda",
            "a3c8011337fb36e3972152a2d18eaea0923dc4a67868f76c46792da7fbcb82f2",
        ),
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, want: str, arch_tag: str = "") -> Path:
    DL.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[1].replace("%2B", "+")
    if arch_tag and arch_tag not in name:
        name = f"{arch_tag}-{name}"
    dst = DL / name
    if not dst.is_file() or sha256(dst) != want:
        print(f"Downloading {name}...", flush=True)
        part = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256(part)
        if got != want:
            part.unlink(missing_ok=True)
            sys.exit(f"{name}: sha256 {got}, expected {want}")
        part.replace(dst)
    return dst


def unpack_top(archive: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(archive) as t:
        t.extractall(tmp, filter="tar")
    (top,) = tmp.iterdir()
    shutil.rmtree(dst, ignore_errors=True)
    top.rename(dst)
    tmp.rmdir()


def unpack_conda(archive: Path, dst: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        (inner,) = [n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")]
        with z.open(inner) as raw, zstd.open(raw) as zf, tarfile.open(fileobj=zf, mode="r|") as t:
            t.extractall(dst, filter="tar")


def unpack_wheel(wheel: Path, site: Path) -> None:
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)


def prune_python(py_dir: Path) -> None:
    """Remove unused test/GUI/static-lib files from python-build-standalone to keep DMG lean."""
    stdlib = py_dir / "lib/python3.14"
    for rel in ("test", "idlelib", "turtledemo", "tkinter", "ensurepip", "pydoc_data"):
        shutil.rmtree(stdlib / rel, ignore_errors=True)
    for d in (py_dir / "include", py_dir / "share"):
        shutil.rmtree(d, ignore_errors=True)
    for p in py_dir.rglob("*.a"):
        p.unlink(missing_ok=True)
    for p in py_dir.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


def relocate_macho_dir(glslc_dir: Path) -> None:
    """Rewrite conda-forge glslc + lib/*.dylib install names to @loader_path and re-sign."""
    for extra in ("include", "share", "lib/cmake", "lib/pkgconfig"):
        shutil.rmtree(glslc_dir / extra, ignore_errors=True)
    for a in (glslc_dir / "lib").glob("*.a"):
        a.unlink()

    lib_dir = glslc_dir / "lib"
    dylibs = {p.name: p for p in lib_dir.glob("*.dylib") if not p.is_symlink()}
    macho_magic = (b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")
    bins = [
        p for p in sorted((glslc_dir / "bin").glob("*"))
        if p.is_file() and not p.is_symlink() and p.open("rb").read(4) in macho_magic
    ]
    targets = [*bins, *dylibs.values()]

    for p in targets:
        if not p.is_file():
            continue
        p.chmod(0o755)
        is_bin = p.parent.name == "bin"
        if not is_bin:
            subprocess.run(["install_name_tool", "-id", f"@loader_path/{p.name}", str(p)], check=True)
        out = subprocess.check_output(["otool", "-L", str(p)], text=True)
        dep_lines = out.splitlines()[1 if is_bin else 2 :]
        for line in dep_lines:
            dep = line.strip().split(" (", 1)[0]
            if dep.startswith(("/usr/lib/", "/System/Library/")):
                continue
            dep_name = Path(dep).name
            # Match exact dylib or symlink in lib_dir (e.g. libshaderc_shared.1.dylib)
            if (lib_dir / dep_name).exists():
                new_ref = f"@loader_path/../lib/{dep_name}" if is_bin else f"@loader_path/{dep_name}"
                if dep != new_ref:
                    subprocess.run(["install_name_tool", "-change", dep, new_ref, str(p)], check=True)
            else:
                sys.exit(f"{p}: unresolved non-system dependency {dep}")

    for p in targets:
        if p.is_file():
            subprocess.run(["codesign", "--force", "--sign", "-", str(p)], check=True)


def copy_sources(src_root: Path) -> None:
    shutil.rmtree(src_root, ignore_errors=True)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "README.md")
    for rel in release_ps5.SOURCES:
        s, d = ROOT / rel, src_root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=skip)
        elif s.is_file():
            shutil.copy2(s, d)
        else:
            sys.exit(f"missing source {rel}")
    files = sorted(p.relative_to(src_root).as_posix() for p in src_root.rglob("*") if p.is_file())
    (src_root / "FILES").write_text("".join(f"{f}\n" for f in files), encoding="utf-8")


REL = "https://github.com/exploitizm/mkw-pit-stop/releases/download/ps5-0.3.0/"
BOOTSTRAP_PINS = {
    "bundle": (REL + "mkw-pit-stop-0.3.0-ps5-builder-linux-x86_64.tar.zst",
               "fa0e94470f48214c6aafc45fcf7c06acde55ed771bb85fe11c7b3f7825049252"),
    "vulkan": (REL + "ps5-vulkan-7f492b4c6f67.tar.gz",
               "10cc0386e203e054d98e7130ae1f04e9b1cc301627abdc561f8f539cfffb81cc"),
    "blink": ("https://github.com/jart/blink/archive/refs/tags/1.1.0.tar.gz",
              "2649793e1ebf12027f5e240a773f452434cefd9494744a858cd8bff8792dba68"),
    "zig-arm64": ("https://ziglang.org/download/0.17.1/zig-aarch64-macos-0.17.1.tar.xz",
                  "75f1bc7a1e9852f17ec9a0c1895f767c4ec56d46bc8a834920eece1d0359dd8d"),
    "zig-x86_64": ("https://ziglang.org/download/0.17.1/zig-x86_64-macos-0.17.1.tar.xz",
                   "9af07b20319fbd0c92ee46ad23e81d6a782b1aaea2cfb8dcbff35f1d2c7ddbb7"),
    "libc6": ("https://archive.debian.org/debian/pool/main/g/glibc/libc6_2.31-13+deb11u11_amd64.deb",
              "05f7264da867b37f4c5ce49266b558ea1e81e05a9464f623152fca70f3550282"),
}


def ensure_ps5_bundle(bundle_arg: str | None) -> None:
    dst = CACHE / "ps5-bundle"
    if (dst / "loader-inventory.o").is_file():
        return
    if bundle_arg:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(bundle_arg, dst)
        return
    url, want = BOOTSTRAP_PINS["bundle"]
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / "ps5-bundle.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # bsdtar handles the zstd archive; skip terminfo, whose links loop on case-insensitive volumes
    subprocess.run(["tar", "-xf", str(fetch(url, want)), "-C", str(tmp), "--exclude", "*/terminfo/*"], check=True)
    tops = list(tmp.iterdir())
    root = tops[0] if len(tops) == 1 and tops[0].is_dir() else tmp
    shutil.rmtree(dst, ignore_errors=True)
    root.rename(dst)
    shutil.rmtree(tmp, ignore_errors=True)


def ensure_ps5_native_tool(vulkan_arg: str | None) -> None:
    out = CACHE / "ps5-native-tool"
    if out.is_file():
        return
    if vulkan_arg:
        src = Path(vulkan_arg)
    else:
        url, want = BOOTSTRAP_PINS["vulkan"]
        src = CACHE / "ps5-vulkan"
        unpack_top(fetch(url, want), src)
    native = src / "tooling/native"
    files = [native / f"{n}.cpp" for n in ("native_app_builder", "self_container", "elf_object", "sce_module_writer")]
    parts = []
    for arch in ("arm64", "x86_64"):
        part = CACHE / f"ps5-native-tool.{arch}"
        subprocess.run(["clang++", "-std=c++20", "-O2", "-arch", arch, "-mmacosx-version-min=13.0",
                        "-I", str(native), *map(str, files), "-lz", "-o", str(part)], check=True)
        parts.append(str(part))
    subprocess.run(["lipo", "-create", *parts, "-output", str(out)], check=True)
    for p in parts:
        Path(p).unlink()


def ensure_universal_zig() -> None:
    out = CACHE / "zig-universal"
    if (out / "zig").is_file():
        return
    bins = []
    for arch in ("arm64", "x86_64"):
        url, want = BOOTSTRAP_PINS[f"zig-{arch}"]
        d = CACHE / f"zig-{arch}"
        unpack_top(fetch(url, want, arch_tag=arch), d)
        bins.append(d / "zig")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(CACHE / "zig-arm64/lib", out / "lib")
    subprocess.run(["lipo", "-create", *map(str, bins), "-output", str(out / "zig")], check=True)
    (out / "zig").chmod(0o755)
    for arch in ("arm64", "x86_64"):
        shutil.rmtree(CACHE / f"zig-{arch}", ignore_errors=True)


def ensure_universal_blink() -> None:
    out = CACHE / "blink"
    if out.is_file():
        return
    gmake = shutil.which("gmake")
    if not gmake:
        sys.exit("building blink needs GNU Make: brew install make")
    url, want = BOOTSTRAP_PINS["blink"]
    src = CACHE / "blink-src"
    unpack_top(fetch(url, want), src)
    parts = []
    for arch in ("arm64", "x86_64"):
        build = CACHE / f"blink-build-{arch}"
        shutil.rmtree(build, ignore_errors=True)
        shutil.copytree(src, build)
        flags = f"-O2 -arch {arch} -mmacosx-version-min=13.0"
        env = {**os.environ, "CFLAGS": flags, "LDFLAGS": f"-arch {arch}"}
        subprocess.run(["./configure", "--disable-strace"], cwd=build, env=env, check=True)
        subprocess.run([gmake, "-j8", "o//blink/blink"], cwd=build, env=env, check=True)
        parts.append(str(build / "o/blink/blink"))
    subprocess.run(["lipo", "-create", *parts, "-output", str(out)], check=True)
    out.chmod(0o755)
    for arch in ("arm64", "x86_64"):
        shutil.rmtree(CACHE / f"blink-build-{arch}", ignore_errors=True)


def ensure_linux_sysroot() -> None:
    lib = CACHE / "linux-sysroot/lib64"
    if (lib / "ld-linux-x86-64.so.2").is_file():
        return
    url, want = BOOTSTRAP_PINS["libc6"]
    deb = fetch(url, want)
    tmp = CACHE / "libc6.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(["tar", "-xf", str(deb), "-C", str(tmp)], check=True)  # bsdtar reads .deb (ar)
    (data,) = tmp.glob("data.tar.*")
    subprocess.run(["tar", "-xf", str(data), "-C", str(tmp)], check=True)
    lib.mkdir(parents=True, exist_ok=True)
    for name in ("ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1"):
        hits = list(tmp.rglob(name))
        if not hits:
            sys.exit(f"{name} not found in libc6 package")
        src = hits[0]
        while src.is_symlink():  # absolute links point inside the package extract
            link = os.readlink(src)
            src = tmp / link.lstrip("/") if link.startswith("/") else (src.parent / link).resolve()
        shutil.copy2(src, lib / name)
    shutil.rmtree(tmp, ignore_errors=True)


def stage_setup(setup_dir: Path, version: str, bundle: str | None = None, vulkan: str | None = None) -> None:
    ensure_ps5_bundle(bundle)
    ensure_ps5_native_tool(vulkan)
    ensure_universal_zig()
    ensure_universal_blink()
    ensure_linux_sysroot()
    for req in (
        CACHE / "ps5-bundle/loader-inventory.o",
        CACHE / "ps5-native-tool",
        CACHE / "zig-universal/zig",
        CACHE / "blink",
        CACHE / "linux-sysroot/lib64/ld-linux-x86-64.so.2",
    ):
        if not req.is_file():
            sys.exit(f"missing prerequisite {req}")

    setup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dual-arch Python 3.14 + wheels and relocated conda-forge glslc/shaderc
    for arch in ("arm64", "x86_64"):
        pins = MACOS_PINS[arch]
        got = {name: fetch(url, want, arch_tag=arch if name in ("shaderc", "glslang", "spirv-tools", "spirv-cross", "libcxx") else "") for name, (url, want) in pins.items()}
        arch_dir = setup_dir / "arch" / arch
        py_dir = arch_dir / "python"
        unpack_top(got["python"], py_dir)
        site = py_dir / "lib/python3.14/site-packages"
        for wheel in ("numpy", "lz4", "fonttools"):
            unpack_wheel(got[wheel], site)
        prune_python(py_dir)

        glslc_dir = arch_dir / "glslc"
        shutil.rmtree(glslc_dir, ignore_errors=True)
        for name in ("shaderc", "glslang", "spirv-tools", "spirv-cross", "libcxx"):
            unpack_conda(got[name], glslc_dir)
        relocate_macho_dir(glslc_dir)

    # 2. Universal zig, universal blink, minimal Linux x86-64 sysroot
    zig_dst = setup_dir / "zig"
    shutil.rmtree(zig_dst, ignore_errors=True)
    shutil.copytree(CACHE / "zig-universal", zig_dst)
    (zig_dst / "zig").chmod(0o755)

    bin_dst = setup_dir / "bin"
    bin_dst.mkdir(exist_ok=True)
    shutil.copy2(CACHE / "blink", bin_dst / "blink")
    (bin_dst / "blink").chmod(0o755)

    sysroot_dst = setup_dir / "linux-sysroot"
    shutil.rmtree(sysroot_dst, ignore_errors=True)
    shutil.copytree(CACHE / "linux-sysroot", sysroot_dst)

    # 3. Game-free ps5-bundle with universal macOS ps5-native-tool
    bundle_dst = setup_dir / "ps5-bundle"
    shutil.rmtree(bundle_dst, ignore_errors=True)
    shutil.copytree(CACHE / "ps5-bundle", bundle_dst)
    shutil.copy2(CACHE / "ps5-native-tool", bundle_dst / "bin/ps5-native-tool")
    (bundle_dst / "bin/ps5-native-tool").chmod(0o755)

    # Copy notices to setup root + include blink ISC license
    if (bundle_dst / "LEGAL.txt").is_file():
        shutil.copy2(bundle_dst / "LEGAL.txt", setup_dir / "LEGAL.txt")
    if (bundle_dst / "licenses").is_dir():
        shutil.copytree(bundle_dst / "licenses", setup_dir / "licenses", dirs_exist_ok=True)
    blink_lic = CACHE / "blink-src/LICENSE"
    if blink_lic.is_file():
        (setup_dir / "licenses").mkdir(exist_ok=True)
        shutil.copy2(blink_lic, setup_dir / "licenses/LICENSE-blink.txt")

    # 4. Patched Python sources
    copy_sources(setup_dir / "src")

    # 5. CLI wrapper for Terminal users
    cli = setup_dir / "prepare_ps5.sh"
    cli.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        'ARCH="$(uname -m)"\n'
        '[[ "$ARCH" == "arm64" ]] || ARCH="x86_64"\n'
        'export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"\n'
        'export BLINK_LOG_FILENAME="/dev/null"\n'
        'exec "$HERE/arch/$ARCH/python/bin/python3" "$HERE/src/tools/ps5/prepare_ps5.py" "$@"\n',
        encoding="utf-8",
    )
    cli.chmod(0o755)

    # 6. Verify 0 game-content violations (check Resources/ so setup/src/tools/romfs matches release_ps5's allowlist)
    errors = release_ps5.check_no_game_ps5(setup_dir.parent)
    if errors:
        sys.exit("Game-content check failed:\n" + "\n".join(errors))
    print(f"Staged setup clean (0 violations): {setup_dir}", flush=True)


def verify_staged_tools(setup_dir: Path) -> None:
    """Verify Python imports and Shaderc.near() under both arm64 and x86_64."""
    shader_test = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import numpy, lz4.block, fontTools.ttLib, compression.zstd, ssl, plistlib; "
        "import shaderc_lib; comp = shaderc_lib.Shaderc.near(sys.argv[2]); "
        "assert comp is not None, 'Shaderc.near returned None'; "
        "spv, err = comp.compile('#version 450\\nlayout(location=0) out vec4 o; void main(){o=vec4(1.0);}', 'ps', 't.frag'); "
        "assert not err and spv[:4] == b'\\x03\\x02#\\x07', f'bad spv: {err}'; "
        "print('OK', len(spv))"
    )
    for arch in ("arm64", "x86_64"):
        py = setup_dir / f"arch/{arch}/python/bin/python3"
        glslc = setup_dir / f"arch/{arch}/glslc/bin/glslc"
        env = dict(os.environ, DYLD_LIBRARY_PATH=str(glslc.parent.parent / "lib"))
        r = subprocess.run(
            ["arch", f"-{arch}", str(py), "-I", "-c", shader_test, str(setup_dir / "src/tools/shaders"), str(glslc)],
            env=env, capture_output=True, text=True, check=True,
        )
        print(f"Verify {arch} Python + Shaderc:", r.stdout.strip(), flush=True)
        r_zig = subprocess.run(["arch", f"-{arch}", str(setup_dir / "zig/zig"), "version"],
                               capture_output=True, text=True, check=True)
        r_nat = subprocess.run(["arch", f"-{arch}", str(setup_dir / "ps5-bundle/bin/ps5-native-tool"), "--help"],
                               capture_output=True, text=True, check=False)
        assert "usage:" in (r_nat.stdout + r_nat.stderr).lower(), f"ps5-native-tool {arch} failed: {r_nat.stderr}"
        print(f"Verify {arch} zig={r_zig.stdout.strip()}, ps5-native-tool=OK", flush=True)


def build_app_icon(resources_dir: Path) -> None:
    """Generate AppIcon.icns and header PNG from docs/images/icon0.png."""
    src_png = ROOT / "docs/images/icon0.png"
    if not src_png.is_file():
        return
    logo_dst = resources_dir / "pitstop-logo.png"
    subprocess.run(["sips", "-z", "256", "256", str(src_png), "--out", str(logo_dst)], check=True,
                   capture_output=True)
    iconset = APP_OUT / "AppIcon.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir(parents=True)
    for sz in (16, 32, 128, 256, 512):
        subprocess.run(["sips", "-z", str(sz), str(sz), str(src_png), "--out", str(iconset / f"icon_{sz}x{sz}.png")],
                       check=True, capture_output=True)
        subprocess.run(["sips", "-z", str(sz * 2), str(sz * 2), str(src_png),
                        "--out", str(iconset / f"icon_{sz}x{sz}@2x.png")], check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(resources_dir / "AppIcon.icns")], check=True)
    shutil.rmtree(iconset, ignore_errors=True)


def compile_swift_app(app_dir: Path, version: str) -> None:
    """Compile Universal (arm64 + x86_64) SwiftUI macOS app bundle."""
    swift_dir = HERE.parent / "app"
    sources = sorted(str(p) for p in swift_dir.glob("*.swift"))
    if not sources:
        sys.exit(f"missing Swift sources in {swift_dir}")

    macos_dir = app_dir / "Contents/MacOS"
    res_dir = app_dir / "Contents/Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    arm_bin = APP_OUT / "MKWPitStop.arm64"
    x64_bin = APP_OUT / "MKWPitStop.x86_64"
    uni_bin = macos_dir / "MKWPitStop"

    for arch, target, out_bin in (
        ("arm64", "arm64-apple-macosx13.0", arm_bin),
        ("x86_64", "x86_64-apple-macosx13.0", x64_bin),
    ):
        print(f"Compiling SwiftUI app for {arch}...", flush=True)
        subprocess.run([
            "swiftc", "-O", "-parse-as-library", "-sdk", sdk, "-target", target,
            "-framework", "SwiftUI", "-framework", "AppKit", "-framework", "UniformTypeIdentifiers",
            *sources, "-o", str(out_bin),
        ], check=True)

    subprocess.run(["lipo", "-create", str(arm_bin), str(x64_bin), "-output", str(uni_bin)], check=True)
    arm_bin.unlink(missing_ok=True)
    x64_bin.unlink(missing_ok=True)
    uni_bin.chmod(0o755)

    plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": "MKWPitStop",
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": "com.exploitizm.mkw-pit-stop",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "Mario Kart World Pit Stop",
        "CFBundleDisplayName": "Mario Kart World: Pit Stop",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "NSLocalNetworkUsageDescription": (
            "Mario Kart World: Pit Stop scans your local network to discover your PS5 "
            "and transfer the built title over TCP (port 19913) or FTP."
        ),
    }
    (app_dir / "Contents/Info.plist").write_bytes(plistlib.dumps(plist))
    build_app_icon(res_dir)


def codesign_app(app_dir: Path) -> None:
    """Ad-hoc sign all Mach-O binaries inside the .app bottom-up, then sign and verify the bundle."""
    for p in sorted(app_dir.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        if p.parent.name in ("MacOS", "bin", "zig") or p.suffix == ".dylib" or p.name in ("blink", "ps5-native-tool", "glslc"):
            with p.open("rb") as f:
                magic = f.read(4)
            if magic in (b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
                subprocess.run(["codesign", "--force", "--sign", "-", str(p)], check=True, capture_output=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(app_dir)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app_dir)], check=True)


def package_dmg(app_dir: Path, version: str) -> Path:
    DMG_OUT.mkdir(parents=True, exist_ok=True)
    dmg_stage = APP_OUT / "dmg-stage"
    shutil.rmtree(dmg_stage, ignore_errors=True)
    dmg_stage.mkdir(parents=True)
    subprocess.run(["ditto", str(app_dir), str(dmg_stage / app_dir.name)], check=True)
    (dmg_stage / "Applications").symlink_to("/Applications")
    (dmg_stage / "README-macOS.txt").write_text(
        f"Mario Kart World: Pit Stop {version} — Universal macOS PS5 Builder\n"
        "=====================================================================\n\n"
        "1. Drag 'Mario Kart World Pit Stop.app' into 'Applications'.\n"
        "2. First launch (ad-hoc signed open-source app):\n"
        "   If macOS Gatekeeper blocks opening an unsigned/ad-hoc app, either:\n"
        "   - Right-click the app -> Open -> Open, OR\n"
        "   - Run in Terminal: xattr -cr \"/Applications/Mario Kart World Pit Stop.app\"\n"
        "3. On your PS5, install the PPSA99193 placeholder from GitHub Releases\n"
        "   and open it so it displays its IP address (port 19913), or plug in\n"
        "   an exFAT USB drive (25+ GB free) for ShadowMountPlus.\n"
        "4. Build outputs, logs, and cached SM86 specs are stored in:\n"
        "   ~/Library/Application Support/Mario Kart World Pit Stop/\n",
        encoding="utf-8",
    )
    dmg_path = DMG_OUT / f"Mario-Kart-World-Pit-Stop-{version}-macOS-Universal.dmg"
    dmg_path.unlink(missing_ok=True)
    subprocess.run([
        "hdiutil", "create",
        "-volname", "Mario Kart World Pit Stop",
        "-srcfolder", str(dmg_stage),
        "-ov", "-format", "UDZO",
        str(dmg_path),
    ], check=True)
    subprocess.run(["hdiutil", "verify", str(dmg_path)], check=True)
    shutil.rmtree(dmg_stage, ignore_errors=True)
    return dmg_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="0.3.0")
    ap.add_argument("--stage-only", action="store_true", help="Only stage setup/ toolchain and run dual-arch checks")
    ap.add_argument("--bundle", help="Use an already extracted ps5-bundle directory instead of downloading")
    ap.add_argument("--vulkan", help="Use an already extracted PS5_Vulkan source tree instead of downloading")
    args = ap.parse_args()

    app_dir = APP_OUT / "Mario Kart World Pit Stop.app"
    setup_dir = app_dir / "Contents/Resources/setup"
    stage_setup(setup_dir, args.version, args.bundle, args.vulkan)
    verify_staged_tools(setup_dir)
    if args.stage_only:
        return 0

    compile_swift_app(app_dir, args.version)
    codesign_app(app_dir)
    dmg_path = package_dmg(app_dir, args.version)
    print(f"Created Universal DMG: {dmg_path} ({dmg_path.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
