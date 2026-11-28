"""Plugins linked into the runtime (runtime/plugin/plugin.c, --plugin DIR of build.py and tools/ps5/build_game.py): the
ninja rules that build each folder into one object, and the table defaults.c gives plugin.c.

Why: a plugin is written as a library (mkw_plugin.h: mods/<mod>/plugin/*.so|dll), but the PS5 loads no libraries, so
there it has to be linked into the executable. A library keeps its own names to itself; linked in, a plugin's globals
(its plat_*, log_*, config_*...) would meet the runtime's and every other plugin's. So each folder is built the way a
library would be and then sealed:
  1. every .c and .cpp of the folder (or the ones its plugin.list names) compiled with the target's compiler, -fvisibility=hidden and -I runtime/plugin
     (the plugin's own headers first, as in its library build; none of the runtime's), MKW_PLUGIN_LINKED defined;
  2. the objects partially linked into one (ld.lld -r): references between the plugin's files are resolved there;
  3. every global it defines made local except mkw_plugin_init (llvm-objcopy --keep-global-symbol), which is then
     renamed mkw_plugin_init__<name>, the name the table in defaults.c lists.
What stays undefined (the C library, pthreads, sockets) is the executable's to resolve. Used by tools/build/build.py
and tools/ps5/build_game.py; not run on its own.
"""

import re
import shlex
from pathlib import Path

RUNTIME_PLUGIN = Path(__file__).resolve().parents[2] / "runtime/plugin"


LIST = "plugin.list"


def list_flags(folder: Path, value: str) -> list[str]:
    """A cflags= / cxxflags= value as arguments, its -I folders made absolute from the list's folder."""
    out = []
    for arg in shlex.split(value):
        if arg.startswith("-I") and len(arg) > 2 and not Path(arg[2:]).is_absolute():
            arg = "-I" + str((folder / arg[2:]).resolve())
        out.append(arg)
    return out


def read_list(folder: Path) -> tuple[str | None, list[Path], list[str], list[str]] | None:
    """A folder's plugin.list, if it has one: `name=NAME` (the mod it stands for), `cflags=` and `cxxflags=` (the
    plugin's own flags for its C and C++, after the glue's; -I relative to the folder), and one source per line,
    relative to the folder ('#' starts a comment, so a flag cannot hold one). For a plugin whose sources sit beside
    others it must not build (another platform's), or whose folder is not named after its mod. None without the file."""
    path = folder / LIST
    if not path.is_file():
        return None
    name, srcs, cflags, cxxflags = None, [], [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("name="):
            name = line[5:].strip()
            continue
        if line.startswith("cflags="):
            cflags += list_flags(folder, line[7:])
            continue
        if line.startswith("cxxflags="):
            cxxflags += list_flags(folder, line[9:])
            continue
        src = (folder / line).resolve()
        if src.suffix not in (".c", ".cpp") or not src.is_file():
            raise SystemExit(f"{path}: {line}: not a .c or .cpp file")
        srcs.append(src)
    return name, srcs, cflags, cxxflags


def plugin_name(folder: Path) -> str:
    """A C identifier: plugin.list's name, else the folder's (the mod the plugin stands for: mods/<name>)."""
    listed = read_list(folder.resolve())
    name = listed[0] if listed and listed[0] else folder.resolve().name
    return re.sub(r"\W", "_", name)


def plugin_sources(folders: list[Path]) -> list[tuple[str, Path, list[Path], list[str], list[str]]]:
    """(name, folder, sources, cflags, cxxflags) of each folder: plugin.list's, else every .c and .cpp under it."""
    out, names = [], set()
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"--plugin {folder}: not a folder")
        name = plugin_name(folder)
        if name in names:
            raise SystemExit(f"--plugin {folder}: another plugin folder has the name {name}")
        names.add(name)
        listed = read_list(folder.resolve())
        srcs = listed[1] if listed else sorted(p for pattern in ("*.c", "*.cpp") for p in folder.resolve().rglob(pattern))
        if not srcs:
            raise SystemExit(f"--plugin {folder}: no .c or .cpp files")
        out.append((name, folder.resolve(), srcs, listed[2] if listed else [], listed[3] if listed else []))
    return out


def plugin_table(folders: list[Path]) -> str:
    """The C lines of the table plugin.c reads (rt_plugins_linked), an empty table when no folder is given."""
    names = [plugin_name(f) for f in folders]
    lines = ['#include "plugin.h"']
    lines += [f"int mkw_plugin_init__{n}(const MkwPluginApi *api, uint32_t api_version, const char **message);"
              for n in names]  # fmt: skip
    rows = ", ".join(f'{{"{n}", mkw_plugin_init__{n}}}' for n in names) or '{0, 0}'
    lines += [f"const RtLinkedPlugin rt_plugins_linked[] = {{{rows}}};",
              f"const int rt_plugins_linked_count = {len(names)};", ""]  # fmt: skip
    return "\n".join(lines)


def plugin_ninja(folders: list[Path], out: Path, cc: str, cxx: str, flags: list[str], ld: str,
                 objcopy: str) -> tuple[list[str], list[str]]:  # fmt: skip
    """Ninja lines (rules and builds) and the sealed object of each folder, under out/plugin/<name>/."""
    if not folders:
        return [], []
    plugflags = " ".join([*flags, "-fvisibility=hidden", "-DMKW_PLUGIN_LINKED=1", f"-I{RUNTIME_PLUGIN}"])
    lines = [
        f"plugflags = {plugflags}",
        (f"rule plugcc\n  command = {cc} -std=gnu11 $plugflags $plugc -I$plugdir -MD -MF $out.d -c $in -o $out"
         "\n  depfile = $out.d\n  deps = gcc\n  description = plugin $in"),
        (f"rule plugcxx\n  command = {cxx} -std=c++17 $plugflags $plugcxx -I$plugdir -MD -MF $out.d -c $in -o $out"
         "\n  depfile = $out.d\n  deps = gcc\n  description = plugin $in"),
        # step 2 and 3 of the module comment; the intermediate keeps a failed step from leaving a half-sealed object
        (f"rule plugseal\n  command = {ld} -r -o $out.part $in"
         f" && {objcopy} --keep-global-symbol=mkw_plugin_init $out.part"
         f" && {objcopy} --redefine-sym=mkw_plugin_init=mkw_plugin_init__$name $out.part $out && rm -f $out.part"
         "\n  description = seal plugin $name"),
    ]  # fmt: skip
    sealed = []
    for name, folder, srcs, cflags, cxxflags in plugin_sources(folders):
        base = out / "plugin" / name
        objs = []
        for src in srcs:
            rel = src.relative_to(folder) if src.is_relative_to(folder) else Path(src.name)
            obj = base / rel.with_suffix(".o")
            objs.append(str(obj))
            rule = "plugcxx" if src.suffix == ".cpp" else "plugcc"
            own = shlex.join(cxxflags if rule == "plugcxx" else cflags).replace("$", "$$")
            var = "plugcxx" if rule == "plugcxx" else "plugc"   # the rule's own-flags variable
            lines.append(f"build {obj}: {rule} {src}\n  plugdir = {src.parent}\n  {var} = {own}")
        obj = base.with_suffix(".o")
        lines.append(f"build {obj}: plugseal {' '.join(objs)}\n  name = {name}")
        sealed.append(str(obj))
    return lines, sealed
