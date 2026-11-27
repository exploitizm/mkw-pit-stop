"""What the slow frames of a race are made of: the frames that set the 1 % and 0.1 % lows, sorted by cause.

Why: the PS5 benches of 28/11/2026 ran at 59 fps on average with a 1 % low of 40 fps (MKW_BENCH's lows are the frame
time 99 % and 99.9 % of frames stay under, so ~90 frames of a 150 s race at 25 ms or more). The 60-frame perf lines
only give window totals, and the 25/11 hitch hunt's scripts were lost with the scratchpad. This reads the per-frame
cost ledger on each "perf: slow frame" line (runtime/core/diag/perf.c ledger_take) against the 60-frame average of the
same costs ("perf: frame N: costs ms/f"), and names for each slow frame the cost that grew the most, down to the most
specific one (precheck > upload > astc). Logs from before the ledger still get the frame-time bands, the render
thread's blocked time and, with MKW_FRAME_TRACE=1, whether the frame ended as the GPU finished or with the GPU idle.

The run it needs: MKW_BENCH (the race window and its lows), MKW_TRACE=perf, MKW_PERF_SLOW_MS at or under the 1 % low's
frame time (20 for a 40-50 fps low), and MKW_FRAME_TRACE=1 for the full frame-time distribution.

Usage: uv run tools/run/slowframes.py build/run/ps5/NAME-mkw.log [--frames] [--target-fps 50] [--clusters 8]
"""

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TICK = 1000 / 60  # ms of a 60 fps frame
SLOW = re.compile(r"perf: slow frame (\d+): ([\d.]+) ms; blocked (.*?)(?:; costs ms:(.*))?$")
WINDOW = re.compile(r"perf: frame (\d+): costs ms/f:(.*)$")
TRACE = re.compile(r"frame: period ([\d.]+) call [\d.]+ meas [\d.]+ cpu ([\d.]+) ts (-?[\d.]+)")
RESULT = re.compile(
    r"bench: result mode=(\w+) avg_ms=([\d.]+) fps=([\d.]+) low1=([\d.]+) low01=([\d.]+) worst_ms=([\d.]+) frames=(\d+)"
)
START = ("bench: window starts", "race: in a race")
END = ("bench: result", "race: out of the race", "bench: no finish seen", "bench: Grand Prix finished")
# A cost's more specific parts (the ledger's names, perf.c): the game waiting for the replay is explained by what the
# replay did; the precheck pass holds uploads and texture-lock waits; an upload holds its ASTC decode.
CHILDREN = {
    "sync_replay": ["replay"],
    "replay": ["precheck", "texlock", "upload", "astc", "sparse", "gov"],
    "precheck": ["upload", "texlock"],
    "upload": ["astc"],
}
BLOCKED = ("pipeline", "texture", "astc", "fence")  # render thread's own waits on the slow-frame line (pacing is idle)
GPU_TAIL_MS = -2.0  # FRAME_TRACE ts above this: the frame ended within 2 ms of the GPU finishing its last work


def pairs(text: str | None) -> dict[str, float]:
    """' name 1.2 other 3.4' -> {name: 1.2, other: 3.4}"""
    toks = (text or "").split()
    return {toks[i]: float(toks[i + 1]) for i in range(0, len(toks) - 1, 2)}


def race_lines(lines: list[str]) -> list[str]:
    a = next((i for i, ln in enumerate(lines) if any(s in ln for s in START)), 0)
    b = next((i for i in range(a + 1, len(lines)) if any(s in lines[i] for s in END)), len(lines))
    return lines[a:b]


def classify(
    costs: dict[str, float], base: dict[str, float], blocked: dict[str, float], excess: float, ts: float | None
):
    """The cause of one slow frame: the cost that grew the most over its normal, followed down to the part that holds
    at least half of that growth. Under a quarter of the frame's excess explained: the game's own CPU or the GPU."""
    extra = {k: v - base.get(k, 0.0) for k, v in costs.items()}
    extra.update({f"render {k}": v for k, v in blocked.items() if k in BLOCKED})
    if extra:
        name = max(extra, key=lambda k: extra[k])
        path = [name]
        while kids := [k for k in CHILDREN.get(name, []) if extra.get(k, 0) >= 0.5 * extra[name]]:
            name = max(kids, key=lambda k: extra[k])
            path.append(name)
        if extra[path[0]] >= 0.25 * excess:
            return " > ".join(path), extra[path[-1]]
    if ts is None:
        return "unexplained", 0.0
    return ("unexplained, GPU tail" if ts > GPU_TAIL_MS else "unexplained, game CPU"), 0.0


def pct(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--frames", action="store_true", help="list every frame of the 1 %% set with its cause")
    ap.add_argument("--target-fps", type=float, default=50.0, help="1 %% low to aim at (frames to remove)")
    ap.add_argument("--clusters", type=int, default=8, help="longest runs of slow frames to show")
    args = ap.parse_args()
    lines = race_lines(args.log.read_text(errors="replace").splitlines())

    result = next((m for ln in lines if (m := RESULT.search(ln))), None)
    if result is None:  # the result line closes the window: look past it
        tail = args.log.read_text(errors="replace").splitlines()
        result = next((m for ln in tail if (m := RESULT.search(ln))), None)
    periods, slow, windows = [], [], []
    last_ts = None
    for ln in lines:
        if m := TRACE.search(ln):
            periods.append(float(m[1]))
            last_ts = float(m[3])
        elif m := SLOW.search(ln):
            slow.append((int(m[1]), float(m[2]), pairs(m[3]), pairs(m[4]), last_ts))
        elif m := WINDOW.search(ln):
            windows.append(pairs(m[2]))
    if not slow:
        sys.exit("no 'perf: slow frame' lines in the race: run with MKW_TRACE=perf and MKW_PERF_SLOW_MS=20")

    # The lows' frame times: MKW_BENCH's own, else the trace's distribution, else the slow lines alone
    n_frames = int(result[7]) if result else len(periods)
    if result:
        p99, p999 = 1000 / float(result[4]), 1000 / float(result[5])
        print(
            f"bench: {result[1]} avg {result[2]} ms ({result[3]} fps), 1% low {result[4]} fps ({p99:.1f} ms), "
            f"0.1% low {result[5]} fps ({p999:.1f} ms), worst {result[6]} ms, {n_frames} frames"
        )
    elif periods:
        p99, p999 = pct(periods, 0.99), pct(periods, 0.999)
        print(f"frame trace: {len(periods)} frames, p99 {p99:.1f} ms, p99.9 {p999:.1f} ms (no bench result line)")
    else:
        p99 = p999 = 0.0
        print("no bench result and no frame trace: the lows' frame times are unknown")
    slow_min = min(ms for _, ms, *_ in slow)
    if p99 and slow_min > p99:
        print(
            f"WARNING: MKW_PERF_SLOW_MS ({slow_min:.0f}+ ms here) is above the 1% low's {p99:.1f} ms: part of the 1% "
            "set has no line"
        )

    if periods:
        print("frame-time bands (frame trace): frames, time over 16.7 ms")
        for lo, hi in ((0, 17.5), (17.5, 20), (20, 25), (25, 33.4), (33.4, 50), (50, 1e9)):
            xs = [p for p in periods if lo <= p < hi]
            lost = sum(p - TICK for p in xs if p > TICK) / 1000
            print(f"  {lo:5.1f}-{'' if hi > 1e8 else f'{hi:.1f}':<5} ms {len(xs):6} {lost:6.2f} s")
    if n_frames and args.target_fps:
        t = 1000 / args.target_fps
        over = sum(1 for p in periods if p > t) if periods else sum(1 for _, ms, *_ in slow if ms > t)
        allowed = n_frames // 100
        print(
            f"for a 1% low of {args.target_fps:.0f} fps ({t:.1f} ms) at most {allowed} frames may be over {t:.1f} ms: "
            f"now {over}{' (slow lines only)' if not periods else ''}, {max(0, over - allowed)} to remove"
        )

    has_ledger = any(c for *_, c, _ in slow)
    base = {k: statistics.median(w.get(k, 0.0) for w in windows) for k in {k for w in windows for k in w}}
    if not has_ledger:
        print(
            "no cost ledger on the slow-frame lines (runtime before perf.c ledger_take): causes from the render "
            "thread's blocked time and the frame trace only"
        )
    elif base:
        print(
            "normal frame (median of the 60-frame costs ms/f): "
            + ", ".join(f"{k} {v:.2f}" for k, v in sorted(base.items(), key=lambda kv: -kv[1]))
        )

    rows = []
    for frame, ms, blocked, costs, ts in slow:
        cause, ms_cause = classify(costs, base, blocked, ms - TICK, ts)
        rows.append((frame, ms, cause, ms_cause, costs, blocked, ts))

    def table(label: str, sel: list[tuple]) -> None:
        if not sel:
            return
        lost = sum(r[1] - TICK for r in sel)
        print(f"\n{label}: {len(sel)} frames, {lost / 1000:.2f} s over 16.7 ms")
        by = defaultdict(list)
        for r in sel:
            by[r[2]].append(r)
        for cause, rs in sorted(by.items(), key=lambda kv: -sum(r[1] - TICK for r in kv[1])):
            l2 = sum(r[1] - TICK for r in rs)
            extra = statistics.mean(r[3] for r in rs)
            print(
                f"  {len(rs):5} {100 * l2 / lost:5.1f}%  avg {statistics.mean(r[1] for r in rs):5.1f} ms"
                f"{f', +{extra:.1f} ms' if extra else '':<14} {cause}"
            )

    in1 = [r for r in rows if not p99 or r[1] >= p99]
    table(f"1% set (frames at {p99:.1f} ms or more)" if p99 else "slow frames", in1)
    if p999:
        table(f"0.1% set (frames at {p999:.1f} ms or more)", [r for r in rows if r[1] >= p999])

    # Runs of slow frames at most 3 frames apart: a hitch (one cause, many frames) shows as one long run
    runs, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][0] <= 3:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    runs = [r for r in runs if len(r) > 1]
    if runs and args.clusters:
        print(f"\nruns of slow frames (at most 3 apart), {len(runs)} in all, longest by time:")
        for run in sorted(runs, key=lambda rs: -sum(r[1] - TICK for r in rs))[: args.clusters]:
            top = Counter(r[2] for r in run).most_common(1)[0][0]
            print(
                f"  frames {run[0][0]}-{run[-1][0]}: {len(run)} slow, {sum(r[1] - TICK for r in run):6.0f} ms over, "
                f"worst {max(r[1] for r in run):.1f} ms, mostly {top}"
            )

    if args.frames:
        print("\nframe      ms  cause (its growth over normal), costs, render blocked, ts")
        for frame, ms, cause, mc, costs, blocked, ts in in1:
            c = " ".join(f"{k} {v:g}" for k, v in sorted(costs.items(), key=lambda kv: -kv[1]))
            b = " ".join(f"{k} {v:g}" for k, v in blocked.items() if v)
            print(f"{frame:7} {ms:6.1f}  {cause}{f' (+{mc:.1f})' if mc else ''} | {c} | {b} | ts {ts}")


if __name__ == "__main__":
    main()
