# tools/macos

Universal macOS (`arm64` + `x86_64`) packaging of the PS5 builder. The builder logic itself stays in `tools/ps5/prepare_ps5.py`; this folder only wraps it in a self-contained `.app` and `.dmg`.

| Path | Purpose |
| --- | --- |
| `dmg/` | Self-bootstrapping build of the Universal `.app` and `.dmg` (see `dmg/README.md`) |
| `app/` | Native SwiftUI front end (see `app/README.md`) that drives `prepare_ps5.py` and `console_probe.py` |

```sh
uv run tools/macos/dmg/build_macos_dmg.py            # full .dmg in build/release/macos/
uv run tools/macos/dmg/build_macos_dmg.py --stage-only
```
