# tools/macos/dmg

Builds the Universal macOS PS5 builder `.app` and `.dmg`.

| File | Purpose |
| --- | --- |
| `build_macos_dmg.py` | Bootstraps SHA-256-pinned prerequisites (`--bundle DIR` / `--vulkan DIR` reuse local copies), stages dual-arch Python 3.14, `glslc`, `zig`, `blink` with a minimal glibc sysroot and `ps5-native-tool`, compiles the SwiftUI app from `../app`, signs it and packs the `.dmg` into `build/release/macos/` |
