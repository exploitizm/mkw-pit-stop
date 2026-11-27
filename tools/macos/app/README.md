# tools/macos/app

Native Universal macOS (`arm64` + `x86_64`) SwiftUI GUI for building and deploying *Mario Kart World: Pit Stop* (`PPSA99193`) to PS5.

| File | Purpose |
| --- | --- |
| `App.swift` | SwiftUI `@main` application entry point and window configuration |
| `ContentView.swift` | Main 3-step builder UI (dump validation, LAN/USB/local destination picker, 9-step pipeline tracker, live log console, setup guide sheet) |
| `ProcessRunner.swift` | `BuilderViewModel` managing bundled Python/glslc/zig/blink invocation, `console_probe.py` LAN scanning, `diskutil` USB filesystem inspection, `setup.log` tailing, and process-group cancellation |
| `Models.swift` | Data models for build steps, target modes, discovered consoles, USB volumes, and validation states |
| `Theme.swift` | Dark motorsport theme palette, cards, telemetry pills, and button styles |
