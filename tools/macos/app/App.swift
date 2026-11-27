import SwiftUI

@main
struct MKWPitStopApp: App {
    @StateObject private var vm = BuilderViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(vm: vm)
                .frame(minWidth: 920, minHeight: 740)
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}
