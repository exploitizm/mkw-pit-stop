import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var vm: BuilderViewModel
    @State private var isDropTargeted = false
    @State private var showAdvanced = false

    var body: some View {
        ZStack {
            PitTheme.bg.ignoresSafeArea()

            VStack(spacing: 0) {
                headerBar
                Divider().background(PitTheme.border)

                ScrollView {
                    VStack(spacing: 16) {
                        dumpCard
                        destinationCard
                        advancedCard
                        progressCard
                    }
                    .padding(20)
                }
            }
        }
        .sheet(isPresented: $vm.showGuide) {
            GuideSheet()
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Header
    private var headerBar: some View {
        HStack(spacing: 14) {
            if let img = vm.logoImage {
                Image(nsImage: img)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: 42, height: 42)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Color.white.opacity(0.15), lineWidth: 1)
                    )
            }
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text("Mario Kart World: Pit Stop")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundColor(PitTheme.textPrimary)
                    TelemetryPill(text: "PS5 • PPSA99193", color: PitTheme.amber)
                }
                Text("Universal macOS Builder • v0.3.0 • forked by VenezuelanBiggie24")
                    .font(.system(size: 12))
                    .foregroundColor(PitTheme.textSecondary)
            }

            Spacer()

            TelemetryPill(text: vm.hostArchLabel, color: PitTheme.cyan, icon: "cpu")

            Button {
                vm.openDataFolder()
            } label: {
                Label("Data", systemImage: "folder")
            }
            .buttonStyle(SecondaryButtonStyle())

            Button {
                vm.showGuide = true
            } label: {
                Label("PS5 Setup Guide", systemImage: "questionmark.circle")
            }
            .buttonStyle(SecondaryButtonStyle())
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(PitTheme.card.opacity(0.85))
    }

    // MARK: - Step 1: Game Dump
    private var dumpCard: some View {
        PitCard(
            step: 1,
            title: "Switch 2 Game Dump",
            subtitle: "Decrypted exefs/ + romfs/ folder (or updates/1.8.0) or .nup package",
            trailing: { dumpBadge }
        ) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    TextField("Drop dump folder or .nup here, or browse...", text: $vm.sourcePath)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12.5, design: .monospaced))
                        .padding(10)
                        .background(PitTheme.cardElevated)
                        .overlay(
                            RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .stroke(isDropTargeted ? PitTheme.amber : PitTheme.border, lineWidth: isDropTargeted ? 2 : 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                        .onSubmit { vm.validateDump() }

                    Button("Choose Folder…") { pickDump(folders: true) }
                        .buttonStyle(SecondaryButtonStyle())
                        .disabled(vm.isBuilding)

                    Button("Choose .nup…") { pickDump(folders: false) }
                        .buttonStyle(SecondaryButtonStyle())
                        .disabled(vm.isBuilding)
                }
                .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
                    handleDrop(providers: providers)
                }

                if case .valid(_, let detail) = vm.dumpValidation, !detail.isEmpty {
                    Text(detail)
                        .font(.system(size: 11.5))
                        .foregroundColor(PitTheme.emerald)
                } else if case .invalid(let err) = vm.dumpValidation {
                    Text(err)
                        .font(.system(size: 11.5))
                        .foregroundColor(PitTheme.crimson)
                }
            }
        }
    }

    @ViewBuilder
    private var dumpBadge: some View {
        switch vm.dumpValidation {
        case .empty:
            TelemetryPill(text: "Waiting for dump", color: PitTheme.textSecondary)
        case .checking:
            TelemetryPill(text: "Checking…", color: PitTheme.cyan)
        case .valid(let version, _):
            TelemetryPill(text: "MKW \(version)", color: PitTheme.emerald, icon: "checkmark.circle.fill")
        case .invalid:
            TelemetryPill(text: "Invalid dump", color: PitTheme.crimson, icon: "exclamationmark.triangle.fill")
        }
    }

    // MARK: - Step 2: Destination
    private var destinationCard: some View {
        PitCard(
            step: 2,
            title: "Destination",
            subtitle: "Send directly to your PS5 over Wi-Fi/LAN, write to an exFAT USB drive, or stage locally"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                Picker("Destination", selection: $vm.targetMode) {
                    ForEach(TargetMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.icon).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(vm.isBuilding)

                switch vm.targetMode {
                case .ps5Network:
                    ps5NetworkSection
                case .usbDrive:
                    usbDriveSection
                case .localFolder:
                    localFolderSection
                }
            }
        }
    }

    private var ps5NetworkSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                TextField("PS5 IP address (e.g. 192.168.1.50)", text: $vm.ps5IP)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13, design: .monospaced))
                    .padding(10)
                    .background(PitTheme.cardElevated)
                    .overlay(RoundedRectangle(cornerRadius: 9).stroke(PitTheme.border, lineWidth: 1))
                    .clipShape(RoundedRectangle(cornerRadius: 9))
                    .disabled(vm.isBuilding)

                Button {
                    vm.scanLAN()
                } label: {
                    HStack(spacing: 6) {
                        if vm.isScanningLAN {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "dot.radiowaves.left.and.right")
                        }
                        Text(vm.isScanningLAN ? "Scanning…" : "Scan LAN")
                    }
                }
                .buttonStyle(SecondaryButtonStyle())
                .disabled(vm.isBuilding || vm.isScanningLAN)

                Button {
                    vm.checkConsole()
                } label: {
                    HStack(spacing: 6) {
                        if vm.isCheckingConsole {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "antenna.radiowaves.left.and.right")
                        }
                        Text("Check PS5")
                    }
                }
                .buttonStyle(SecondaryButtonStyle())
                .disabled(vm.isBuilding || vm.isCheckingConsole || vm.ps5IP.isEmpty)
            }

            if !vm.foundConsoles.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(vm.foundConsoles) { console in
                            let selected = vm.ps5IP == console.ip
                            Button {
                                vm.ps5IP = console.ip
                                vm.checkConsole()
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 6) {
                                        Text(console.ip)
                                            .font(.system(size: 12.5, weight: .bold, design: .monospaced))
                                            .foregroundColor(PitTheme.textPrimary)
                                        TelemetryPill(
                                            text: console.kind == "receiver" ? "PIT1" : "FTP",
                                            color: console.kind == "receiver" ? PitTheme.emerald : PitTheme.amber
                                        )
                                    }
                                    Text(console.summary)
                                        .font(.system(size: 11))
                                        .foregroundColor(PitTheme.textSecondary)
                                        .lineLimit(1)
                                }
                                .padding(10)
                                .background(selected ? PitTheme.amber.opacity(0.14) : PitTheme.cardElevated)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(selected ? PitTheme.amber : PitTheme.border, lineWidth: 1)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            if let chk = vm.consoleCheck, !chk.line.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: chk.kind == "none" ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .foregroundColor(chk.kind == "none" ? PitTheme.amber : PitTheme.emerald)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(chk.line)
                            .font(.system(size: 12))
                            .foregroundColor(PitTheme.textPrimary)
                        if !chk.games.isEmpty {
                            Text("On console: " + chk.games.joined(separator: ", "))
                                .font(.system(size: 11))
                                .foregroundColor(PitTheme.textSecondary)
                        }
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(PitTheme.cardElevated)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            VStack(alignment: .leading, spacing: 8) {
                Toggle("My game files are already on the PS5", isOn: $vm.useConsoleGame)
                    .toggleStyle(.switch)
                    .font(.system(size: 12.5))
                    .disabled(vm.isBuilding)
                if vm.useConsoleGame {
                    HStack {
                        Text("PS5 path:")
                            .font(.system(size: 12))
                            .foregroundColor(PitTheme.textSecondary)
                        TextField("/data/mkw", text: $vm.consoleGamePath)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12.5, design: .monospaced))
                            .padding(8)
                            .background(PitTheme.cardElevated)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(PitTheme.border, lineWidth: 1))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
    }

    private var usbDriveSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("External drives in /Volumes (exFAT recommended for 23 GB romfs.nup)")
                    .font(.system(size: 12))
                    .foregroundColor(PitTheme.textSecondary)
                Spacer()
                Button {
                    vm.refreshUSBVolumes()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .buttonStyle(SecondaryButtonStyle())

                Button("Other Folder…") {
                    pickFolder { vm.customUSBPath = $0 }
                }
                .buttonStyle(SecondaryButtonStyle())
            }

            if vm.usbVolumes.isEmpty && vm.customUSBPath.isEmpty {
                Text("No external USB drive detected. Plug in an exFAT drive (25+ GB free) or choose a folder.")
                    .font(.system(size: 12))
                    .foregroundColor(PitTheme.amber)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(PitTheme.cardElevated)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                VStack(spacing: 8) {
                    ForEach(vm.usbVolumes) { vol in
                        let selected = vm.effectiveUSBPath == vol.path
                        Button {
                            vm.customUSBPath = ""
                            vm.selectedUSBPath = vol.path
                        } label: {
                            HStack {
                                Image(systemName: "externaldrive.fill")
                                    .foregroundColor(vol.isSupported ? PitTheme.cyan : PitTheme.crimson)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(vol.name)
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundColor(PitTheme.textPrimary)
                                    Text("\(vol.path) • \(vol.freeText) free")
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(PitTheme.textSecondary)
                                }
                                Spacer()
                                TelemetryPill(
                                    text: vol.filesystem,
                                    color: vol.isExFAT ? PitTheme.emerald : (vol.isFAT32 ? PitTheme.amber : PitTheme.crimson)
                                )
                            }
                            .padding(10)
                            .background(selected ? PitTheme.cyan.opacity(0.12) : PitTheme.cardElevated)
                            .overlay(
                                RoundedRectangle(cornerRadius: 9)
                                    .stroke(selected ? PitTheme.cyan : PitTheme.border, lineWidth: 1)
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 9))
                        }
                        .buttonStyle(.plain)
                    }
                    if !vm.customUSBPath.isEmpty {
                        HStack {
                            Text("Custom target: \(vm.customUSBPath)/homebrew/PPSA99193")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(PitTheme.cyan)
                            Spacer()
                            Button("Clear") { vm.customUSBPath = "" }
                                .buttonStyle(SecondaryButtonStyle())
                        }
                        .padding(8)
                        .background(PitTheme.cardElevated)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
    }

    private var localFolderSection: some View {
        HStack(spacing: 10) {
            Text(vm.effectiveOutDir.path)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(PitTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(PitTheme.cardElevated)
                .overlay(RoundedRectangle(cornerRadius: 9).stroke(PitTheme.border, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 9))

            Button("Choose Output Folder…") {
                pickFolder { vm.localOutPath = $0 }
            }
            .buttonStyle(SecondaryButtonStyle())
            .disabled(vm.isBuilding)
        }
    }

    // MARK: - Advanced Options
    private var advancedCard: some View {
        DisclosureGroup(isExpanded: $showAdvanced) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("RomFS packaging:")
                        .font(.system(size: 12.5))
                        .foregroundColor(PitTheme.textSecondary)
                        .frame(width: 130, alignment: .leading)
                    Picker("", selection: $vm.romfsMode) {
                        ForEach(RomFSMode.allCases) { m in
                            Text(m.label).tag(m)
                        }
                    }
                    .labelsHidden()
                    .disabled(vm.isBuilding)
                }

                HStack {
                    Text("Parallel jobs:")
                        .font(.system(size: 12.5))
                        .foregroundColor(PitTheme.textSecondary)
                        .frame(width: 130, alignment: .leading)
                    Stepper(
                        vm.jobsCount == 0 ? "Auto (recommended)" : "\(vm.jobsCount) threads",
                        value: $vm.jobsCount,
                        in: 0...max(4, ProcessInfo.processInfo.processorCount)
                    )
                    .disabled(vm.isBuilding)
                }

                HStack(spacing: 10) {
                    Button("Open setup.log") { vm.openLogFile() }
                        .buttonStyle(SecondaryButtonStyle())
                    Button("Reveal Staged PPSA99193") { vm.revealStagedOutput() }
                        .buttonStyle(SecondaryButtonStyle())
                }
            }
            .padding(.top, 10)
        } label: {
            Text("Advanced Options")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(PitTheme.textSecondary)
        }
        .padding(14)
        .background(PitTheme.card)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(PitTheme.border, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Step 3: Build & Progress
    private var progressCard: some View {
        PitCard(
            step: 3,
            title: "Build & Transfer",
            subtitle: "Lifts AArch64 ExeFS, links eboot.bin for PS5, translates 11,561 SM86 shaders, and deploys",
            trailing: {
                if vm.buildFinishedOK {
                    TelemetryPill(text: "Complete", color: PitTheme.emerald, icon: "checkmark.seal.fill")
                } else if vm.isBuilding {
                    TelemetryPill(text: "Building", color: PitTheme.amber, icon: "gearshape.2.fill")
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 14) {
                // 9-step pipeline pills
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 9), spacing: 8) {
                    ForEach(vm.steps) { step in
                        stepTile(step)
                    }
                }

                if vm.isBuilding || vm.progressTotal > 0 {
                    VStack(alignment: .leading, spacing: 6) {
                        if vm.progressTotal > 0 {
                            ProgressView(value: min(vm.progressValue, vm.progressTotal), total: vm.progressTotal)
                                .tint(PitTheme.amber)
                        } else if vm.isBuilding {
                            ProgressView()
                                .progressViewStyle(.linear)
                                .tint(PitTheme.amber)
                        }
                        HStack {
                            Text(vm.currentNote.isEmpty ? "Working…" : vm.currentNote)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(PitTheme.textPrimary)
                            Spacer()
                            if vm.progressTotal > 0 {
                                Text("\(Int(vm.progressValue)) / \(Int(vm.progressTotal))")
                                    .font(.system(size: 11.5, design: .monospaced))
                                    .foregroundColor(PitTheme.textSecondary)
                            }
                        }
                    }
                    .padding(10)
                    .background(PitTheme.cardElevated)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                // Info banners from prepare_ps5.py
                ForEach(["dump", "console", "game", "plan"], id: \.self) { key in
                    if let text = vm.infoBanners[key] {
                        HStack(spacing: 8) {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(PitTheme.cyan)
                            Text(text)
                                .font(.system(size: 12))
                                .foregroundColor(PitTheme.textPrimary)
                        }
                    }
                }

                if let err = vm.buildError {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "xmark.octagon.fill")
                            .foregroundColor(PitTheme.crimson)
                        Text(err)
                            .font(.system(size: 12.5))
                            .foregroundColor(PitTheme.crimson)
                            .textSelection(.enabled)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(PitTheme.crimson.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                HStack(spacing: 12) {
                    if vm.isBuilding {
                        Button("Cancel Build") { vm.cancelBuild() }
                            .buttonStyle(SecondaryButtonStyle())
                    } else {
                        Button {
                            vm.startBuild()
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: ctaIcon)
                                Text(ctaTitle)
                            }
                        }
                        .buttonStyle(RacingPrimaryButtonStyle(disabled: !vm.canStartBuild))
                        .disabled(!vm.canStartBuild)
                    }

                    Spacer()

                    Button {
                        withAnimation { vm.showLogs.toggle() }
                    } label: {
                        Label(vm.showLogs ? "Hide Live Log" : "Show Live Log", systemImage: "terminal")
                    }
                    .buttonStyle(SecondaryButtonStyle())
                }

                if vm.showLogs {
                    logConsole
                }
            }
        }
    }

    private func stepTile(_ step: BuildStepItem) -> some View {
        let color: Color = {
            switch step.status {
            case .pending: return PitTheme.textSecondary.opacity(0.5)
            case .running: return PitTheme.amber
            case .done: return PitTheme.emerald
            case .skipped: return PitTheme.textSecondary
            case .failed: return PitTheme.crimson
            }
        }()

        return VStack(spacing: 5) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.16))
                    .frame(width: 28, height: 28)
                if step.status == .running {
                    ProgressView().controlSize(.small)
                } else if step.status == .done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(color)
                } else if step.status == .skipped {
                    Image(systemName: "minus")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(color)
                } else if step.status == .failed {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(color)
                } else {
                    Image(systemName: step.icon)
                        .font(.system(size: 11))
                        .foregroundColor(color)
                }
            }
            Text(step.title)
                .font(.system(size: 10.5, weight: step.status == .running ? .bold : .medium))
                .foregroundColor(step.status == .pending ? PitTheme.textSecondary : PitTheme.textPrimary)
                .lineLimit(1)
        }
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity)
        .background(PitTheme.cardElevated.opacity(step.status == .running ? 1.0 : 0.6))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(step.status == .running ? PitTheme.amber.opacity(0.7) : PitTheme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var logConsole: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("setup.log")
                    .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                    .foregroundColor(PitTheme.textSecondary)
                Spacer()
                Button("Copy") { vm.copyLogsToClipboard() }
                    .buttonStyle(SecondaryButtonStyle())
                Button("Open File") { vm.openLogFile() }
                    .buttonStyle(SecondaryButtonStyle())
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(vm.logLines) { item in
                            Text(item.text)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(item.isError ? PitTheme.crimson : PitTheme.textSecondary)
                                .textSelection(.enabled)
                                .id(item.id)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                }
                .frame(height: 180)
                .background(Color.black.opacity(0.45))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(PitTheme.border, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .onChange(of: vm.logLines.count) { _ in
                    if let last = vm.logLines.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var ctaTitle: String {
        switch vm.targetMode {
        case .ps5Network: return "Build & Send to PS5"
        case .usbDrive: return "Build to USB Drive"
        case .localFolder: return "Build PPSA99193 Locally"
        }
    }

    private var ctaIcon: String {
        switch vm.targetMode {
        case .ps5Network: return "paperplane.fill"
        case .usbDrive: return "externaldrive.badge.checkmark"
        case .localFolder: return "hammer.fill"
        }
    }

    // MARK: - Pickers & Drag-Drop
    private func pickDump(folders: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = folders
        panel.canChooseFiles = !folders
        panel.allowsMultipleSelection = false
        panel.prompt = folders ? "Select Dump Folder" : "Select .nup File"
        if !folders {
            panel.allowedContentTypes = [UTType(filenameExtension: "nup") ?? .data]
        }
        if panel.runModal() == .OK, let url = panel.url {
            vm.sourcePath = url.path
            vm.validateDump()
        }
    }

    private func pickFolder(onPick: (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            onPick(url.path)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let item = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }
        item.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
            guard let urlData = data as? Data,
                  let url = URL(dataRepresentation: urlData, relativeTo: nil) else { return }
            Task { @MainActor in
                vm.sourcePath = url.path
                vm.validateDump()
            }
        }
        return true
    }
}

struct GuideSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("PS5 Setup Guide")
                    .font(.system(size: 18, weight: .bold))
                Spacer()
                Button("Close") { dismiss() }
            }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    guideStep("1. Install Placeholder on PS5",
                              "Download mkw-pit-stop-0.3.0-ps5-placeholder-PPSA99193.zip from GitHub Releases and copy PPSA99193/ to /data/homebrew/ on your jailbroken PS5 (etaHEN/elfldr + kstuff + ShadowMountPlus 1.7 Beta 5+).")
                    guideStep("2. Open the Receiver Screen",
                              "Launch Mario Kart World: Pit Stop on the PS5. It shows 'Game files missing' and your Console IP (port 19913). To update an existing install later, hold L1 + R1 while launching the title.")
                    guideStep("3. Build & Send from Mac",
                              "Choose your decrypted Switch 2 Mario Kart World dump (1.8.0 or 1.0.0) or .nup file, click Scan LAN (or enter your PS5 IP), and click Build & Send to PS5. About 30 GB of free space is required on your Mac.")
                    guideStep("USB Alternative (No Network)",
                              "Format a USB drive as exFAT (25+ GB free), select USB Drive in Step 2, build directly to the drive (homebrew/PPSA99193), and plug it into the PS5 for ShadowMountPlus.")
                }
            }
            HStack {
                Spacer()
                Link("Open GitHub Releases", destination: URL(string: "https://github.com/exploitizm/mkw-pit-stop/releases")!)
            }
        }
        .padding(20)
        .frame(width: 540, height: 400)
        .preferredColorScheme(.dark)
    }

    private func guideStep(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 13.5, weight: .semibold))
                .foregroundColor(PitTheme.amber)
            Text(body)
                .font(.system(size: 12.5))
                .foregroundColor(PitTheme.textPrimary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(PitTheme.card)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
