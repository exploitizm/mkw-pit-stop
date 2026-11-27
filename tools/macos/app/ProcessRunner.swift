import AppKit
import Combine
import Darwin
import Foundation

@MainActor
final class BuilderViewModel: ObservableObject {
    // Persisted user choices
    @Published var sourcePath: String = UserDefaults.standard.string(forKey: "sourcePath") ?? "" {
        didSet { UserDefaults.standard.set(sourcePath, forKey: "sourcePath") }
    }
    @Published var ps5IP: String = UserDefaults.standard.string(forKey: "ps5IP") ?? "" {
        didSet { UserDefaults.standard.set(ps5IP, forKey: "ps5IP") }
    }
    @Published var targetMode: TargetMode = TargetMode(
        rawValue: UserDefaults.standard.string(forKey: "targetMode") ?? "ps5"
    ) ?? .ps5Network {
        didSet { UserDefaults.standard.set(targetMode.rawValue, forKey: "targetMode") }
    }
    @Published var romfsMode: RomFSMode = RomFSMode(
        rawValue: UserDefaults.standard.string(forKey: "romfsMode") ?? "nup"
    ) ?? .nup {
        didSet { UserDefaults.standard.set(romfsMode.rawValue, forKey: "romfsMode") }
    }
    @Published var useConsoleGame: Bool = UserDefaults.standard.bool(forKey: "useConsoleGame") {
        didSet { UserDefaults.standard.set(useConsoleGame, forKey: "useConsoleGame") }
    }
    @Published var consoleGamePath: String = UserDefaults.standard.string(forKey: "consoleGamePath") ?? "/data/mkw" {
        didSet { UserDefaults.standard.set(consoleGamePath, forKey: "consoleGamePath") }
    }
    @Published var localOutPath: String = UserDefaults.standard.string(forKey: "localOutPath") ?? "" {
        didSet { UserDefaults.standard.set(localOutPath, forKey: "localOutPath") }
    }
    @Published var customUSBPath: String = UserDefaults.standard.string(forKey: "customUSBPath") ?? "" {
        didSet { UserDefaults.standard.set(customUSBPath, forKey: "customUSBPath") }
    }
    @Published var jobsCount: Int = 0 // 0 = Auto

    // Runtime state
    @Published var dumpValidation: DumpValidation = .empty
    @Published var foundConsoles: [FoundConsole] = []
    @Published var isScanningLAN: Bool = false
    @Published var isCheckingConsole: Bool = false
    @Published var consoleCheck: ConsoleCheckResult?
    @Published var usbVolumes: [USBVolume] = []
    @Published var selectedUSBPath: String = ""

    @Published var steps: [BuildStepItem] = BuildStepItem.allSteps
    @Published var isBuilding: Bool = false
    @Published var buildFinishedOK: Bool = false
    @Published var buildError: String?
    @Published var progressValue: Double = 0
    @Published var progressTotal: Double = 0
    @Published var currentNote: String = ""
    @Published var infoBanners: [String: String] = [:]
    @Published var detectedVersion: String = ""
    @Published var logLines: [LogLine] = []
    @Published var showGuide: Bool = false
    @Published var showLogs: Bool = false

    let hostArchLabel: String = {
        #if arch(arm64)
        return "Apple Silicon (arm64)"
        #else
        return "Intel (x86_64)"
        #endif
    }()

    private var activeProcess: Process?
    private var activePGID: pid_t = 0
    private var sleepAssertion: NSObjectProtocol?
    private var logTailTimer: Timer?
    private var logReadOffset: UInt64 = 0

    let appSupportDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("Mario Kart World Pit Stop", isDirectory: true)
    }()

    var workDir: URL { appSupportDir.appendingPathComponent("work", isDirectory: true) }
    var defaultOutDir: URL { appSupportDir.appendingPathComponent("stage/PPSA99193", isDirectory: true) }
    var logFileURL: URL { appSupportDir.appendingPathComponent("logs/setup.log") }

    var effectiveOutDir: URL {
        if targetMode == .localFolder, !localOutPath.isEmpty {
            return URL(fileURLWithPath: localOutPath).appendingPathComponent("PPSA99193", isDirectory: true)
        }
        return defaultOutDir
    }

    var setupDir: URL? {
        if let res = Bundle.main.resourceURL?.appendingPathComponent("setup", isDirectory: true),
           FileManager.default.fileExists(atPath: res.appendingPathComponent("src/tools/ps5/prepare_ps5.py").path) {
            return res
        }
        // Fallback when running from repo checkout during development
        let dev = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("build/macos-app/Mario Kart World Pit Stop.app/Contents/Resources/setup")
        if FileManager.default.fileExists(atPath: dev.appendingPathComponent("src/tools/ps5/prepare_ps5.py").path) {
            return dev
        }
        return nil
    }

    var archTag: String {
        #if arch(arm64)
        return "arm64"
        #else
        return "x86_64"
        #endif
    }

    var pythonURL: URL? { setupDir?.appendingPathComponent("arch/\(archTag)/python/bin/python3") }
    var glslcURL: URL? { setupDir?.appendingPathComponent("arch/\(archTag)/glslc/bin/glslc") }
    var bundleDirURL: URL? { setupDir?.appendingPathComponent("ps5-bundle", isDirectory: true) }
    var prepareScriptURL: URL? { setupDir?.appendingPathComponent("src/tools/ps5/prepare_ps5.py") }
    var probeScriptURL: URL? { setupDir?.appendingPathComponent("src/tools/ps5/console_probe.py") }

    var logoImage: NSImage? {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("pitstop-logo.png"),
           let img = NSImage(contentsOf: url) {
            return img
        }
        return NSApp.applicationIconImage
    }

    var canStartBuild: Bool {
        guard !isBuilding, setupDir != nil else { return false }
        if case .valid = dumpValidation {} else { return false }
        switch targetMode {
        case .ps5Network:
            return !ps5IP.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .usbDrive:
            return !effectiveUSBPath.isEmpty
        case .localFolder:
            return true
        }
    }

    var effectiveUSBPath: String {
        !customUSBPath.isEmpty ? customUSBPath : selectedUSBPath
    }

    init() {
        try? FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: appSupportDir.appendingPathComponent("logs"), withIntermediateDirectories: true)
        refreshUSBVolumes()
        if !sourcePath.isEmpty {
            validateDump()
        }
    }

    private func toolEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        if FileManager.default.fileExists(atPath: "/etc/ssl/cert.pem") {
            env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
        }
        let zcache = workDir.appendingPathComponent("zig-cache").path
        try? FileManager.default.createDirectory(atPath: zcache, withIntermediateDirectories: true)
        env["ZIG_GLOBAL_CACHE_DIR"] = zcache
        env["ZIG_LOCAL_CACHE_DIR"] = zcache
        env["BLINK_LOG_FILENAME"] = "/dev/null"
        env["PYTHONUNBUFFERED"] = "1"
        if let glslcLib = glslcURL?.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("lib").path {
            let old = env["DYLD_LIBRARY_PATH"] ?? ""
            env["DYLD_LIBRARY_PATH"] = old.isEmpty ? glslcLib : "\(glslcLib):\(old)"
        }
        return env
    }

    // MARK: - Dump validation
    func validateDump() {
        let trimmed = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            dumpValidation = .empty
            return
        }
        guard FileManager.default.fileExists(atPath: trimmed) else {
            dumpValidation = .invalid("Path does not exist")
            return
        }
        guard let py = pythonURL, let script = prepareScriptURL, let bundle = bundleDirURL else {
            dumpValidation = .invalid("Bundled setup toolchain is missing")
            return
        }
        dumpValidation = .checking

        let checkLog = appSupportDir.appendingPathComponent("logs/check.log").path
        let env = toolEnvironment()
        let args = [
            script.path,
            "--source", trimmed,
            "--bundle", bundle.path,
            "--work", workDir.path,
            "--out", effectiveOutDir.path,
            "--log", checkLog,
            "--check-only",
        ]

        Task.detached {
            let res = Self.runSync(exe: py, args: args, env: env)
            await MainActor.run {
                var ver = ""
                var detail = ""
                var errMsg = ""
                for line in res.stdout.components(separatedBy: .newlines) {
                    if line.hasPrefix("version ") {
                        ver = String(line.dropFirst("version ".count))
                    } else if line.hasPrefix("info dump ") {
                        detail = String(line.dropFirst("info dump ".count))
                    } else if line.hasPrefix("error ") {
                        let rest = String(line.dropFirst("error ".count))
                        if let sp = rest.firstIndex(of: " ") {
                            errMsg = String(rest[rest.index(after: sp)...])
                        } else {
                            errMsg = rest
                        }
                    }
                }
                if res.code == 0, !ver.isEmpty {
                    self.dumpValidation = .valid(version: ver, detail: detail)
                    self.detectedVersion = ver
                } else if !errMsg.isEmpty {
                    self.dumpValidation = .invalid(errMsg)
                } else {
                    let msg = res.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.dumpValidation = .invalid(msg.isEmpty ? "Not a valid Mario Kart World 1.0.0 or 1.8.0 dump" : msg)
                }
            }
        }
    }

    // MARK: - LAN Discovery & Check
    func scanLAN() {
        guard !isScanningLAN, let py = pythonURL, let probe = probeScriptURL else { return }
        isScanningLAN = true
        foundConsoles = []
        let env = toolEnvironment()
        Task.detached {
            let res = Self.runSync(exe: py, args: [probe.path, "find"], env: env)
            var found: [FoundConsole] = []
            for line in res.stdout.components(separatedBy: .newlines) {
                let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
                if parts.count >= 4, parts[0] == "found" {
                    found.append(FoundConsole(ip: parts[1], kind: parts[2], summary: parts[3]))
                }
            }
            let result = found
            await MainActor.run {
                self.foundConsoles = result
                self.isScanningLAN = false
                if self.ps5IP.isEmpty, let first = result.first {
                    self.ps5IP = first.ip
                    self.checkConsole()
                }
            }
        }
    }

    func checkConsole() {
        let host = ps5IP.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !host.isEmpty, !isCheckingConsole, let py = pythonURL, let probe = probeScriptURL else { return }
        isCheckingConsole = true
        consoleCheck = nil
        let env = toolEnvironment()
        Task.detached {
            let res = Self.runSync(exe: py, args: [probe.path, "check", host], env: env)
            var parsed = ConsoleCheckResult()
            for line in res.stdout.components(separatedBy: .newlines) {
                let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
                if parts.count >= 2 {
                    switch parts[0] {
                    case "kind": parsed.kind = parts[1]
                    case "line": parsed.line = parts[1]
                    case "game":
                        let folder = parts.count >= 3 ? parts[2] : ""
                        parsed.games.append("\(parts[1]) (\(folder))")
                    default: break
                    }
                }
            }
            let result = parsed
            await MainActor.run {
                self.consoleCheck = result
                self.isCheckingConsole = false
            }
        }
    }

    // MARK: - USB Drive Detection
    func refreshUSBVolumes() {
        Task.detached {
            let fm = FileManager.default
            let keys: [URLResourceKey] = [
                .volumeNameKey,
                .volumeIsInternalKey,
                .volumeIsRemovableKey,
                .volumeIsReadOnlyKey,
                .volumeAvailableCapacityForImportantUsageKey,
                .volumeTotalCapacityKey,
            ]
            let urls = fm.mountedVolumeURLs(includingResourceValuesForKeys: keys, options: [.skipHiddenVolumes]) ?? []
            var list: [USBVolume] = []
            for url in urls {
                guard url.path != "/" else { continue }
                let rv = try? url.resourceValues(forKeys: Set(keys))
                if rv?.volumeIsReadOnly == true { continue }
                let isInternal = rv?.volumeIsInternal ?? false
                let isRemovable = rv?.volumeIsRemovable ?? !isInternal
                if isInternal && !isRemovable { continue }

                let fs = Self.queryDiskutilFilesystem(path: url.path)
                let free = rv?.volumeAvailableCapacityForImportantUsage ?? 0
                let total = Int64(rv?.volumeTotalCapacity ?? 0)
                let name = rv?.volumeName ?? url.lastPathComponent
                list.append(USBVolume(
                    name: name,
                    path: url.path,
                    filesystem: fs.isEmpty ? "Unknown" : fs,
                    freeBytes: free,
                    totalBytes: total,
                    isRemovable: isRemovable
                ))
            }
            list.sort { ($0.isExFAT ? 0 : 1, $0.name) < ($1.isExFAT ? 0 : 1, $1.name) }
            let result = list
            await MainActor.run {
                self.usbVolumes = result
                if self.selectedUSBPath.isEmpty || !result.contains(where: { $0.path == self.selectedUSBPath }) {
                    self.selectedUSBPath = result.first(where: { $0.isSupported })?.path ?? result.first?.path ?? ""
                }
            }
        }
    }

    nonisolated private static func queryDiskutilFilesystem(path: String) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/diskutil")
        p.arguments = ["info", "-plist", path]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            guard p.terminationStatus == 0,
                  let dict = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
                return ""
            }
            let raw = (dict["FilesystemType"] as? String ?? dict["FileSystemType"] as? String ?? "").lowercased()
            switch raw {
            case "exfat": return "exFAT"
            case "msdos", "fat32": return "FAT32"
            case "apfs": return "APFS"
            case "hfs": return "HFS+"
            case "ntfs": return "NTFS"
            default: return raw.uppercased()
            }
        } catch {
            return ""
        }
    }

    // MARK: - Build & Deploy
    func startBuild() {
        guard canStartBuild,
              let py = pythonURL,
              let script = prepareScriptURL,
              let bundle = bundleDirURL,
              let glslc = glslcURL else { return }

        steps = BuildStepItem.allSteps
        isBuilding = true
        buildFinishedOK = false
        buildError = nil
        progressValue = 0
        progressTotal = 0
        currentNote = "Starting build pipeline..."
        infoBanners = [:]
        logLines = []
        logReadOffset = 0

        try? FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: logFileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logFileURL.path, contents: nil)

        sleepAssertion = ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled, .userInitiated],
            reason: "Building Mario Kart World: Pit Stop for PS5"
        )

        var args = [
            script.path,
            "--source", sourcePath.trimmingCharacters(in: .whitespacesAndNewlines),
            "--bundle", bundle.path,
            "--work", workDir.path,
            "--out", effectiveOutDir.path,
            "--glslc", glslc.path,
            "--romfs", romfsMode.rawValue,
            "--log", logFileURL.path,
            "--events",
        ]
        if jobsCount > 0 {
            args += ["--jobs", "\(jobsCount)"]
        }
        switch targetMode {
        case .ps5Network:
            args += ["--send", ps5IP.trimmingCharacters(in: .whitespacesAndNewlines)]
            if useConsoleGame {
                let remote = consoleGamePath.trimmingCharacters(in: .whitespacesAndNewlines)
                if !remote.isEmpty {
                    args += ["--console-game", remote]
                }
            }
        case .usbDrive:
            args += ["--usb", effectiveUSBPath]
        case .localFolder:
            break
        }

        let proc = Process()
        proc.executableURL = py
        proc.arguments = args
        proc.environment = toolEnvironment()
        let outPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = FileHandle.nullDevice

        activeProcess = proc
        startLogTailTimer()

        let fh = outPipe.fileHandleForReading
        final class LineAccumulator: @unchecked Sendable {
            private let lock = NSLock()
            private var buffer = Data()

            func appendAndDrain(_ chunk: Data) -> [String] {
                lock.lock()
                defer { lock.unlock() }
                buffer.append(chunk)
                var lines: [String] = []
                while let nl = buffer.firstIndex(of: 0x0A) {
                    let slice = buffer.prefix(upTo: nl)
                    buffer.removeSubrange(...nl)
                    if let text = String(data: slice, encoding: .utf8) {
                        lines.append(text)
                    }
                }
                return lines
            }
        }
        let acc = LineAccumulator()
        fh.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            let lines = acc.appendAndDrain(chunk)
            if !lines.isEmpty {
                Task { @MainActor [weak self] in
                    for text in lines {
                        self?.handleEventLine(text)
                    }
                }
            }
        }

        proc.terminationHandler = { [weak self] p in
            fh.readabilityHandler = nil
            Task { @MainActor [weak self] in
                self?.finishBuild(exitCode: p.terminationStatus)
            }
        }

        do {
            try proc.run()
        } catch {
            finishBuild(exitCode: 1, launchError: error.localizedDescription)
        }
    }

    func cancelBuild() {
        guard isBuilding, let proc = activeProcess else { return }
        currentNote = "Cancelling build..."
        let pid = proc.processIdentifier
        let pgid = getpgid(pid)
        let ownPgrp = getpgrp()
        if pgid > 0 && pgid != ownPgrp {
            kill(-pgid, SIGTERM)
        }
        proc.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
            if pgid > 0 && pgid != ownPgrp { kill(-pgid, SIGKILL) }
            kill(pid, SIGKILL)
        }
        buildError = "Cancelled by user"
    }

    private func handleEventLine(_ line: String) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if trimmed.hasPrefix("step ") {
            let name = String(trimmed.dropFirst(5))
            markRunningStep(name)
            progressValue = 0
            progressTotal = 0
        } else if trimmed.hasPrefix("skip ") {
            let name = String(trimmed.dropFirst(5))
            setStep(name, status: .skipped)
        } else if trimmed.hasPrefix("progress ") {
            let parts = trimmed.split(separator: " ")
            if parts.count >= 3, let d = Double(parts[1]), let t = Double(parts[2]), t > 0 {
                progressValue = d
                progressTotal = t
            }
        } else if trimmed.hasPrefix("version ") {
            detectedVersion = String(trimmed.dropFirst(8))
        } else if trimmed.hasPrefix("note ") {
            currentNote = String(trimmed.dropFirst(5))
        } else if trimmed.hasPrefix("info ") {
            let rest = String(trimmed.dropFirst(5))
            if let sp = rest.firstIndex(of: " ") {
                let key = String(rest[..<sp])
                let val = String(rest[rest.index(after: sp)...])
                infoBanners[key] = val
            }
        } else if trimmed == "done" {
            for i in steps.indices where steps[i].status == .running {
                steps[i].status = .done
            }
            if targetMode == .localFolder {
                setStep("deploy", status: .skipped)
            }
            buildFinishedOK = true
            currentNote = targetMode == .ps5Network
                ? "Transfer complete! Launch Mario Kart World: Pit Stop on your PS5."
                : "Build complete!"
        } else if trimmed.hasPrefix("error ") {
            let rest = String(trimmed.dropFirst(6))
            if let sp = rest.firstIndex(of: " ") {
                buildError = String(rest[rest.index(after: sp)...])
            } else {
                buildError = rest
            }
            for i in steps.indices where steps[i].status == .running {
                steps[i].status = .failed
            }
        }
    }

    private func markRunningStep(_ name: String) {
        for i in steps.indices {
            if steps[i].status == .running {
                steps[i].status = .done
            }
            if steps[i].id == name {
                steps[i].status = .running
            }
        }
    }

    private func setStep(_ name: String, status: StepStatus) {
        if let idx = steps.firstIndex(where: { $0.id == name }) {
            steps[idx].status = status
        }
    }

    private func finishBuild(exitCode: Int32, launchError: String? = nil) {
        stopLogTailTimer()
        readNewLogChunk()
        if let act = sleepAssertion {
            ProcessInfo.processInfo.endActivity(act)
            sleepAssertion = nil
        }
        activeProcess = nil
        activePGID = 0
        isBuilding = false

        if let launchError {
            buildError = launchError
        } else if exitCode == 0 && buildError == nil {
            for i in steps.indices where steps[i].status == .running {
                steps[i].status = .done
            }
            buildFinishedOK = true
        } else if buildError == nil {
            buildError = "Build exited with code \(exitCode). Inspect setup.log for details."
            for i in steps.indices where steps[i].status == .running {
                steps[i].status = .failed
            }
        }
    }

    private func startLogTailTimer() {
        stopLogTailTimer()
        logTailTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.readNewLogChunk()
            }
        }
    }

    private func stopLogTailTimer() {
        logTailTimer?.invalidate()
        logTailTimer = nil
    }

    private func readNewLogChunk() {
        guard let fh = try? FileHandle(forReadingFrom: logFileURL) else { return }
        defer { try? fh.close() }
        do {
            try fh.seek(toOffset: logReadOffset)
            let data = fh.readDataToEndOfFile()
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            logReadOffset += UInt64(data.count)
            let newLines = chunk.components(separatedBy: .newlines).filter { !$0.isEmpty }.map { txt in
                let isErr = txt.localizedCaseInsensitiveContains("error") || txt.localizedCaseInsensitiveContains("failed")
                return LogLine(text: txt, isError: isErr)
            }
            logLines.append(contentsOf: newLines)
            if logLines.count > 1200 {
                logLines.removeFirst(logLines.count - 1200)
            }
        } catch {}
    }

    // MARK: - Helpers
    func openDataFolder() {
        NSWorkspace.shared.open(appSupportDir)
    }

    func openLogFile() {
        if FileManager.default.fileExists(atPath: logFileURL.path) {
            NSWorkspace.shared.open(logFileURL)
        } else {
            NSWorkspace.shared.open(appSupportDir.appendingPathComponent("logs"))
        }
    }

    func revealStagedOutput() {
        if FileManager.default.fileExists(atPath: effectiveOutDir.path) {
            NSWorkspace.shared.activateFileViewerSelecting([effectiveOutDir])
        } else {
            NSWorkspace.shared.open(effectiveOutDir.deletingLastPathComponent())
        }
    }

    func copyLogsToClipboard() {
        let text = logLines.map(\.text).joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    nonisolated private static func runSync(exe: URL, args: [String], env: [String: String]) -> (code: Int32, stdout: String, stderr: String) {
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        p.environment = env
        let outP = Pipe()
        let errP = Pipe()
        p.standardOutput = outP
        p.standardError = errP
        do {
            try p.run()
            let outData = outP.fileHandleForReading.readDataToEndOfFile()
            let errData = errP.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            return (
                p.terminationStatus,
                String(data: outData, encoding: .utf8) ?? "",
                String(data: errData, encoding: .utf8) ?? ""
            )
        } catch {
            return (1, "", error.localizedDescription)
        }
    }
}
