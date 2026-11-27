import Foundation

enum StepStatus: Equatable {
    case pending
    case running
    case done
    case skipped
    case failed
}

struct BuildStepItem: Identifiable, Equatable {
    let id: String
    let title: String
    let icon: String
    var status: StepStatus = .pending

    static let allSteps: [BuildStepItem] = [
        BuildStepItem(id: "check", title: "Verify", icon: "checkmark.shield"),
        BuildStepItem(id: "nup", title: "Unpack NUP", icon: "archivebox"),
        BuildStepItem(id: "exefs", title: "ExeFS", icon: "cpu"),
        BuildStepItem(id: "romfs", title: "RomFS", icon: "opticaldisc"),
        BuildStepItem(id: "generate", title: "Lift C", icon: "chevron.left.forwardslash.chevron.right"),
        BuildStepItem(id: "compile", title: "Compile", icon: "hammer"),
        BuildStepItem(id: "link", title: "Link SELF", icon: "link"),
        BuildStepItem(id: "shaders", title: "Shaders", icon: "sparkles"),
        BuildStepItem(id: "deploy", title: "Transfer", icon: "paperplane"),
    ]
}

enum TargetMode: String, CaseIterable, Identifiable {
    case ps5Network = "ps5"
    case usbDrive = "usb"
    case localFolder = "local"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .ps5Network: return "PS5 (Network)"
        case .usbDrive: return "USB Drive"
        case .localFolder: return "Local Folder"
        }
    }

    var icon: String {
        switch self {
        case .ps5Network: return "wifi"
        case .usbDrive: return "externaldrive.connected.to.line.below"
        case .localFolder: return "folder"
        }
    }
}

enum RomFSMode: String, CaseIterable, Identifiable {
    case nup = "nup"
    case copy = "copy"
    case skip = "skip"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .nup: return "romfs.nup (1 file — Recommended)"
        case .copy: return "Loose RomFS files"
        case .skip: return "Skip RomFS"
        }
    }
}

struct FoundConsole: Identifiable, Equatable {
    let ip: String
    let kind: String // "receiver" | "ftp"
    let summary: String
    var id: String { ip }
}

struct USBVolume: Identifiable, Equatable {
    let name: String
    let path: String
    let filesystem: String
    let freeBytes: Int64
    let totalBytes: Int64
    let isRemovable: Bool

    var id: String { path }

    var isSupported: Bool {
        let fs = filesystem.uppercased()
        return fs == "EXFAT" || fs == "FAT32" || fs == "FAT"
    }

    var isExFAT: Bool {
        filesystem.uppercased() == "EXFAT"
    }

    var isFAT32: Bool {
        let fs = filesystem.uppercased()
        return fs == "FAT32" || fs == "FAT"
    }

    var freeText: String {
        ByteCountFormatter.string(fromByteCount: freeBytes, countStyle: .file)
    }
}

enum DumpValidation: Equatable {
    case empty
    case checking
    case valid(version: String, detail: String)
    case invalid(String)
}

struct ConsoleCheckResult: Equatable {
    var kind: String = ""
    var line: String = ""
    var games: [String] = []
}

struct LogLine: Identifiable, Equatable {
    let id = UUID()
    let text: String
    let isError: Bool
}
