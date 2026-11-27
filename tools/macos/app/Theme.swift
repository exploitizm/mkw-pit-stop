import SwiftUI

enum PitTheme {
    static let bg = Color(red: 0.047, green: 0.059, blue: 0.086)
    static let card = Color(red: 0.086, green: 0.106, blue: 0.153)
    static let cardElevated = Color(red: 0.118, green: 0.145, blue: 0.208)
    static let border = Color.white.opacity(0.08)
    static let crimson = Color(red: 0.910, green: 0.224, blue: 0.275)
    static let amber = Color(red: 0.980, green: 0.549, blue: 0.196)
    static let cyan = Color(red: 0.220, green: 0.741, blue: 0.973)
    static let emerald = Color(red: 0.063, green: 0.725, blue: 0.506)
    static let textPrimary = Color(red: 0.945, green: 0.961, blue: 0.988)
    static let textSecondary = Color(red: 0.592, green: 0.643, blue: 0.733)

    static let racingGradient = LinearGradient(
        colors: [crimson, amber],
        startPoint: .leading,
        endPoint: .trailing
    )
}

struct PitCard<Content: View, Trailing: View>: View {
    let stepNumber: Int?
    let title: String
    let subtitle: String?
    let trailing: Trailing
    let content: Content

    init(
        step: Int? = nil,
        title: String,
        subtitle: String? = nil,
        @ViewBuilder trailing: () -> Trailing = { EmptyView() },
        @ViewBuilder content: () -> Content
    ) {
        self.stepNumber = step
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing()
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 10) {
                if let stepNumber {
                    Text("\(stepNumber)")
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .frame(width: 24, height: 24)
                        .background(PitTheme.racingGradient)
                        .clipShape(Circle())
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(PitTheme.textPrimary)
                    if let subtitle {
                        Text(subtitle)
                            .font(.system(size: 12))
                            .foregroundColor(PitTheme.textSecondary)
                    }
                }
                Spacer()
                trailing
            }
            content
        }
        .padding(18)
        .background(PitTheme.card)
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(PitTheme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct TelemetryPill: View {
    let text: String
    let color: Color
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 5) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .semibold))
            }
            Text(text)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
        }
        .foregroundColor(color)
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(color.opacity(0.14))
        .overlay(
            Capsule().stroke(color.opacity(0.32), lineWidth: 1)
        )
        .clipShape(Capsule())
    }
}

struct RacingPrimaryButtonStyle: ButtonStyle {
    var disabled: Bool = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 14, weight: .bold))
            .foregroundColor(.white)
            .padding(.horizontal, 20)
            .padding(.vertical, 11)
            .background(
                Group {
                    if disabled {
                        Color.white.opacity(0.12)
                    } else {
                        PitTheme.racingGradient
                    }
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(configuration.isPressed ? 0.84 : (disabled ? 0.55 : 1.0))
            .scaleEffect(configuration.isPressed ? 0.99 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundColor(PitTheme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(PitTheme.cardElevated)
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .opacity(configuration.isPressed ? 0.75 : 1.0)
    }
}
