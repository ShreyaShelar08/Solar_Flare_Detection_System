"use client";

import { Shield, AlertTriangle, Zap } from "lucide-react";

interface AlertBannerProps {
  alertLevel: "normal" | "elevated" | "critical";
}

const ALERT_CONFIG = {
  normal: {
    bg: "bg-[var(--alert-green-bg)] border-[var(--alert-green)]/30",
    text: "text-[var(--alert-green)]",
    icon: Shield,
    title: "BACKGROUND ACTIVITY: NOMINAL",
    subtitle: "All instruments within expected parameters. No significant solar activity detected.",
    animate: "",
  },
  elevated: {
    bg: "bg-[var(--alert-amber-bg)] border-[var(--alert-amber)]/30",
    text: "text-[var(--alert-amber)]",
    icon: AlertTriangle,
    title: "FORECAST ALERT: Elevated Thermal Heating — M-Class Potential",
    subtitle: "SoLEXS soft X-ray flux is rising. BiLSTM model indicates increasing flare probability.",
    animate: "",
  },
  critical: {
    bg: "bg-[var(--alert-red-bg)] border-[var(--alert-red)]/30",
    text: "text-[var(--alert-red)]",
    icon: Zap,
    title: "CRITICAL WARNING: Impulsive HXR Spike Detected — Flare Imminent",
    subtitle: "HEL1OS hard X-ray surge detected. CNN nowcasting model reports >80% flare probability.",
    animate: "animate-pulse-critical",
  },
};

export default function AlertBanner({ alertLevel }: AlertBannerProps) {
  const config = ALERT_CONFIG[alertLevel];
  const Icon = config.icon;

  return (
    <div
      className={`alert-banner ${config.bg} ${config.animate} border rounded-lg mx-4 mt-3 px-5 py-3 flex items-center gap-4`}
    >
      {/* Icon */}
      <div className={`${config.text} flex-shrink-0`}>
        <Icon className="w-6 h-6" strokeWidth={2.5} />
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p
          className={`${config.text} text-sm font-bold tracking-wider uppercase truncate`}
        >
          {config.title}
        </p>
        <p className="text-xs text-[var(--text-muted)] mt-0.5 truncate">
          {config.subtitle}
        </p>
      </div>

      {/* Level Badge */}
      <div
        className={`${config.text} flex-shrink-0 text-[10px] font-bold tracking-widest uppercase border ${
          alertLevel === "normal"
            ? "border-[var(--alert-green)]/40"
            : alertLevel === "elevated"
            ? "border-[var(--alert-amber)]/40"
            : "border-[var(--alert-red)]/40"
        } rounded px-2 py-1`}
      >
        {alertLevel === "normal"
          ? "LEVEL 1"
          : alertLevel === "elevated"
          ? "LEVEL 2"
          : "LEVEL 3"}
      </div>
    </div>
  );
}
