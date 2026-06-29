"use client";

import { Zap, Radio } from "lucide-react";

interface NowcastPanelProps {
  probability: number;
  isRunning: boolean;
}

export default function NowcastPanel({
  probability,
  isRunning,
}: NowcastPanelProps) {
  const pct = probability * 100;
  const isCritical = pct >= 80;
  const isElevated = pct >= 50;

  // SVG gauge calculations
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const dashoffset = arcLength - (arcLength * Math.min(pct, 100)) / 100;

  // Color based on probability
  const gaugeColor = isCritical
    ? "var(--alert-red)"
    : isElevated
    ? "var(--alert-amber)"
    : "var(--alert-green)";

  const statusColor = isCritical
    ? "text-[var(--alert-red)]"
    : isElevated
    ? "text-[var(--alert-amber)]"
    : "text-[var(--alert-green)]";

  const statusText = isCritical
    ? "SPIKE DETECTED"
    : isElevated
    ? "ELEVATED"
    : "MONITORING";

  return (
    <div className="glass-card rounded-xl p-5 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-4 h-4 text-[var(--helios-cyan)]" />
        <h3 className="text-xs font-semibold tracking-widest uppercase text-[var(--text-primary)]">
          Nowcasting Engine
        </h3>
        <span className="ml-auto text-[10px] font-mono text-[var(--text-muted)] bg-[var(--bg-secondary)] px-2 py-0.5 rounded">
          1D-CNN
        </span>
      </div>

      {/* Gauge */}
      <div className="flex justify-center items-center py-2">
        <div className="relative">
          <svg
            width="170"
            height="140"
            viewBox="0 0 170 170"
            className="overflow-visible"
          >
            {/* Background arc */}
            <circle
              cx="85"
              cy="85"
              r={radius}
              fill="none"
              stroke="var(--border-subtle)"
              strokeWidth="8"
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeDashoffset={0}
              strokeLinecap="round"
              transform="rotate(135 85 85)"
            />

            {/* Value arc */}
            <circle
              cx="85"
              cy="85"
              r={radius}
              fill="none"
              stroke={gaugeColor}
              strokeWidth="8"
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeDashoffset={dashoffset}
              strokeLinecap="round"
              transform="rotate(135 85 85)"
              className="gauge-arc"
              style={{
                filter: isCritical
                  ? `drop-shadow(0 0 8px ${gaugeColor})`
                  : "none",
              }}
            />

            {/* Tick marks */}
            {[0, 25, 50, 75, 100].map((tick) => {
              const angle = 135 + (tick / 100) * 270;
              const rad = (angle * Math.PI) / 180;
              const innerR = radius - 14;
              const outerR = radius - 10;
              return (
                <line
                  key={tick}
                  x1={85 + innerR * Math.cos(rad)}
                  y1={85 + innerR * Math.sin(rad)}
                  x2={85 + outerR * Math.cos(rad)}
                  y2={85 + outerR * Math.sin(rad)}
                  stroke="var(--text-muted)"
                  strokeWidth={1}
                />
              );
            })}
          </svg>

          {/* Center value */}
          <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ paddingTop: "10px" }}>
            <span
              className="font-mono text-3xl font-bold font-tabular"
              style={{ color: gaugeColor }}
            >
              {isRunning ? pct.toFixed(1) : "--.-"}
              <span className="text-lg">%</span>
            </span>
            <span className="text-[10px] text-[var(--text-muted)] tracking-wider uppercase mt-0.5">
              CNN Probability
            </span>
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center justify-center gap-2 mt-2">
        <Radio
          className={`w-3 h-3 ${statusColor} ${
            isCritical ? "animate-glow" : ""
          }`}
        />
        <span
          className={`text-xs font-bold tracking-widest uppercase ${statusColor} ${
            isCritical ? "animate-pulse" : ""
          }`}
        >
          {isRunning ? statusText : "STANDBY"}
        </span>
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] text-center">
        <p className="text-[10px] text-[var(--text-muted)] tracking-wider">
          HEL1OS Hard X-Ray Analysis · 0–15 min window
        </p>
      </div>
    </div>
  );
}
