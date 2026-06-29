"use client";

import { BrainCircuit, Clock, TrendingUp } from "lucide-react";
import type { ForecastData } from "@/hooks/useTelemetrySimulator";

interface ForecastPanelProps {
  forecast: ForecastData;
  isRunning: boolean;
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(value * 100, 100);
  const color =
    pct > 70
      ? "var(--alert-red)"
      : pct > 40
      ? "var(--alert-amber)"
      : "var(--alert-green)";

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
          Model Confidence
        </span>
        <span className="text-xs font-mono font-bold" style={{ color }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div className="w-full h-1.5 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

export default function ForecastPanel({
  forecast,
  isRunning,
}: ForecastPanelProps) {
  const isHighRisk =
    forecast.predictedClass.startsWith("M") ||
    forecast.predictedClass.startsWith("X");

  return (
    <div className="glass-card rounded-xl p-5 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <BrainCircuit className="w-4 h-4 text-[var(--solexs-orange)]" />
        <h3 className="text-xs font-semibold tracking-widest uppercase text-[var(--text-primary)]">
          Forecasting Engine
        </h3>
        <span className="ml-auto text-[10px] font-mono text-[var(--text-muted)] bg-[var(--bg-secondary)] px-2 py-0.5 rounded">
          BiLSTM
        </span>
      </div>

      {/* Predicted Class */}
      <div className="flex flex-col items-center py-3">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Predicted Flare Class
        </span>
        <div
          className={`relative px-6 py-3 rounded-lg border-2 font-mono text-3xl font-bold tracking-wider transition-all duration-500 ${
            isHighRisk ? "animate-slide-in" : ""
          }`}
          style={{
            color: isRunning ? forecast.classColor : "var(--text-muted)",
            borderColor: isRunning
              ? `${forecast.classColor}60`
              : "var(--border-subtle)",
            backgroundColor: isRunning
              ? `${forecast.classColor}10`
              : "transparent",
            boxShadow: isRunning && isHighRisk
              ? `0 0 20px ${forecast.classColor}20`
              : "none",
          }}
        >
          {isRunning ? forecast.predictedClass : "--.-"}
        </div>

        {/* GOES Classification Scale */}
        <div className="flex items-center gap-1 mt-3">
          {["B", "C", "M", "X"].map((cls) => {
            const isActive =
              isRunning && forecast.predictedClass.startsWith(cls);
            const clsColor =
              cls === "X"
                ? "var(--alert-red)"
                : cls === "M"
                ? "var(--alert-amber)"
                : cls === "C"
                ? "#eab308"
                : "var(--alert-green)";
            return (
              <span
                key={cls}
                className="text-[10px] font-mono font-bold px-2 py-0.5 rounded transition-all duration-300"
                style={{
                  color: isActive ? clsColor : "var(--text-muted)",
                  backgroundColor: isActive ? `${clsColor}20` : "transparent",
                  border: `1px solid ${
                    isActive ? `${clsColor}40` : "var(--border-subtle)"
                  }`,
                }}
              >
                {cls}
              </span>
            );
          })}
        </div>
      </div>

      {/* Time to Peak */}
      <div className="flex items-center gap-3 mt-3 p-3 bg-[var(--bg-secondary)] rounded-lg">
        <Clock className="w-4 h-4 text-[var(--text-muted)] flex-shrink-0" />
        <div className="flex-1">
          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
            Time to Peak
          </span>
          <p className="font-mono text-lg font-bold font-tabular text-[var(--text-primary)]">
            {isRunning ? forecast.timeToPeak : "--h --m"}
          </p>
        </div>
        <TrendingUp
          className={`w-4 h-4 flex-shrink-0 ${
            isHighRisk
              ? "text-[var(--alert-amber)]"
              : "text-[var(--text-muted)]"
          }`}
        />
      </div>

      {/* Confidence Bar */}
      <div className="mt-3">
        <ConfidenceBar value={isRunning ? forecast.confidence : 0} />
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] text-center">
        <p className="text-[10px] text-[var(--text-muted)] tracking-wider">
          SoLEXS Soft X-Ray Analysis · 1–24 hour window
        </p>
      </div>
    </div>
  );
}
