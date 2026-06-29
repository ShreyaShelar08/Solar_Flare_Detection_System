"use client";

import { Play, Square, Download, Database, Clock, Hash } from "lucide-react";
import type { TelemetryPoint } from "@/hooks/useTelemetrySimulator";
import { exportToCSV } from "@/utils/csvExport";

interface ActionFooterProps {
  isRunning: boolean;
  onToggle: () => void;
  data: TelemetryPoint[];
  tickCount: number;
  lastUpdate: string;
}

export default function ActionFooter({
  isRunning,
  onToggle,
  data,
  tickCount,
  lastUpdate,
}: ActionFooterProps) {
  const handleDownload = () => {
    if (data.length === 0) return;

    const csvData = data.map((d) => ({
      time: d.time,
      sxr: d.sxr,
      hxr: d.hxr,
      sxrRollingMean: d.sxrRollingMean,
      hxrRollingMean: d.hxrRollingMean,
      nowcastProb: d.nowcastProb,
      forecastClass: d.forecastClass,
      alertLevel: d.alertLevel,
    }));

    exportToCSV(csvData);
  };

  return (
    <footer className="glass-card border-t border-[var(--border-subtle)] px-6 py-3">
      <div className="flex items-center justify-between flex-wrap gap-3">
        {/* Buttons */}
        <div className="flex items-center gap-3">
          {/* Start / Stop */}
          <button
            onClick={onToggle}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold tracking-wider uppercase transition-all duration-200 cursor-pointer ${
              isRunning
                ? "bg-[var(--alert-red)]/15 text-[var(--alert-red)] border border-[var(--alert-red)]/30 hover:bg-[var(--alert-red)]/25"
                : "bg-[var(--alert-green)]/15 text-[var(--alert-green)] border border-[var(--alert-green)]/30 hover:bg-[var(--alert-green)]/25"
            }`}
          >
            {isRunning ? (
              <>
                <Square className="w-4 h-4" />
                Stop Simulation
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Simulation
              </>
            )}
          </button>

          {/* Download CSV */}
          <button
            onClick={handleDownload}
            disabled={data.length === 0}
            className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold tracking-wider uppercase bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border-subtle)] hover:border-[var(--border-accent)] hover:text-[var(--text-primary)] transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 text-[10px] text-[var(--text-muted)] font-mono">
          <span className="flex items-center gap-1.5">
            <Database className="w-3 h-3" />
            {data.length} pts
          </span>
          <span className="flex items-center gap-1.5">
            <Hash className="w-3 h-3" />
            Tick #{tickCount}
          </span>
          <span className="flex items-center gap-1.5">
            <Clock className="w-3 h-3" />
            {lastUpdate}
          </span>
          <span className="px-2 py-0.5 bg-[var(--bg-primary)] rounded text-[var(--text-muted)]">
            1Hz
          </span>
        </div>
      </div>
    </footer>
  );
}
