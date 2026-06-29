"use client";

import { useState, useEffect } from "react";
import { Satellite, Wifi, WifiOff } from "lucide-react";

interface HeaderBarProps {
  isRunning: boolean;
}

export default function HeaderBar({ isRunning }: HeaderBarProps) {
  const [utcTime, setUtcTime] = useState("--:--:-- UTC");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setUtcTime(
        now.toISOString().replace("T", " ").substring(0, 19) + " UTC"
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="glass-card flex items-center justify-between px-6 py-3 border-b border-[var(--border-subtle)]">
      {/* Left — Title */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <Satellite className="w-7 h-7 text-[var(--solexs-orange)]" />
          <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-[var(--alert-green)] rounded-full animate-pulse-dot" />
        </div>
        <div>
          <h1 className="text-sm font-semibold tracking-widest uppercase text-[var(--text-primary)]">
            Aditya-L1 Solar Flare Early Warning System
          </h1>
          <p className="text-[10px] tracking-wider text-[var(--text-muted)] uppercase">
            ISRO · Indian Space Research Organisation
          </p>
        </div>
      </div>

      {/* Center — UTC Clock */}
      <div className="hidden md:flex flex-col items-center">
        <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
          Mission Clock
        </span>
        <span className="font-mono text-lg font-tabular text-[var(--helios-cyan)] tracking-wide">
          {utcTime}
        </span>
      </div>

      {/* Right — Status */}
      <div className="flex items-center gap-2">
        {isRunning ? (
          <>
            <Wifi className="w-4 h-4 text-[var(--alert-green)]" />
            <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--alert-green)] uppercase tracking-wider">
              <span className="w-2 h-2 bg-[var(--alert-green)] rounded-full animate-pulse-dot" />
              Simulating
            </span>
          </>
        ) : (
          <>
            <WifiOff className="w-4 h-4 text-[var(--text-muted)]" />
            <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              <span className="w-2 h-2 bg-[var(--text-muted)] rounded-full" />
              Offline
            </span>
          </>
        )}
      </div>
    </header>
  );
}
