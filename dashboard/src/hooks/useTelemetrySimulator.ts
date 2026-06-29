"use client";

import { useState, useEffect, useCallback, useRef } from "react";

/* ── Types ──────────────────────────────────────────────── */
export interface TelemetryPoint {
  time: string;
  timestamp: number;
  sxr: number;
  hxr: number;
  sxrRollingMean: number;
  hxrRollingMean: number;
  nowcastProb: number;
  forecastClass: string;
  forecastTimeToPeak: string;
  forecastConfidence: number;
  alertLevel: "normal" | "elevated" | "critical";
}

export interface ForecastData {
  predictedClass: string;
  timeToPeak: string;
  confidence: number;
  classColor: string;
}

export interface TelemetryState {
  data: TelemetryPoint[];
  isRunning: boolean;
  nowcastProb: number;
  forecastData: ForecastData;
  alertLevel: "normal" | "elevated" | "critical";
  tickCount: number;
  lastUpdate: string;
  toggle: () => void;
}

/* ── Constants ──────────────────────────────────────────── */
const WINDOW_SIZE = 80;
const TICK_INTERVAL = 1000; // ms
const SPIKE_INTERVAL_MIN = 40;
const SPIKE_INTERVAL_MAX = 80;
const THERMAL_RAMP_DURATION = 30; // ticks
const CRITICAL_THRESHOLD = 0.80;

/* ── Helpers ────────────────────────────────────────────── */
function gaussianRandom(mean = 0, std = 1): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function formatUTC(date: Date): string {
  return date.toISOString().replace("T", " ").substring(11, 19);
}

function getFlareClass(sxrMean: number): { cls: string; color: string } {
  if (sxrMean > 12) return { cls: "X1.2", color: "#ef4444" };
  if (sxrMean > 9) return { cls: "M5.3", color: "#f59e0b" };
  if (sxrMean > 7) return { cls: "M2.1", color: "#f59e0b" };
  if (sxrMean > 5.5) return { cls: "C8.4", color: "#eab308" };
  if (sxrMean > 4) return { cls: "C3.7", color: "#22c55e" };
  return { cls: "B2.1", color: "#10b981" };
}

function getTimeToPeak(sxrMean: number): string {
  if (sxrMean > 10) {
    const mins = Math.floor(Math.random() * 30 + 10);
    return `0h ${mins}m`;
  }
  if (sxrMean > 7) {
    const hrs = Math.floor(Math.random() * 3 + 1);
    const mins = Math.floor(Math.random() * 60);
    return `${hrs}h ${mins.toString().padStart(2, "0")}m`;
  }
  if (sxrMean > 5) {
    const hrs = Math.floor(Math.random() * 8 + 4);
    const mins = Math.floor(Math.random() * 60);
    return `${hrs}h ${mins.toString().padStart(2, "0")}m`;
  }
  return "N/A";
}

/* ── Hook ───────────────────────────────────────────────── */
export function useTelemetrySimulator(): TelemetryState {
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [nowcastProb, setNowcastProb] = useState(0);
  const [forecastData, setForecastData] = useState<ForecastData>({
    predictedClass: "B2.1",
    timeToPeak: "N/A",
    confidence: 0.12,
    classColor: "#10b981",
  });
  const [alertLevel, setAlertLevel] = useState<"normal" | "elevated" | "critical">("normal");
  const [tickCount, setTickCount] = useState(0);
  const [lastUpdate, setLastUpdate] = useState("--:--:--");

  // Mutable refs for simulation state (no re-renders)
  const tickRef = useRef(0);
  const nextSpikeRef = useRef(
    Math.floor(Math.random() * (SPIKE_INTERVAL_MAX - SPIKE_INTERVAL_MIN) + SPIKE_INTERVAL_MIN)
  );
  const prevHxrRef = useRef(2);
  const thermalRampRef = useRef(0);
  const thermalActiveRef = useRef(false);
  const sxrRollingRef = useRef<number[]>([]);
  const hxrRollingRef = useRef<number[]>([]);
  const decayProbRef = useRef(0);

  const toggle = useCallback(() => {
    setIsRunning((prev) => !prev);
  }, []);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      tickRef.current += 1;
      const t = tickRef.current;
      const now = new Date();
      const timeStr = formatUTC(now);

      /* ── SXR (SoLEXS) ─── Slow thermal sine wave ────────── */
      let sxrBase =
        5 + 2.5 * Math.sin(t * 0.05) + 0.8 * Math.sin(t * 0.02 + 1.5);

      // Occasional thermal ramp (elevated heating)
      if (!thermalActiveRef.current && Math.random() < 0.008) {
        thermalActiveRef.current = true;
        thermalRampRef.current = 0;
      }
      if (thermalActiveRef.current) {
        thermalRampRef.current += 1;
        const rampProgress = thermalRampRef.current / THERMAL_RAMP_DURATION;
        sxrBase += 5 * Math.sin(rampProgress * Math.PI); // bell curve ramp
        if (thermalRampRef.current >= THERMAL_RAMP_DURATION) {
          thermalActiveRef.current = false;
          thermalRampRef.current = 0;
        }
      }

      const sxr = Math.max(0, sxrBase + gaussianRandom(0, 0.4));

      /* ── HXR (HEL1OS) ─── Flat baseline + impulsive spikes ─ */
      let hxrBase = 2 + Math.abs(gaussianRandom(0, 1.2));
      let isSpike = false;

      if (t >= nextSpikeRef.current) {
        // Inject a massive spike
        const spikeAmplitude = 150 + Math.random() * 250; // 150-400
        hxrBase = spikeAmplitude;
        isSpike = true;
        nextSpikeRef.current =
          t +
          Math.floor(
            Math.random() * (SPIKE_INTERVAL_MAX - SPIKE_INTERVAL_MIN) +
              SPIKE_INTERVAL_MIN
          );
      }

      const hxr = Math.max(0, hxrBase);

      /* ── Rolling Means ─────────────────────────────────────── */
      sxrRollingRef.current.push(sxr);
      hxrRollingRef.current.push(hxr);
      if (sxrRollingRef.current.length > 16) sxrRollingRef.current.shift();
      if (hxrRollingRef.current.length > 16) hxrRollingRef.current.shift();

      const sxrMean =
        sxrRollingRef.current.reduce((a, b) => a + b, 0) /
        sxrRollingRef.current.length;
      const hxrMean =
        hxrRollingRef.current.reduce((a, b) => a + b, 0) /
        hxrRollingRef.current.length;

      /* ── Nowcast Probability ─── Derivative-based ──────────── */
      const derivative = Math.abs(hxr - prevHxrRef.current);
      prevHxrRef.current = hxr;

      let prob: number;
      if (isSpike || derivative > 50) {
        prob = sigmoid((derivative - 20) * 0.08);
        prob = Math.max(prob, 0.85 + Math.random() * 0.13); // 85-98%
        decayProbRef.current = prob;
      } else {
        // Exponential decay from last spike
        decayProbRef.current *= 0.92;
        prob = Math.max(decayProbRef.current, 0.02 + Math.random() * 0.08);
      }
      prob = Math.min(prob, 0.99);

      /* ── Forecast ─── Based on SXR thermal state ───────────── */
      const { cls, color } = getFlareClass(sxrMean);
      const timeToPeak = getTimeToPeak(sxrMean);
      const confidence =
        sxrMean > 7 ? 0.7 + Math.random() * 0.25 : 0.1 + Math.random() * 0.3;

      /* ── Alert Level ─────────────────────────────────────────── */
      let alert: "normal" | "elevated" | "critical" = "normal";
      if (prob >= CRITICAL_THRESHOLD) {
        alert = "critical";
      } else if (
        cls.startsWith("M") ||
        cls.startsWith("X") ||
        thermalActiveRef.current
      ) {
        alert = "elevated";
      }

      /* ── Build Data Point ────────────────────────────────────── */
      const point: TelemetryPoint = {
        time: timeStr,
        timestamp: now.getTime(),
        sxr: parseFloat(sxr.toFixed(2)),
        hxr: parseFloat(hxr.toFixed(2)),
        sxrRollingMean: parseFloat(sxrMean.toFixed(2)),
        hxrRollingMean: parseFloat(hxrMean.toFixed(2)),
        nowcastProb: parseFloat(prob.toFixed(4)),
        forecastClass: cls,
        forecastTimeToPeak: timeToPeak,
        forecastConfidence: parseFloat(confidence.toFixed(2)),
        alertLevel: alert,
      };

      /* ── Update State ────────────────────────────────────────── */
      setData((prev) => {
        const next = [...prev, point];
        return next.length > WINDOW_SIZE ? next.slice(-WINDOW_SIZE) : next;
      });

      setNowcastProb(prob);
      setForecastData({
        predictedClass: cls,
        timeToPeak: timeToPeak,
        confidence: confidence,
        classColor: color,
      });
      setAlertLevel(alert);
      setTickCount(t);
      setLastUpdate(timeStr);
    }, TICK_INTERVAL);

    return () => clearInterval(interval);
  }, [isRunning]);

  return {
    data,
    isRunning,
    nowcastProb,
    forecastData,
    alertLevel,
    tickCount,
    lastUpdate,
    toggle,
  };
}
