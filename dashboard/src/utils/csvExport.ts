/**
 * Converts an array of telemetry data objects to a CSV string
 * and triggers a browser download.
 */

export interface TelemetryDataPoint {
  time: string;
  sxr: number;
  hxr: number;
  sxrRollingMean: number;
  hxrRollingMean: number;
  nowcastProb: number;
  forecastClass: string;
  alertLevel: string;
}

export function exportToCSV(
  data: TelemetryDataPoint[],
  filename: string = "aditya_l1_telemetry.csv"
): void {
  if (data.length === 0) return;

  const headers = [
    "timestamp_utc",
    "sxr_flux_solexs",
    "hxr_flux_helios",
    "sxr_rolling_mean",
    "hxr_rolling_mean",
    "nowcast_probability",
    "forecast_class",
    "alert_level",
  ];

  const csvRows = [
    headers.join(","),
    ...data.map((point) =>
      [
        point.time,
        point.sxr.toFixed(4),
        point.hxr.toFixed(4),
        point.sxrRollingMean.toFixed(4),
        point.hxrRollingMean.toFixed(4),
        point.nowcastProb.toFixed(4),
        point.forecastClass,
        point.alertLevel,
      ].join(",")
    ),
  ];

  const csvString = csvRows.join("\n");
  const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();

  // Cleanup
  setTimeout(() => {
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, 100);
}
