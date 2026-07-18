export interface Alert {
  id: string;
  severity: "info" | "warning" | "critical";
  message: string;                // e.g. "Seizure event detected — session 4821"
  createdAt: string;
}
