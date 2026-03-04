/**
 * RiskBadge – Compact colour-coded risk indicator.
 *
 * Props:
 *   riskLevel – "LOW" | "MEDIUM" | "HIGH"
 *   hri       – numeric score (0–1)
 */

import React from "react";

const RISK_COLORS = {
  LOW: "#22c55e",
  MEDIUM: "#f59e0b",
  HIGH: "#ef4444",
};

export default function RiskBadge({ riskLevel, hri }) {
  const color = RISK_COLORS[riskLevel] || "#6b7280";

  return (
    <span className="risk-badge-inline" style={{ borderColor: color }}>
      <span className="risk-dot" style={{ backgroundColor: color }} />
      <span className="risk-label">{riskLevel}</span>
      {hri != null && (
        <span className="risk-hri" style={{ color }}>
          {hri.toFixed(2)}
        </span>
      )}
    </span>
  );
}
