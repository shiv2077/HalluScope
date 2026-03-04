/**
 * RiskIndicator – Displays the Hallucination Risk Index (HRI)
 * with colour-coded risk level.
 *
 * Props:
 *   hri       – Numeric risk score (0–1).
 *   riskLevel – "LOW" | "MEDIUM" | "HIGH".
 */

import React from "react";

const RISK_COLORS = {
  LOW: "#22c55e",    // green
  MEDIUM: "#f59e0b", // amber
  HIGH: "#ef4444",   // red
};

export default function RiskIndicator({ hri, riskLevel }) {
  const color = RISK_COLORS[riskLevel] || "#6b7280";

  return (
    <div className="panel risk-indicator" style={{ borderColor: color }}>
      <h2>Risk Level</h2>
      <div className="risk-score" style={{ color }}>
        {hri.toFixed(2)}
      </div>
      <span className="risk-badge" style={{ backgroundColor: color }}>
        {riskLevel}
      </span>
    </div>
  );
}
