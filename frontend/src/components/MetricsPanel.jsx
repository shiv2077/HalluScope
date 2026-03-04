/**
 * MetricsPanel v2 – Grid of all uncertainty metrics.
 *
 * Props:
 *   metrics   – MetricsDetail object from API
 *   hri       – overall HRI score
 *   riskLevel – "LOW" | "MEDIUM" | "HIGH"
 */

import React from "react";

const RISK_COLORS = {
  LOW: "#22c55e",
  MEDIUM: "#f59e0b",
  HIGH: "#ef4444",
};

export default function MetricsPanel({ metrics, hri, riskLevel }) {
  const items = [
    { label: "HRI Score", value: hri, fmt: (v) => v?.toFixed(4), highlight: true },
    { label: "Risk Level", value: riskLevel, fmt: (v) => v, highlight: true },
    { label: "Mean Log-Prob", value: metrics.mean_logprob, fmt: (v) => v?.toFixed(4) },
    { label: "Log-Prob Std", value: metrics.logprob_std, fmt: (v) => v?.toFixed(4) },
    { label: "Log-Prob Variance", value: metrics.logprob_variance, fmt: (v) => v?.toFixed(4) },
    { label: "Entropy", value: metrics.entropy, fmt: (v) => v?.toFixed(4) },
    { label: "Normalized Entropy", value: metrics.normalized_entropy, fmt: (v) => v?.toFixed(4) },
    { label: "Neg Log Likelihood", value: metrics.neg_log_likelihood, fmt: (v) => v?.toFixed(4) },
    { label: "Perplexity", value: metrics.perplexity, fmt: (v) => v?.toFixed(4) },
    { label: "Confidence Decay", value: metrics.confidence_decay, fmt: (v) => v?.toFixed(4) },
    { label: "Token Count", value: metrics.token_count, fmt: (v) => v },
  ];

  const color = RISK_COLORS[riskLevel] || "#6b7280";

  return (
    <div className="metrics-grid">
      {items.map((item) => (
        <div
          key={item.label}
          className={`metric-card ${item.highlight ? "metric-highlight" : ""}`}
          style={item.highlight ? { borderColor: color } : undefined}
        >
          <span className="metric-card-label">{item.label}</span>
          <span
            className="metric-card-value"
            style={item.label === "Risk Level" ? { color } : undefined}
          >
            {item.fmt(item.value)}
          </span>
        </div>
      ))}
    </div>
  );
}
