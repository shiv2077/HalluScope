/**
 * MetricsPage – Detailed analytics for a single message.
 *
 * Fetches full metrics from GET /metrics/:messageId and displays:
 *   - Metric grid (9 metrics)
 *   - Per-token HRI trajectory chart
 *   - Token risk heatmap
 */

import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getMetrics } from "../api";
import MetricsPanel from "../components/MetricsPanel";
import TokenHeatmap from "../components/TokenHeatmap";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export default function MetricsPage() {
  const { messageId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics(messageId)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [messageId]);

  if (error) {
    return (
      <div className="metrics-page">
        <div className="error-banner">{error}</div>
        <button className="back-btn" onClick={() => navigate("/")}>
          ← Back to Chat
        </button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="metrics-page">
        <p className="loading-text">Loading metrics…</p>
      </div>
    );
  }

  // Prepare chart data
  const chartData = data.per_token_hri.map((hri, i) => ({
    index: i,
    hri,
  }));

  return (
    <div className="metrics-page">
      <header className="metrics-header">
        <button className="back-btn" onClick={() => navigate("/")}>
          ← Back to Chat
        </button>
        <h1>Message Analytics</h1>
        <span className="msg-id">ID: {messageId}</span>
      </header>

      {/* Response text */}
      <div className="panel metrics-answer">
        <h2>Generated Response</h2>
        <p>{data.content}</p>
        <span className="model-badge">{data.model}</span>
      </div>

      {/* Metrics grid */}
      <MetricsPanel metrics={data.metrics} hri={data.hri} riskLevel={data.risk_level} />

      {/* HRI Trajectory Chart */}
      <div className="panel chart-panel">
        <h2>HRI Trajectory (per token)</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
            <XAxis
              dataKey="index"
              label={{ value: "Token Index", position: "insideBottom", offset: -5, fill: "#9ca3af" }}
              stroke="#9ca3af"
            />
            <YAxis
              domain={[0, 1]}
              label={{ value: "HRI", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
              stroke="#9ca3af"
            />
            <Tooltip
              contentStyle={{ background: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: 8 }}
              labelFormatter={(v) => `Token ${v}`}
            />
            {/* Threshold lines */}
            <ReferenceLine y={0.35} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "MED", fill: "#f59e0b", fontSize: 11 }} />
            <ReferenceLine y={0.46} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "HIGH", fill: "#ef4444", fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="hri"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#818cf8" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Token Heatmap */}
      <div className="panel heatmap-panel">
        <h2>Token Risk Heatmap</h2>
        <TokenHeatmap tokens={data.token_details} />
      </div>
    </div>
  );
}
