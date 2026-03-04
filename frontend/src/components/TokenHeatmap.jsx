/**
 * TokenHeatmap – Visualises per-token risk as colour-coded spans.
 *
 * green  = confident (low risk contribution)
 * yellow = moderate uncertainty
 * red    = high uncertainty
 *
 * Hovering shows a tooltip with token probability, entropy,
 * and risk contribution.
 *
 * Props:
 *   tokens – array of { token, logprob, entropy, risk_contribution }
 */

import React, { useState } from "react";

function riskColor(contribution) {
  // Map risk_contribution to a green→yellow→red gradient
  // Typical range ~0–3, normalise to 0–1
  const t = Math.min(contribution / 2.5, 1);
  if (t < 0.4) {
    // green → yellow
    const r = Math.round(34 + (245 - 34) * (t / 0.4));
    const g = Math.round(197 + (158 - 197) * (t / 0.4));
    const b = Math.round(94 + (11 - 94) * (t / 0.4));
    return `rgb(${r},${g},${b})`;
  }
  // yellow → red
  const r2 = Math.round(245 + (239 - 245) * ((t - 0.4) / 0.6));
  const g2 = Math.round(158 - 158 * ((t - 0.4) / 0.6));
  const b2 = Math.round(11 + (68 - 11) * ((t - 0.4) / 0.6));
  return `rgb(${r2},${g2},${b2})`;
}

export default function TokenHeatmap({ tokens }) {
  const [hover, setHover] = useState(null);

  if (!tokens || tokens.length === 0) {
    return <p className="no-data">No token data available.</p>;
  }

  return (
    <div className="token-heatmap">
      {tokens.map((tok, i) => (
        <span
          key={i}
          className="heatmap-token"
          style={{
            backgroundColor: riskColor(tok.risk_contribution),
            color: tok.risk_contribution > 1.5 ? "#fff" : "#1a1d27",
          }}
          onMouseEnter={() => setHover(i)}
          onMouseLeave={() => setHover(null)}
        >
          {tok.token}
          {hover === i && (
            <span className="heatmap-tooltip">
              <strong>{tok.token}</strong>
              <br />
              Prob: {Math.exp(tok.logprob).toFixed(4)}
              <br />
              Entropy: {tok.entropy.toFixed(4)}
              <br />
              Risk: {tok.risk_contribution.toFixed(4)}
            </span>
          )}
        </span>
      ))}
    </div>
  );
}
