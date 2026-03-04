/**
 * MessageBubble – A single chat message with optional risk info.
 *
 * User messages: plain bubble.
 * Assistant messages: bubble + RiskBadge + "View Metrics" link.
 */

import React from "react";
import RiskBadge from "./RiskBadge";

export default function MessageBubble({ message, onViewMetrics }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        {message.intervention && (
          <span className="intervention-tag">⚠️ Intervention</span>
        )}
        <p className="bubble-text">{message.content}</p>

        {!isUser && (
          <div className="bubble-footer">
            <RiskBadge riskLevel={message.risk_level} hri={message.hri_score} />
            <button
              className="view-metrics-btn"
              onClick={() => onViewMetrics(message.id)}
            >
              View Metrics →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
