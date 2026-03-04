/**
 * ChatWindow – Renders the scrollable list of messages.
 *
 * Props:
 *   messages       – array of { id, role, content, risk_level, hri_score, ... }
 *   loading        – show typing indicator
 *   onViewMetrics  – callback(messageId) when "View Metrics" is clicked
 */

import React from "react";
import MessageBubble from "./MessageBubble";

export default function ChatWindow({ messages, loading, onViewMetrics }) {
  return (
    <div className="chat-window">
      {messages.length === 0 && !loading && (
        <div className="empty-state">
          <p className="empty-title">🔬 HalluScope Chat</p>
          <p className="empty-sub">
            Ask anything. Every response is monitored for hallucination risk.
          </p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          onViewMetrics={onViewMetrics}
        />
      ))}

      {loading && (
        <div className="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      )}
    </div>
  );
}
