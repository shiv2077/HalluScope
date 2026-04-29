/**
 * ChatPage – Main chatbot interface with hallucination monitoring.
 *
 * Maintains conversation history in state.  Every assistant reply
 * carries its HRI score, risk level, and a "View Metrics" link.
 */

import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { sendChatMessageWithMode, resetChat } from "../api";
import ChatWindow from "../components/ChatWindow";
import RiskBadge from "../components/RiskBadge";

const LOADING_MESSAGES = [
  "Calling model…",      // step 0  (0–2s)
  "Computing metrics…",  // step 1  (2–4s)
  "Analyzing risk…",     // step 2  (4s+)
];

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [evaluationMode, setEvaluationMode] = useState("api");
  const [loadingStep, setLoadingStep] = useState(0);
  const bottomRef = useRef(null);
  const navigate = useNavigate();

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Cycle the loading status message every 2s while a request is in flight.
  // Caps at step 2 so it shows "Analyzing risk…" indefinitely until the
  // response arrives (or an error fires) and resets to step 0 each turn.
  useEffect(() => {
    if (!loading) {
      setLoadingStep(0);
      return;
    }
    setLoadingStep(0);
    const id = setInterval(() => {
      setLoadingStep((prev) => Math.min(prev + 1, LOADING_MESSAGES.length - 1));
    }, 2000);
    return () => clearInterval(id);
  }, [loading]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const data = await sendChatMessageWithMode(text, evaluationMode);

      const assistantMsg = {
        id: data.message_id,
        role: "assistant",
        content: data.answer,
        risk_level: data.risk_level,
        hri_score: data.hri,
        intervention: data.intervention,
        metrics: data.metrics,
        model: data.model,
        evaluation_mode: data.evaluation_mode,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleReset = async () => {
    await resetChat();
    setMessages([]);
    setError(null);
  };

  return (
    <div className="chat-page">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          <h1>🔬 HalluScope</h1>
          <span className="version-badge">v2</span>
        </div>
        <div className="chat-header-right">
          <label className="mode-picker-label" htmlFor="mode-picker">Mode</label>
          <select
            id="mode-picker"
            className="mode-picker"
            value={evaluationMode}
            onChange={(e) => setEvaluationMode(e.target.value)}
            disabled={loading}
          >
            <option value="api">API</option>
            <option value="local">Local</option>
          </select>
          <button className="reset-btn" onClick={handleReset}>
            New Chat
          </button>
        </div>
      </header>

      {/* Messages */}
      <ChatWindow
        messages={messages}
        loading={loading}
        onViewMetrics={(id) => navigate(`/metrics/${id}`)}
      />
      <div ref={bottomRef} />

      {/* Error */}
      {error && <div className="error-banner">{error}</div>}

      {/* Loading status (cycling) — shown above the input row while a request is in flight */}
      {loading && (
        <div
          className="loading-status"
          style={{
            padding: "6px 12px",
            color: "#9ca3af",
            fontSize: 13,
            fontStyle: "italic",
          }}
        >
          {LOADING_MESSAGES[loadingStep]}
        </div>
      )}

      {/* Input */}
      <div className="chat-input-area">
        <textarea
          className="chat-input"
          rows={2}
          placeholder="Send a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          →
        </button>
      </div>
    </div>
  );
}
