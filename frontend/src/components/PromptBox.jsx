/**
 * PromptBox – User input area with an Analyze button.
 *
 * Props:
 *   onAnalyze(prompt) – callback fired when the user clicks Analyze.
 *   loading           – disables the button while a request is in flight.
 */

import React, { useState } from "react";

export default function PromptBox({ onAnalyze, loading }) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim()) {
      onAnalyze(prompt.trim());
    }
  };

  return (
    <form className="prompt-box" onSubmit={handleSubmit}>
      <textarea
        className="prompt-input"
        rows={4}
        placeholder="Enter your prompt here…"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={loading}
      />
      <button className="analyze-btn" type="submit" disabled={loading || !prompt.trim()}>
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </form>
  );
}
