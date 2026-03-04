/**
 * HalluScope v2 – API Client
 *
 * Thin wrapper around the Fetch API for communicating with the
 * FastAPI backend.  All network calls live here so components
 * stay free of HTTP concerns.
 */

const API_BASE = "http://localhost:8000";

/**
 * Send a chat message and get the analysed response.
 */
export async function sendChatMessage(message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed");
  }
  return res.json();
}

/**
 * Get detailed metrics for a specific message.
 */
export async function getMetrics(messageId) {
  const res = await fetch(`${API_BASE}/metrics/${messageId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Metrics request failed");
  }
  return res.json();
}

/**
 * Reset conversation history.
 */
export async function resetChat() {
  const res = await fetch(`${API_BASE}/chat/reset`, { method: "POST" });
  if (!res.ok) throw new Error("Reset failed");
  return res.json();
}
