/**
 * HalluScope v2 – Root Application
 *
 * Sets up React Router with two pages:
 *   /          → ChatPage (main chatbot interface)
 *   /metrics/:id → MetricsPage (detailed analytics)
 */

import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import MetricsPage from "./pages/MetricsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/metrics/:messageId" element={<MetricsPage />} />
      </Routes>
    </BrowserRouter>
  );
}
