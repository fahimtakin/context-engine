import React, { useState } from "react";
import ChatConsole from "./components/ChatConsole";
import Sidebar from "./components/Sidebar";

export default function App() {
  const [sessionId, setSessionId] = useState("web-default-session");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "System connection established. ContextEngine Multi-Agent RAG interface online.",
      route: "system",
    },
  ]);
  const [activeRoute, setActiveRoute] = useState("unknown");

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100">
      {/* Dynamic Session Settings Sidebar */}
      <Sidebar
        sessionId={sessionId}
        setSessionId={setSessionId}
        activeRoute={activeRoute}
        setMessages={setMessages}
      />

      {/* Live Agent Conversation Console */}
      <ChatConsole
        sessionId={sessionId}
        messages={messages}
        setMessages={setMessages}
        setActiveRoute={setActiveRoute}
      />
    </div>
  );
}
