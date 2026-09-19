import React, { useState, useRef, useEffect } from "react";
import { Send, CornerDownLeft, Loader2 } from "lucide-react";

export default function ChatConsole({
  sessionId,
  messages,
  setMessages,
  setActiveRoute,
}) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const triggerAgentQuery = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: "human", content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input.trim();
    setInput("");
    setIsLoading(true);

    try {
      // Formatted into a complete loopback gateway URL pointing to FastAPI port
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: currentInput, // Synchronized with ChatRequest.message schema
          thread_id: sessionId, // Synchronized with ChatRequest.thread_id schema
        }),
      });

      if (!response.ok) throw new Error("Gateway Connection Interrupted.");

      const data = await response.json();

      setActiveRoute(data.router_decision);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          route: data.router_decision,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Network Failure Error: ${err.message}. Ensure your local API server is running on port 8000.`,
          route: "error",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      triggerAgentQuery(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-950 justify-between h-full">
      {/* Messages Feed View */}
      <div className="flex-1 overflow-y-auto px-12 py-8 space-y-6">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.role === "human" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-3xl rounded-2xl p-5 border text-sm leading-relaxed shadow-sm ${
                msg.role === "human"
                  ? "bg-emerald-600 border-emerald-500 text-white rounded-br-none"
                  : msg.route === "error"
                    ? "bg-rose-950/40 border-rose-900/50 text-rose-200 rounded-bl-none"
                    : "bg-slate-900/40 border-slate-800 text-slate-200 rounded-bl-none"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-900/40 border border-slate-800 rounded-2xl rounded-bl-none p-5 text-sm flex items-center gap-3 text-slate-400">
              <Loader2 size={16} className="animate-spin text-emerald-400" />
              <span className="font-mono text-xs">
                Agents evaluating execution choices...
              </span>
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* Input Action Form */}
      <form
        onSubmit={triggerAgentQuery}
        className="p-8 border-t border-slate-900 bg-slate-950"
      >
        <div className="relative bg-slate-900 rounded-2xl border border-slate-800 focus-within:border-emerald-500/40 transition-all duration-200 px-4 py-3.5 flex items-center shadow-inner">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question over internal architecture layout..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 focus:outline-none resize-none max-h-32 pr-12 font-sans py-1"
            disabled={isLoading}
          />
          <div className="absolute right-4 flex items-center gap-2">
            <span className="text-[10px] text-slate-500 font-mono hidden md:flex items-center gap-0.5">
              <span>Enter</span>
              <CornerDownLeft size={10} />
            </span>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="p-2 rounded-xl bg-emerald-500 text-slate-950 hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-600 transition-all duration-150"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
