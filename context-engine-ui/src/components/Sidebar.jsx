import React, { useState } from "react";
import {
  Github,
  Loader2,
  AlertCircle,
  Cpu,
  Network,
} from "lucide-react";

export default function Sidebar({ sessionId, activeRoute, setMessages }) {
  const [repoUrl, setRepoUrl] = useState("");
  const [status, setStatus] = useState({ type: "idle", message: "" });

  const handleGitIndex = async (e) => {
    e.preventDefault();
    const targetUrl = repoUrl.trim();
    if (!targetUrl || status.type === "loading") return;

    setStatus({
      type: "loading",
      message: "Cloning repo and processing embedding layers...",
    });
    try {
      const response = await fetch("http://localhost:8000/api/index-git", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_url: targetUrl, branch: "main" }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.detail || "Ingestion gateway error.");

      setStatus({
        type: "success",
        message: data.message || "Vector index successfully processed!",
      });
      setRepoUrl("");

      setMessages((prev) => [
        ...prev,
        {
          role: "human",
          content: `Connect and index repository: ${targetUrl}`,
        },
        {
          role: "assistant",
          content: `Successfully vectorized and indexed the codebase from ${targetUrl} into Qdrant index. I am now fully grounded in this project's architecture. What would you like to analyze or explore?`,
          route: "vector_db",
        },
      ]);
    } catch (err) {
      setStatus({ type: "error", message: `Refusal: ${err.message}` });
    }
  };

  return (
    <div className="w-80 bg-slate-900 border-r border-slate-800 p-6 flex flex-col justify-between h-full text-slate-200 select-none">
      <div className="space-y-6">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Cpu size={16} />
            </div>
            <h2 className="text-base font-bold text-white tracking-tight">
              ContextEngine
            </h2>
          </div>
          <p className="text-[10px] text-slate-500 font-mono mt-1 uppercase tracking-wider">
            Multi-Agent RAG System
          </p>
        </div>

        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800/60 font-mono text-xs space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-slate-500 flex items-center gap-1">
              <Network size={12} /> Node Active:
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${activeRoute === "vector_db" ? "bg-emerald-950/40 text-emerald-400 border border-emerald-900/30" : "bg-slate-800 text-slate-400"}`}
            >
              {activeRoute}
            </span>
          </div>
        </div>

        <div className="border-t border-slate-800/80 pt-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
            <Github size={14} /> Connect Workspace
          </h3>
          <form onSubmit={handleGitIndex} className="space-y-3">
            <input
              type="url"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder="https://github.com..."
              required
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs font-mono"
              disabled={status.type === "loading"}
            />
            <button
              type="submit"
              className="w-full bg-emerald-500 text-slate-950 rounded-xl py-2.5 text-xs font-semibold hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-600 transition-all duration-150"
            >
              {status.type === "loading"
                ? "Vectorizing Code..."
                : "Index Git Repository"}
            </button>
          </form>

          {status.type !== "idle" && (
            <div
              className={`mt-4 p-3 rounded-xl border text-[11px] font-mono flex gap-2 ${status.type === "loading" ? "bg-slate-950 text-slate-400 border-slate-800 animate-pulse" : status.type === "success" ? "bg-emerald-950/20 text-emerald-300 border-emerald-900/30" : "bg-rose-950/20 text-rose-300 border-rose-900/30"}`}
            >
              {status.type === "loading" && (
                <Loader2
                  size={13}
                  className="animate-spin text-emerald-400 shrink-0 mt-0.5"
                />
              )}
              <span>{status.message}</span>
            </div>
          )}
        </div>
      </div>
      <div className="text-[10px] text-slate-600 font-mono">
        v1.2.0 • Local Production Host
      </div>
    </div>
  );
}
