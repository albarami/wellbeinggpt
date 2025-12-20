"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { AskMode, AskUiResponse, ChunkResponse, EdgeEvidenceResponse } from "@/lib/types";

type ChatTurn = {
  role: "user" | "assistant";
  text: string;
};

async function postAskUi(question: string, mode: AskMode): Promise<AskUiResponse> {
  const resp = await fetch("/api/ask/ui", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language: "ar", mode, engine: "muhasibi" }),
  });
  const text = await resp.text();
  if (!resp.ok) throw new Error(text || `HTTP ${resp.status}`);
  return JSON.parse(text) as AskUiResponse;
}

async function fetchChunk(chunkId: string): Promise<ChunkResponse> {
  const resp = await fetch(`/api/chunk/${encodeURIComponent(chunkId)}`, { cache: "no-store" });
  const text = await resp.text();
  if (!resp.ok) throw new Error(text || `HTTP ${resp.status}`);
  return JSON.parse(text) as ChunkResponse;
}

async function fetchEdgeEvidence(edgeId: string): Promise<EdgeEvidenceResponse> {
  const resp = await fetch(`/api/graph/edge/${encodeURIComponent(edgeId)}/evidence`, { cache: "no-store" });
  const text = await resp.text();
  if (!resp.ok) throw new Error(text || `HTTP ${resp.status}`);
  return JSON.parse(text) as EdgeEvidenceResponse;
}

// Icons as components
const IconChat = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
);

const IconEvidence = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const IconGraph = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
  </svg>
);

const IconSettings = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const IconFlow = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
  </svg>
);

const IconSend = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

const IconClose = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const IconChevron = () => (
  <svg className="w-4 h-4 chevron transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const IconThumbUp = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
  </svg>
);

const IconThumbDown = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
  </svg>
);

// Drawer component
function Drawer(props: { title: string; open: boolean; onClose: () => void; children: React.ReactNode }) {
  if (!props.open) return null;
  return (
    <div className="fixed inset-0 z-[100]">
      <div className="drawer-overlay absolute inset-0" onClick={props.onClose} />
      <div className="drawer-panel absolute inset-y-0 left-0 w-full max-w-2xl flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-teal-600 to-teal-700">
          <h3 className="text-lg font-semibold text-white">{props.title}</h3>
          <button onClick={props.onClose} className="p-2 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition-colors">
            <IconClose />
          </button>
        </div>
        <div className="flex-1 overflow-auto p-6 bg-slate-50">{props.children}</div>
      </div>
    </div>
  );
}

export default function ChatClient() {
  const [mode, setMode] = useState<AskMode>("natural_chat");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [run, setRun] = useState<AskUiResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [chunkId, setChunkId] = useState<string | null>(null);
  const [chunk, setChunk] = useState<ChunkResponse | null>(null);
  const [edgeId, setEdgeId] = useState<string | null>(null);
  const [edgeEvidence, setEdgeEvidence] = useState<EdgeEvidenceResponse | null>(null);

  const submit = useCallback(async () => {
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setError(null);
    setTurns((t) => [...t, { role: "user", text: q }]);
    setQuestion("");
    try {
      const res = await postAskUi(q, mode);
      setRun(res);
      setTurns((t) => [...t, { role: "assistant", text: res.final.answer_ar }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [question, mode, busy]);

  useEffect(() => {
    if (!chunkId) return;
    setChunk(null);
    fetchChunk(chunkId).then(setChunk).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [chunkId]);

  useEffect(() => {
    if (!edgeId) return;
    setEdgeEvidence(null);
    fetchEdgeEvidence(edgeId).then(setEdgeEvidence).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [edgeId]);

  const evidenceItems = useMemo(() => run?.citations_spans ?? [], [run]);
  const usedEdges = useMemo(() => run?.graph_trace.used_edges ?? [], [run]);
  const argChains = useMemo(() => run?.graph_trace.argument_chains ?? [], [run]);
  const trace = useMemo(() => run?.muhasibi_trace ?? [], [run]);

  const sendFeedback = useCallback(async (rating: number) => {
    if (!run?.request_id) return;
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: run.request_id, rating }),
    });
  }, [run?.request_id]);

  const getContractBadge = () => {
    if (!run) return null;
    const outcome = run.contract_outcome;
    if (outcome === "PASS_FULL") return <span className="badge badge-success">✓ إجابة كاملة</span>;
    if (outcome === "PASS_PARTIAL") return <span className="badge badge-warning">◐ إجابة جزئية</span>;
    return <span className="badge badge-error">✗ فشل</span>;
  };

  return (
    <div className="grid gap-6 lg:grid-cols-12">
      {/* Chat Panel - Main */}
      <section className="lg:col-span-5">
        <div className="card overflow-hidden h-[calc(100vh-200px)] flex flex-col">
          {/* Header */}
          <div className="panel-header">
            <IconChat />
            <span>الحوار</span>
            <div className="mr-auto flex items-center gap-3">
              <div className="flex items-center gap-1 bg-white/20 rounded-lg p-1">
                <button
                  onClick={() => setMode("answer")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    mode === "answer" ? "bg-white text-teal-700 shadow" : "text-white/80 hover:text-white"
                  }`}
                >
                  صارم
                </button>
                <button
                  onClick={() => setMode("natural_chat")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    mode === "natural_chat" ? "bg-white text-teal-700 shadow" : "text-white/80 hover:text-white"
                  }`}
                >
                  طبيعي
                </button>
              </div>
            </div>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-auto p-4 space-y-4 bg-gradient-to-b from-slate-50 to-white">
            {turns.length === 0 ? (
              <div className="empty-state h-full">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-100 to-teal-200 flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-slate-700 mb-2">أهلاً بك في المُحاسبي</h3>
                <p className="text-sm text-slate-500 max-w-sm">
                  اسأل أي سؤال متعلق بإطار الحياة الطيبة، وسأجيبك بإجابة مؤصّلة مع الأدلة والمصادر.
                </p>
              </div>
            ) : (
              turns.map((t, idx) => (
                <div key={idx} className={`flex ${t.role === "user" ? "justify-start" : "justify-end"}`}>
                  <div className={t.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"}>
                    <div className="text-xs mb-2 opacity-70">
                      {t.role === "user" ? "أنت" : "المُحاسبي"}
                    </div>
                    <div className="whitespace-pre-wrap leading-relaxed">{t.text}</div>
                  </div>
                </div>
              ))
            )}
            {busy && (
              <div className="flex justify-end">
                <div className="chat-bubble-assistant">
                  <div className="flex items-center gap-3">
                    <div className="spinner" />
                    <span className="text-sm text-slate-500">جارٍ التفكير...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-slate-200 bg-white">
            {error && (
              <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
                {error}
              </div>
            )}
            <div className="flex gap-3">
              <input
                className="input flex-1"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="اكتب سؤالك هنا..."
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                  }
                }}
                disabled={busy}
              />
              <button className="btn-primary flex items-center gap-2" onClick={submit} disabled={busy}>
                <IconSend />
                <span>إرسال</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Evidence Panel */}
      <section className="lg:col-span-3">
        <div className="card overflow-hidden h-[calc(100vh-200px)] flex flex-col">
          <div className="panel-header-secondary flex items-center gap-2">
            <IconEvidence />
            <span>الأدلة والاقتباسات</span>
            {evidenceItems.length > 0 && (
              <span className="mr-auto bg-white/20 px-2 py-0.5 rounded-full text-xs">
                {evidenceItems.length}
              </span>
            )}
          </div>
          <div className="flex-1 overflow-auto p-4 space-y-3">
            {evidenceItems.length === 0 ? (
              <div className="empty-state h-full">
                <IconEvidence />
                <p className="text-sm mt-3">ستظهر الأدلة هنا بعد السؤال</p>
              </div>
            ) : (
              evidenceItems.map((c, idx) => (
                <div key={idx} className="evidence-card">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="badge badge-neutral text-xs">{c.chunk_id}</span>
                    <button onClick={() => setChunkId(c.chunk_id)} className="btn-ghost text-xs">
                      فتح المقطع ←
                    </button>
                  </div>
                  <p className="text-sm leading-7 text-slate-700">{c.quote || "(لم يُحدد النص)"}</p>
                  <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
                    <span className={`status-dot ${c.span_resolution_status === "resolved" ? "status-dot-success" : "status-dot-warning"}`} />
                    <span>{c.span_resolution_status === "resolved" ? "محدد" : "غير محدد"}</span>
                    <span className="mx-1">·</span>
                    <span>{c.span_resolution_method}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      {/* Graph & Quality Panel */}
      <section className="lg:col-span-4">
        <div className="card overflow-hidden h-[calc(100vh-200px)] flex flex-col">
          <div className="panel-header-secondary flex items-center gap-2">
            <IconGraph />
            <span>الخريطة والحجج</span>
          </div>
          <div className="flex-1 overflow-auto p-4 space-y-4">
            {/* Used Edges */}
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-teal-500" />
                الروابط المستخدمة
              </h4>
              {usedEdges.length === 0 ? (
                <p className="text-sm text-slate-400">لا توجد روابط</p>
              ) : (
                <div className="space-y-2">
                  {usedEdges.map((e) => (
                    <button key={e.edge_id} onClick={() => setEdgeId(e.edge_id)} className="edge-card w-full text-right">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="badge badge-info text-xs">{e.relation_type}</span>
                      </div>
                      <p className="text-sm text-slate-600">
                        {e.from_node} → {e.to_node}
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Argument Chains */}
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                سلاسل الحجج
              </h4>
              {argChains.length === 0 ? (
                <p className="text-sm text-slate-400">لا توجد سلاسل</p>
              ) : (
                <div className="space-y-2">
                  {argChains.map((a) => (
                    <div key={a.edge_id} className="bg-amber-50 border border-amber-200 rounded-xl p-3">
                      <p className="text-sm font-medium text-amber-800 mb-1">{a.claim_ar}</p>
                      <span className="badge badge-warning text-xs">{a.inference_type}</span>
                      {a.boundary_ar && (
                        <p className="mt-2 text-xs text-amber-600 border-t border-amber-200 pt-2">
                          الحد: {a.boundary_ar}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quality / Contract Panel */}
            <details className="bg-slate-50 border border-slate-200 rounded-xl overflow-hidden">
              <summary className="px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-slate-100 transition-colors">
                <div className="flex items-center gap-2">
                  <IconSettings />
                  <span className="text-sm font-semibold text-slate-700">الجودة والعقد</span>
                </div>
                <div className="flex items-center gap-2">
                  {getContractBadge()}
                  <IconChevron />
                </div>
              </summary>
              <div className="px-4 py-3 border-t border-slate-200 bg-white">
                {run ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="bg-slate-50 rounded-lg p-3">
                        <div className="text-xs text-slate-500 mb-1">النتيجة</div>
                        <div className="font-semibold text-slate-700">{run.contract_outcome}</div>
                      </div>
                      <div className="bg-slate-50 rounded-lg p-3">
                        <div className="text-xs text-slate-500 mb-1">زمن الاستجابة</div>
                        <div className="font-semibold text-slate-700">{run.latency_ms} مللي ثانية</div>
                      </div>
                    </div>
                    {run.contract_reasons?.length ? (
                      <div>
                        <div className="text-xs text-slate-500 mb-2">الأسباب</div>
                        <ul className="space-y-1">
                          {run.contract_reasons.map((r, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                              <span className="text-amber-500 mt-1">•</span>
                              {r}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {run.abstain_reason && (
                      <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div className="text-xs text-red-500 mb-1">سبب الامتناع</div>
                        <p className="text-sm text-red-700">{run.abstain_reason}</p>
                      </div>
                    )}
                    <div className="text-xs text-slate-400 font-mono">
                      {run.request_id}
                    </div>
                    <div className="flex gap-2 pt-2 border-t border-slate-100">
                      <button onClick={() => sendFeedback(1)} className="btn-secondary flex items-center gap-2 flex-1 justify-center">
                        <IconThumbUp />
                        <span>مفيد</span>
                      </button>
                      <button onClick={() => sendFeedback(-1)} className="btn-secondary flex items-center gap-2 flex-1 justify-center">
                        <IconThumbDown />
                        <span>غير مفيد</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400 text-center py-4">اسأل سؤالاً لرؤية التفاصيل</p>
                )}
              </div>
            </details>

            {/* Muhasibi Flow Panel */}
            <details className="bg-gradient-to-br from-teal-50 to-emerald-50 border border-teal-200 rounded-xl overflow-hidden" open>
              <summary className="px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-teal-100/50 transition-colors">
                <div className="flex items-center gap-2">
                  <IconFlow />
                  <span className="text-sm font-semibold text-teal-800">مسار تفكير المُحاسبي</span>
                </div>
                <div className="flex items-center gap-2">
                  {trace.length > 0 && (
                    <span className="badge badge-success">{trace.length} خطوة</span>
                  )}
                  <IconChevron />
                </div>
              </summary>
              <div className="px-4 py-3 border-t border-teal-200 bg-white/50">
                {trace.length === 0 ? (
                  <p className="text-sm text-teal-600 text-center py-4">سيظهر مسار التفكير هنا</p>
                ) : (
                  <div className="space-y-0 relative">
                    {trace.map((t, i) => {
                      // State-specific display configuration
                      const stateConfig: Record<string, { label: string; icon: string; description: string }> = {
                        LISTEN: { label: "الاستماع", icon: "👂", description: "فهم السؤال وتحديد الكيانات" },
                        PURPOSE: { label: "الغاية", icon: "🎯", description: "تحديد هدف الإجابة" },
                        PATH: { label: "المسار", icon: "🛤️", description: "تخطيط خطوات الإجابة" },
                        RETRIEVE: { label: "الاسترجاع", icon: "📚", description: "البحث في قاعدة المعرفة" },
                        ACCOUNT: { label: "المحاسبة", icon: "⚖️", description: "تقييم كفاية الأدلة" },
                        INTERPRET: { label: "التفسير", icon: "💡", description: "صياغة الإجابة من الأدلة" },
                        REFLECT: { label: "التأمل", icon: "🔍", description: "مراجعة وتحسين الإجابة" },
                        FINALIZE: { label: "الإتمام", icon: "✅", description: "إنهاء الإجابة" },
                      };
                      const cfg = stateConfig[t.state] || { label: t.state, icon: "📌", description: "" };
                      
                      // State-specific fields to display
                      const getStateFields = () => {
                        switch (t.state) {
                          case "LISTEN":
                            return [
                              { label: "الكيانات المكتشفة", value: t.detected_entities_count, icon: "🏷️" },
                              { label: "الكلمات المفتاحية", value: t.keywords_count, icon: "🔑" },
                            ];
                          case "PURPOSE":
                            return [
                              { label: "الهدف", value: t.ultimate_goal_ar ? "✓ محدد" : "—", icon: "🎯" },
                              { label: "القيود", value: t.constraints_count, icon: "📋" },
                            ];
                          case "RETRIEVE":
                            return [
                              { label: "حزم الأدلة", value: t.evidence_packets_count, icon: "📦" },
                              { label: "وُجد تعريف", value: t.has_definition ? "✓" : "✗", icon: "📖" },
                              { label: "وُجدت أدلة", value: t.has_evidence ? "✓" : "✗", icon: "📜" },
                            ];
                          case "ACCOUNT":
                            return [
                              { label: "غير موجود", value: t.not_found ? "✗ نعم" : "✓ لا", icon: t.not_found ? "⚠️" : "✅" },
                            ];
                          case "INTERPRET":
                            return [
                              { label: "الاقتباسات", value: t.citations_count, icon: "📝" },
                              { label: "الثقة", value: t.confidence || "—", icon: "📊" },
                              { label: "غير موجود", value: t.not_found ? "✗" : "✓", icon: t.not_found ? "⚠️" : "✅" },
                            ];
                          case "REFLECT":
                            return [
                              { label: "أُضيف تأمل", value: t.reflection_added ? "✓ نعم" : "✗ لا", icon: "🔍" },
                            ];
                          default:
                            return [];
                        }
                      };
                      const fields = getStateFields();
                      const isActive = t.elapsed_s > 0.01;
                      
                      return (
                        <div key={i} className={`trace-step mb-4 ${!isActive ? "opacity-60" : ""}`}>
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{cfg.icon}</span>
                              <div>
                                <span className="text-sm font-semibold text-teal-700">{cfg.label}</span>
                                <p className="text-xs text-teal-500">{cfg.description}</p>
                              </div>
                            </div>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${isActive ? "text-teal-600 bg-teal-100" : "text-slate-400 bg-slate-100"}`}>
                              {t.elapsed_s.toFixed(3)} ث
                            </span>
                          </div>
                          {fields.length > 0 && (
                            <div className={`grid gap-2 text-xs ${fields.length <= 2 ? "grid-cols-2" : "grid-cols-3"}`}>
                              {fields.map((f, fi) => (
                                <div key={fi} className="bg-white rounded-lg p-2 border border-teal-100 flex items-center gap-1">
                                  <span>{f.icon}</span>
                                  <span className="text-teal-500">{f.label}:</span>
                                  <span className="mr-1 font-medium text-slate-700">{f.value ?? "—"}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          {t.issues?.length ? (
                            <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-lg">
                              <div className="text-xs text-amber-600 font-medium mb-1">⚠️ المشكلات:</div>
                              <ul className="space-y-0.5">
                                {t.issues.slice(0, 4).map((x, j) => (
                                  <li key={j} className="text-xs text-amber-700">• {x}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </details>
          </div>
        </div>
      </section>

      {/* Chunk Drawer */}
      <Drawer title={`المقطع: ${chunkId ?? ""}`} open={!!chunkId} onClose={() => setChunkId(null)}>
        {!chunk ? (
          <div className="flex items-center justify-center h-32">
            <div className="spinner" />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="badge badge-info">{chunk.entity_type}</span>
              <span className="badge badge-neutral">{chunk.entity_id}</span>
              <span className="badge badge-neutral">{chunk.chunk_type}</span>
            </div>
            <div className="bg-white rounded-xl p-4 border border-slate-200">
              <p className="text-base leading-8 text-slate-700 whitespace-pre-wrap">{chunk.text_ar}</p>
            </div>
            {chunk.refs?.length ? (
              <div>
                <h4 className="text-sm font-semibold text-slate-700 mb-2">المراجع</h4>
                <ul className="space-y-2">
                  {chunk.refs.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="badge badge-info">{r.ref_type}</span>
                      <span className="text-slate-600">{r.ref}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </Drawer>

      {/* Edge Evidence Drawer */}
      <Drawer title={`أدلة الرابط: ${edgeId ?? ""}`} open={!!edgeId} onClose={() => setEdgeId(null)}>
        {!edgeEvidence ? (
          <div className="flex items-center justify-center h-32">
            <div className="spinner" />
          </div>
        ) : edgeEvidence.spans.length === 0 ? (
          <div className="empty-state h-32">
            <p className="text-sm">لا توجد نصوص مرتبطة</p>
          </div>
        ) : (
          <div className="space-y-3">
            {edgeEvidence.spans.map((s, i) => (
              <div key={i} className="evidence-card">
                <div className="flex items-center justify-between mb-2">
                  <span className="badge badge-neutral">{s.chunk_id}</span>
                  <button onClick={() => setChunkId(s.chunk_id)} className="btn-ghost text-xs">
                    فتح المقطع ←
                  </button>
                </div>
                <p className="text-sm leading-7 text-slate-700">{s.quote}</p>
              </div>
            ))}
          </div>
        )}
      </Drawer>
    </div>
  );
}
