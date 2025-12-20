"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Graph from "graphology";
import type { EdgeEvidenceResponse, GraphExpandResponse } from "@/lib/types";

type SigmaClass = typeof import("sigma").default;
type SigmaInstance = InstanceType<SigmaClass>;

function nodeKey(t: string, id: string) {
  return `${t}:${id}`;
}

function stablePoint(key: string): { x: number; y: number } {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const a = ((h >>> 0) % 1000) / 1000;
  const b = (((h >>> 0) / 1000) % 1000) / 1000;
  return { x: a * 10 - 5, y: b * 10 - 5 };
}

// Beautiful pillar colors
const PILLAR_COLORS: Record<string, { bg: string; text: string; glow: string }> = {
  P001: { bg: "#3b82f6", text: "#1e40af", glow: "rgba(59, 130, 246, 0.3)" },
  P002: { bg: "#ec4899", text: "#be185d", glow: "rgba(236, 72, 153, 0.3)" },
  P003: { bg: "#8b5cf6", text: "#6d28d9", glow: "rgba(139, 92, 246, 0.3)" },
  P004: { bg: "#22c55e", text: "#15803d", glow: "rgba(34, 197, 94, 0.3)" },
  P005: { bg: "#f97316", text: "#c2410c", glow: "rgba(249, 115, 22, 0.3)" },
};

function colorForPillar(pillarId?: string | null): string {
  const pid = (pillarId ?? "").toUpperCase();
  return PILLAR_COLORS[pid]?.bg ?? "#64748b";
}

function getPillarInfo(pillarId?: string | null) {
  const pid = (pillarId ?? "").toUpperCase();
  return PILLAR_COLORS[pid] ?? { bg: "#64748b", text: "#475569", glow: "rgba(100, 116, 139, 0.3)" };
}

async function fetchExpand(params: Record<string, string>): Promise<GraphExpandResponse> {
  const url = new URL("/api/graph/expand", window.location.origin);
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const resp = await fetch(url.toString(), { cache: "no-store" });
  const text = await resp.text();
  if (!resp.ok) throw new Error(text || `HTTP ${resp.status}`);
  return JSON.parse(text) as GraphExpandResponse;
}

async function fetchEdgeEvidence(edgeId: string): Promise<EdgeEvidenceResponse> {
  const resp = await fetch(`/api/graph/edge/${encodeURIComponent(edgeId)}/evidence`, { cache: "no-store" });
  const text = await resp.text();
  if (!resp.ok) throw new Error(text || `HTTP ${resp.status}`);
  return JSON.parse(text) as EdgeEvidenceResponse;
}

// Icons
const IconSearch = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

const IconExpand = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
  </svg>
);

const IconInfo = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const IconEdge = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
  </svg>
);

export default function GraphClient() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sigmaRef = useRef<SigmaInstance | null>(null);
  const graphRef = useRef<Graph | null>(null);

  const [nodeType, setNodeType] = useState("pillar");
  const [nodeId, setNodeId] = useState("");
  const [depth, setDepth] = useState("2");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [edgeEvidence, setEdgeEvidence] = useState<EdgeEvidenceResponse | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!alive) return;
      if (!containerRef.current) return;
      if (sigmaRef.current) return;
      if (!graphRef.current) graphRef.current = new Graph();

      const mod = await import("sigma");
      const SigmaCtor = mod.default as unknown as SigmaClass;
      if (!alive) return;

      const sigma = new SigmaCtor(graphRef.current, containerRef.current, {
        allowInvalidContainer: false,
        renderLabels: true,
        labelDensity: 0.07,
        labelGridCellSize: 60,
        labelRenderedSizeThreshold: 6,
        defaultNodeColor: "#64748b",
        defaultEdgeColor: "#cbd5e1",
      });
      sigma.on("clickEdge", ({ edge }: { edge: string }) => {
        setEdgeEvidence(null);
        setSelectedEdgeId(String(edge));
      });
      sigmaRef.current = sigma;
    })().catch((e) => setError(e instanceof Error ? e.message : String(e)));

    return () => {
      alive = false;
      sigmaRef.current?.kill();
      sigmaRef.current = null;
      graphRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!selectedEdgeId) return;
    fetchEdgeEvidence(selectedEdgeId)
      .then(setEdgeEvidence)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [selectedEdgeId]);

  const applySubgraph = useCallback((data: GraphExpandResponse) => {
    const g = graphRef.current;
    if (!g) return;

    g.clear();

    for (const n of data.nodes) {
      const id = nodeKey(n.node_type, n.node_id);
      const { x, y } = stablePoint(id);
      const pillarInfo = getPillarInfo(n.pillar_id);
      g.addNode(id, {
        label: n.label_ar ?? id,
        x,
        y,
        size: n.node_type === "pillar" ? 12 : n.node_type === "core_value" ? 10 : 8,
        color: pillarInfo.bg,
        pillar_id: n.pillar_id ?? null,
        node_type: n.node_type,
        node_id: n.node_id,
      });
    }

    for (const e of data.edges) {
      const source = nodeKey(e.from_type, e.from_id);
      const target = nodeKey(e.to_type, e.to_id);
      if (!g.hasNode(source) || !g.hasNode(target)) continue;
      if (g.hasEdge(e.edge_id)) continue;
      g.addEdgeWithKey(e.edge_id, source, target, {
        label: e.relation_type ?? e.rel_type,
        size: e.has_evidence ? 2 : 1,
        color: e.has_evidence ? "#0d9488" : "#cbd5e1",
        has_evidence: e.has_evidence,
      });
    }

    sigmaRef.current?.refresh();
  }, []);

  const expand = useCallback(async () => {
    if (!nodeId.trim()) {
      setError("يرجى إدخال معرّف العقدة");
      return;
    }
    setError(null);
    setLoading(true);
    setStatus(null);
    try {
      const data = await fetchExpand({
        node_type: nodeType,
        node_id: nodeId.trim(),
        depth,
        grounded_only: "true",
      });
      applySubgraph(data);
      setStatus(
        `العقد: ${data.returned_nodes}/${data.total_nodes} · الروابط: ${data.returned_edges}/${data.total_edges}` +
          (data.truncated ? " · تم الاقتطاع" : "")
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [nodeType, nodeId, depth, applySubgraph]);

  const evidenceLines = useMemo(() => edgeEvidence?.spans ?? [], [edgeEvidence]);

  return (
    <div className="grid gap-6 lg:grid-cols-12 h-[calc(100vh-200px)]">
      {/* Control Panel */}
      <section className="lg:col-span-3">
        <div className="card overflow-hidden h-full flex flex-col">
          <div className="panel-header">
            <IconSearch />
            <span>مستكشف الخريطة</span>
          </div>
          <div className="flex-1 p-4 space-y-4">
            {/* Node Type */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">نوع العقدة</label>
              <select
                className="select"
                value={nodeType}
                onChange={(e) => setNodeType(e.target.value)}
              >
                <option value="pillar">ركيزة (Pillar)</option>
                <option value="core_value">قيمة جوهرية (Core Value)</option>
                <option value="sub_value">قيمة فرعية (Sub Value)</option>
              </select>
            </div>

            {/* Node ID */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">معرّف العقدة</label>
              <input
                className="input"
                value={nodeId}
                onChange={(e) => setNodeId(e.target.value)}
                placeholder="مثال: P001 أو CV001"
              />
            </div>

            {/* Depth */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">العمق</label>
              <div className="flex items-center gap-2">
                {["1", "2", "3"].map((d) => (
                  <button
                    key={d}
                    onClick={() => setDepth(d)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                      depth === d
                        ? "bg-teal-600 text-white shadow-md"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>

            {/* Expand Button */}
            <button className="btn-primary w-full flex items-center justify-center gap-2" onClick={expand} disabled={loading}>
              {loading ? (
                <div className="spinner" />
              ) : (
                <>
                  <IconExpand />
                  <span>استكشاف</span>
                </>
              )}
            </button>

            {/* Status */}
            {status && (
              <div className="p-3 bg-teal-50 border border-teal-200 rounded-xl text-sm text-teal-700 flex items-start gap-2">
                <IconInfo />
                <span>{status}</span>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
                {error}
              </div>
            )}

            {/* Legend */}
            <div className="mt-auto pt-4 border-t border-slate-200">
              <h4 className="text-xs font-semibold text-slate-500 mb-3">دليل الألوان</h4>
              <div className="space-y-2">
                {Object.entries(PILLAR_COLORS).map(([pid, colors]) => (
                  <div key={pid} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.bg }} />
                    <span className="text-xs text-slate-600">{pid}</span>
                  </div>
                ))}
                <div className="flex items-center gap-2 mt-3">
                  <div className="w-8 h-0.5 bg-teal-600" />
                  <span className="text-xs text-slate-600">رابط مُوثّق</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5 bg-slate-300" />
                  <span className="text-xs text-slate-600">رابط غير مُوثّق</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Graph Canvas */}
      <section className="lg:col-span-6">
        <div className="card overflow-hidden h-full flex flex-col">
          <div className="panel-header-secondary flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
            <span>الخريطة المعرفية</span>
            <span className="mr-auto text-xs text-slate-400">انقر على رابط لرؤية الأدلة</span>
          </div>
          <div ref={containerRef} className="flex-1 bg-gradient-to-br from-slate-50 to-slate-100" />
        </div>
      </section>

      {/* Edge Evidence Panel */}
      <section className="lg:col-span-3">
        <div className="card overflow-hidden h-full flex flex-col">
          <div className="panel-header-secondary flex items-center gap-2">
            <IconEdge />
            <span>أدلة الرابط</span>
          </div>
          <div className="flex-1 overflow-auto p-4">
            {!selectedEdgeId ? (
              <div className="empty-state h-full">
                <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center mb-3">
                  <IconEdge />
                </div>
                <p className="text-sm text-slate-500">انقر على رابط في الخريطة</p>
                <p className="text-xs text-slate-400 mt-1">لعرض الأدلة والنصوص المرتبطة</p>
              </div>
            ) : !edgeEvidence ? (
              <div className="flex items-center justify-center h-full">
                <div className="spinner" />
              </div>
            ) : evidenceLines.length === 0 ? (
              <div className="empty-state h-full">
                <p className="text-sm text-slate-500">لا توجد نصوص مرتبطة</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="p-3 bg-slate-100 rounded-xl mb-4">
                  <div className="text-xs text-slate-500 mb-1">معرّف الرابط</div>
                  <div className="text-sm font-mono text-slate-700 break-all">{selectedEdgeId}</div>
                </div>
                {evidenceLines.map((s, i) => (
                  <div key={i} className="evidence-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="badge badge-neutral text-xs">{s.chunk_id}</span>
                    </div>
                    <p className="text-sm leading-7 text-slate-700">{s.quote}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
