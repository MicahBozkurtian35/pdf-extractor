import React, { useState } from "react";
import "./App.css";
import {
  ResponsiveContainer,
  ComposedChart, Bar, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  LineChart, BarChart, ScatterChart, Scatter
} from "recharts";

type Scalar = string | number | boolean | null;

interface SeriesMeta { name?: string; axis?: "left" | "right" | "none"; render?: "bar" | "line" | "area" | string; }
interface TableRow { [key: string]: Scalar; }
interface TableEntry {
  page: number;
  region: number;
  image?: string;
  data: TableRow[];
  note?: string | null;
  chart_type?: string | null;
  category?: string | null;
  series_hints?: string[];
  confidence?: "low" | "medium" | "high";
  series_meta?: SeriesMeta[]; // from backend
}
interface DebugEntry { page: number; image?: string; raw?: string; raw_fix?: string; }
interface ProcessData { tables: TableEntry[]; debug_raw: DebugEntry[]; }

interface UploadResponse {
  message: string;
  filename: string;
  page_count: number;
  thumbnails: string[];
}
interface ProcessPageResponse { message: string; data: ProcessData; }

const backend = (import.meta as any).env?.VITE_BACKEND_URL || "http://localhost:5000";

function urlWithBust(path: string) {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}v=${Date.now()}`;
}

function DataTable({ rows }: { rows: TableRow[] }) {
  if (!rows || rows.length === 0) return <div className="muted">No data</div>;
  // Preserve backend column order; 'X' should be first
  const columns = Object.keys(rows[0] ?? {});
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => <td key={`${c}-${i}`}>{String(r[c] ?? "")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartFromData({ rows, chartType, seriesMeta }: { rows: TableRow[]; chartType?: string | null; seriesMeta?: SeriesMeta[] }) {
  if (!rows || rows.length === 0) return null;

  const cols = Object.keys(rows[0] ?? {});
  const xKey = cols.includes("X") ? "X" : (cols.find(c => rows.some(r => isNaN(Number(r[c])))) || cols[0]);

  // numeric columns (excluding X)
  const numericCols = cols.filter((c) =>
    c !== xKey && rows.every((r) => r[c] === null || r[c] === "" || !isNaN(Number(r[c])))
  );

  const meta = (seriesMeta || []).filter(m => m && m.name && numericCols.includes(String(m.name)));
  const useComposed =
    meta.length > 0 ||
    (chartType || "").toLowerCase() === "dual_axis" ||
    (chartType || "").toLowerCase() === "combo";

  const palette = ["#2f81f7","#ef4444","#22c55e","#f59e0b","#a855f7","#14b8a6","#eab308","#f97316"];
  const colorFor = (name: string, i: number) => palette[i % palette.length];

  if (useComposed) {
    // If no meta is present, default first series to bar, rest to line on left axis
    const completeMeta: SeriesMeta[] = meta.length
      ? meta
      : numericCols.map((n, i) => ({ name: n, axis: "left", render: i === 0 ? "bar" : "line" }));

    const anyRight = completeMeta.some(m => (m.axis || "left") === "right");
    return (
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis yAxisId="left" tick={{ fill: "#cbd5e1" }} />
            {anyRight && <YAxis yAxisId="right" orientation="right" tick={{ fill: "#cbd5e1" }} />}
            <Tooltip />
            <Legend />
            {completeMeta.map((m, i) => {
              const key = String(m.name);
              const axisId = (m.axis || "left") === "right" ? "right" : "left";
              const color = colorFor(key, i);
              const render = (m.render || "line").toLowerCase();
              if (render === "bar") {
                return <Bar key={key} yAxisId={axisId} dataKey={key} fill={color} />;
              }
              return <Line key={key} yAxisId={axisId} type="monotone" dataKey={key} stroke={color} dot={false} strokeWidth={2} />;
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // Fallbacks (non-meta)
  const yKeys = numericCols;
  if (yKeys.length === 0) {
    return <div className="muted">Not enough numeric columns to render a chart.</div>;
  }

  let type = (chartType || "").toLowerCase();
  if (!type) type = yKeys.length >= 2 ? "bar" : "line";

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        {type === "bar" ? (
          <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis tick={{ fill: "#cbd5e1" }} />
            <Tooltip />
            <Legend />
            {yKeys.map((k, i) => <Bar key={k} dataKey={k} fill={colorFor(k, i)} />)}
          </BarChart>
        ) : (
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis tick={{ fill: "#cbd5e1" }} />
            <Tooltip />
            <Legend />
            {yKeys.map((k, i) => <Line key={k} type="monotone" dataKey={k} dot={false} stroke={colorFor(k, i)} strokeWidth={2} />)}
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [serverFilename, setServerFilename] = useState<string>("");
  const [pageCount, setPageCount] = useState<number>(0);
  const [thumbs, setThumbs] = useState<string[]>([]);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);

  const [result, setResult] = useState<ProcessData | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [processing, setProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setServerFilename("");
    setPageCount(0);
    setThumbs([]);
    setSelectedPage(null);
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    setError("");
    if (!file) { setError("Choose a PDF first."); return; }
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${backend}/upload`, { method: "POST", body: fd });
      const json = (await res.json()) as UploadResponse & { error?: string };
      if (!res.ok) throw new Error(json.error || "Upload failed");
      setServerFilename(json.filename);
      setPageCount(json.page_count);
      setThumbs(json.thumbnails);
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const processPage = async (page: number) => {
    setError("");
    if (!serverFilename) { setError("Upload a PDF first."); return; }
    setProcessing(true);
    setSelectedPage(page);
    setResult(null);
    try {
      const res = await fetch(`${backend}/process_page`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: serverFilename, page })
      });
      const json = (await res.json()) as ProcessPageResponse & { error?: string };
      if (!res.ok) throw new Error(json.error || "Processing failed");

      const sorted = {
        ...json.data,
        tables: [...(json.data?.tables ?? [])].sort((a, b) =>
          a.page !== b.page ? a.page - b.page : a.region - b.region
        )
      } as ProcessData;

      setResult(sorted);
    } catch (e: any) {
      setError(e?.message || "Processing failed");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="container">
      <h1>PDF Graph Extractor</h1>

      <div className="panel">
        <label className="file-picker">
          <span>Choose PDF</span>
          <input type="file" accept="application/pdf" onChange={onFileChange} />
        </label>
        <button onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {serverFilename && (
        <div className="panel" style={{ display: "block" }}>
          <div className="info" style={{ marginBottom: 12 }}>
            <div>Uploaded: <strong>{serverFilename}</strong></div>
            <div>Pages: <strong>{pageCount}</strong></div>
          </div>
          <div className="thumb-grid">
            {thumbs.map((t, i) => (
              <button
                key={t}
                onClick={() => processPage(i + 1)}
                className={selectedPage === (i + 1) ? "thumb-btn active" : "thumb-btn"}
                title={`Page ${i + 1}`}
              >
                <img
                  src={urlWithBust(`${backend}/thumbnail/${encodeURIComponent(t)}`)}
                  alt={`Page ${i + 1}`}
                />
                <div className="thumb-caption">Page {i + 1}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {processing && <div className="panel">Processing page {selectedPage}…</div>}

      {result?.tables?.length ? (
        <div className="results">
          {result.tables.map((t, idx) => {
            const cropUrl = t.image ? urlWithBust(`${backend}/images/enhanced/${encodeURIComponent(t.image)}`) : "";
            return (
              <div className="card" key={`${t.page}-${t.region}-${idx}`}>
                <div className="card-header">
                  <h3>
                    Page {t.page} • Graph {t.region + 1}
                    {t.chart_type ? <span className="muted" style={{ marginLeft: 10 }}>({t.chart_type})</span> : null}
                    {t.confidence ? <span className="muted" style={{ marginLeft: 10 }}>confidence: {t.confidence}</span> : null}
                  </h3>
                </div>

                <div className="row-2col">
                  <div className="image-block">
                    <div className="img-label">Cropped graph</div>
                    {cropUrl ? <img src={cropUrl} alt="Cropped graph" /> : <div className="muted">No crop preview</div>}
                    {t.note ? <div className="muted" style={{ marginTop: 6 }}>{t.note}</div> : null}
                  </div>

                  <div className="image-block">
                    <div className="img-label">Remade from extracted data</div>
                    <ChartFromData rows={t.data} chartType={t.chart_type} seriesMeta={t.series_meta} />
                  </div>
                </div>

                <div className="table-block">
                  <DataTable rows={t.data} />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        selectedPage && !processing && <div className="empty">No graphs detected or no tables parsed.</div>
      )}
    </div>
  );
}
