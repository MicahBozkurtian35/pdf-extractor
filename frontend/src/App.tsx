import React, { useState } from "react";
import "./App.css";
import {
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter
} from "recharts";

type Scalar = string | number | boolean | null;

interface TableRow { [key: string]: Scalar; }

interface TableEntry {
  page: number;
  region: number;
  image: string;         // enhanced crop filename from backend
  data: TableRow[];      // parsed table/series rows
  note?: string | null;
  chart_type?: string | null;   // "line" | "bar" | "stacked_bar" | "scatter" | ...
  category?: string | null;
  series_hints?: string[];      // optional, may be undefined
}

interface DebugEntry { page: number; image: string; raw: string; }
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

function ChartFromData({ rows, chartType }: { rows: TableRow[]; chartType?: string | null }) {
  if (!rows || rows.length === 0) return null;

  const cols = Object.keys(rows[0] ?? {});
  const numericCols = cols.filter((c) =>
    rows.every((r) => r[c] === null || r[c] === "" || !isNaN(Number(r[c])))
  );
  const xKey = cols.find((c) => !numericCols.includes(c)) ?? cols[0];
  const yKeys = numericCols.filter((c) => c !== xKey);

  let type = (chartType || "").toLowerCase();
  if (!type) {
    if (numericCols.length === 2 && numericCols.includes("x") && numericCols.includes("y")) type = "scatter";
    else if (yKeys.length >= 2) type = "bar";
    else type = "line";
  }

  if (yKeys.length === 0) {
    return <div className="muted">Not enough numeric columns to render a chart.</div>;
  }

  // High-contrast palette and special-series styling
  const palette = ["#2f81f7","#ef4444","#22c55e","#f59e0b","#a855f7","#14b8a6","#eab308","#f97316"];
  const styleFor = (name: string, i: number) => {
    const n = (name || "").toLowerCase();
    if (n.includes("current")) return { stroke: "#ef4444", strokeWidth: 2.5 };
    if (n.includes("avg") || n.includes("average")) return { stroke: "#9ca3af", strokeDasharray: "5 5", strokeWidth: 2 };
    if (n.includes("sd")) return { stroke: "#cbd5e1", strokeDasharray: "4 6", strokeWidth: 2 };
    return { stroke: palette[i % palette.length], strokeWidth: 2 };
  };

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        {type === "scatter" ? (
          <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey === "x" ? "x" : xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis dataKey={yKeys[0] ?? "y"} tick={{ fill: "#cbd5e1" }} />
            <Tooltip />
            <Legend />
            <Scatter data={rows} />
          </ScatterChart>
        ) : type === "bar" || type === "stacked_bar" ? (
          <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis tick={{ fill: "#cbd5e1" }} />
            <Tooltip />
            <Legend />
            {yKeys.map((k, i) => {
              const s = styleFor(String(k), i);
              return (
                <Bar
                  key={k}
                  dataKey={k}
                  stackId={type === "stacked_bar" ? "a" : undefined}
                  fill={s.stroke}
                />
              );
            })}
          </BarChart>
        ) : (
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fill: "#cbd5e1" }} />
            <YAxis tick={{ fill: "#cbd5e1" }} />
            <Tooltip />
            <Legend />
            {yKeys.map((k, i) => {
              const s = styleFor(String(k), i);
              return (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={k}
                  dot={false}
                  stroke={s.stroke}
                  strokeWidth={s.strokeWidth as number}
                  strokeDasharray={s.strokeDasharray as string | undefined}
                />
              );
            })}
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
      const fd = new FormData();
      fd.append("filename", serverFilename);
      fd.append("page_number", String(page));
      const res = await fetch(`${backend}/process_page`, { method: "POST", body: fd });
      const json = (await res.json()) as ProcessPageResponse & { error?: string };
      if (!res.ok) throw new Error(json.error || "Processing failed");
      setResult(json.data);
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
                  src={urlWithBust(`${backend}/images/thumbs/${encodeURIComponent(t)}`)}
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
            const cropUrl = urlWithBust(`${backend}/images/enhanced/${encodeURIComponent(t.image)}`);
            return (
              <div className="card" key={`${t.page}-${t.region}-${idx}`}>
                <div className="card-header">
                  <h3>Page {t.page} • Graph {t.region + 1}</h3>
                </div>

                <div className="row-2col">
                  <div className="image-block">
                    <div className="img-label">Cropped graph</div>
                    <img src={cropUrl} alt="Cropped graph" />
                    {t.chart_type ? <div className="muted" style={{ marginTop: 6 }}>Detected type: {t.chart_type}</div> : null}
                    {t.note ? <div className="muted" style={{ marginTop: 6 }}>{t.note}</div> : null}
                  </div>

                  <div className="image-block">
                    <div className="img-label">Remade from extracted data</div>
                    <ChartFromData rows={t.data} chartType={t.chart_type} />
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
