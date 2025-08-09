import { useMemo, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

// ------------------ Types ------------------
interface TableData {
  image: string;
  data: Record<string, string | number>[];
  image_url?: string;
}

interface DebugRaw {
  image: string;
  raw: string;
  image_url?: string;
}

interface ExtractResult {
  tables: TableData[];
  debug_raw: DebugRaw[];
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ExtractResult | null>(null);

  const onUpload = async () => {
    setError("");
    setLoading(true);
    setResult(null);
    try {
      if (!file) {
        setError("Choose a PDF first.");
        setLoading(false);
        return;
      }

      const form = new FormData();
      form.append("file", file);

      const { data } = await axios.post<{ data: ExtractResult }>(
        `${API_BASE}/upload`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(data.data);
    } catch (e: any) {
      setError(
        e?.response?.data?.error || e?.message || "Upload failed. Check backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <header style={styles.header}>PDF Extractor</header>

      <section style={styles.controls}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button style={styles.button} disabled={loading} onClick={onUpload}>
          {loading ? "Processing..." : "Upload PDF"}
        </button>
      </section>

      {error && <div style={styles.error}>{error}</div>}

      {result && <ResultsPanel apiBase={API_BASE} result={result} />}
    </div>
  );
}

// ------------------ Results Panel ------------------
interface ResultsPanelProps {
  apiBase: string;
  result: ExtractResult;
}

function ResultsPanel({ apiBase, result }: ResultsPanelProps) {
  const tables = result?.tables || [];
  const raw = result?.debug_raw || [];

  if (!tables.length && !raw.length) {
    return <div style={styles.notice}>No tables found.</div>;
  }

  return (
    <div style={styles.panel}>
      {tables.map((t, idx) => (
        <ChartCard
          key={idx}
          apiBase={apiBase}
          imageName={t.image}
          imageUrl={t.image_url || `${apiBase}/images/${t.image}`}
          rows={t.data}
          title={`Detected Chart #${idx + 1}`}
        />
      ))}

      {raw?.length > 0 && (
        <details style={styles.details}>
          <summary style={styles.summary}>Model raw output</summary>
          <pre style={styles.pre}>{JSON.stringify(raw, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}

// ------------------ Chart Card ------------------
interface ChartCardProps {
  apiBase: string;
  imageName: string;
  imageUrl: string;
  rows: Record<string, string | number>[];
  title: string;
}

function ChartCard({ imageName, imageUrl, rows, title }: ChartCardProps) {
  const { xKey, seriesKeys } = useMemo(() => inferSchema(rows), [rows]);

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>{title}</h3>

      <div style={styles.sideBySide}>
        {/* Original chart */}
        <figure style={styles.figure}>
          <img src={imageUrl} alt={imageName} style={styles.img} />
          <figcaption style={styles.caption}>Original (from PDF)</figcaption>
        </figure>

        {/* Regenerated chart */}
        <div style={styles.chartWrap}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={rows}
              margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              {xKey && <XAxis dataKey={xKey} />}
              <YAxis />
              <Tooltip />
              <Legend />
              {seriesKeys.map((k) => (
                <Bar key={k} dataKey={k} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          <div style={styles.caption}>Regenerated (from extracted data)</div>
        </div>
      </div>

      <DataTable rows={rows} />

      <div style={styles.actions}>
        <button style={styles.secondaryBtn} onClick={() => downloadCSV(rows)}>
          Download CSV
        </button>
      </div>
    </div>
  );
}

// ------------------ Data Table ------------------
interface DataTableProps {
  rows: Record<string, string | number>[];
}

function DataTable({ rows }: DataTableProps) {
  if (!rows?.length) return null;
  const columns = Object.keys(rows[0]);
  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} style={styles.th}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} style={styles.td}>
                  {String(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------ Utilities ------------------
function inferSchema(rows: Record<string, string | number>[]) {
  if (!rows?.length) return { xKey: null, seriesKeys: [] };
  const keys = Object.keys(rows[0]);
  const numeric = keys.filter((k) => typeof rows[0][k] === "number");
  const nonNumeric = keys.filter((k) => typeof rows[0][k] !== "number");
  const xKey = nonNumeric[0] || keys[0];
  const seriesKeys = numeric.length ? numeric : keys.filter((k) => k !== xKey);
  return { xKey, seriesKeys };
}

function downloadCSV(rows: Record<string, string | number>[]) {
  if (!rows?.length) return;
  const cols = Object.keys(rows[0]);
  const header = cols.join(",");
  const lines = rows.map((r) => cols.map((c) => r[c]).join(","));
  const csv = [header, ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "extracted_table.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// ------------------ Styles ------------------
const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "#0f1115",
    color: "#e5e7eb",
    padding: "32px 24px",
    fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
  },
  header: { fontSize: 36, fontWeight: 800, marginBottom: 20 },
  controls: {
    display: "flex",
    gap: 12,
    alignItems: "center",
    marginBottom: 20,
  },
  button: {
    padding: "10px 16px",
    borderRadius: 10,
    background: "#2563eb",
    color: "white",
    border: "none",
    fontWeight: 600,
    cursor: "pointer",
  },
  secondaryBtn: {
    padding: "8px 12px",
    borderRadius: 8,
    background: "#111827",
    color: "#e5e7eb",
    border: "1px solid #374151",
    cursor: "pointer",
  },
  error: {
    color: "#fecaca",
    background: "#7f1d1d",
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  notice: { opacity: 0.8 },
  panel: { display: "grid", gap: 20 },
  card: {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: 16,
    padding: 16,
    boxShadow: "0 6px 24px rgba(0,0,0,0.25)",
  },
  cardTitle: { fontSize: 18, fontWeight: 700, marginBottom: 12 },
  sideBySide: {
    display: "grid",
    gridTemplateColumns: "1fr 1.2fr",
    gap: 16,
    alignItems: "center",
  },
  figure: { margin: 0, textAlign: "center" },
  img: { maxWidth: "100%", borderRadius: 12, border: "1px solid #374151" },
  caption: { marginTop: 6, fontSize: 12, opacity: 0.8 },
  chartWrap: { width: "100%", height: 340 },
  tableWrap: { overflowX: "auto", marginTop: 16 },
  table: {
    width: "100%",
    borderCollapse: "separate",
    borderSpacing: 0,
    border: "1px solid #1f2937",
    borderRadius: 12,
  },
  th: {
    textAlign: "left",
    padding: "10px 12px",
    background: "#0b1220",
    borderBottom: "1px solid #1f2937",
    position: "sticky",
    top: 0,
  },
  td: {
    padding: "10px 12px",
    borderBottom: "1px solid #1f2937",
  },
  actions: { marginTop: 12, display: "flex", gap: 8 },
  details: {
    marginTop: 8,
    background: "#0b1220",
    border: "1px solid #1f2937",
    borderRadius: 12,
    padding: 12,
  },
  summary: { cursor: "pointer", fontWeight: 600 },
  pre: { overflowX: "auto", whiteSpace: "pre-wrap", marginTop: 8 },
};
