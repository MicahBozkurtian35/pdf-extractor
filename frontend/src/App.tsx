import { useState } from 'react';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>("");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);



  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  setLoading(true);
  setMessage("");
  setData(null);

  try {
    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    setMessage(result.message || result.error || "Unknown response");
    setData(result.data || result.excel_file || null);
  } catch (err) {
    console.error(err);
    setMessage("Upload failed");
  } finally {
    setLoading(false);
  }
};

  

  return (
    <div style={{ padding: "2rem" }}>
      <h1>PDF Extractor</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button type="submit">Upload PDF</button>
      </form>
      {loading && <p>Processing... Please wait.</p>}
      {message && <p>{message}</p>}
      <pre>{JSON.stringify(message, null, 2)}</pre>
      <pre>{JSON.stringify(data, null, 2)}</pre>

      {Array.isArray(data) && data.length > 0 && (
        <div>
          {data.map((table: any, idx: number) => (
            <div key={idx} style={{ marginBottom: "2rem" }}>
              <h3>{table.tab_name}</h3>
              <table border={1} cellPadding={5} style={{ borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {Object.keys(table.data[0] || {}).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.data.map((row: any, rIdx: number) => (
                    <tr key={rIdx}>
                      {Object.values(row).map((val, cIdx) => (
                        <td key={cIdx}>{String(val)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
