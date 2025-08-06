import { useState } from 'react';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>("");
  const [data, setData] = useState<any>(null);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      setMessage(result.message || result.error || "Unknown response");
      setData(result.data || null);

    } catch (err) {
      console.error(err);
      setMessage("Upload failed");
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
      {message && <p>{message}</p>}
      {data && (
        <div>
          <h2>Extracted Data:</h2>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
