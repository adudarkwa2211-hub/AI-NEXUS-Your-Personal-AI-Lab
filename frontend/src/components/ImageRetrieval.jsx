import { useState } from "react";

export default function ImageRetrieval() {
  const [mode, setMode] = useState("text"); // "text" | "image"
  const [queryText, setQueryText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_BASE = "http://127.0.0.1:8000";

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResults([]);

    try {
      if (mode === "text") {
        if (!queryText.trim()) {
          setError("Vui lòng nhập từ khóa tìm kiếm.");
          setLoading(false);
          return;
        }

        const res = await fetch(`${API_BASE}/api/retrieval/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryText, top_k: 6 }),
        });

        if (!res.ok) throw new Error("Tìm kiếm bằng văn bản thất bại");
        const data = await res.json();
        setResults(data.results || []);
      } else {
        if (!selectedFile) {
          setError("Vui lòng chọn 1 file ảnh để tìm kiếm.");
          setLoading(false);
          return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("top_k", 6);

        const res = await fetch(`${API_BASE}/api/retrieval/image`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error("Tìm kiếm bằng hình ảnh thất bại");
        const data = await res.json();
        setResults(data.results || []);
      }
    } catch (err) {
      setError(err.message || "Đã xảy ra lỗi kết nối Backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "900px", margin: "0 auto" }}>
      <h2>Image Retrieval (CLIP + FAISS)</h2>
      <p style={{ color: "#666" }}>
        Tìm kiếm ảnh tương quan dựa trên mô tả văn bản hoặc upload ảnh mẫu.
      </p>

      {/* Switch Mode Buttons */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        <button
          className={mode === "text" ? "active" : ""}
          onClick={() => {
            setMode("text");
            setError("");
          }}
          style={{ padding: "8px 16px", cursor: "pointer" }}
        >
          Text → Image
        </button>
        <button
          className={mode === "image" ? "active" : ""}
          onClick={() => {
            setMode("image");
            setError("");
          }}
          style={{ padding: "8px 16px", cursor: "pointer" }}
        >
          Image → Image
        </button>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} style={{ marginBottom: "20px" }}>
        {mode === "text" ? (
          <div style={{ display: "flex", gap: "10px" }}>
            <input
              type="text"
              placeholder="Nhập mô tả ảnh (ví dụ: a red flower, a dog, a car...)"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              style={{ flex: 1, padding: "10px", fontSize: "14px" }}
            />
            <button type="submit" disabled={loading} style={{ padding: "10px 20px" }}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <input type="file" accept="image/*" onChange={handleFileChange} />
            {previewUrl && (
              <div style={{ marginTop: "10px" }}>
                <p>Ảnh tìm kiếm mẫu:</p>
                <img
                  src={previewUrl}
                  alt="Query preview"
                  style={{ maxHeight: "150px", borderRadius: "8px" }}
                />
              </div>
            )}
            <button
              type="submit"
              disabled={loading}
              style={{ padding: "10px 20px", alignSelf: "flex-start", marginTop: "10px" }}
            >
              {loading ? "Searching..." : "Find Similar Images"}
            </button>
          </div>
        )}
      </form>

      {error && <div style={{ color: "red", marginBottom: "15px" }}>{error}</div>}

      {/* Results Grid */}
      <h3>Kết quả tìm kiếm ({results.length})</h3>
      {results.length === 0 && !loading && (
        <p style={{ color: "#888" }}>Chưa có kết quả. Hãy thử nhập từ khóa hoặc upload ảnh.</p>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: "15px",
          marginTop: "15px",
        }}
      >
        {results.map((item) => (
          <div
            key={item.id}
            style={{
              border: "1px solid #ccc",
              borderRadius: "8px",
              overflow: "hidden",
              textAlign: "center",
              paddingBottom: "10px",
              background: "#fafafa",
            }}
          >
            <img
              src={`${API_BASE}${item.relative_url}`}
              alt={item.filename}
              style={{
                width: "100%",
                height: "160px",
                objectFit: "cover",
              }}
            />
            <div style={{ padding: "8px 5px" }}>
              <div style={{ fontSize: "12px", color: "#555", wordBreak: "break-all" }}>
                {item.filename}
              </div>
              <div style={{ fontSize: "13px", fontWeight: "bold", color: "#2e7d32", marginTop: "4px" }}>
                Similarity: {(item.score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}