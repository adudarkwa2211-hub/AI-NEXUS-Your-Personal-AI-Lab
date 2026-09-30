import { useEffect, useState } from "react";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

export default function FlowerClassifier() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Choose a valid image file.");
      return;
    }
    if (nextFile.size > MAX_IMAGE_BYTES) {
      setError("Image must be 8 MB or smaller.");
      return;
    }
    setFile(nextFile);
    setResult(null);
    setError("");
  }

  async function submit(event) {
    event.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    body.append("top_k", "5");
    try {
      const response = await fetch("/api/classify", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Flower classification failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flower-classifier">
      <section className="flower-controls">
        <h2>Flower species</h2>
        <p className="status">ResNet-18 · daisy, dandelion, roses, sunflowers and tulips</p>
        <form onSubmit={submit}>
          <label
            className="flower-upload"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            <span>Choose or drop a flower image</span>
            <span className="status small">JPG, PNG or WebP · up to 8 MB</span>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              aria-label="Choose a flower image"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
          </label>
          {file && <p className="file-name">{file.name}</p>}
          <button className="detect-button" type="submit" disabled={!file || busy}>
            {busy ? "Analyzing…" : "Classify flower"}
          </button>
        </form>
      </section>

      <section className="flower-result" aria-live="polite">
        {previewUrl && <img className="flower-preview" src={previewUrl} alt="Selected flower" />}
        {busy && <p className="status" role="status">Analyzing image…</p>}
        {error && <p className="error-message" role="alert">{error}</p>}
        {result && (
          <>
            {!result.confident && (
              <p className="flower-warning" role="status">
                Low confidence. This image may not show one of the flower species the model knows.
              </p>
            )}
            <h3>{result.predictions[0]?.label || "No prediction"}</h3>
            {result.predictions.map((prediction) => (
              <div className="flower-score" key={prediction.label}>
                <span>{prediction.label}</span>
                <div className="flower-track" role="img" aria-label={`${prediction.label}: ${(prediction.score * 100).toFixed(1)} percent`}>
                  <div className="flower-fill" style={{ width: `${prediction.score * 100}%` }} />
                </div>
                <span>{(prediction.score * 100).toFixed(1)}%</span>
              </div>
            ))}
            <p className="status small">{result.latency_ms} ms</p>
          </>
        )}
      </section>
    </main>
  );
}
