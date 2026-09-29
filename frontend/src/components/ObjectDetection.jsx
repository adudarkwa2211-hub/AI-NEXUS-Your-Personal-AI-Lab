import { useState } from "react";

export default function ObjectDetection() {
  const [file, setFile] = useState(null);
  const [confidence, setConfidence] = useState(0.25);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    body.append("confidence", String(confidence));
    try {
      const response = await fetch("/api/detect", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      setResult(data);
    } catch (err) {
      setError(err.message || "Detection failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="detector">
      <section className="detector-controls">
        <h2>Object detection</h2>
        <p className="status">YOLO11n pretrained on COCO · 80 object categories</p>
        <form onSubmit={submit}>
          <label className="upload-label">
            Choose an image
            <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          </label>
          {file && <p className="file-name">{file.name}</p>}
          <label className="confidence-control">
            Confidence threshold: <strong>{confidence.toFixed(2)}</strong>
            <input type="range" min="0.05" max="0.95" step="0.05" value={confidence}
              onChange={(event) => setConfidence(Number(event.target.value))} />
          </label>
          <button className="detect-button" type="submit" disabled={!file || busy}>
            {busy ? "Detecting…" : "Detect objects"}
          </button>
        </form>
        <p className="status small">Maximum image size: 8 MB. First request downloads the model weights.</p>
      </section>
      <section className="detector-result" aria-live="polite">
        {error && <p className="error-message">{error}</p>}
        {result && <>
          <img className="detected-image" src={result.annotated_image} alt="Image annotated with detected objects" />
          <h3>{result.detections.length} object{result.detections.length === 1 ? "" : "s"} detected</h3>
          {result.detections.length > 0 ? <div className="detection-table-wrap">
            <table className="detection-table">
              <thead><tr><th>Class</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead>
              <tbody>{result.detections.map((item, index) => <tr key={`${item.class_id}-${index}`}>
                <td>{item.label}</td><td>{(item.confidence * 100).toFixed(1)}%</td><td>{item.box_xyxy.join(", ")}</td>
              </tr>)}</tbody>
            </table>
          </div> : <p className="status">No objects above this confidence threshold.</p>}
        </>}
      </section>
    </main>
  );
}
