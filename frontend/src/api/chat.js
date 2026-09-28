export async function sendChat(message, history = []) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const text = Array.isArray(detail)
      ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
      : detail || `Request failed (${response.status})`;
    throw new Error(text);
  }
  return data;
}

export async function fetchHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error("API is not ready");
  }
  return response.json();
}
