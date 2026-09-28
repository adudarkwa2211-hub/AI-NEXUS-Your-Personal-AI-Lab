import { useEffect, useState } from "react";
import MessageInput from "./components/MessageInput.jsx";
import MessageList from "./components/MessageList.jsx";
import { fetchHealth, sendChat } from "./api/chat.js";

let nextId = 1;

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: "assistant",
      content:
        "Hello. I am the campus RAG assistant. Ask in English or Vietnamese about the library, admissions, or how this chatbot works.",
      sources: [],
    },
  ]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Checking API...");

  useEffect(() => {
    fetchHealth()
      .then((info) => {
        setStatus(
          `Ready · ${info.embedding_model.split("/").pop()} · ${info.llm_model.split("/").pop()} · ${info.index_size} chunks`
        );
      })
      .catch(() => {
        setStatus("API offline. Start FastAPI on port 8000.");
      });
  }, []);

  async function handleSend(text) {
    const userMessage = { id: nextId++, role: "user", content: text, sources: [] };
    setMessages((current) => [...current, userMessage]);
    setBusy(true);
    try {
      const history = messages
        .filter((item) => item.role === "user" || item.role === "assistant")
        .slice(-8)
        .map((item) => ({ role: item.role, content: item.content }));
      const result = await sendChat(text, history);
      setMessages((current) => [
        ...current,
        {
          id: nextId++,
          role: "assistant",
          content: result.answer,
          sources: result.sources || [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: nextId++,
          role: "assistant",
          content: error.message || "Request failed.",
          sources: [],
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header>
        <h1>Greenfield RAG Chatbot</h1>
        <p className="status">{status}</p>
      </header>
      <MessageList messages={messages} />
      <MessageInput onSend={handleSend} disabled={busy} />
    </div>
  );
}
