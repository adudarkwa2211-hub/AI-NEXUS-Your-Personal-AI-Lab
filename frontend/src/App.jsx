import { useEffect, useState } from "react";
import MessageInput from "./components/MessageInput.jsx";
import MessageList from "./components/MessageList.jsx";
import { fetchHealth, sendChat } from "./api/chat.js";
import ObjectDetection from "./components/ObjectDetection.jsx";
import FlowerClassifier from "./components/FlowerClassifier.jsx";
import ImageRetrieval from "./components/ImageRetrieval.jsx";

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
  const [page, setPage] = useState("chat");

  useEffect(() => {
    fetchHealth()
      .then((info) => {
        setStatus(
          `Ready · ${info.embedding_model.split("/").pop()} · ${info.llm_model.split("/").pop()} · ${info.index_size} chunks`
        );
      })
      .catch(() => setStatus("Backend offline"));
  }, []);

  async function handleSend(text) {
    if (!text.trim() || busy) return;

    const userMsg = { id: nextId++, role: "user", content: text };
    setMessages((current) => [...current, userMsg]);
    setBusy(true);

    try {
      const data = await sendChat(text);
      const botMsg = {
        id: nextId++,
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
      };
      setMessages((current) => [...current, botMsg]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: nextId++,
          role: "assistant",
          content: "Sorry, I ran into an error connecting to the backend server.",
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
        <h1>AI NEXUS</h1>
        <p className="status">{status}</p>
        <nav className="feature-nav" aria-label="AI features">
          <button
            className={page === "chat" ? "active" : ""}
            onClick={() => setPage("chat")}
          >
            Campus assistant
          </button>
          <button
            className={page === "detect" ? "active" : ""}
            onClick={() => setPage("detect")}
          >
            Object detection
          </button>
          <button
            className={page === "flowers" ? "active" : ""}
            onClick={() => setPage("flowers")}
          >
            Flower classifier
          </button>
          <button
            className={page === "retrieval" ? "active" : ""}
            onClick={() => setPage("retrieval")}
          >
            Image retrieval
          </button>
        </nav>
      </header>

      {page === "chat" ? (
        <>
          <MessageList messages={messages} />
          <MessageInput onSend={handleSend} disabled={busy} />
        </>
      ) : page === "detect" ? (
        <ObjectDetection />
      ) : page === "flowers" ? (
        <FlowerClassifier />
      ) : (
        <ImageRetrieval />
      )}
    </div>
  );
}