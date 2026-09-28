import { useState } from "react";

export default function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    const value = text.trim();
    if (!value || disabled) {
      return;
    }
    onSend(value);
    setText("");
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <input
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Ask about library hours, admissions, or this RAG module..."
        disabled={disabled}
        maxLength={2000}
        aria-label="Chat message"
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        Send
      </button>
    </form>
  );
}
