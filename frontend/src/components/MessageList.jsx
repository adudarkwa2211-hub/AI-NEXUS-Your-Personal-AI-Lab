export default function MessageList({ messages }) {
  return (
    <div className="message-list" aria-live="polite">
      {messages.map((item) => (
        <article key={item.id} className={`bubble ${item.role}`}>
          <p>{item.content}</p>
          {item.sources?.length ? (
            <ul className="sources">
              {item.sources.map((source) => (
                <li key={`${item.id}-${source.id}`}>
                  {source.source} ({source.score.toFixed(2)})
                </li>
              ))}
            </ul>
          ) : null}
        </article>
      ))}
    </div>
  );
}
