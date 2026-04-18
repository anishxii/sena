import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "@xyflow/react/dist/style.css";
import "./styles/index.css";

const rootElement = document.getElementById("root");

function renderFatal(message) {
  if (!rootElement) return;
  rootElement.innerHTML = `
    <main style="min-height:100vh;background:#0b0b10;color:#f5f5f7;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Helvetica Neue',Arial,sans-serif;padding:40px;">
      <div style="max-width:960px;margin:0 auto;border:1px solid #262633;background:#101017;padding:32px;">
        <p style="margin:0 0 8px;color:#a1a1aa;text-transform:uppercase;letter-spacing:0.08em;font-size:12px;">Cognitive Middleware Observatory</p>
        <h1 style="margin:0 0 12px;font-size:32px;line-height:1.05;">React runtime error</h1>
        <pre style="margin:0;white-space:pre-wrap;color:#ffb4ab;line-height:1.5;">${String(message)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")}</pre>
      </div>
    </main>
  `;
}

window.addEventListener("error", (event) => {
  renderFatal(event.error?.stack || event.message || "Unknown runtime error");
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason?.stack || event.reason?.message || String(event.reason || "Unhandled promise rejection");
  renderFatal(reason);
});

try {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} catch (error) {
  renderFatal(error?.stack || error?.message || String(error));
}
