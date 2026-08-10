import React from "react";
import ReactDOM from "react-dom/client";

import "./styles/global.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <main className="shell">
      <h1>JobPilot</h1>
      <p>Local-first interview and application assistant scaffold.</p>
    </main>
  </React.StrictMode>,
);

