import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ValuersDashboard from "../twinval_valuers_dashboard.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ValuersDashboard />
  </StrictMode>
);
