import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Phone } from "./Phone";
import "../styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Phone />
  </StrictMode>,
);
