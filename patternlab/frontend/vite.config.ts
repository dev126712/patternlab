import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API routes are unprefixed on the FastAPI side; proxy each root path in dev.
const API = process.env.VITE_API_PROXY ?? "http://localhost:8000";
const proxy = Object.fromEntries(
  ["/live", "/sessions", "/match", "/analyze", "/compare", "/health"].map((p) => [p, API]),
);

export default defineConfig({
  plugins: [react()],
  server: { port: 3000, proxy },
});
