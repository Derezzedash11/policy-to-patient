import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs on :5173 and forwards API calls to the FastAPI backend.
// `npm run build` writes dist/, which the backend serves at /ui/ (same origin, no CORS).
const backend = process.env.API_URL ?? "http://localhost:8000";
const apiPaths = ["/health", "/policies", "/treatments", "/estimate", "/coverage"];

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: { proxy: Object.fromEntries(apiPaths.map((p) => [p, backend])) },
});
