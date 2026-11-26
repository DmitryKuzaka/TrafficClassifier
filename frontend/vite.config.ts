import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "TRAFFICCLASSIFIER_");
  const apiUrl = env.TRAFFICCLASSIFIER_API_URL || "http://127.0.0.1:8000";
  const proxy = {
    "/api": { target: apiUrl },
    "/ready": { target: apiUrl },
  };

  return {
    plugins: [react()],
    server: { host: "127.0.0.1", proxy },
    preview: { host: "127.0.0.1", proxy },
  };
});
