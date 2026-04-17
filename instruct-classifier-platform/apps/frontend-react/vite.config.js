import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 개발 중 CORS 없이 백엔드 호출 가능 (선택적)
      // "/api": { target: "http://localhost:8000", rewrite: (path) => path.replace(/^\/api/, "") }
    },
  },
});