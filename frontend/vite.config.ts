/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Port 5173 is the origin the backend's CORS rule allows.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  preview: { port: 5173, strictPort: true },
  test: { environment: "jsdom", css: { include: [/tokens\.css/], modules: { classNameStrategy: "non-scoped" } } },
});
