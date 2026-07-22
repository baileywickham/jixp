import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import tailwindcss from "@tailwindcss/vite";

// Everything inlines into dist/index.html: the compiler treats that file as a
// template, replacing __TITLE__ and the __DATA__ inside #model-data.
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
});
