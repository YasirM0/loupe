import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative asset paths — the same build output has to work from three
  // different roots: GitHub Pages' subpath (yasirm0.github.io/loupe/),
  // a custom domain root, and Tauri's file:// load of ../dist. An absolute
  // base ('/') only works for one of those; './' works for all three.
  base: './',
})
