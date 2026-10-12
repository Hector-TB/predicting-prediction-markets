import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development /api/* goes to the FastAPI app (uvicorn api.main:app, port 8000)
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
