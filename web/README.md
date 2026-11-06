# Athena web dashboard

React + TypeScript + Tailwind (Vite). Talks only to the local Python API in `app/server.py`.

```bash
npm install
npm run build          # output in web/dist, served by: python app/server.py  → http://localhost:8000
npm run dev            # live reload on http://127.0.0.1:5173 (run python app/server.py alongside)
```

Pages: Input Note · Triage · Med Diff · Medications · Audit & System. Light and pitch-black dark themes.
