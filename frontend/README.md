# Sift frontend

React + Vite frontend for the Sift log QA API.

## Run locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

The default API URL is `http://localhost:8001`. Set `VITE_API_URL` in `.env.local` to use the deployed Cloud Run API.

## Build

```bash
npm run build
```
