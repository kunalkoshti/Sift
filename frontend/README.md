# Sift frontend

React + Vite frontend for the Sift log QA API.

## Run locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

The default API URL is `http://localhost:8001`. Set `VITE_API_URL` in `.env.local` to use the deployed Cloud Run API.

## Authentication

The frontend uses Supabase Auth for both supported sign-in methods:

- Email and password
- Google OAuth

Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` in `.env.local`.
The older Supabase dashboard may call the publishable key the `anon` key. Enable
Email and Google under Supabase Dashboard → Authentication → Providers. Add
`http://localhost:5173` and the deployed Vercel URL to Supabase's allowed
redirect URLs. The frontend sends the Supabase access token to `/ask` and
`/stats/log-levels`; the API verifies it before serving those routes. The
architecture page remains available without signing in, while the chat and
live corpus panels require an authenticated session.

If you are the administrator, set `VITE_ADMIN_EMAILS` to your comma-separated
Supabase email address(es) to display the private Production metrics page. The
backend independently enforces the `ADMIN_EMAILS`/`ADMIN_USER_IDS` allowlist on
`/admin/metrics`; the frontend variable only controls visibility of the link.

## Build

```bash
npm run build
```
