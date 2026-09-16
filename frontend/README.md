# Doc2MD Platform — Frontend

Next.js App Router shell for the Parser module. Talks to the FastAPI parser service.

## Prerequisites

- Node.js 20+
- Parser API running at `http://localhost:8000` (or set `NEXT_PUBLIC_API_URL`)

## Run

```bash
cd frontend
cp .env.example .env.local   # defaults to http://localhost:8000
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) — redirects to `/parser`.

## Scripts

| Command | Purpose |
|---------|---------|
| `npm run dev` | Dev server (Turbopack) |
| `npm run build` | Production build |
| `npm run start` | Serve production build |
| `npm run lint` | ESLint |

## API client

Typed client in `lib/api.ts` / `lib/types.ts`:

- `POST /api/v1/parse/file` — multipart upload
- `POST /api/v1/parse/text` — `{ text, format }`
- `POST /api/v1/parse/url` — `{ url }`
- `GET /api/v1/parse/{job_id}` — poll async jobs (HTTP 202 path)
