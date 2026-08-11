# OfferMaster Nginx

This project uses the local Windows Nginx package at `F:\pythonProject\spring-ai-nginx`.

## Why this differs from the reference config

The reference config in `F:\pythonProject\spring-ai-nginx\conf\nginx.conf` is useful, but it cannot be copied as-is:

- It rewrites `/api/(.*)` to `/$1`; OfferMaster FastAPI routes already start with `/api/v1`, so rewriting would break API calls.
- It serves static files with `root html` but does not use `try_files ... /index.html`; React Router deep links would 404.
- It logs and stores runtime files inside the Nginx package. OfferMaster keeps runtime output under `F:\pythonProject\OfferMaster\runtime\nginx`, which is ignored by Git.

## Development mode

Use Nginx as the single browser entrypoint on `http://127.0.0.1:5173`.

```powershell
# terminal 1: backend
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --app-dir apps/api

# terminal 2: Vite dev server, internal only
D:\node.js\npx.cmd --yes pnpm@11.16.0 --dir apps/web dev

# terminal 3: Nginx entrypoint
New-Item -ItemType Directory -Force -Path F:\pythonProject\OfferMaster\logs,F:\pythonProject\OfferMaster\runtime\nginx\logs | Out-Null
F:\pythonProject\spring-ai-nginx\nginx.exe -p F:\pythonProject\OfferMaster -c infra/nginx/nginx.dev.conf
```

Routing in development:

```text
Browser -> Nginx 5173 -> Vite 5174
Browser -> Nginx 5173 -> FastAPI 8000 for /api/
```

## Preview mode

After building the frontend, Nginx serves `apps/web/dist` directly.

```powershell
D:\node.js\npx.cmd --yes pnpm@11.16.0 --dir apps/web build
New-Item -ItemType Directory -Force -Path F:\pythonProject\OfferMaster\logs,F:\pythonProject\OfferMaster\runtime\nginx\logs | Out-Null
F:\pythonProject\spring-ai-nginx\nginx.exe -p F:\pythonProject\OfferMaster -c infra/nginx/nginx.preview.conf
```

Routing in preview:

```text
Browser -> Nginx 5173 -> apps/web/dist
Browser -> Nginx 5173 -> FastAPI 8000 for /api/
```

## Stop or reload

```powershell
F:\pythonProject\spring-ai-nginx\nginx.exe -p F:\pythonProject\OfferMaster -s stop
F:\pythonProject\spring-ai-nginx\nginx.exe -p F:\pythonProject\OfferMaster -s reload
```
