# SatVision

SatVision — frontend and backend for real-time flood detection.

## Local frontend

1. Open a terminal and install dependencies:

```powershell
cd "c:\Users\Abdur Rehman\Documents\Final Year Proj\frontend"
npm install
```

2. Run the dev server:

```powershell
npm start
# Open http://localhost:3000
```

3. Create a production build:

```powershell
npm run build
# build/ will contain the production files
```

## Deploy to Vercel (recommended)

Option A — via GitHub (recommended):
- Push your repository to GitHub (already done).
- Go to https://vercel.com, import the `SatVision` project, and set the root directory to `frontend`.
- Vercel will use `npm run build` and deploy the `build/` directory.

Option B — Vercel CLI:

```powershell
npm i -g vercel
cd frontend
vercel login
vercel --prod
```

Vercel will detect the React app. If asked, set the framework to `create-react-app` and the output directory to `build`.

## Notes
- `vercel.json` already exists in `frontend/` to handle SPA routing.
- `package.json` contains the build and start scripts used by Vercel.

If you want, I can trigger a deploy from this machine via the Vercel CLI (requires you to login interactively).