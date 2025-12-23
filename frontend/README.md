# Agentic Console - Frontend

Beautiful terminal-style interface for the Agentic Backend.

## Features

- 🎨 Terminal-inspired cyberpunk UI
- 🔄 Real-time health monitoring
- 🛠️ Tool usage visualization
- 📊 RAG source browser
- 💬 Persistent conversation sessions
- ⚡ Live API integration

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```env
VITE_API_URL=http://localhost:8000
VITE_API_KEY=your-internal-api-key-here
```

**Get your API key** from the backend `.env` file (`INTERNAL_API_KEY`)

### 3. Start Development Server

```bash
npm run dev
```

Frontend will be available at http://localhost:5173

### 4. Make Sure Backend is Running

The frontend connects to the backend API at `http://localhost:8000`. Ensure your backend is running:

```bash
# In the root directory
./quick-start.sh

# Or manually
docker compose up -d
```

## Build for Production

```bash
# Build
npm run build

# Preview build
npm run preview
```

The built files will be in the `dist/` directory.

## Development

### API Configuration

The frontend uses environment variables for API configuration:

- `VITE_API_URL`: Backend API base URL (default: `http://localhost:8000`)
- `VITE_API_KEY`: API key for authentication

These can be set in `.env` or `.env.local` for local development.

### Vite Proxy

The Vite dev server is configured to proxy `/api/*` requests to the backend to avoid CORS issues during development. See `vite.config.ts`.

## Features Overview

### Health Monitoring

The console automatically checks the health of:
- FastAPI backend
- Qdrant vector database
- Redis session store

Status indicators update in real-time in the header.

### Tool Visualization

When the agent uses tools, they flash in the toolbar:
- CALC - Calculator
- WEB - Web search
- RAG - Document search
- FILE - File operations
- CODE - Python execution
- HTTP - HTTP requests

### RAG Sources

When documents are retrieved for RAG, you can click "VIEW SOURCES" to see:
- Source document names
- Similarity scores
- Content snippets

### Session Persistence

Conversations are persistent across page reloads using session IDs. The session ID is displayed in the toolbar.

## Troubleshooting

### "Failed to fetch" errors

**Check backend is running:**
```bash
curl http://localhost:8000/api/v1/health
```

**Check CORS settings** in backend `app/main.py`

**Check API key** in `.env` matches backend

### Backend connection refused

Make sure Docker services are running:
```bash
docker compose ps
```

All services should show as "healthy" or "running".

### Port conflicts

If port 5173 is in use, change it in `vite.config.ts`:

```ts
server: {
  port: 3000, // or any available port
  // ...
}
```

## Tech Stack

- **React 19.2.0** - UI library
- **TypeScript 5.9.3** - Type safety
- **Vite 7.2.4** - Build tool and dev server
- **No CSS framework** - Pure inline styles for terminal aesthetic

## License

Same as parent project
