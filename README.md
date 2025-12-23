## Running with Docker

This project provides Dockerfiles for both the backend (Python) and frontend (TypeScript/React), along with a `docker-compose.yml` for easy orchestration. The setup also includes a Redis service for caching.

### Requirements
- **Python backend:** Uses Python 3.11 (slim image)
- **Frontend:** Uses Node.js v22.13.1 (slim image)
- **Redis:** Official `redis:latest` image

### Environment Variables
- Backend: Copy `.env.example` to `.env` and adjust as needed. Uncomment the `env_file` line in `docker-compose.yml` if you use a `.env` file.
- Frontend: Copy `frontend/.env.example` to `frontend/.env` if needed. Uncomment the `env_file` line in the frontend service if you use a `.env` file.

### Build and Run
1. Ensure Docker and Docker Compose are installed.
2. From the project root, run:

   ```sh
   docker compose up --build
   ```

   This will build and start all services: backend, frontend, and Redis.

### Service Ports
- **Backend (python-app):** [http://localhost:8000](http://localhost:8000)
- **Frontend (typescript-frontend):** [http://localhost:3000](http://localhost:3000)
- **Redis:** Internal only (not exposed to host)

### Special Configuration
- The backend expects a `.env` file for configuration. See `.env.example` for required variables.
- Redis persistence is optional. To enable, uncomment the `volumes` section for Redis in `docker-compose.yml`.
- The backend and frontend Dockerfiles use multi-stage builds for smaller production images and improved caching.

### Dependencies
- The backend installs Python dependencies from `requirements.txt` inside a virtual environment.
- The frontend installs dependencies via `npm ci` and builds the app before serving.

---

For more details on configuration and advanced usage, see the documentation in the `docs/` directory.