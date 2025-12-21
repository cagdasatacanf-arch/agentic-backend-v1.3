@echo off
REM ============================================================================
REM Agentic Backend v1.3 - Quick Start Script (Windows)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ===============================================================================
echo   Agentic Backend v1.3 - Quick Start
echo ===============================================================================
echo.

REM ============================================================================
REM Step 1: Check Prerequisites
REM ============================================================================
echo [Step 1] Checking Prerequisites...
echo.

where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed
    echo Please install Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/
    exit /b 1
)
echo [OK] Docker is installed

docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker daemon is not running
    echo Please start Docker Desktop
    exit /b 1
)
echo [OK] Docker daemon is running

docker compose version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    docker-compose --version >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Docker Compose is not installed
        exit /b 1
    )
    set COMPOSE_CMD=docker-compose
) else (
    set COMPOSE_CMD=docker compose
)
echo [OK] Docker Compose is installed
echo.

REM ============================================================================
REM Step 2: Environment Configuration
REM ============================================================================
echo [Step 2] Environment Configuration...
echo.

if exist .env (
    echo [WARNING] .env file already exists
    set /p RECONFIGURE="Do you want to reconfigure it? (y/N): "
    if /i not "!RECONFIGURE!"=="y" (
        echo [INFO] Using existing .env file
        goto skip_env_setup
    )
)

copy .env.example .env >nul
echo [OK] Created .env file from template
echo.

echo Required Configuration:
echo -------------------------------------------------------------------------------
set /p OPENAI_KEY="Enter your OpenAI API Key (from https://platform.openai.com): "

if "!OPENAI_KEY!"=="" (
    echo [ERROR] OpenAI API Key is required
    exit /b 1
)

REM Generate random Internal API Key
set "chars=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
set "INTERNAL_KEY="
for /L %%i in (1,1,64) do (
    set /a "rand=!RANDOM! %% 62"
    for %%j in (!rand!) do set "INTERNAL_KEY=!INTERNAL_KEY!!chars:~%%j,1!"
)

REM Update .env file
powershell -Command "(gc .env) -replace 'OPENAI_API_KEY=.*', 'OPENAI_API_KEY=!OPENAI_KEY!' | Out-File -encoding ASCII .env"
powershell -Command "(gc .env) -replace 'INTERNAL_API_KEY=.*', 'INTERNAL_API_KEY=!INTERNAL_KEY!' | Out-File -encoding ASCII .env"

echo [OK] OpenAI API Key configured
echo [OK] Generated Internal API Key: !INTERNAL_KEY!
echo.
echo [WARNING] Save this Internal API Key - you'll need it to make API requests!
echo.

:skip_env_setup

REM ============================================================================
REM Step 3: Stop Existing Services
REM ============================================================================
echo [Step 3] Stopping Existing Services...
echo.

%COMPOSE_CMD% ps >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Stopping existing containers...
    %COMPOSE_CMD% down
    echo [OK] Existing services stopped
) else (
    echo [INFO] No existing services to stop
)
echo.

REM ============================================================================
REM Step 4: Build and Start Services
REM ============================================================================
echo [Step 4] Building and Starting Services...
echo.

echo [INFO] Building Docker images (this may take a few minutes on first run)...
%COMPOSE_CMD% build

echo [INFO] Starting services: Redis, Qdrant, Jaeger, FastAPI...
%COMPOSE_CMD% up -d

echo [OK] Services started
echo.

REM ============================================================================
REM Step 5: Wait for Services
REM ============================================================================
echo [Step 5] Waiting for Services to Initialize...
echo.

echo Waiting for services to be ready (30 seconds)...
timeout /t 30 /nobreak >nul
echo.

REM ============================================================================
REM Step 6: Health Checks
REM ============================================================================
echo [Step 6] Running Health Checks...
echo.

echo Checking FastAPI health endpoint...
set RETRY=0
:health_check_loop
if %RETRY% GEQ 10 (
    echo [ERROR] API health check failed
    echo Checking API logs...
    %COMPOSE_CMD% logs --tail=50 api
    exit /b 1
)

curl -s http://localhost:8000/api/v1/health >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    set /a RETRY+=1
    timeout /t 2 /nobreak >nul
    goto health_check_loop
)

echo [OK] FastAPI is healthy

curl -s http://localhost:6333/healthz >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Qdrant is healthy
)

echo.
echo [OK] All health checks passed!
echo.

REM ============================================================================
REM Step 7: Display Service Information
REM ============================================================================
echo.
echo ===============================================================================
echo   System is Ready!
echo ===============================================================================
echo.

REM Get the Internal API Key from .env
for /f "tokens=2 delims==" %%a in ('findstr "INTERNAL_API_KEY=" .env') do set INTERNAL_API_KEY=%%a

echo All services are up and running!
echo.
echo -------------------------------------------------------------------------------
echo Service URLs:
echo -------------------------------------------------------------------------------
echo.
echo   API Server:        http://localhost:8000
echo   API Documentation: http://localhost:8000/docs
echo   Jaeger Tracing:    http://localhost:16686
echo   Qdrant Dashboard:  http://localhost:6333/dashboard
echo   Redis:             localhost:6379
echo.
echo -------------------------------------------------------------------------------
echo Your API Key:
echo -------------------------------------------------------------------------------
echo.
echo   %INTERNAL_API_KEY%
echo.
echo   [WARNING] Save this key! You'll need it for all API requests.
echo.
echo -------------------------------------------------------------------------------
echo Quick Test Commands:
echo -------------------------------------------------------------------------------
echo.
echo 1. Health Check:
echo    curl http://localhost:8000/api/v1/health
echo.
echo 2. Simple Query:
echo    curl -X POST http://localhost:8000/api/v1/langgraph/query ^
echo      -H "Content-Type: application/json" ^
echo      -H "X-API-Key: %INTERNAL_API_KEY%" ^
echo      -d "{\"question\": \"What is 25 * 47?\", \"use_rag\": false}"
echo.
echo -------------------------------------------------------------------------------
echo Useful Commands:
echo -------------------------------------------------------------------------------
echo.
echo   View logs:        %COMPOSE_CMD% logs -f api
echo   Restart services: %COMPOSE_CMD% restart
echo   Stop services:    %COMPOSE_CMD% down
echo   Service status:   %COMPOSE_CMD% ps
echo.
echo ===============================================================================
echo Happy building with Agentic Backend!
echo ===============================================================================
echo.

endlocal
