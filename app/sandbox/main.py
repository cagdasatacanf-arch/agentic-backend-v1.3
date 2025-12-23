import sys
import io
import contextlib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

@app.post("/execute")
async def execute_code(request: CodeRequest):
    """
    Executes Python code in a contained environment.
    Captures stdout/stderr.
    """
    # Create string buffers to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        # Redirect stdout and stderr
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            # Safe-ish execution globals
            # We remove easy access to dangerous things like 'quit' or 'exit' if possible,
            # though inside a container, the risk is contained.
            exec_globals = {} 
            exec(request.code, exec_globals)
            
        return {
            "output": stdout_buffer.getvalue(),
            "error": stderr_buffer.getvalue(),
            "status": "success"
        }
    except Exception as e:
        return {
            "output": stdout_buffer.getvalue(),
            "error": str(e),
            "status": "error"
        }
