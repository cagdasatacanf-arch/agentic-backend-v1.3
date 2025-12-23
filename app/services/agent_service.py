from typing import Dict, List, Tuple
from uuid import uuid4
import json
import logging
import redis
from pathlib import Path

from qdrant_client.http.models import PointStruct

from app.config import settings
from app.rag import get_agent_answer, client, COLLECTION, embed
from app.utils.chunking import default_chunker

logger = logging.getLogger("app")


class AgentService:
    """
    Facade for agent-related operations.
    Includes tools, memory (Redis), and document indexing strategies.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, callable] = {}
        self.workspace_root = Path("/data/workspace")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis client
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        
        # Register default tools
        self.register_tool("calculator", self._calculator_tool, self._tool_definitions[0])
        self.register_tool("web_search", self._web_search_tool, self._tool_definitions[1])
        self.register_tool("list_files", self._list_files_tool, self._tool_definitions[2])
        self.register_tool("read_file", self._read_file_tool, self._tool_definitions[3])
        self.register_tool("write_file", self._write_file_tool, self._tool_definitions[4])
        self.register_tool("run_python", self._run_python_tool, self._tool_definitions[5])
        self.register_tool("make_http_request", self._http_request_tool, self._tool_definitions[6])
        self.register_tool("create_agent", self._create_agent_tool, self._tool_definitions[7])
        self.register_tool("list_agents", self._list_agents_tool, self._tool_definitions[8])
        self.register_tool("get_agent", self._get_agent_tool, self._tool_definitions[9])
        self.register_tool("deep_research", self._deep_research_tool, self._tool_definitions[10])
        self.register_tool("delegate_to_agent", self._delegate_to_agent_tool, self._tool_definitions[11])
        self.register_tool("ask_human", self._ask_human_tool, self._tool_definitions[12])

    def register_tool(self, name: str, fn: callable, schema: Dict) -> None:
        self.tools[name] = {"fn": fn, "schema": schema}

    def _calculator_tool(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Use eval with restricted globals for safety
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def _web_search_tool(self, query: str) -> str:
        """Search the web using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No results found."
                
                output = []
                for i, r in enumerate(results, 1):
                    output.append(f"{i}. {r['title']}\\n   {r['body']}\\n   URL: {r['href']}")
                return "\\n\\n".join(output)
        except Exception as e:
            return f"Error performing web search: {e}"

    # ... (other tools) ...

    def _ask_human_tool(self, question: str) -> str:
        """Placeholder. The Orchestrator handles this specially."""
        return "User input required."

    # ... (other methods same) ...

    def _list_files_tool(self, path: str = ".") -> str:
        """List files in the workspace."""
        try:
            target_path = self._validate_path(path)
            if not target_path.exists():
                return f"Error: Path {path} does not exist."
            
            items = []
            for item in target_path.iterdir():
                type_ = "DIR" if item.is_dir() else "FILE"
                items.append(f"{type_}: {item.name}")
            return "\n".join(items)
        except Exception as e:
            return f"Error listing files: {e}"

    def _read_file_tool(self, path: str) -> str:
        """Read content of a file (Supports: txt, md, json, pdf, csv, xlsx)."""
        try:
            target_path = self._validate_path(path)
            if not target_path.is_file():
                return f"Error: {path} is not a file."
            
            suffix = target_path.suffix.lower()
            
            # 1. JSON (Pretty Print)
            if suffix == ".json":
                data = json.loads(target_path.read_text(encoding="utf-8"))
                return json.dumps(data, indent=2)
                
            # 2. PDF (Extract Text)
            elif suffix == ".pdf":
                from pypdf import PdfReader
                try:
                    reader = PdfReader(str(target_path))
                    text = []
                    for i, page in enumerate(reader.pages):
                        text.append(f"--- Page {i+1} ---\n{page.extract_text()}")
                    return "\n".join(text)
                except Exception as e:
                    return f"Error reading PDF: {e}"

            # 3. CSV / Excel (Markdown Table)
            elif suffix in [".csv", ".xls", ".xlsx"]:
                import pandas as pd
                try:
                    if suffix == ".csv":
                        df = pd.read_csv(target_path)
                    else:
                        df = pd.read_excel(target_path)
                    
                    if len(df) > 50:
                        return f"File too large ({len(df)} rows). showing first 50:\n" + df.head(50).to_markdown(index=False)
                    return df.to_markdown(index=False)
                except Exception as e:
                    return f"Error reading Data Table: {e}"

            # 4. Fallback (Text)
            return target_path.read_text(encoding="utf-8")
            
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file_tool(self, path: str, content: str) -> str:
        """Write content to a file."""
        try:
            target_path = self._validate_path(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _validate_path(self, path: str) -> Path:
        """Ensure path is within workspace."""
        base = self.workspace_root.resolve()
        target = (base / path).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError(f"Access denied: {path} is outside workspace.")
        return target

    def _run_python_tool(self, code: str) -> str:
        """Execute python code via separate Sandbox service."""
        import httpx
        try:
            # Call the Sandbox Container
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    "http://sandbox:8000/execute", 
                    json={"code": code}
                )
            
            if resp.status_code != 200:
                return f"Sandbox Error ({resp.status_code}): {resp.text}"
                
            data = resp.json()
            output = data.get("output", "")
            error = data.get("error", "")
            
            result = output
            if error:
                result += f"\nError Output:\n{error}"
                
            return result if result.strip() else "(No output)"
            
        except Exception as e:
            return f"Error executing code in sandbox: {e}"

    def _http_request_tool(self, url: str, method: str = "GET", headers: Dict = None, body: Dict = None) -> str:
        """Make an HTTP request to an external API."""
        import httpx
        try:
            method = method.upper()
            if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                return f"Error: Unsupported method {method}"

            with httpx.Client(timeout=10.0) as client:
                response = client.request(method, url, headers=headers, json=body)
                
            # Truncate large responses
            resp_text = response.text
            if len(resp_text) > 2000:
                resp_text = resp_text[:2000] + "... (truncated)"
            
            return f"Status: {response.status_code}\nContent: {resp_text}"
        except Exception as e:
            return f"Error making HTTP request: {e}"

    def _get_agents_db(self) -> Path:
        """Helper to get agents DB path."""
        return self.workspace_root / "agents.json"

    def _create_agent_tool(self, name: str, system_prompt: str, tools: List[str]) -> str:
        """Create or update a new AI agent profile."""
        try:
            db_path = self._get_agents_db()
            if db_path.exists():
                data = json.loads(db_path.read_text(encoding="utf-8"))
            else:
                data = {}
            
            # Validate tools
            valid_tools = list(self.tools.keys())
            for t in tools:
                if t not in valid_tools:
                    return f"Error: Tool '{t}' does not exist. Valid: {valid_tools}"

            data[name] = {
                "name": name,
                "system_prompt": system_prompt,
                "tools": tools,
                "created_at": str(uuid4())
            }
            
            db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return f"Agent '{name}' created successfully."
        except Exception as e:
            return f"Error creating agent: {e}"

    def _list_agents_tool(self) -> str:
        """List all available agents."""
        try:
            db_path = self._get_agents_db()
            if not db_path.exists(): return "No agents found."
            
            data = json.loads(db_path.read_text(encoding="utf-8"))
            return "\n".join([f"- {name} (Tools: {a['tools']})" for name, a in data.items()])
        except Exception as e:
            return f"Error listing agents: {e}"

    def _get_agent_tool(self, name: str) -> str:
        """Get details of a specific agent."""
        try:
            db_path = self._get_agents_db()
            if not db_path.exists(): return "Error: No agents DB found."
            
            data = json.loads(db_path.read_text(encoding="utf-8"))
            if name not in data:
                return f"Error: Agent '{name}' not found."
            
            return json.dumps(data[name], indent=2)
        except Exception as e:
            return f"Error getting agent: {e}"

    def _deep_research_tool(self, query: str) -> str:
        """Perform a deep research query using Perplexity API."""
        if not settings.perplexity_api_key:
            return "Error: Perplexity API key is not configured in settings."

        import httpx
        try:
            url = "https://api.perplexity.ai/chat/completions"
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": "Be precise and provide citations."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1
            }
            headers = {
                "Authorization": f"Bearer {settings.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                return f"Perplexity API Error: {response.status_code} - {response.text}"

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            citations = data.get("citations", [])
            
            if citations:
                content += "\n\nSources:\n" + "\n".join([f"- {c}" for c in citations])
                
            return content
        except Exception as e:
            return f"Error performing deep research: {e}"

    async def _delegate_to_agent_tool(self, agent_name: str, task: str) -> str:
        """Delegate a task to a specific specialized agent."""
        try:
            # 1. Load Agent Profile
            db_path = self._get_agents_db()
            if not db_path.exists():
                return "Error: No agents database found."
            
            data = json.loads(db_path.read_text(encoding="utf-8"))
            if agent_name not in data:
                return f"Error: Agent '{agent_name}' not found. Use list_agents to see available agents."
            
            agent_profile = data[agent_name]
            
            # 2. Construct Sub-Agent Tools
            allowed_tool_names = agent_profile["tools"]
            sub_agent_tools_map = {}
            sub_agent_tool_defs = []
            
            # Filter from available tools
            for name in allowed_tool_names:
                if name in self.tools:
                    sub_agent_tools_map[name] = self.tools[name]
                    sub_agent_tool_defs.append(self.tools[name]["schema"])
                else:
                    logger.warning(f"Agent {agent_name} wants missing tool {name}")

            # 3. Construct Messages
            system_msg = {
                "role": "system", 
                "content": f"{agent_profile['system_prompt']}\n\nYou are a specialized agent named '{agent_name}'."
            }
            user_msg = {"role": "user", "content": task}
            messages = [system_msg, user_msg]
            
            logger.info(f"DELEGATING to {agent_name}: {task}")
            
            # 4. Run Sub-Loop
            # Note: We must await this because we are inside an async tool (called via `await fn_res` in loop)
            final_ans, usage = await self._run_orchestration_loop(
                messages, 
                tool_definitions=sub_agent_tool_defs, 
                tools_map=sub_agent_tools_map
            )
            
            return f"Agent '{agent_name}' Result:\n{final_ans}"

        except Exception as e:
            logger.error(f"Delegation failed: {e}")
            return f"Error delegating to {agent_name}: {e}"

    # Tool Schemas for OpenAI
    @property
    def _tool_definitions(self) -> List[Dict]:
        tools = [
            # ... (calculator, web_search, list_files, read_file, write_file, run_python same as before) ...
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a mathematical expression. Useful for calculations.",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The math expression to evaluate, e.g. '2 + 2'"}}, "required": ["expression"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet for real-time information, news, or facts.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories in the workspace.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path to list (default: '.')"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the content of a file.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path to the file."}}, "required": ["path"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write text content to a file (overwrites existing).",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path to the file."},"content": {"type": "string","description": "The full text content to write."}},"required": ["path", "content"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Execute Python code to test logic, calculations, or scripts. Code runs in the workspace.",
                    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The valid Python code to execute."}}, "required": ["code"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "make_http_request",
                    "description": "Make an HTTP request to an external API/URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The full URL (e.g. https://api.coingecko.com/api/v3/ping)"},
                            "method": {"type": "string", "description": "HTTP Method (GET, POST, etc.). Default: GET", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                            "headers": {"type": "object", "description": "JSON object of request headers.", "additionalProperties": True},
                            "body": {"type": "object", "description": "JSON object for request body (if POST/PUT).", "additionalProperties": True}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Define and save a new AI agent profile.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique name of the agent (e.g. 'CryptoAnalyst')."},
                            "system_prompt": {"type": "string", "description": "The instructions defining the agent's persona and rules."},
                            "tools": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "List of tool names this agent can use (e.g. ['web_search', 'calculator'])."
                            }
                        },
                        "required": ["name", "system_prompt", "tools"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "List all saved agent profiles.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_agent",
                    "description": "Get valid details for a specific agent.",
                    "parameters": {
                        "type": "object", 
                        "properties": {"name": {"type": "string", "description": "Name of the agent."}},
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "deep_research",
                    "description": "Perform deep, multi-step web research with citations using Perplexity AI.",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "query": {"type": "string", "description": "Complex research question or topic."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_agent",
                    "description": "Delegate a sub-task to a specialized agent.",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "agent_name": {"type": "string", "description": "Name of the agent to call (must exist in list_agents)."},
                            "task": {"type": "string", "description": "Specific instruction for the sub-agent."}
                        },
                        "required": ["agent_name", "task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_human",
                    "description": "Ask the user a question or request approval for a dangerous action.",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "question": {"type": "string", "description": "The question or confirmation request."}
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
        return tools

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        session_id: str | None = None,
        response_format: str | None = None,
    ) -> Tuple[str, List[Dict], Dict]:
        """Legacy wrapper for non-streaming callers."""
        final_ans = ""
        total_usage = {}
        docs = []
        
        async for event in self.answer_stream(question, top_k, session_id, response_format):
            if event["type"] == "answer":
                final_ans = event["content"]
            elif event["type"] == "usage":
                total_usage = event["content"]
            elif event["type"] == "docs":
                docs = event["content"]
                
        return final_ans, docs, total_usage

    async def answer_stream(
        self,
        question: str,
        top_k: int = 5,
        session_id: str | None = None,
        response_format: str | None = None,
    ):
        """Yields events: {type: 'thought'|'tool'|'answer'|'error', content: ...}"""
        
        # 1. Retrieve history
        history = []
        if session_id:
            history = self._get_history(session_id)

        # CHECK FOR RESUME
        last_msg = history[-1] if history else None
        is_resume = False
        
        if last_msg and last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            last_tool = last_msg["tool_calls"][-1]
            if last_tool["function"]["name"] == "ask_human":
                # Treat 'question' as user input to the tool
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": last_tool["id"],
                    "name": "ask_human",
                    "content": f"User Answer: {question}"
                }
                history.append(tool_msg)
                self._add_to_history(session_id, "tool", json.dumps(tool_msg))
                messages = history
                is_resume = True
                yield {"type": "info", "content": "Resuming from human input..."}

        if not is_resume:
            # RAG
            docs = await get_agent_answer(question, top_k=top_k, return_docs_only=True)
            yield {"type": "docs", "content": docs}
            
            context_text = "\n\n".join(d["text"] for d in docs)
            
            system_instruction = (
                "You are an expert AI Architect and Agent Builder.\n"
                "Your goal is to help the user design, debug, and build other AI agents.\n"
                "You have access to tools and a file system (/data/workspace).\n\n"
                "## INSTRUCTIONS:\n"
                "1. **THINK**: Prioritize safety. User `ask_human` for critical actions.\n"
                "2. **ACT**: Use tools iteratively.\n"
                "3. **ANSWER**: Final response.\n"
            )
            
            if response_format == "json_object":
                system_instruction += "\n**IMPORTANT**: You MUST output valid JSON only for the final answer."

            system_prompt = f"{system_instruction}\nContext:\n{context_text}"

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})

        # 4. Stream Loop
        final_answer = ""
        user_interrupted = False
        
        async for event in self._run_orchestration_stream(messages, response_format=response_format):
            yield event
            if event["type"] == "answer":
                final_answer = event["content"]
            elif event["type"] == "stop":
                if "INPUT REQUIRED" in event.get("content", ""):
                    user_interrupted = True

        # 5. Update history (Redis)
        if session_id:
            if not is_resume:
                self._add_to_history(session_id, "user", question)
            
            if user_interrupted:
                last_assistant = [m for m in messages if m["role"] == "assistant"][-1]
                self._add_to_history(session_id, "assistant", json.dumps(last_assistant))
            else:
                self._add_to_history(session_id, "assistant", final_answer)

    async def _run_orchestration_loop(self, messages: List[Dict], **kwargs) -> Tuple[str, Dict]:
        """Backward compatible wrapper for delegations."""
        final = ""
        usage = {}
        async for event in self._run_orchestration_stream(messages, **kwargs):
            if event["type"] == "answer": final = event["content"]
            if event["type"] == "usage": usage = event["content"]
        return final, usage

    async def _run_orchestration_stream(
        self, 
        messages: List[Dict], 
        tool_definitions: List[Dict] = None, 
        tools_map: Dict = None,
        response_format: str | None = None
    ):
        """Core ReAct Generator."""
        import httpx
        from app.rag import OPENAI_API_KEY, CHAT_MODEL
        
        current_tool_defs = tool_definitions if tool_definitions is not None else self._tool_definitions
        current_tools_map = tools_map if tools_map is not None else self.tools
        
        max_steps = 7
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for step in range(max_steps):
            yield {"type": "step", "content": f"Step {step+1}/{max_steps}"}
            
            # Call LLM
            async with httpx.AsyncClient(timeout=60.0) as http:
                payload = {
                    "model": CHAT_MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                }
                
                # Check for JSON mode
                if response_format == "json_object":
                    payload["response_format"] = {"type": "json_object"}
                
                if current_tool_defs:
                    payload["tools"] = current_tool_defs
                    payload["tool_choice"] = "auto"

                resp = await http.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json=payload,
                )
            
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            usage = data.get("usage", {})
            
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            
            messages.append(message)
            
            tool_calls = message.get("tool_calls")

            if tool_calls:
                yield {"type": "thought", "content": f"Used {len(tool_calls)} tools."}
                
                # Check Interrupt
                for tc in tool_calls:
                    if tc["function"]["name"] == "ask_human":
                        q = "User Input Required"
                        try: q = json.loads(tc["function"]["arguments"]).get("question", q)
                        except: pass
                        yield {"type": "answer", "content": f"🔴 **INPUT REQUIRED**: {q}"}
                        yield {"type": "stop", "content": f"INPUT REQUIRED"}
                        return

                for tool_call in tool_calls:
                    fn_name = tool_call["function"]["name"]
                    fn_args_str = tool_call["function"]["arguments"]
                    call_id = tool_call["id"]
                    
                    yield {"type": "tool_start", "name": fn_name, "args": fn_args_str}
                    
                    result_content = ""
                    if fn_name in current_tools_map:
                        try:
                            fn_args = json.loads(fn_args_str)
                            fn_res = current_tools_map[fn_name]["fn"](**fn_args)
                            
                            import inspect
                            if inspect.iscoroutine(fn_res):
                                result_content = str(await fn_res)
                            else:
                                result_content = str(fn_res)
                        except Exception as e:
                            logger.error(f"Tool error: {e}")
                            result_content = f"Error: {e}"
                    else:
                        result_content = "Error: Tool not found."

                    yield {"type": "tool_result", "name": fn_name, "content": result_content}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": fn_name,
                        "content": result_content
                    })
            else:
                content = message.get("content", "")
                yield {"type": "answer", "content": content}
                break

        yield {"type": "usage", "content": total_usage}

    def _is_unsafe(self, text: str) -> bool:
        """
        Basic safety filter. In production, use a dedicated model or Guardrails AI.
        """
        unsafe_keywords = ["hack", "exploit", "steal", "illegal", "malware", "keylogger"]
        return any(k in text.lower() for k in unsafe_keywords)

    async def index_document(self, text: str, metadata: Dict | None = None) -> str:
        """
        Split text into chunks and index them in Qdrant.
        """
        chunks = default_chunker.chunk_text(text)
        logger.info(f"Indexing document as {len(chunks)} chunks")
        
        # Embed chunks
        embeddings = await embed(chunks)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "metadata": metadata or {},
                        "chunk_index": i
                    },
                )
            )

        if points:
            client.upsert(collection_name=COLLECTION, points=points)
            
        return f"Indexed {len(points)} chunks."

    async def ingest_file(self, filename: str, content: bytes, content_type: str) -> str:
        """Parse and index a file (PDF, MD, TXT)."""
        text = ""
        
        try:
            if content_type == "application/pdf" or filename.endswith(".pdf"):
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract:
                        text += extract + "\n"
            else:
                # Assume text/markdown
                text = content.decode("utf-8")
                
            if not text.strip():
                return "Error: Empty file content."
                
            return await self.index_document(text, metadata={"filename": filename, "type": content_type})
            
        except Exception as e:
            logger.error(f"File ingestion error: {e}")
            return f"Error processing file: {e}"


agent_service = AgentService()
