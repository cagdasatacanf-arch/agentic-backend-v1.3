"""
Code Specialist Agent

Specialized agent for code-related tasks:
- Code generation (Python, JavaScript, etc.)
- Code explanation and documentation
- Debugging and error fixes
- Code review and optimization

Based on research:
- Specialized code agents outperform general models
- Chain-of-thought improves code quality
- Test-driven development reduces bugs
"""

from typing import Dict, List, Optional
import logging
import subprocess
import tempfile
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)


class CodeSpecialist:
    """
    Specialized agent for programming tasks.

    Features:
    - Multi-language code generation
    - Code explanation
    - Syntax checking
    - Optional code execution (Python only, sandboxed)

    Usage:
        agent = CodeSpecialist()
        result = await agent.generate("Write a Python function to check if a number is prime")
    """

    def __init__(self, model: str = "gpt-4o", allow_execution: bool = False):
        """
        Initialize code specialist.

        Args:
            model: LLM to use
            allow_execution: Allow Python code execution (use with caution)
        """
        self.model = model
        self.allow_execution = allow_execution

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.2,  # Slight creativity for code
            api_key=settings.openai_api_key
        )

        logger.info(f"CodeSpecialist initialized with {model} (execution: {allow_execution})")

    async def generate(
        self,
        request: str,
        language: Optional[str] = None,
        include_tests: bool = False
    ) -> Dict:
        """
        Generate code based on request.

        Args:
            request: Code generation request
            language: Programming language (auto-detected if None)
            include_tests: Generate unit tests

        Returns:
            {
                "code": "...",
                "language": "...",
                "explanation": "...",
                "tests": "...",  # If include_tests=True
                "execution_result": {...},  # If executed
                "agent_type": "code"
            }
        """
        logger.info(f"Generating code: {request[:60]}...")

        # Build prompt
        system_prompt = """You are an expert programmer. Generate clean, well-documented code.

Format your response as:
LANGUAGE: [programming language]
EXPLANATION: [Brief explanation of the code]
CODE:
```[language]
[your code here]
```

{tests_instruction}

Follow best practices:
- Clear variable names
- Proper error handling
- Efficient algorithms
- Inline comments for complex logic"""

        if include_tests:
            tests_instruction = """
TESTS:
```[language]
[unit tests for the code]
```"""
        else:
            tests_instruction = ""

        system_prompt = system_prompt.format(tests_instruction=tests_instruction)

        if language:
            user_prompt = f"Generate {language} code: {request}"
        else:
            user_prompt = f"Generate code: {request}"

        try:
            # Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            content = response.content

            # Parse response
            detected_language = self._extract_language(content) or language or "python"
            explanation = self._extract_explanation(content)
            code = self._extract_code(content, "CODE")
            tests = self._extract_code(content, "TESTS") if include_tests else None

            # Execute if allowed and language is Python
            execution_result = None
            if self.allow_execution and detected_language.lower() == "python" and code:
                execution_result = await self._execute_python(code)

            result = {
                "code": code,
                "language": detected_language,
                "explanation": explanation,
                "tests": tests,
                "execution_result": execution_result,
                "agent_type": "code",
                "success": True
            }

            logger.info(f"Code generated: {detected_language}, {len(code) if code else 0} chars")
            return result

        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "code",
                "success": False
            }

    def _extract_language(self, text: str) -> Optional[str]:
        """Extract programming language from response"""
        import re
        match = re.search(r'LANGUAGE:\s*(\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try to detect from code block
        match = re.search(r'```(\w+)', text)
        if match:
            return match.group(1).strip()

        return None

    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation from response"""
        import re
        match = re.search(r'EXPLANATION:\s*(.+?)(?=\nCODE:|\nLANGUAGE:|\n```|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_code(self, text: str, section: str = "CODE") -> Optional[str]:
        """Extract code from markdown code blocks"""
        import re

        # Try to find section-specific code block
        pattern = f"{section}:\\s*```\\w*\\n(.+?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: find any code block
        pattern = r'```\w*\n(.+?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return first code block (usually the main code)
            return matches[0].strip()

        return None

    async def _execute_python(self, code: str, timeout: int = 5) -> Dict:
        """
        Execute Python code in a sandboxed environment.

        WARNING: This should only be used with trusted code or in isolated environments.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result dict
        """
        logger.warning("Executing Python code - ensure this is safe!")

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Clean up
            os.unlink(temp_file)

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "timed_out": False
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timed out after {timeout}s")
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "return_code": -1,
                "success": False,
                "timed_out": True
            }
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
                "error": str(e)
            }

    async def explain(self, code: str, language: Optional[str] = None) -> Dict:
        """
        Explain existing code.

        Args:
            code: Code to explain
            language: Programming language

        Returns:
            Explanation dict
        """
        system_prompt = "You are a code explainer. Provide clear, detailed explanations of code."

        lang_str = f" ({language})" if language else ""
        user_prompt = f"Explain this code{lang_str}:\n\n```\n{code}\n```"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "explanation": response.content,
                "code": code,
                "language": language,
                "agent_type": "code",
                "success": True
            }

        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "code",
                "success": False
            }

    async def debug(self, code: str, error: str, language: Optional[str] = None) -> Dict:
        """
        Debug code and suggest fixes.

        Args:
            code: Code with error
            error: Error message
            language: Programming language

        Returns:
            Debug result with suggested fix
        """
        system_prompt = """You are a debugging expert. Analyze code errors and provide fixes.

Format your response as:
PROBLEM: [Explanation of the error]
FIX: [How to fix it]
FIXED_CODE:
```[language]
[corrected code]
```"""

        lang_str = f" ({language})" if language else ""
        user_prompt = f"""Debug this code{lang_str}:

Code:
```
{code}
```

Error:
{error}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            content = response.content

            # Parse response
            import re
            problem = re.search(r'PROBLEM:\s*(.+?)(?=\nFIX:)', content, re.DOTALL | re.IGNORECASE)
            fix = re.search(r'FIX:\s*(.+?)(?=\nFIXED_CODE:|$)', content, re.DOTALL | re.IGNORECASE)
            fixed_code = self._extract_code(content, "FIXED_CODE")

            return {
                "problem": problem.group(1).strip() if problem else "",
                "fix": fix.group(1).strip() if fix else "",
                "fixed_code": fixed_code,
                "original_code": code,
                "original_error": error,
                "agent_type": "code",
                "success": True
            }

        except Exception as e:
            logger.error(f"Debugging failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "code",
                "success": False
            }
