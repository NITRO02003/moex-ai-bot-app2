import os
import json
import re
import ast
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI

# -----------------------------
# Configuration (edit if needed)
# -----------------------------
PROJECT_ROOT = Path(r"C:\Users\nitro\moex-ai-bot")
CLI_CWD = PROJECT_ROOT  # run from repo root for reliable module imports

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen2.5-coder-32b")
# LM Studio typically doesn't require an API key for localhost; OpenAI client expects one anyway.
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

app = FastAPI(title="MOEX Agent (MVP)")

# Static UI
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


class ChatRequest(BaseModel):
    message: str
    # Optional: allow per-request model override
    model: Optional[str] = None


def tool_read_text(path: str, max_chars: int = 12000) -> Dict[str, Any]:
    p = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    # Prevent escaping outside project root unless absolute path is explicitly provided
    if not Path(path).is_absolute():
        try:
            p.relative_to(PROJECT_ROOT.resolve())
        except Exception:
            raise HTTPException(status_code=400, detail=f"Path escapes project root: {path}")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {p}")

    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        data = data[:max_chars] + "\n\n...[truncated]..."
    return {"path": str(p), "content": data}


def tool_run_cli(args: List[str], timeout_sec: int = 600) -> Dict[str, Any]:
    """
    Runs python -m app2.cli ... in CLI_CWD.
    Example args:
      ["range-v3-backtest", "--symbols", "SBER", ...]
    """
    if not CLI_CWD.exists():
        raise HTTPException(status_code=500, detail=f"CLI_CWD not found: {CLI_CWD}")

    cmd = [sys.executable, "-m", "app2.cli"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(CLI_CWD),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "cmd": cmd, "timeout_sec": timeout_sec, "stdout": "", "stderr": "TIMEOUT"}

    # Keep tails to avoid huge payloads
    def tail(s: str, n: int = 12000) -> str:
        return s[-n:] if len(s) > n else s

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "cwd": str(CLI_CWD),
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Read a text file from the project (relative to project root) and return its content (possibly truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to project root (e.g. app2/range/config.json)."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": 12000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": "Run a moex-ai-bot CLI command: python -m app2.cli <args...> from the app2 directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments after `python -m app2.cli` (e.g. ['range-v3-backtest','--symbols','SBER']).",
                    },
                    "timeout_sec": {"type": "integer", "default": 600},
                },
                "required": ["args"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a senior quantitative developer and code reviewer helping with a MOEX trading system.

Priorities:
- correctness over speed
- verify using tools instead of guessing
- minimal invasive changes; prefer pointing to exact files/params
- reproducibility via CLI commands

Rules:
- Do not output install scripts or unrelated commands.
- Before making claims, use read_text/run_cli as needed.
- When diagnosing '0 trades', always inspect config and strategy code via tools.
- Keep answers structured and concise.
"""

# Simple in-memory session (per server process). For a real app, persist this per user.
SESSION_MESSAGES: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]



def extract_explicit_tools(user_text: str) -> Dict[str, Any]:
    """
    Heuristic fallback for environments where the model/server does not support OpenAI tool-calling.
    Supports:
      - "Прочитай файл: <path>" or "Read file: <path>"
      - "run_cli args = [ ... ]" (Python list syntax) OR a command line starting with "python -m app2.cli ..."
    """
    out: Dict[str, Any] = {"read_paths": [], "cli_args": None}

    for m in re.finditer(r'(?:Прочитай\s+файл|Read\s+file)\s*:\s*([^\n\r]+)', user_text, flags=re.IGNORECASE):
        out["read_paths"].append(m.group(1).strip())

    m = re.search(r'run_cli\s+args\s*=\s*(\[[^\]]*\])', user_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            out["cli_args"] = ast.literal_eval(m.group(1))
        except Exception:
            out["cli_args"] = None

    if out["cli_args"] is None:
        m2 = re.search(r'python\s+-m\s+app2\.cli\s+([^\n\r]+)', user_text, flags=re.IGNORECASE)
        if m2:
            import shlex
            out["cli_args"] = shlex.split(m2.group(1))

    return out

def call_model(messages: List[Dict[str, Any]], model: str) -> Any:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
        top_p=0.9,
    )


def handle_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    name = tc["function"]["name"]
    args_json = tc["function"].get("arguments") or "{}"
    try:
        args = json.loads(args_json)
    except Exception:
        args = {}

    if name == "read_text":
        return tool_read_text(path=args.get("path", ""), max_chars=int(args.get("max_chars", 12000)))
    if name == "run_cli":
        return tool_run_cli(args=args.get("args", []), timeout_sec=int(args.get("timeout_sec", 600)))

    return {"error": f"Unknown tool: {name}"}


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/api/chat")
def chat(req: ChatRequest):
    model = req.model or LMSTUDIO_MODEL

    # Append user message
    SESSION_MESSAGES.append({"role": "user", "content": req.message})

    # Heuristic tool execution (fallback if model/server does not support tool-calling)
    explicit = extract_explicit_tools(req.message)
    explicit_tool_log: List[Dict[str, Any]] = []
    if explicit.get("read_paths"):
        for p in explicit["read_paths"]:
            try:
                out = tool_read_text(path=p, max_chars=12000)
            except Exception as e:
                out = {"error": str(e)}
            explicit_tool_log.append({"tool": "read_text", "arguments": json.dumps({"path": p}, ensure_ascii=False), "result": out})
            SESSION_MESSAGES.append({"role": "tool", "tool_call_id": f"explicit_read_{len(explicit_tool_log)}", "content": json.dumps(out, ensure_ascii=False)})

    if explicit.get("cli_args"):
        out = tool_run_cli(args=explicit["cli_args"], timeout_sec=600)
        explicit_tool_log.append({"tool": "run_cli", "arguments": json.dumps({"args": explicit["cli_args"]}, ensure_ascii=False), "result": out})
        SESSION_MESSAGES.append({"role": "tool", "tool_call_id": "explicit_run_cli", "content": json.dumps(out, ensure_ascii=False)})

    tool_log: List[Dict[str, Any]] = []

    # Tool-calling loop (cap to avoid infinite loops)
    for _ in range(6):
        resp = call_model(SESSION_MESSAGES, model=model)
        msg = resp.choices[0].message

        # If no tool calls, finalize
        if not getattr(msg, "tool_calls", None):
            SESSION_MESSAGES.append({"role": "assistant", "content": msg.content or ""})
            return {"answer": msg.content or "", "tool_log": (explicit_tool_log + tool_log)}

        # Add assistant message with tool calls
        SESSION_MESSAGES.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )

        # Execute each tool call
        for tc in msg.tool_calls:
            tc_d = tc.model_dump()
            out = handle_tool_call(tc_d)
            tool_log.append({"tool": tc_d["function"]["name"], "arguments": tc_d["function"].get("arguments", "{}"), "result": out})

            # Provide tool output back to model
            SESSION_MESSAGES.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_d["id"],
                    "content": json.dumps(out, ensure_ascii=False),
                }
            )

    raise HTTPException(status_code=500, detail="Tool loop exceeded limit (possible runaway tool-calling).")
