# MOEX Agent (Local) — MVP

This is a minimal web UI + tool-calling wrapper for your local LM Studio server.

## Prerequisites
- LM Studio server running (OpenAI-compatible) on http://127.0.0.1:1234/v1
- Model loaded: qwen/qwen2.5-coder-32b

## Install (Windows / PowerShell)
From your project root (C:\Users\nitro\moex-ai-bot):

```powershell
python -m venv .venv_agent
.\.venv_agent\Scripts\Activate.ps1
pip install fastapi uvicorn openai
```

## Run
```powershell
python -m uvicorn out.agent.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:
http://127.0.0.1:8000

## Environment variables (optional)
- LMSTUDIO_BASE_URL (default http://127.0.0.1:1234/v1)
- LMSTUDIO_MODEL (default qwen/qwen2.5-coder-32b)
- LMSTUDIO_API_KEY (default lm-studio)
