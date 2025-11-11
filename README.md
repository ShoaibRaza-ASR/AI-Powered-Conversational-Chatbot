# AI Conversational Chatbot


This repository contains a FastAPI-based conversational chatbot using LangChain and OpenAI. It provides REST API endpoints for chat, context management, sentiment analysis, and entity extraction. A small frontend is included for manual testing.


## Features
- `/chat` — conversational chat with context
- `/context` — get/set conversation context
- `/sentiment` — sentiment analysis (positive/negative/neutral)
- `/entities` — entity extraction returns JSON array
- Optional Redis-backed context persistence (via `REDIS_URL`)
- Non-blocking calls to LangChain using `run_in_threadpool`



## Files
See repository root for `app.py`, `requirements.txt`, `.env.example`, and the `frontend/` folder.


## Setup (local)
1. Clone the repo:
```bash
git clone <repo-url>
cd <repo-folder>

2.Create & activate virtual environment (example):

python -m venv .venv
source .venv/bin/activate # macOS / Linux
.\.venv\Scripts\activate # Windows (cmd)

3.Install dependencies:
   pip install -r requirements.txt

4.Create .env (based on .env.example) and set OPENAI_API_KEY and optionally REDIS_URL.

5.Run the app:

uvicorn app:app --reload --port 8000

6.Open API docs:

Swagger UI: http://127.0.0.1:8000/docs

Frontend: http://127.0.0.1:8000/frontend/index.html


Docker (optional)
Build and run with docker-compose (includes optional Redis):

docker-compose up --build

Deploy to GitHub / Platform
1.Initialize git and push to GitHub:

git init
git add .
git commit -m "Initial chatbot project"
git branch -M main
git remote add origin <your-git-repo-url>
git push -u origin main

2.For Heroku / Render / Railway, follow platform-specific steps. Use Dockerfile or Procfile.



Notes:-->

Costs: using gpt-4 is expensive. Change LLM_MODEL or model parameter in .env if needed.

For production, secure your API endpoints with authentication, add rate limiting, and persist context with Redis/DB.