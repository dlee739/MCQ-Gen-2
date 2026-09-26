# MCQ-Gen 2

A focused local application for generating and practicing medical multiple-choice questions from a PDF.

## What it does

- Sends one complete PDF directly to the OpenAI Responses API—no page chunking or local text extraction.
- Generates exactly 1–100 four-choice MCQs using Structured Outputs.
- Provides High-Volume and High-Quality generation modes.
- Supports `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`.
- Records actual input/output token usage and estimates cost from a dated pricing snapshot.
- Saves question sets and quiz attempts locally in SQLite.
- Tracks incorrect questions until they are answered correctly on a later attempt.
- Exports test results to PDF.

## Setup

Requires Python 3.11 or newer.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` and add your API key:

```text
OPENAI_API_KEY=sk-your-key-here
```

`.env` is ignored by Git. The application never writes the key to its database or logs.

## Run

Double-click `run_app.bat`, or run:

```powershell
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Application data is stored in `data/mcqgen.sqlite3`. Uploaded PDFs are sent directly from memory and are not retained locally after generation.

## Test

```powershell
.venv\Scripts\python -m pytest
```

Automated tests use a fake OpenAI client and do not make paid API calls.

## Pricing note

The built-in pricing table was verified on 2026-09-25 against the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing). Costs shown by the app are estimates based on API-reported token usage and the stored pricing snapshot, not invoices. Update `mcqgen2/pricing.py` when OpenAI pricing changes.
