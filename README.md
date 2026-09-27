# MCQ-Gen 2

A focused local application for generating and practicing source-grounded questions from a PDF.

> **Beta:** Version `0.1.0b1` is intended for trusted, single-user Windows testing.
> It is not a hosted multi-user service.

## What it does

- Sends the complete source in one OpenAI Responses API request—never page chunks.
- Extracts all PDF text locally by default to reduce token cost, with direct-PDF visual input available when needed.
- Generates exactly 1–100 MCQ or SATA questions using Structured Outputs.
- Provides High-Volume and High-Quality generation modes.
- Provides editable, reusable instruction profiles with a separate default for each mode.
- Supports `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`.
- Records actual input/output token usage and estimates cost from a dated pricing snapshot.
- Saves question sets and quiz attempts locally in SQLite.
- Tracks incorrect questions until they are answered correctly on a later attempt.
- Supports persistent bookmarks, including mixed bookmarked-question practice.
- Can delete all generated runs and bookmarks while preserving instruction profiles.
- Exports test results to PDF.

## Before setup

Generation uses the paid OpenAI API. Create your own OpenAI project API key and set a
[project spend alert or hard monthly spend limit](https://developers.openai.com/api/docs/guides/spend-limits)
before testing. Never share a personal key or commit it to Git.

Source material is sent to OpenAI for generation. The application sets `store=False`, does
not retain uploaded PDF bytes or extracted text locally after generation, and stores only
generated questions, usage information, profiles, bookmarks, and quiz history in SQLite.
Do not upload material you are not permitted to share or process.

## Windows setup

The tested beta setup uses Python 3.13.

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/windows/)
   with the Python Launcher enabled.
2. Download or clone this repository.
3. Double-click `setup_beta.bat` once.
4. Open `.env` and add your own OpenAI project API key:

   ```text
   OPENAI_API_KEY=your-project-key-here
   ```

`.env` is ignored by Git. The application never writes the key to its database or logs.

For manual setup instead:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

## Run

Double-click `run_app.bat`, or run:

```powershell
.venv\Scripts\python -m streamlit run streamlit_app.py
```

Application data is stored in `data/mcqgen.sqlite3`. Uploaded PDFs and extracted text are handled in memory and are not retained after generation.

Generation is intentionally one-shot. MCQ-Gen 2 never splits a source into page batches
or issues separate generation requests.

## Input processing

- **Extract text (default):** extracts every page locally, adds page markers, and sends all
  extracted text in one request. This is the recommended lower-cost option.
- **Send original PDF:** sends the PDF through the API file-input path so page images and
  diagrams are available. This can use substantially more input tokens.

If a PDF has too little extractable text, the app asks the user to choose the original-PDF
option. It never switches to the more expensive input mode silently.

## Question types and instruction profiles

- **MCQ** questions have exactly one correct answer.
- **SATA** (Select All That Apply) questions may have zero to four correct source choices
  and are graded by exact-set matching. Every SATA question includes a fixed fifth option,
  **None of the above**, which is correct only when none of A–D is correct.
- Each generation mode has its own library of editable instruction profiles. The selected
  rules can be edited for the current request, saved over the profile, saved as a new profile,
  or made the default for that mode. Fixed source-grounding and response-structure rules are
  not part of the editable text.

Fixed runtime prompt wording and factory-default mode rules are centralized in
`mcqgen2/prompts.py`. Saved profile rules live in SQLite and can be managed from the app.

## Test

```powershell
.venv\Scripts\python -m pytest
```

Automated tests use a fake OpenAI client and do not make paid API calls.

## Pricing note

The built-in pricing table was verified on 2026-09-25 against the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing). Costs shown by the app are estimates based on API-reported token usage and the stored pricing snapshot, not invoices. Update `mcqgen2/pricing.py` when OpenAI pricing changes.

## Beta limitations and bug reports

This release is local and single-user. It does not provide accounts, shared hosting, cloud
sync, automatic backups, or automatic updates. Generated questions can contain errors and
should be checked against the source.

See [`BETA_RELEASE_NOTES.md`](BETA_RELEASE_NOTES.md) for the exact information to include
when reporting a bug. Never include an API key or private source material in a report.
