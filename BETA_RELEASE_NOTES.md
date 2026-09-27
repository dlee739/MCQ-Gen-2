# MCQ-Gen 2 v0.1.0-beta.1

This is a local, single-user Windows beta. Each tester should use their own OpenAI
project API key and configure a
[project spend alert or hard monthly spend limit](https://developers.openai.com/api/docs/guides/spend-limits).

## Install and run

1. Install Python 3.13 for Windows with the Python Launcher enabled.
2. Download or clone this repository.
3. Run `setup_beta.bat` once.
4. Add your own project API key to `.env`.
5. Run `run_app.bat`.

## Important limitations

- Generation uses the paid OpenAI API. The displayed cost is an estimate, not an invoice.
- Uploaded source material is sent to OpenAI for generation. Do not upload material you
  are not permitted to share or process.
- Extracted-text input is the lower-cost default. Direct PDF input includes page visuals
  and can use substantially more input tokens.
- Generated questions can contain errors and should be checked against the source.
- This beta is local and single-user. It has no accounts, shared hosting, cloud sync,
  automatic backups, or automatic updates.
- Application data is stored in `data/mcqgen.sqlite3` on the local computer.

## Reporting a bug

Include all of the following:

- App version (`0.1.0b1`)
- Generation mode and API model
- MCQ or SATA
- Extracted text or direct PDF input
- Number of requested questions
- The exact visible error message or unexpected behavior
- Reproduction steps, without including your API key or private source material
