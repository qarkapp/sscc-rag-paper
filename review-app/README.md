# HetDocQA Review App

A single-file web app for the human-validation pass over generated HetDocQA
candidates. No build step and no backend — just open the HTML file.

## Use

1. Generate the review bundle (done automatically by `scripts/build_hetdocqa.py`,
   which writes `data/hetdocqa/review.json`).
2. Open `review-app/review.html` in a browser (it loads React and Tailwind from a
   CDN, so an internet connection is needed on first load).
3. Click **Load review.json** and select `data/hetdocqa/review.json`.

## Reviewing

Each question shows the drafted question, an editable answer, the type label, and
the **gold evidence highlighted in context** within each source document. For every
candidate, decide:

- Is it answerable from the shown evidence?
- Is the gold-evidence set correct and minimal?
- Is the type label right?
- Is the question natural and unambiguous?

Then **Accept** or **Reject**. You can edit the answer and type inline before
accepting. Decisions persist in the browser (localStorage), so you can stop and
resume.

Keyboard: `a` accept · `r` reject · `j`/`k` or `↑`/`↓` navigate.

## Export

- **Export accepted.jsonl** — the validated questions (accepted only, with your
  edits applied), in the same schema as `questions.jsonl`. This is the released
  benchmark file.
- **Export decisions** — the full decision record (`{qid: {status, type, answer,
  notes}}`) for provenance and to re-derive the dataset.
