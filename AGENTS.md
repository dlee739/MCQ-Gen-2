# MCQ-Gen 2 project context

These decisions are intentional and should be preserved unless the user explicitly changes them.

- Generation is one-shot: send the complete source in one request. Do not reintroduce page chunking or batching.
- Cost is a primary product constraint. Aim for legacy-level generation cost or only a modest increase.
- Extracting the complete PDF text is the default input path. Users may explicitly choose direct PDF input when page visuals matter.
- High-Volume is the default mode and its current question quality is considered successful. Avoid adding complexity without a concrete need.
- High-Quality should improve reasoning and distractor quality within the original generation request.
- Do not add an LLM reviewer, grader, semantic quality pass, or automatic paid retry without explicit user approval.
- SATA questions may intentionally have zero, one, two, three, or all four source choices correct. Grade them by exact-set matching.
- New SATA sets enforce a center-weighted 5/20/40/30/5 distribution across zero through four correct source choices within the same one-shot response.
- Every SATA question has a fixed fifth choice E, "None of the above." It is correct only when none of A-D is correct and must never be shuffled away from the last position.
- The model returns letter-free correct and incorrect choice buckets with a rationale attached to each choice. The application shuffles those tagged choice objects, assigns A-D, and derives the answer key from bucket membership; never ask the model to provide answer letters.
- Editable instruction profiles are separate by generation mode, with one default per mode. They replace the old preference block but never control the output schema or fixed structural rules.
- Bulk run deletion removes sets, attempts, and bookmarks but preserves instruction profiles.
- Keep all runtime prompt wording in `mcqgen2/prompts.py`, separate from API request logic.
- Keep product copy domain-neutral; do not repeatedly label the app or questions with a particular profession or subject.
