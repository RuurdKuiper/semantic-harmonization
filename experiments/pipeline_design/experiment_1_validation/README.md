# Experiment 1: validation and design choices

Inputs: ADEM, acute myocardial infarction, eczema vaccinatum, erythema multiforme,
Guillain–Barré syndrome, kidney disease, and myocarditis. References enter scoring
and validation-only selection, rather than model prompts.

1. Rank the full terminology corpus using BM25 alone, MiniLM alone, and hybrid
   lexical weights 0.05–0.95 in increments of 0.05. Select the highest mean EDF
   average precision. Report recall at 200, 1,000, and 5,000, and per-EDF values.
   This selects a tested retrieval weight; it does not establish a global optimum
   or the best final classification score for every possible retrieval method.
2. Compare Jev, OpenAI Decisions, and JPT-4B on
   that ranking. Use binary Narrow/Exclude questions in batches of 20. Stop after
   ten consecutive batches with zero Narrow decisions, after at least three
   batches. A shared 10,000-candidate ceiling per EDF bounds runs when the
   adaptive condition never triggers. The user chose this ceiling after Laya
   continued including unrelated codes. Ceiling hits are reported separately;
   previous verified ranked-prefix outputs are reused. Select the highest mean
   EDF F1 overall and within each branch.
   Laya's multilingual checkpoint was withdrawn at the user's request after
   excessive Narrow decisions, including unrelated codes, in early validation
   runs. Its incomplete diagnostic outputs are retained and reported separately,
   rather than entering model selection. JPT-4B is the remaining local decision model.
3. Compare GPT-6 Luna, GPT-6.1 Sol, and Qwen3.5-9B 8-bit on the same best overall
   preliminary answers. Review confidence <0.3, at most 300 per EDF, lowest
   confidence first. Reasoning is enabled: medium effort for GPT, a 1,024-token
   thinking budget for Qwen (temperature 0.6, top-p 0.95, top-k 20, seed 42).
   Also evaluate reviewers on the best compatible preliminary model for
   each complete branch. Select proprietary reviewers within their branch;
   Qwen supplies the local reviewer.
4. Apply CodeMapper after review to each branch's selected output. Link direct
   shared CUIs across selected vocabularies, with no descendants or recursion.
   Cache results and pace all operations at six requests/minute. Retain mapping
   in a frozen branch only if it increases validation mean EDF F1.

Results: `retrieval.csv`, `decision_metrics.csv`, `review_metrics.csv`,
`mapping_metrics.csv`, and the corresponding selection JSON files in the configured
results directory. Partial files contain only completed cases and never establish
final winners. Each decision and review file retains batch durations and runtime
information. The concise [report](../report.md) is generated from these artifacts.
