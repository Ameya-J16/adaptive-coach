# Eval results

Every live run of the two golden sets, kept as history. **The current numbers
are the 2026-11-20 runs**; the August files record how the critic was built
and are kept deliberately.

Critic set: 14 hand-labelled plans (10 with a pass/fail verdict, 4 testing a
single flaw dimension). Routing set: 14 labelled athlete requests.
"False pass" = a bad plan scored as passing, the dangerous direction.

## Current (deployed configuration)

| File | Models | Embeddings | Result |
|---|---|---|---|
| `critic_eval_2026-11-20T12-14-09.json` | agents gpt-5.4-mini, **critic gpt-5.4**, reasoning effort low | local bge-base | **9/10 verdicts (90%), 0 false passes**, 1 false fail (`good_endurance_mix`, 0.725) |
| `routing_eval_2026-11-20T12-07-17.json` | supervisor gpt-5.4-mini, reasoning effort low | n/a | **14/14 routed correctly** |

## History: building the critic (2026-08-05, GPT-4o)

| File | What it was | Embeddings | Verdicts |
|---|---|---|---|
| `critic_eval_2026-08-05T03-00-18.json` | First live run of the golden set | OpenAI* | 70% (3 false fails) |
| `critic_eval_2026-08-05T03-02-34.json` | Critic iteration | OpenAI* | 90% |
| `critic_eval_2026-08-05T14-05-55.json` | Critic iteration | OpenAI* | 90% |
| `critic_eval_2026-08-05T14-06-41.json` | **Single-case debug run** (`--only`), not the full set | OpenAI* | 1/1 |
| `critic_eval_2026-08-05T14-09-39.json` | Last full run before the embedding comparison | OpenAI* | 100% |
| `critic_eval_2026-08-05T18-27-30.json` | Same critic, free local embeddings (small) | bge-small | 80% |
| `critic_eval_2026-08-05T20-25-10.json` | Same critic, free local embeddings (base), the setting the app ships | bge-base | **90%** |

\* Runs before 18:27 did not yet record their backend in the file. Commit
`09c5f40` reports the embedding comparison these runs make up (OpenAI 1.00,
bge-base 0.90, bge-small 0.80, GPT-4o held constant), so the 14:09 run is the
OpenAI-embeddings result; the earlier ones are inferred to match. Between the
first run and 14:09 the critic got three fixes (a retry on unparseable JSON, a
materiality bar so it stopped overcorrecting, and a hard cap so a rule
violation can't be averaged away); the files don't record which run had which.

**Every run has 0 false passes.** The only 100% result used paid OpenAI
embeddings; with the embeddings the app actually ships, verdict accuracy is
90% on both GPT-4o and GPT-5.4.

Re-run with `python -m evals.run_eval` and `python -m evals.run_routing_eval`
(both need an API key). New reports land here with a timestamp.
