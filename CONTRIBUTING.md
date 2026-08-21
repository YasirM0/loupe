# Contributing

Loupe's UI is one React component (`src/Loupe.jsx`), with a few extracted pieces: `src/lib/` (BM25, text extraction, sentence/claim splitting, shared NLI classification logic), `src/workers/inference.worker.js` (the local ML pipeline, off the main thread), and `bench/` (the model-quality benchmark). There's no backend and no build step beyond Vite.

## Setup

```bash
npm install
npm run dev
```

## Adding an LLM provider (API-key based)

If it speaks the OpenAI-compatible `chat/completions` shape, add an entry to `PROVIDER_DEFAULTS` in `src/Loupe.jsx` with `kind: 'openai'`, a default `baseUrl`, a default `model`, and an `aliases` array (lowercase names someone might type into the "bring your own API key" box — see `matchApiProviders`). That's the whole change — the request/response handling in `callLLM` is shared across every `kind: 'openai'` provider.

If it needs a genuinely different request shape (like Anthropic's `content` blocks and PDF document support), it needs a new branch in `callLLM`.

## Adding a local embedding or NLI model

Add an entry to `EMBED_MODELS` or `NLI_MODELS` in `src/Loupe.jsx` with the Hugging Face model id, download size, and a `quality` field. **That `quality` number must come from actually running `npm run bench`** (see [`bench/`](bench)) against the new model — never hand-write a percentage. The benchmark imports `classifyPair`/`verdictFromScores` from `src/lib/nli.js`, the same code the app runs in production, so the number reflects reality. If you add a case to `bench/testset.mjs`, re-run the benchmark for every existing model too so the comparison stays apples-to-apples, and update the numbers in `README.md` alongside `src/Loupe.jsx`.

## Changing the verification prompt

The grounding rules live in `ABSOLUTE_RULES`, shared between the claim-checking prompt (`claimInstructions`) and the contradiction-hunting prompt (`CONTRADICTION_INSTRUCTIONS`). Keep both prompts returning the same JSON shape the UI expects (see the `citedClaims` / `uncitedClaims` / `contradictions` schemas inline) — changing a field name means updating the corresponding render code (`ClaimCard`, `UncitedClaimCard`, `ContradictionCard`) too.

## Pull requests

Small, focused changes are easiest to review. If you're adding a feature, a one-paragraph description of the use case in the PR body is more useful than a long implementation writeup — the diff speaks for the implementation.
