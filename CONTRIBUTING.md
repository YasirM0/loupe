# Contributing

Loupe is a single-file React component (`src/Loupe.jsx`) plus a thin Vite scaffold around it. There's no backend and no build step beyond Vite.

## Setup

```bash
npm install
npm run dev
```

## Adding a provider

If it speaks the OpenAI-compatible `chat/completions` shape, add an entry to `PROVIDER_DEFAULTS` in `src/Loupe.jsx` with `kind: 'openai'`, a default `baseUrl`, and a default `model`. That's the whole change — the request/response handling in `callLLM` is shared across every `kind: 'openai'` provider.

If it needs a genuinely different request shape (like Anthropic's `content` blocks and PDF document support), it needs a new branch in `callLLM`.

## Changing the verification prompt

The grounding rules live in `ABSOLUTE_RULES`, shared between the claim-checking prompt (`claimInstructions`) and the contradiction-hunting prompt (`CONTRADICTION_INSTRUCTIONS`). Keep both prompts returning the same JSON shape the UI expects (see the `citedClaims` / `uncitedClaims` / `contradictions` schemas inline) — changing a field name means updating the corresponding render code (`ClaimCard`, `UncitedClaimCard`, `ContradictionCard`) too.

## Pull requests

Small, focused changes are easiest to review. If you're adding a feature, a one-paragraph description of the use case in the PR body is more useful than a long implementation writeup — the diff speaks for the implementation.
