# Hosted hero demo

Static chat-style interface for GitHub Pages. It has no build step and contains an
honestly labeled recorded fallback. The live mode calls the separate FastAPI service
in `gemma_sv.demo_server`.

## Conversation flow

1. The visitor states one clearly fictional fact (or picks a preset conversation).
2. They mark the protected value by highlighting it inside their own message.
3. Their first question becomes the registered audit probe. The client scaffolds a
   `Question: … Answer: …` record and a stem probe under the hood (the recipe in
   `app.js` mirrors `gemma_sv/demo_server/scenarios.py`) because the base model is a
   completion model, not a chat model. Server-side, the record is packed into a
   conversation log padded with fictional `User:`/`Assistant:` sample exchanges so
   the marked span lands beyond the 4B model's 1024-token local window
   (`gemma_engine._QA_FILLERS`).
4. After the recall gate passes, the visitor revokes the value; behavioral deletion
   returns immediately and the float64 certificate streams in as a background job.
5. Post-deletion questions run as labeled empirical attacks (exact vs the ICUL
   prompt-retraction baseline), and the twin test asks the visitor to distinguish
   deletion from a refit that never contained the value.

In recorded replay mode free typing is disabled; the validated fictional patient
field replays using the committed 4B float64 run. The richer cyber conversation is
kept as a live stress test because it does not pass this checkpoint's greedy recall
gate. Live mode additionally exposes the fixed eight-record manifest used by the
certified 1B whole-record benchmark; six passed and two remain visibly labeled
admission failures. Record-scale certificates use a coupled block decrement that
passes post-verification on most affected head-gates (276/320 hero, 260/272 cyber);
the remaining gates fall back to disclosed exact fixed-C refit. Removing the separate
ingestion-time neighbor imprint still requires a full repack.

## Test locally

Install the demo dependencies once:

```bash
.venv311/bin/python -m pip install -r gemma_sv/requirements-demo.txt
```

Terminal 1 — run the API in fast recorded mode:

```bash
HERO_ENGINE=replay \
HERO_ALLOWED_ORIGINS=http://127.0.0.1:8000 \
.venv311/bin/python -m gemma_sv.demo_server
```

Terminal 2 — serve the static files:

```bash
.venv311/bin/python -m http.server 8000 --directory gemma_sv/demo_site
```

Open <http://127.0.0.1:8000/?api=http://127.0.0.1:8001>. The query parameter selects
the local API without editing the committed replay-safe configuration.

If you serve the static files on a different port, the API must allow that origin,
e.g. `HERO_ALLOWED_ORIGINS=http://127.0.0.1:4173` for port 4173; otherwise the
browser blocks the cross-origin calls and the page falls back to recorded replay.
When connecting through an IDE port forward, open the page via `localhost` and
allow both spellings:
`HERO_ALLOWED_ORIGINS=http://localhost:4173,http://127.0.0.1:4173`.

To run the real M3 Ultra backend:

```bash
HERO_ENGINE=gemma \
HERO_RESOLVER_MODE=gemini \
HERO_RESOLVER_MODEL=gemini-3.6-flash \
GEMINI_API_KEY=... \
HERO_ALLOWED_ORIGINS=http://127.0.0.1:8000 \
.venv311/bin/python -m gemma_sv.demo_server
```

The first live request lazily loads the fp32/MPS model. The certificate job separately
loads the fp64/CPU model. The API serializes interactive forwards and runs one
certificate worker. The Gemini key remains server-side; the resolver may
propose only catalog IDs, and deletion still requires an explicit confirmation
against the complete owned exchange. Omit the resolver variables to run the
live model with manual ID-based deletion only.

## Privacy and scope

- Enter synthetic text only; never use credentials or patient data.
- Session text and gate state are process-local, expire after 30 minutes, and can be
  explicitly purged.
- Request bodies are not logged by the application; Uvicorn access logging is off by
  default.
- Behavioral attacks and the float64 registered-probe certificate are labeled
  separately in the transcript.

Production tunnels, review/release identities, browser E2E, containers, and AWS
deployment are deliberately deferred until the local experience is accepted.
