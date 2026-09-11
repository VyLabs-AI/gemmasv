const deployment = window.HERO_DEMO_CONFIG || {};

const $ = (selector) => document.querySelector(selector);

const state = {
  client: null,
  apiConfig: null,
  replay: true,
  replayKey: null,
  sessionId: null,
  phase: "idle", // idle | mark | ask | busy | recalled | forgotten | ended
  domain: "custom",
  fact: null,
  factBubble: null,
  spanStart: null, // offsets within the fact text
  spanEnd: null,
  auditTarget: null,
  targetStart: null,
  deleteScope: "span",
  recordId: null,
  deletionRanges: null,
  recordAudits: null,
  recordNeighbor: null,
  deletionLabel: null,
  auditProbeOverride: null,
  question: null,
  registeredProbe: null,
  certificateDone: false,
  certificateJobId: null,
  twinPanes: null,
  toastTimer: null,
};

/* ------------------------------------------------------------------ *
 * Scaffolding recipe — must mirror gemma_sv/demo_server/scenarios.py *
 * ------------------------------------------------------------------ */

function normalizeQuestion(question) {
  const cleaned = question.split(/\s+/).filter(Boolean).join(" ");
  if (!cleaned) return "";
  return /[?.!]$/.test(cleaned) ? cleaned : `${cleaned}?`;
}

function scaffoldMemory(question, fact) {
  return `Question: ${question} Answer: ${fact}`;
}

function scaffoldOffset(question) {
  return `Question: ${question} Answer: `.length;
}

function stemProbe(question, fact, spanStartInFact) {
  const stem = fact.slice(0, spanStartInFact).replace(/\s+$/, "");
  if (stem) return `\n\nQuestion: ${question}\nAnswer: ${stem}`;
  return `\n\nQuestion: ${question}\nAnswer:`;
}

function bareProbe(question) {
  return `\n\nQuestion: ${question}\nAnswer:`;
}

/* ------------------- API clients ------------------- */

class DemoApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async initialize() {
    await this.request("/healthz", { timeoutMs: 4_000 });
    return this.request("/api/v1/config", { timeoutMs: 4_000 });
  }

  createSession(domain) {
    return this.request("/api/v1/sessions", { method: "POST", body: { domain } });
  }

  ingest(sessionId, payload) {
    return this.request(`/api/v1/sessions/${sessionId}/ingest`, {
      method: "POST",
      body: payload,
    });
  }

  recall(sessionId) {
    return this.request(`/api/v1/sessions/${sessionId}/recall`, { method: "POST" });
  }

  forget(sessionId) {
    return this.request(`/api/v1/sessions/${sessionId}/forget`, { method: "POST" });
  }

  job(sessionId, jobId) {
    return this.request(`/api/v1/jobs/${jobId}`, {
      headers: { "X-Demo-Session-ID": sessionId },
    });
  }

  attack(sessionId, payload) {
    return this.request(`/api/v1/sessions/${sessionId}/attacks`, {
      method: "POST",
      body: payload,
    });
  }

  twin(sessionId) {
    return this.request(`/api/v1/sessions/${sessionId}/twin`, { method: "POST" });
  }

  guess(sessionId, pane) {
    return this.request(`/api/v1/sessions/${sessionId}/twin/guess`, {
      method: "POST",
      body: { pane },
    });
  }

  purge(sessionId, keepalive = false) {
    return this.request(`/api/v1/sessions/${sessionId}`, {
      method: "DELETE",
      keepalive,
    });
  }

  async request(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      options.timeoutMs || 15 * 60_000,
    );
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    const init = {
      method: options.method || "GET",
      headers,
      signal: controller.signal,
      cache: "no-store",
      keepalive: Boolean(options.keepalive),
    };
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    try {
      const response = await fetch(`${this.baseUrl}${path}`, init);
      if (response.status === 204) return null;
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(data?.detail?.code || `http_${response.status}`);
        error.code = data?.detail?.code || `http_${response.status}`;
        throw error;
      }
      return data;
    } finally {
      clearTimeout(timeout);
    }
  }
}

class RecordedReplayClient {
  constructor(data) {
    this.data = data;
    this.sessions = new Map();
  }

  initialize() {
    return Promise.resolve(this.data.config);
  }

  createSession(domain, replayKey = domain) {
    if (!this.data.runs[replayKey]) throw codedError("custom_requires_live_backend");
    const sessionId = `replay-${crypto.randomUUID()}`;
    this.sessions.set(sessionId, { domain, replayKey, deletedPane: null });
    return Promise.resolve({ session_id: sessionId, domain, expires_in_seconds: 1800 });
  }

  ingest(sessionId, payload) {
    const session = this.get(sessionId);
    session.payload = payload;
    const run = this.run(session);
    const ranges = payload.deletion_ranges || [
      { start: payload.secret_start, end: payload.secret_end },
    ];
    const selectedValues = ranges.map(({ start, end }) =>
      payload.memory_text.slice(start, end),
    );
    return Promise.resolve({
      evidence: "recorded_replay",
      selected_value: selectedValues[0],
      selected_values: selectedValues,
      audit_target:
        payload.audit_target ||
        selectedValues[0],
      deletion_scope: payload.delete_scope === "span" ? "field" : payload.delete_scope,
      record_id: payload.record_id || null,
      field_audits: run.field_audits || null,
      neighbor: run.neighbor || null,
      memory_tokens: run.memory_tokens,
      memory_copies: run.memory_copies,
      selected_positions: run.selected_positions,
      distance_beyond_window: run.distance_beyond_window,
      local_window: run.local_window,
      conversation_log: run.conversation_log || null,
      message: "Recorded run loaded. Your text was not sent to a model.",
    });
  }

  recall(sessionId) {
    const run = this.run(this.get(sessionId));
    return Promise.resolve({ evidence: "recorded_replay", ...run.recall });
  }

  forget(sessionId) {
    const session = this.get(sessionId);
    const run = this.run(session);
    session.jobId = `replay-job-${crypto.randomUUID()}`;
    return Promise.resolve({
      forget: {
        evidence: "recorded_replay",
        method: "exact",
        ...run.forget,
        deletion_ms: null,
        message: "Recorded behavioral deletion.",
      },
      certificate_job: { job_id: session.jobId, status: "succeeded", progress: 1 },
    });
  }

  job(sessionId, jobId) {
    const session = this.get(sessionId);
    if (session.jobId !== jobId) throw codedError("job_not_found");
    return Promise.resolve({
      job_id: jobId,
      status: "succeeded",
      progress: 1,
      error_code: null,
      result: {
        evidence: "recorded_replay",
        probe: session.payload.audit_probe,
        ...this.run(session).certificate,
      },
    });
  }

  attack(sessionId, payload) {
    const session = this.get(sessionId);
    const recorded = this.run(session).attacks[payload.method];
    if (!recorded || payload.kind !== "extraction") {
      throw codedError("unsupported_attack_method");
    }
    const ratio = recorded.target_probability / Math.max(recorded.floor_probability, 1e-300);
    return Promise.resolve({
      evidence: "recorded_replay",
      method: payload.method,
      kind: payload.kind,
      target_probability: recorded.target_probability,
      floor_probability: recorded.floor_probability,
      probability_ratio_to_floor: ratio,
      at_floor: ratio <= 1.5,
      generated_text: recorded.generated_text,
      message: "Recorded attack outcome; no visitor prompt was executed.",
    });
  }

  twin(sessionId) {
    const session = this.get(sessionId);
    const run = this.run(session);
    session.deletedPane = crypto.getRandomValues(new Uint8Array(1))[0] % 2 ? "A" : "B";
    const refitPane = session.deletedPane === "A" ? "B" : "A";
    return Promise.resolve({
      evidence: "recorded_replay",
      panes: {
        [session.deletedPane]: { top_tokens: run.twins.exact },
        [refitPane]: { top_tokens: run.twins.refit },
      },
      scope: "registered_audit_probe",
    });
  }

  guess(sessionId, pane) {
    const session = this.get(sessionId);
    return Promise.resolve({
      correct: pane === session.deletedPane,
      deleted_pane: session.deletedPane,
      scope: "registered_audit_probe",
      certificate: this.run(session).certificate,
    });
  }

  purge(sessionId) {
    this.sessions.delete(sessionId);
    return Promise.resolve(null);
  }

  get(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) throw codedError("session_not_found");
    return session;
  }

  run(session) {
    return this.data.runs[session.replayKey || session.domain];
  }
}

/* ------------------- boot ------------------- */

async function initialize() {
  wireEvents();
  try {
    if (deployment.forceReplay || !deployment.apiBase) throw new Error("replay_configured");
    const client = new DemoApiClient(deployment.apiBase);
    const config = await client.initialize();
    state.client = client;
    state.apiConfig = config;
    state.replay = !config.live_compute;
    const modelLabel = (config.model_id || "Gemma-3-4B")
      .replace(/^google\//, "")
      .replace(/-pt$/, "");
    if (!state.replay) {
      setRuntime("live", `live compute · ${modelLabel} on M3 Ultra`);
    } else {
      setRuntime("replay", "API online · recorded engine");
      showModeBanner(
        "The API is running its recorded engine, so numbers replay the committed float64 run. " +
          "Start the server with HERO_ENGINE=gemma for live compute.",
      );
    }
  } catch (_error) {
    const response = await fetch("assets/replay.json", { cache: "no-store" });
    if (!response.ok) throw new Error("replay_unavailable");
    const client = new RecordedReplayClient(await response.json());
    state.client = client;
    state.apiConfig = await client.initialize();
    state.replay = true;
    setRuntime("replay", "recorded replay · backend offline");
    showModeBanner(
      "Recorded fallback: every number below comes from a committed float64 Gemma run. " +
        "Free typing is disabled and nothing you enter is executed.",
    );
  }
  $("#identity-copy").textContent =
    deployment.reviewMode ?? state.apiConfig.review_mode
      ? "Anonymous research demonstration."
      : "Exact Memory Lab research demonstration.";
  resetConversation();
}

function wireEvents() {
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    onSend();
  });
  $("#composer-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  });
  $("#purge-session").addEventListener("click", () => purgeSession(true));
  document.addEventListener("selectionchange", maybeShowMarkPopover);
  $("#mark-confirm").addEventListener("click", confirmMark);
  window.addEventListener("pagehide", () => {
    if (state.sessionId && state.client instanceof DemoApiClient) {
      state.client.purge(state.sessionId, true).catch(() => {});
    }
  });
}

/* ------------------- transcript helpers ------------------- */

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  if (role !== "system") {
    const label = document.createElement("span");
    label.className = "role";
    label.textContent = role === "user" ? "you" : "model";
    bubble.append(label);
  }
  bubble.append(document.createTextNode(text));
  $("#transcript").append(bubble);
  bubble.scrollIntoView({ block: "end" });
  return bubble;
}

function addSystem(text, { error = false } = {}) {
  const bubble = addMessage("system", text);
  if (error) bubble.classList.add("error");
  return bubble;
}

function addMetrics(bubble, rows, verdict) {
  const metrics = document.createElement("div");
  metrics.className = "metrics";
  rows.forEach(([label, value]) => {
    const line = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = `${label}: `;
    const strong = document.createElement("b");
    strong.textContent = value;
    line.append(name, strong);
    metrics.append(line);
  });
  bubble.append(metrics);
  if (verdict) {
    const badge = document.createElement("span");
    badge.className = `verdict ${verdict.tone}`;
    badge.textContent = verdict.text;
    bubble.append(badge);
  }
  bubble.scrollIntoView({ block: "end" });
}

function setChips(chips) {
  const row = $("#chip-row");
  row.replaceChildren();
  chips.forEach(({ label, onClick, danger = false, disabled = false, title = "" }) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = danger ? "chip danger" : "chip";
    chip.textContent = label;
    chip.disabled = disabled;
    if (title) chip.title = title;
    chip.addEventListener("click", onClick);
    row.append(chip);
  });
}

function setComposer({ enabled, placeholder }) {
  const input = $("#composer-input");
  input.disabled = !enabled;
  input.placeholder = placeholder;
  $("#send-button").disabled = !enabled;
}

function firstLine(text) {
  const line = (text || "").split("\n")[0].trim();
  return line || "(empty completion)";
}

/* ------------------- persistent-memory log ------------------- */

function renderConversationLog(ingest) {
  const log = ingest.conversation_log;
  if (!Array.isArray(log) || log.length === 0) return null;
  const entries = log.filter((segment) => segment.kind !== "header");
  const records = entries.filter((segment) => segment.kind === "record").length;
  const exchanges = entries.length - records;

  const details = document.createElement("details");
  details.className = "memory-log";
  const summary = document.createElement("summary");
  summary.textContent =
    `Persistent memory — audited field ${ingest.distance_beyond_window} tokens back ` +
    `(${ingest.memory_tokens} total tokens; ${exchanges} synthetic padding exchanges)`;
  details.append(summary);

  const body = document.createElement("div");
  body.className = "log-body";
  const location = document.createElement("div");
  location.className = "target-location";
  location.textContent =
    `The highlighted field is beyond Gemma's ${ingest.local_window}-token local window. ` +
    "The source record spans the boundary, so it is shown as one readable block below.";
  body.append(location);
  let boundaryPlaced = false;
  for (const segment of entries) {
    if (!boundaryPlaced && !segment.beyond_window) {
      body.append(logBoundary(ingest.local_window));
      boundaryPlaced = true;
    }
    body.append(logEntry(segment));
  }
  const probeNote = document.createElement("div");
  probeNote.className = "log-probe-note";
  probeNote.textContent =
    "▼ questions are appended here, at the most recent end of the log";
  const copyNote = document.createElement("div");
  copyNote.className = "log-probe-note";
  copyNote.textContent =
    `The record appears ${records} times to stabilize recall; deletion removes ` +
    "the marked tokens from every copy.";
  body.append(copyNote, probeNote);
  details.append(body);
  $("#transcript").append(details);
  details.scrollIntoView({ block: "end" });
  return details;
}

function logBoundary(window) {
  const divider = document.createElement("div");
  divider.className = "log-boundary";
  divider.textContent =
    `everything above is beyond the ${window}-token local attention window — ` +
    "only the SV gate can reach it";
  return divider;
}

function logEntry(segment) {
  const entry = document.createElement("div");
  if (segment.kind === "record") {
    entry.className = "log-entry record";
    const tag = document.createElement("span");
    tag.className = "log-tag";
    tag.textContent = `your record · ${segment.tokens_from_end} tokens from the end`;
    entry.append(tag, recordTextWithMark(segment.text));
    return entry;
  }
  entry.className = "log-entry";
  const split = segment.text.split(" Assistant: ");
  if (split.length === 2) {
    const user = document.createElement("div");
    user.textContent = split[0];
    const assistant = document.createElement("div");
    assistant.textContent = `Assistant: ${split[1]}`;
    entry.append(user, assistant);
  } else {
    entry.textContent = segment.text;
  }
  return entry;
}

function recordTextWithMark(text) {
  const wrapper = document.createElement("div");
  const start = scaffoldOffset(state.question) + state.spanStart;
  const end = scaffoldOffset(state.question) + state.spanEnd;
  if (
    Number.isFinite(start) &&
    Number.isFinite(end) &&
    end > start &&
    text.slice(start, end) === state.fact.slice(state.spanStart, state.spanEnd)
  ) {
    const mark = document.createElement("mark");
    mark.textContent = text.slice(start, end);
    wrapper.append(
      document.createTextNode(text.slice(0, start)),
      mark,
      document.createTextNode(text.slice(end)),
    );
  } else {
    wrapper.textContent = text;
  }
  return wrapper;
}

/* ------------------- conversation flow ------------------- */

function resetConversation() {
  state.sessionId = null;
  state.replayKey = null;
  state.phase = "idle";
  state.domain = "custom";
  state.fact = null;
  state.factBubble = null;
  state.spanStart = null;
  state.spanEnd = null;
  state.auditTarget = null;
  state.targetStart = null;
  state.deleteScope = "span";
  state.recordId = null;
  state.deletionRanges = null;
  state.recordAudits = null;
  state.recordNeighbor = null;
  state.deletionLabel = null;
  state.auditProbeOverride = null;
  state.question = null;
  state.registeredProbe = null;
  state.certificateDone = false;
  state.certificateJobId = null;
  $("#transcript").replaceChildren();
  $("#purge-session").disabled = true;
  hideMarkPopover();
  addSystem(
    state.replay
      ? "Start with a guided example: remember one made-up record, delete it, then audit the result."
      : "Recommended: run a guided example first. Custom synthetic facts are exploratory and may fail the recall check.",
  );
  const presets = [...(state.apiConfig?.presets || [])].sort(
    (left, right) => Number(Boolean(right.guided)) - Number(Boolean(left.guided)),
  );
  setChips([
    ...presets.map((preset) => ({
      label: preset.guided
        ? `Guided example: ${preset.title}`
        : `Stress test: ${preset.title.replace(/^Experimental:\s*/, "")}`,
      onClick: () => runPreset(preset),
      disabled: state.replay && !preset.replay_key,
      title:
        state.replay && !preset.replay_key
          ? "This scenario requires the live backend."
          : preset.description || "",
    })),
  ]);
  setComposer({
    enabled: !state.replay,
    placeholder: state.replay
      ? "Recorded replay — free typing is disabled."
      : "Tell the model one clearly fictional fact…",
  });
}

async function onSend() {
  const input = $("#composer-input");
  const text = input.value.trim();
  if (!text || input.disabled) return;
  if (state.phase === "idle") {
    input.value = "";
    acceptFact(text, state.domain);
  } else if (state.phase === "ask") {
    input.value = "";
    await askQuestion(text);
  } else if (state.phase === "forgotten") {
    input.value = "";
    await freeformAttack(text);
  }
}

function acceptFact(fact, domain, preset = null) {
  if (fact.length > 600) {
    toast("Keep the fact under 600 characters.");
    return;
  }
  state.phase = "mark";
  state.domain = domain;
  state.fact = fact;
  state.auditTarget = preset?.audit_target || null;
  state.targetStart = state.auditTarget ? fact.indexOf(state.auditTarget) : null;
  state.deleteScope = preset?.delete_scope || "span";
  state.recordId = preset?.record_id || preset?.slug || null;
  state.replayKey = preset?.replay_key || null;
  state.deletionRanges = preset?.deletion_ranges || null;
  state.deletionLabel = preset?.deletion_label || null;
  state.auditProbeOverride = preset?.audit_probe || null;
  state.factBubble = addMessage("user", fact);
  state.factBubble.dataset.factBubble = "1";
  if (preset) {
    applyMark(preset.fact.indexOf(preset.selected_value), preset.fact.indexOf(preset.selected_value) + preset.selected_value.length);
  } else {
    addSystem("Select the exact value to protect inside your message above, then confirm the mark.");
    setComposer({ enabled: false, placeholder: "Mark the protected value first…" });
    setChips([{ label: "Start over", onClick: () => purgeSession(false) }]);
  }
}

function addHistoryTurn(turn, markRecord = false) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${turn.side === "model" ? "model" : "user"} history-turn`;
  if (markRecord) bubble.classList.add("record-scope");
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = turn.speaker;
  bubble.append(label, document.createTextNode(turn.text));
  $("#transcript").append(bubble);
  return bubble;
}

function addStructuredIndex(fact, auditTarget) {
  const indexText = fact.split("Conversation transcript:")[0].trim();
  if (!indexText) return;
  const card = document.createElement("div");
  card.className = "record-index";
  const title = document.createElement("strong");
  title.textContent = "Structured index derived from the conversation";
  const content = document.createElement("code");
  const targetStart = indexText.indexOf(auditTarget);
  if (targetStart >= 0) {
    const mark = document.createElement("mark");
    mark.textContent = auditTarget;
    content.append(
      document.createTextNode(indexText.slice(0, targetStart)),
      mark,
      document.createTextNode(indexText.slice(targetStart + auditTarget.length)),
    );
  } else {
    content.textContent = indexText;
  }
  card.append(title, content);
  $("#transcript").append(card);
}

function acceptPresetConversation(preset) {
  state.phase = "ask";
  state.domain = preset.domain;
  state.fact = preset.fact;
  state.factBubble = null;
  state.spanStart = preset.fact.indexOf(preset.selected_value);
  state.spanEnd = state.spanStart + preset.selected_value.length;
  state.auditTarget = preset.audit_target || preset.selected_value;
  state.targetStart = preset.fact.indexOf(state.auditTarget);
  state.deleteScope = preset.delete_scope || "span";
  state.recordId = preset.record_id || preset.slug || null;
  state.replayKey = preset.replay_key || null;
  state.deletionRanges = preset.deletion_ranges || null;
  state.deletionLabel = preset.deletion_label || preset.selected_value;
  state.auditProbeOverride = preset.audit_probe || null;

  addSystem(
    "Loaded synthetic conversation history. The bracketed turns are the source record; " +
      "the service stores a structured index beside it.",
  );
  const turns = preset.conversation || [];
  turns.forEach((turn) => addHistoryTurn(turn, true));
  addStructuredIndex(preset.fact, state.auditTarget);
  addSystem(
    `Selected for certified deletion: ${state.deletionLabel}. ` +
      `The audit will ask for ${JSON.stringify(state.auditTarget)} before and after deletion.`,
  );
  if (preset.domain === "medicine") {
    addSystem(
      "This fast path deletes the indexed diagnosis field. Purging every patient turn " +
        "and any ingestion-time imprint requires rebuilding memory without the whole record.",
    );
  }
  setComposer({ enabled: false, placeholder: "Running the registered recall question…" });
  setChips([{ label: "Start over", onClick: () => purgeSession(false) }]);
}

function applyMark(start, end) {
  state.spanStart = start;
  state.spanEnd = end;
  const fact = state.fact;
  if (!state.auditTarget) {
    state.auditTarget = fact.slice(start, end);
    state.targetStart = start;
    state.deleteScope = "span";
    state.deletionRanges = null;
    state.deletionLabel = state.auditTarget;
  }
  const bubble = state.factBubble;
  // Rebuild the bubble body: role label + text with <mark> around the span.
  bubble.replaceChildren();
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = "you";
  const mark = document.createElement("mark");
  mark.textContent = fact.slice(start, end);
  bubble.append(
    label,
    document.createTextNode(fact.slice(0, start)),
    mark,
    document.createTextNode(fact.slice(end)),
  );
  hideMarkPopover();
  state.phase = "ask";
  addSystem(
    `Marked "${fact.slice(start, end)}". Now ask the model a question whose answer is the marked value.`,
  );
  setComposer({
    enabled: !state.replay,
    placeholder: "Ask the model about the fact…",
  });
  setChips([{ label: "Start over", onClick: () => purgeSession(false) }]);
}

async function askQuestion(rawQuestion) {
  const question = normalizeQuestion(rawQuestion);
  if (!question) return;
  if (question.length > 300) {
    toast("Keep the question under 300 characters.");
    return;
  }
  const value = state.fact.slice(state.spanStart, state.spanEnd);
  const auditTarget = state.auditTarget || value;
  if (question.toLocaleLowerCase().includes(auditTarget.toLocaleLowerCase())) {
    toast("The question must not contain the protected value itself.");
    return;
  }
  addMessage("user", rawQuestion.trim());
  state.question = question;
  state.registeredProbe =
    state.auditProbeOverride ||
    stemProbe(
      question,
      state.fact,
      state.targetStart ?? state.spanStart,
    );
  state.phase = "busy";
  setComposer({ enabled: false, placeholder: "Working…" });
  setChips([]);
  const thinking = addSystem("Placing the record in persistent memory…");
  try {
    const created = await state.client.createSession(
      state.domain,
      state.replayKey || state.domain,
    );
    state.sessionId = created.session_id;
    $("#purge-session").disabled = false;
    const memoryText = scaffoldMemory(question, state.fact);
    const ingest = await state.client.ingest(state.sessionId, {
      memory_text: memoryText,
      secret_start: scaffoldOffset(question) + state.spanStart,
      secret_end: scaffoldOffset(question) + state.spanEnd,
      deletion_ranges: state.deletionRanges,
      audit_probe: state.registeredProbe,
      audit_target: auditTarget,
      delete_scope: state.deleteScope,
      record_id: state.recordId,
    });
    state.recordAudits = ingest.field_audits || null;
    state.recordNeighbor = ingest.neighbor || null;
    thinking.textContent =
      `Stored as one deletable ${state.deleteScope}: ${ingest.memory_tokens} memory tokens, ` +
      `${ingest.selected_positions} selected token positions, ending ` +
      `${ingest.distance_beyond_window} tokens beyond the ` +
      `${ingest.local_window}-token local window. Querying…`;
    renderConversationLog(ingest);
    const recall = await state.client.recall(state.sessionId);
    renderRecall(recall);
    if (state.recordAudits?.length) {
      addSystem(
        `Whole-record audit registered for ${state.recordAudits.length} fields: ` +
          state.recordAudits.map((field) => field.name).join(", ") +
          ". Every field is checked against its never-stored floor.",
      );
    }
  } catch (error) {
    thinking.remove();
    reportError(error);
    await cleanupFailedSession();
    state.phase = "ask";
    setComposer({ enabled: !state.replay, placeholder: "Ask the model about the fact…" });
    setChips([{ label: "Start over", onClick: () => purgeSession(false) }]);
  }
}

function renderRecall(result) {
  const bubble = addMessage("model", firstLine(result.generated_text));
  const admission = result.admission || {};
  const recalled = admission.status === "recalled";
  addMetrics(
    bubble,
    [
      ["answer-span probability", formatProbability(result.target_probability)],
      ["never-told floor", formatProbability(result.floor_probability)],
      ["lift over floor", `${formatNumber(admission.probability_ratio)}×`],
      ["raw completion", JSON.stringify(result.generated_text || "")],
    ],
    recalled
      ? { tone: "ok", text: "recall gate passed" }
      : { tone: "warn", text: "recall gate failed — no deletion story will be claimed" },
  );
  if (recalled) {
    state.phase = "recalled";
    const label =
      state.deletionLabel || state.fact.slice(state.spanStart, state.spanEnd);
    addSystem(
      `The model retrieved the audit answer from ${label}. You can now delete that ${state.deleteScope}.`,
    );
    setComposer({ enabled: false, placeholder: "Delete the selected record to continue…" });
    setChips([
      { label: `Delete ${label}`, danger: true, onClick: revoke },
      { label: "Start over", onClick: () => purgeSession(false) },
    ]);
  } else {
    addSystem(
      admission.message ||
        "Recall was too weak; reformulate the fact or the question and try again.",
      { error: true },
    );
    addSystem(
      "Tip: facts that pass put a distinctive made-up name right after a describing " +
        "word (visitor, operator, threat actor, diagnosed with\u2026) and ask for exactly " +
        "that name — e.g. \u201CFictional visitor Yesh Okafor is here at dock nine.\u201D " +
        "\u2192 \u201CWho is here at dock nine?\u201D",
    );
    setChips([{ label: "Start over", onClick: () => purgeSession(false) }]);
    setComposer({ enabled: false, placeholder: "Purge and try a clearer fact." });
  }
}

async function revoke() {
  if (state.phase !== "recalled") return;
  state.phase = "busy";
  const label =
    state.deletionLabel || state.fact.slice(state.spanStart, state.spanEnd);
  addMessage("user", `Delete ${label} from persistent memory.`);
  setChips([]);
  const line = addSystem("Evicting the marked keys from the gate state…");
  try {
    const response = await state.client.forget(state.sessionId);
    const forget = response.forget;
    line.textContent =
      `Behavioral deletion done${forget.deletion_ms != null ? ` in ${(forget.deletion_ms / 1000).toFixed(2)} s` : ""}. ` +
      `Answer-span probability ${formatProbability(forget.before_probability)} → ` +
      `${formatProbability(forget.after_probability)} (never-told floor ` +
      `${formatProbability(forget.floor_probability)}).`;
    if (state.recordAudits?.length) {
      const fields = state.recordAudits
        .map(
          (field) =>
            `${field.name}: ${formatProbability(field.present_probability)} → ` +
            `${formatProbability(field.deleted_probability)} ` +
            `(never ${formatProbability(field.never_probability)})`,
        )
        .join("; ");
      const neighbor = state.recordNeighbor
        ? ` Neighbor retained: ${formatProbability(state.recordNeighbor.present_probability)} → ` +
          `${formatProbability(state.recordNeighbor.deleted_probability)}.`
        : "";
      addSystem(`All record fields audited — ${fields}.${neighbor}`);
    }
    state.certificateJobId = response.certificate_job.job_id;
    state.phase = "forgotten";
    renderAttackChips();
    setComposer({
      enabled: !state.replay,
      placeholder: "Try to get it back — ask anything (without typing the value)…",
    });
    pollCertificate();
  } catch (error) {
    line.remove();
    reportError(error);
    state.phase = "recalled";
    renderAttackChips();
  }
}

function renderAttackChips() {
  if (state.phase !== "forgotten") return;
  const chips = [];
  if (!state.replay) {
    chips.push({
      label: "Ask the same question again",
      onClick: () => runAttack("exact", "freeform", state.registeredProbe, "same question, after deletion"),
    });
  }
  chips.push(
    {
      label: "Extraction attack vs deletion",
      onClick: () => runAttack("exact", "extraction", null, "extraction attack vs exact deletion"),
    },
    {
      label: "Extraction attack vs ICUL baseline",
      title: "ICUL leaves the fact in memory and merely instructs the model to retract it.",
      onClick: () => runAttack("icul", "extraction", null, "extraction attack vs ICUL retraction"),
    },
  );
  chips.push({
    label: state.certificateDone ? "Twin test" : "Twin test (waiting for certificate)",
    disabled: !state.certificateDone,
    onClick: startTwin,
  });
  chips.push({ label: "Start over", onClick: () => purgeSession(false) });
  setChips(chips);
}

async function freeformAttack(text) {
  const value = state.fact.slice(state.spanStart, state.spanEnd);
  if (text.toLocaleLowerCase().includes(value.toLocaleLowerCase())) {
    toast("Typing the value into the prompt creates a fresh in-window copy — that is outside the deletion claim.");
    return;
  }
  const question = normalizeQuestion(text);
  addMessage("user", text);
  await runAttack("exact", "freeform", bareProbe(question), "your question, after deletion", false);
}

async function runAttack(method, kind, prompt, label, echoUser = true) {
  if (state.phase !== "forgotten") return;
  if (echoUser) {
    addMessage("user", label.charAt(0).toUpperCase() + label.slice(1) + ".");
  }
  const waiting = addSystem("Running one empirical stress test…");
  try {
    const result = await state.client.attack(state.sessionId, {
      method,
      kind,
      budget: null,
      prompt,
    });
    waiting.remove();
    const bubble = addMessage("model", firstLine(result.generated_text));
    addMetrics(
      bubble,
      [
        ["condition", label],
        ["answer-span probability", formatProbability(result.target_probability)],
        ["never-told floor", formatProbability(result.floor_probability)],
        ["ratio to floor", `${formatNumber(result.probability_ratio_to_floor)}×`],
      ],
      result.at_floor
        ? { tone: "ok", text: "at the never-told floor" }
        : { tone: "bad", text: "still above the floor — leaked" },
    );
    renderAttackChips();
  } catch (error) {
    waiting.remove();
    reportError(error);
    renderAttackChips();
  }
}

/* ------------------- certificate + twin ------------------- */

async function pollCertificate() {
  const line = document.createElement("div");
  line.className = "cert-line";
  line.textContent = "float64 verification against refit: queued…";
  $("#transcript").append(line);
  try {
    for (;;) {
      if (!state.sessionId) return;
      let job;
      try {
        job = await state.client.job(state.sessionId, state.certificateJobId);
      } catch (error) {
        if (error?.code === "rate_limited") {
          await sleep(5_000);
          continue;
        }
        throw error;
      }
      const percent = Math.round((job.progress || 0) * 100);
      line.textContent = `float64 verification against refit: ${percent}%`;
      if (job.status === "succeeded") {
        const result = job.result;
        state.certificateDone = true;
        line.classList.add(
          result.band === "outside_demo_tolerance" || result.band === "invalid"
            ? "warn"
            : "done",
        );
        line.textContent =
          `certificate at the registered question: KL = ${formatScientific(result.kl_nats)} nats ` +
          `(${result.decrement_fallbacks || 0} refit fallbacks). ${result.message || ""}`;
        renderAttackChips();
        return;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        throw codedError(job.error_code || `certificate_${job.status}`);
      }
      await sleep(1_000);
    }
  } catch (error) {
    line.classList.add("warn");
    line.textContent = `verification unavailable: ${friendlyError(error)}`;
  }
}

async function startTwin() {
  if (!state.certificateDone) return;
  try {
    const result = await state.client.twin(state.sessionId);
    state.twinPanes = result.panes;
    const card = document.createElement("div");
    card.className = "twin-card";
    const heading = document.createElement("h3");
    heading.textContent = "Blind comparison: deleted vs. never stored";
    const intro = document.createElement("p");
    intro.textContent =
      "We asked the same registered question of two memories. One originally contained " +
      "the indexed field and then deleted it. The other was refit without that field.";
    const goal = document.createElement("p");
    goal.className = "twin-goal";
    goal.textContent =
      "If deletion is exact, the two answer distributions should be indistinguishable. " +
      "Try to identify the deleted one—but a correct guess is only luck, not evidence.";
    card.append(heading, intro, goal);
    const panes = document.createElement("div");
    panes.className = "twin-panes";
    for (const paneId of ["A", "B"]) {
      const pane = document.createElement("button");
      pane.type = "button";
      pane.className = "twin-pane";
      pane.dataset.pane = paneId;
      const title = document.createElement("strong");
      title.textContent = `Anonymous output ${paneId}`;
      const columns = document.createElement("div");
      columns.className = "token-columns";
      columns.innerHTML = "<span>possible next word</span><span>probability</span>";
      pane.append(title, columns);
      (result.panes[paneId].top_tokens || []).forEach((item) => {
        const row = document.createElement("div");
        row.className = "token-line";
        const token = document.createElement("span");
        token.textContent = JSON.stringify(item.token);
        const probability = document.createElement("span");
        probability.textContent = formatProbability(item.probability);
        row.append(token, probability);
        pane.append(row);
      });
      pane.addEventListener("click", () => guessTwin(card, paneId));
      panes.append(pane);
    }
    const note = document.createElement("p");
    note.className = "twin-note";
    note.textContent =
      "These rows are the five most likely next words—not five stored records. " +
      "The certificate compares every vocabulary probability, not just this preview.";
    card.append(panes, note);
    $("#transcript").append(card);
    card.scrollIntoView({ block: "end" });
  } catch (error) {
    reportError(error);
  }
}

async function guessTwin(card, paneId) {
  try {
    const result = await state.client.guess(state.sessionId, paneId);
    card.querySelectorAll(".twin-pane").forEach((pane) => {
      pane.disabled = true;
      pane.classList.toggle("selected", pane.dataset.pane === paneId);
      pane.classList.toggle("deleted-pane", pane.dataset.pane === result.deleted_pane);
      const reveal = document.createElement("span");
      reveal.className = "pane-reveal";
      reveal.textContent =
        pane.dataset.pane === result.deleted_pane
          ? "field deleted"
          : "field never stored";
      pane.append(reveal);
    });
    const outcome = document.createElement("p");
    outcome.className = "twin-result";
    const kl = result.certificate?.kl_nats;
    outcome.textContent =
      `You picked ${paneId}; the randomized deleted-memory pane was ${result.deleted_pane}. ` +
      `${result.correct ? "Your guess happened to match." : "They were designed to be hard to distinguish."} ` +
      `The evidence is the full-distribution KL${Number.isFinite(kl) ? ` = ${formatScientific(kl)} nats` : ""}, ` +
      "and it applies to this registered question only.";
    card.append(outcome);
  } catch (error) {
    reportError(error);
  }
}

/* ------------------- presets ------------------- */

async function runPreset(preset) {
  if (state.phase !== "idle") return;
  state.domain = preset.domain;
  state.replayKey = preset.replay_key || null;
  if (Array.isArray(preset.conversation) && preset.conversation.length) {
    acceptPresetConversation(preset);
  } else {
    acceptFact(preset.fact, preset.domain, preset);
  }
  await sleep(350);
  await askQuestion(preset.question);
}

/* ------------------- span marking ------------------- */

function maybeShowMarkPopover() {
  if (state.phase !== "mark" || !state.factBubble) return hideMarkPopover();
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return hideMarkPopover();
  }
  const range = selection.getRangeAt(0);
  const textNode = factTextNode();
  if (
    !textNode ||
    range.startContainer !== textNode ||
    range.endContainer !== textNode
  ) {
    return hideMarkPopover();
  }
  const start = range.startOffset;
  const end = range.endOffset;
  const value = state.fact.slice(start, end);
  if (!value.trim() || value.length > 240) return hideMarkPopover();
  const popover = $("#mark-popover");
  const button = $("#mark-confirm");
  button.textContent = `Protect "${value.length > 40 ? `${value.slice(0, 37)}…` : value}"`;
  popover.dataset.start = String(start);
  popover.dataset.end = String(end);
  const rect = range.getBoundingClientRect();
  popover.style.left = `${window.scrollX + rect.left}px`;
  popover.style.top = `${window.scrollY + rect.bottom + 6}px`;
  popover.classList.remove("hidden");
}

function factTextNode() {
  const nodes = [...state.factBubble.childNodes].filter(
    (node) => node.nodeType === Node.TEXT_NODE,
  );
  return nodes.length === 1 ? nodes[0] : null;
}

function confirmMark() {
  const popover = $("#mark-popover");
  const start = Number(popover.dataset.start);
  const end = Number(popover.dataset.end);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
    applyMark(start, end);
  }
}

function hideMarkPopover() {
  $("#mark-popover").classList.add("hidden");
}

/* ------------------- session management ------------------- */

async function purgeSession(announce) {
  const sessionId = state.sessionId;
  state.sessionId = null;
  if (sessionId) {
    try {
      await state.client.purge(sessionId);
    } catch (_error) {
      // Server TTL is the backstop.
    }
  }
  resetConversation();
  if (announce) toast("Ephemeral session purged.");
}

async function cleanupFailedSession() {
  if (!state.sessionId) return;
  const id = state.sessionId;
  state.sessionId = null;
  $("#purge-session").disabled = true;
  await state.client.purge(id).catch(() => {});
}

/* ------------------- formatting + errors ------------------- */

function formatProbability(value) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return value < 0.001 ? value.toExponential(2) : value.toPrecision(3);
}

function formatScientific(value) {
  if (!Number.isFinite(value)) return "invalid";
  return value.toExponential(2);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 100) return Math.round(value).toLocaleString();
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function setRuntime(mode, label) {
  const status = $("#runtime-status");
  status.className = `runtime-status ${mode}`;
  status.textContent = label;
}

function showModeBanner(message) {
  const banner = $("#mode-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function reportError(error) {
  const message = friendlyError(error);
  addSystem(message, { error: true });
  toast(message);
}

function friendlyError(error) {
  const code = error?.code || error?.message || "unknown_error";
  const messages = {
    custom_requires_live_backend: "Writing your own fact requires the live backend.",
    invalid_secret_span: "The marked span is not valid inside the stored record.",
    selected_span_too_many_tokens: "Mark a shorter value (at most 16 model tokens).",
    selected_span_inside_local_window: "The record could not be placed beyond the local window.",
    memory_too_long: "The fact is too long for this demo.",
    session_capacity_reached: "The demo is full. Try again shortly.",
    certificate_queue_full: "The verification queue is full. Try again shortly.",
    rate_limited: "Request limit reached. Wait a moment and retry.",
    session_not_found: "This ephemeral session expired. Start again.",
    certificate_required: "Wait for the certificate first.",
    freeform_prompt_required: "Type a question first.",
  };
  return messages[code] || `Demo error: ${String(code).replaceAll("_", " ")}`;
}

function codedError(code) {
  const error = new Error(code);
  error.code = code;
  return error;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.remove("hidden");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => element.classList.add("hidden"), 5_000);
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

initialize().catch((error) => {
  setRuntime("replay", "demo unavailable");
  showModeBanner(`The live API and recorded fallback could not initialize: ${error.message}`);
});
