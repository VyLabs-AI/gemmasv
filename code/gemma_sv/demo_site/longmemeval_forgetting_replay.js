const PAYLOAD_URL = "assets/longmemeval_forgetting_final.json";
const PAYLOAD_SCHEMA = "gemma-sv-longmemeval-chat-forgetting-payload-v2";
const PREVIEW_LABEL = "INCOMPLETE CASE STUDY PREVIEW";
const FINAL_REPORT_SHA256 =
  "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6";
const LAB_CONFIG_URL = "/api/lab/config";
const LAB_RESOLVE_URL = "/api/lab/resolve";
const LAB_CONFIRM_URL = "/api/lab/confirm";
const labResolverEnabled =
  new URLSearchParams(window.location.search).get("resolver") === "lab";
const STAGE_IDS = [
  "chat",
  "recall",
  "resolve-request",
  "resolve-proposal",
  "confirm",
  "forget",
  "re-query",
  "retained",
  "certificate",
];

const transcript = document.querySelector("#replay-transcript");
const stageList = document.querySelector("#stage-list");
const nextButton = document.querySelector("#replay-next");
const resetButton = document.querySelector("#replay-reset");
const confirmButton = document.querySelector("#replay-confirm");
const banner = document.querySelector("#replay-banner");
const provenance = document.querySelector("#replay-provenance");
const subtitle = document.querySelector("#replay-subtitle");
const resolverNote = document.querySelector("#resolver-note");

let payload = null;
let labConfig = null;
let labResolution = null;
let labConfirmation = null;
let nextStage = 0;
let awaitingConfirmation = false;
let deletionConfirmed = false;
let requestInFlight = false;


function assert(condition, message) {
  if (!condition) throw new Error(message);
}


async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    throw new Error(`${url} returned a non-JSON response`);
  }
  if (!response.ok) {
    const code = body?.detail?.code || `HTTP ${response.status}`;
    const reason = body?.detail?.reason;
    throw new Error(reason ? `${code} (${reason})` : code);
  }
  return body;
}


function validateLabConfig(value) {
  assert(
    value?.schema === "gemma-sv-lab-meeting-config-v1" &&
      value?.case_record_id === payload.case.record_id,
    "lab app is serving a different case",
  );
  assert(
    ["recorded", "gemini"].includes(value?.resolver?.mode) &&
      typeof value.resolver.model_id === "string" &&
      typeof value.resolver.evidence === "string" &&
      typeof value.resolver.label === "string" &&
      value.resolver.outside_deletion_certificate === true,
    "lab resolver config is invalid",
  );
  assert(
    value?.deletion_evidence?.kind === "recorded_replay" &&
      value.deletion_evidence.live_model_deletion_run === false &&
      value.deletion_evidence.resolver_outside_certificate === true &&
      value.deletion_evidence.payload_sha256 === payload.integrity.sha256 &&
      value.deletion_evidence.method_report_sha256 ===
        payload.provenance.source_artifacts.method_report.file_sha256,
    "lab deletion-evidence boundary differs",
  );
  return value;
}


function measured(bound, label) {
  assert(
    bound &&
      ["method_report", "admission_report"].includes(bound.artifact) &&
      typeof bound.source_pointer === "string" &&
      bound.source_pointer.startsWith("/records/") &&
      typeof bound.value === "number" &&
      Number.isFinite(bound.value),
    `${label} is not a bound finite measurement`,
  );
  return bound.value;
}


function validatePayload(value) {
  assert(
    value?.schema === PAYLOAD_SCHEMA && value?.schema_version === 2,
    "recorded payload schema differs",
  );
  assert(
    /^[0-9a-f]{64}$/.test(value?.integrity?.sha256 || ""),
    "recorded payload integrity descriptor is invalid",
  );
  assert(
    value.contains_model_generated_text === false &&
      value.contains_unquoted_source_text === false &&
      value.raw_context_text_present === false,
    "recorded payload text boundary differs",
  );
  assert(
    Array.isArray(value.scope?.aggregate_claims) &&
      value.scope.aggregate_claims.length === 0 &&
      value.scope.aggregate_claims_allowed === false,
    "recorded case must not contain aggregate claims",
  );
  assert(
    value.status !== "incomplete_case_study_preview" ||
      value.preview_label === PREVIEW_LABEL,
    "partial replay is not visibly labeled",
  );
  assert(
    value.status === "incomplete_case_study_preview" ||
      (value.status === "final_case_study" &&
        value.preview_label === null &&
        value.scope?.final_all16_complete === true &&
        value.provenance?.source_artifacts?.method_report?.file_sha256 ===
          FINAL_REPORT_SHA256),
    "final replay is not bound to the completed all16 report",
  );
  assert(
    value.scope?.behavioral_reference ===
      "fresh raw round-omitted repack" &&
      value.scope?.certificate_reference === "fixed-C retained-key refit" &&
      value.scope?.full_repack_certificate_claimed === false,
    "behavioral and certificate references are not separated",
  );
  assert(
    value.demo?.live_compute === false &&
      value.demo?.network_api_required === false &&
      value.demo?.sequence?.map((step) => step.id).join("|") ===
        STAGE_IDS.join("|"),
    "recorded replay sequence differs",
  );
  assert(
    value.case?.dialogue?.target?.quotes?.length === 2 &&
      value.case?.dialogue?.retained?.quotes?.length === 2 &&
      value.case?.dialogue?.intervening?.quotes?.length === 2,
    "public dialogue excerpts are incomplete",
  );
  for (const section of ["target", "retained", "intervening"]) {
    for (const quote of value.case.dialogue[section].quotes) {
      assert(
        quote.origin === "public_longmemeval_dialogue" &&
          ["user", "assistant"].includes(quote.role) &&
          typeof quote.display_text === "string" &&
          !quote.display_text.includes("...") &&
          !quote.display_text.includes("…") &&
          quote.truncated === false &&
          quote.excerpt_policy?.selection ===
            "exact UTF-8 source substring" &&
          /^[0-9a-f]{64}$/.test(quote.source_turn_sha256),
        "public dialogue quote is not source-bound",
      );
    }
  }
  assert(
    value.case?.deletion_action?.origin ===
      "evaluator_added_deletion_action" &&
      value.case?.deletion_action?.public_longmemeval_source === false &&
      value.case?.deletion_action?.out_of_band === true,
    "evaluation deletion action provenance differs",
  );
  const resolution = value.case?.memory_resolution;
  const catalog = resolution?.candidate_catalog;
  assert(
    resolution?.schema === "gemma-sv-recorded-memory-resolver-fixture-v1" &&
      resolution?.schema_version === 1 &&
      resolution?.mode === "deterministic_recorded_fixture" &&
      resolution?.label ===
        "Deterministic recorded resolver fixture — no provider call" &&
      resolution?.provider?.call_performed === false &&
      resolution?.provider?.artifact_present === false &&
      resolution?.provider?.artifact === null &&
      resolution?.network_request_performed === false &&
      resolution?.model_configuration?.default_model_id ===
        "gemini-3.6-flash" &&
      resolution?.model_configuration?.model_produced_fixture === false &&
      resolution?.outside_certificate_boundary === true &&
      /^[0-9a-f]{64}$/.test(resolution?.catalog_hash || "") &&
      resolution?.request?.text ===
        value.case.deletion_action.display_text &&
      resolution?.request?.sha256 ===
        value.case.deletion_action.display_text_sha256,
    "recorded resolver fixture provenance differs",
  );
  assert(
    Array.isArray(catalog) &&
      catalog.length === 2 &&
      catalog.every(
        (candidate) =>
          /^memory_[0-9a-f]{24}$/.test(candidate.record_id || "") &&
          candidate.deletion_scope === "complete_exchange" &&
          Array.isArray(candidate.source_turn_ids) &&
          candidate.source_turn_ids.length === 2,
      ) &&
      new Set(catalog.map((candidate) => candidate.record_id)).size === 2,
    "recorded resolver catalog is invalid",
  );
  const targetCandidate = catalog.find(
    (candidate) =>
      candidate.owned_exchange_ref === "/case/dialogue/target/quotes",
  );
  const retainedCandidate = catalog.find(
    (candidate) =>
      candidate.owned_exchange_ref === "/case/dialogue/retained/quotes",
  );
  const decision = resolution.decision;
  const confirmation = resolution.confirmation;
  assert(
    targetCandidate &&
      retainedCandidate &&
      decision?.status === "resolved" &&
      decision?.selected_record_id === targetCandidate.record_id &&
      decision?.selected_record_id !== retainedCandidate.record_id &&
      Array.isArray(decision?.alternative_record_ids) &&
      decision.alternative_record_ids.length === 0 &&
      decision.confidence === null &&
      decision.requires_confirmation === true &&
      decision.deletion_executed === false &&
      decision.selection_rule ===
        "predeclared_exact_bicycle_update_fixture" &&
      catalog.some(
        (candidate) =>
          candidate.record_id === decision.selected_record_id,
      ),
    "recorded resolver decision is not catalog-bound",
  );
  assert(
    confirmation?.selected_record_id === targetCandidate.record_id &&
      confirmation?.catalog_hash === resolution.catalog_hash &&
      confirmation?.requires_explicit_click === true &&
      confirmation?.confirmed_in_payload === false &&
      Array.isArray(confirmation?.owned_exchange) &&
      confirmation.owned_exchange.length ===
        value.case.dialogue.target.quotes.length &&
      confirmation.owned_exchange.every(
        (quote, index) =>
          quote.role === value.case.dialogue.target.quotes[index].role &&
          quote.display_text ===
            value.case.dialogue.target.quotes[index].display_text &&
          quote.source_turn_sha256 ===
            value.case.dialogue.target.quotes[index].source_turn_sha256,
      ),
    "recorded resolver confirmation exchange is incomplete",
  );
  assert(
    value.provenance?.public_dialogue?.assembly_disclosure?.includes(
      "not one organic continuous conversation",
    ),
    "benchmark assembly provenance is missing",
  );
  return value;
}


function createElement(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}


function appendMessage(role, label, text, className = "") {
  const message = createElement(
    "div",
    `msg ${role} ${className}`.trim(),
  );
  message.append(createElement("span", "role", label));
  message.append(document.createTextNode(text));
  transcript.append(message);
  transcript.scrollTop = transcript.scrollHeight;
  return message;
}


function appendQuote(quote, section) {
  const message = appendMessage(
    quote.role === "user" ? "user" : "model",
    `${quote.role} · public LongMemEval · ${section}`,
    quote.display_text,
    "history-turn",
  );
  const note = createElement(
    "span",
    "source-note",
    `source turn ${quote.source_turn_index} · sha256 ${quote.source_turn_sha256.slice(0, 12)}…`,
  );
  message.append(note);
}


function appendMeasurement(label, lines, className = "") {
  const message = createElement(
    "div",
    `msg measurement ${className}`.trim(),
  );
  message.append(createElement("span", "role", label));
  lines.forEach((line, index) => {
    if (index) message.append(document.createElement("br"));
    const strong = createElement("strong", "", line.label);
    message.append(strong, document.createTextNode(` ${line.value}`));
  });
  transcript.append(message);
  transcript.scrollTop = transcript.scrollHeight;
  return message;
}


function formatScientific(value, digits = 2) {
  if (value === 0) return "0";
  return Number(value).toExponential(digits).replace("e+", "e");
}


function formatProbability(value) {
  return value >= 0.001 ? Number(value).toPrecision(4) : formatScientific(value);
}


function outcome(conditionId) {
  const match = payload.case.outcomes.find(
    (row) => row.condition_id === conditionId,
  );
  assert(match, `missing recorded condition ${conditionId}`);
  return match;
}


function renderChatStage() {
  appendMessage(
    "system",
    "recorded source",
    "Loading exact public excerpts from separate LongMemEval examples selected and assembled by the benchmark. They are not one organic continuous chat.",
  );
  for (const quote of payload.case.dialogue.target.quotes) {
    appendQuote(quote, "owned update");
  }
  for (const quote of payload.case.dialogue.retained.quotes) {
    appendQuote(quote, "retained control");
  }
  for (const quote of payload.case.dialogue.intervening.quotes) {
    appendQuote(quote, "later gap round");
  }
  const gap = measured(
    payload.case.chronology.tokens_strictly_after_owned,
    "gap tokens",
  );
  const window = measured(
    payload.case.chronology.local_window_tokens,
    "local window",
  );
  const gapLine = createElement(
    "div",
    "replay-gap",
    `${gap.toLocaleString("en-US")} tokens after the owned round · outside the ${window.toLocaleString("en-US")}-token local window`,
  );
  transcript.append(gapLine);
}


function renderRecallStage() {
  appendMessage(
    "user",
    "registered target query · public LongMemEval",
    payload.case.queries.target.display_text,
  );
  const present = outcome("present");
  appendMeasurement("recorded measurement · not a generation", [
    {
      label: "P(target)",
      value: formatProbability(
        measured(
          present.target.geometric_mean_probability,
          "present target probability",
        ),
      ),
    },
    {
      label: "first-token rank",
      value: String(
        measured(present.target.first_target_token_rank, "present target rank"),
      ),
    },
    {
      label: "admission",
      value: "passed in the committed lock",
    },
  ]);
}


function selectedResolverCandidate() {
  if (labResolverEnabled) {
    const selected = labResolution?.selected_record;
    assert(
      labResolution?.status === "resolved" &&
        labResolution?.proposal_id &&
        selected?.deletion_scope === "complete_exchange" &&
        Array.isArray(selected?.owned_exchange) &&
        selected.owned_exchange.length === 2,
      "server resolver did not return one confirmable complete exchange",
    );
    return selected;
  }
  const resolution = payload.case.memory_resolution;
  const selectedId = resolution.decision.selected_record_id;
  const selected = resolution.candidate_catalog.find(
    (candidate) => candidate.record_id === selectedId,
  );
  assert(selected, "selected resolver candidate is unavailable");
  return selected;
}


async function renderResolutionRequestStage() {
  const resolution = payload.case.memory_resolution;
  appendMessage(
    "user",
    "natural-language deletion request · evaluator-added fixture",
    resolution.request.text,
    "evaluation-action",
  );
  appendMessage(
    "system",
    "resolver boundary",
    "Identification happens before deletion. No deletion has occurred.",
    "resolver-boundary",
  );
  if (!labResolverEnabled) return;

  appendMessage(
    "system",
    `${labConfig.resolver.mode} resolver request`,
    "Calling the isolated lab server with the committed request. The server loads the two candidates; the browser sends no catalog or candidate ID.",
    "resolver-fixture",
  );
  labResolution = await fetchJson(LAB_RESOLVE_URL, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({request_text: resolution.request.text}),
  });
  assert(
    labResolution.deletion_executed === false &&
      labResolution.resolver?.model_id === labConfig.resolver.model_id &&
      labResolution.resolver?.mode === labConfig.resolver.mode &&
      labResolution.resolver?.outside_deletion_certificate === true &&
      labResolution.deletion_evidence?.kind === "recorded_replay" &&
      labResolution.deletion_evidence?.live_model_deletion_run === false,
    "server resolver crossed the recorded-evidence boundary",
  );
  selectedResolverCandidate();
}


function renderResolverProposalStage() {
  const resolution = payload.case.memory_resolution;
  const selected = selectedResolverCandidate();
  if (labResolverEnabled) {
    const resolver = labResolution.resolver;
    appendMessage(
      "system",
      resolver.label,
      "The server returned this catalog-bound proposal. Resolver identification remains outside the deletion certificate.",
      "resolver-fixture",
    );
    appendMeasurement(
      "server resolver proposal · outside certificate",
      [
        {
          label: "proposed memory",
          value: selected.label,
        },
        {
          label: "summary",
          value: selected.summary,
        },
        {
          label: "opaque record ID",
          value: selected.record_id,
        },
        {
          label: "model",
          value: resolver.model_id,
        },
        {
          label: "mode",
          value: resolver.mode,
        },
        {
          label: "evidence",
          value: resolver.evidence,
        },
        {
          label: "provider request",
          value: resolver.network_request_performed ? "performed" : "not performed",
        },
        {
          label: "deletion",
          value: "not executed",
        },
      ],
      "resolver-proposal",
    );
    return;
  }
  appendMessage(
    "system",
    resolution.label,
    "This deterministic fixture made no provider call. The configured model field below is configuration only and did not produce this decision.",
    "resolver-fixture",
  );
  appendMeasurement(
    "recorded resolver proposal · outside certificate",
    [
      {
        label: "proposed memory",
        value: selected.label,
      },
      {
        label: "summary",
        value: selected.summary,
      },
      {
        label: "opaque record ID",
        value: selected.record_id,
      },
      {
        label: "configured default (not used)",
        value: resolution.model_configuration.default_model_id,
      },
      {
        label: "provider call",
        value: "none",
      },
      {
        label: "deletion",
        value: "not executed",
      },
    ],
    "resolver-proposal",
  );
}


function renderConfirmationStage() {
  const selected = selectedResolverCandidate();
  const resolution = payload.case.memory_resolution;
  const quotes = labResolverEnabled
    ? selected.owned_exchange
    : resolution.confirmation.owned_exchange;
  if (labResolverEnabled) {
    assert(
      selected.source_turn_ids.join("|") ===
        quotes.map((quote) => quote.turn_id).join("|"),
      "server confirmation turn binding differs",
    );
  } else {
    assert(
      selected.owned_exchange_ref === "/case/dialogue/target/quotes" &&
        resolution.confirmation.selected_record_id === selected.record_id &&
        quotes.length === 2,
      "confirmation exchange binding differs",
    );
  }

  const card = createElement("section", "confirmation-card");
  card.append(
    createElement("span", "role", "confirmation required"),
    createElement("h3", "", selected.label),
    createElement(
      "p",
      "confirmation-summary",
      "Review the complete owned user–assistant exchange before deletion.",
    ),
  );
  for (const quote of quotes) {
    const text = quote.display_text || quote.text;
    const turn = createElement("div", `confirmation-turn ${quote.role}`);
    turn.append(
      createElement("span", "confirmation-role", quote.role),
      createElement("p", "", text),
      createElement(
        "span",
        "source-note",
        `source turn ${quote.source_turn_index} · sha256 ${quote.source_turn_sha256.slice(0, 12)}…`,
      ),
    );
    card.append(turn);
  }
  card.append(
    createElement(
      "p",
      "confirmation-record-id",
      `Opaque record ID: ${selected.record_id}`,
    ),
    createElement(
      "p",
      "confirmation-warning",
      "Nothing is deleted until you press the dedicated confirmation button.",
    ),
  );
  transcript.append(card);
  transcript.scrollTop = transcript.scrollHeight;
  awaitingConfirmation = true;
  confirmButton.hidden = false;
  confirmButton.disabled = false;
  confirmButton.textContent = labResolverEnabled
    ? "Confirm and show recorded deletion evidence"
    : "Confirm and delete this memory";
}


function renderForgetStage() {
  assert(deletionConfirmed, "deletion requires explicit confirmation");
  const selected = selectedResolverCandidate();
  if (labResolverEnabled) {
    assert(
      labConfirmation?.confirmed === true &&
        labConfirmation?.selected_record_id === selected.record_id &&
        labConfirmation?.deletion_executed === false &&
        labConfirmation?.subsequent_deletion_evidence?.kind ===
          "recorded_replay" &&
        labConfirmation.subsequent_deletion_evidence.live_model_deletion_run ===
          false,
      "server confirmation did not preserve the recorded-evidence boundary",
    );
  }
  const boundary = createElement(
    "div",
    "evidence-boundary",
    "RESOLVER BOUNDARY · identification and confirmation end here · deletion evidence begins below",
  );
  transcript.append(boundary);
  appendMessage(
    "system",
    labResolverEnabled
      ? "recorded deletion evidence · no live deletion run"
      : "confirmed recorded deletion",
    labResolverEnabled
      ? `Confirmed ${selected.record_id}. Showing the committed replay bound to payload ${labConfirmation.subsequent_deletion_evidence.payload_sha256.slice(0, 12)}…; the resolver did not generate or alter these results.`
      : `Applying the locked recorded operation to ${selected.record_id}.`,
    "confirmed-deletion",
  );
  const exact = outcome("exact_decrement");
  const fallbacks = measured(
    payload.case.certificate.decrement_fallbacks,
    "decrement fallbacks",
  );
  const headGateSolves = measured(
    payload.case.certificate.head_gate_solves,
    "attempted head-gate solves",
  );
  const headGates = measured(
    payload.case.certificate.head_gates,
    "head gates",
  );
  appendMeasurement("recorded operation", [
    {
      label: "path",
      value: "exact decrement with fixed-C refit fallback",
    },
    {
      label: "refit fallbacks",
      value: `${fallbacks}/${headGateSolves} attempted solves (${headGates} gates enumerated)`,
    },
    {
      label: "full raw repack",
      value: exact.execution.full_repack_fallback.value ? "used" : "not used",
    },
  ]);
}


function renderRequeryStage() {
  appendMessage(
    "user",
    "official LongMemEval target question · replayed by evaluator",
    payload.case.queries.target.display_text,
  );
  const exact = outcome("exact_decrement");
  const raw = outcome("fresh_raw_omission");
  appendMeasurement("recorded target measurement · not a generation", [
    {
      label: "exact P(target)",
      value: formatProbability(
        measured(
          exact.target.geometric_mean_probability,
          "exact target probability",
        ),
      ),
    },
    {
      label: "exact first-token rank",
      value: String(
        measured(exact.target.first_target_token_rank, "exact target rank"),
      ),
    },
    {
      label: "raw-omission P(target)",
      value: formatProbability(
        measured(raw.target.geometric_mean_probability, "raw target probability"),
      ),
    },
    {
      label: "KL(raw ‖ exact)",
      value: `${formatScientific(
        measured(
          exact.target.full_vocabulary_kl_to_raw_repack_nats,
          "exact target KL",
        ),
      )} nats`,
    },
  ]);
}


function renderRetainedStage() {
  appendMessage(
    "user",
    "official LongMemEval retained question · replayed by evaluator",
    payload.case.queries.retained.display_text,
  );
  const retained = outcome("exact_decrement").retained;
  const drift = measured(
    retained.mean_log_probability_drift_from_raw_repack_nats,
    "retained drift",
  );
  appendMeasurement(
    "recorded retained-control measurement · not a generation",
    [
      {
        label: "P(retained target)",
        value: measured(
          retained.geometric_mean_probability,
          "retained probability",
        ).toFixed(6),
      },
      {
        label: "first-token rank",
        value: String(
          measured(retained.first_target_token_rank, "retained rank"),
        ),
      },
      {
        label: "mean-log-P drift from raw",
        value: `${drift >= 0 ? "+" : ""}${formatScientific(drift)} nats`,
      },
    ],
    "retained",
  );
}


function renderCertificateStage() {
  const certificate = payload.case.certificate;
  appendMeasurement(
    "recorded certificate",
    [
      {
        label: "max KL(exact ‖ fixed-C refit)",
        value: formatScientific(
          measured(certificate.max_output_kl_nats, "certificate KL"),
        ),
      },
      {
        label: "scope",
        value: "full vocabulary, first target token, both registered probes",
      },
      {
        label: "raw-omission equality",
        value: "not certified; behavioral reference shown separately",
      },
    ],
    "certificate",
  );
}


const renderers = [
  renderChatStage,
  renderRecallStage,
  renderResolutionRequestStage,
  renderResolverProposalStage,
  renderConfirmationStage,
  renderForgetStage,
  renderRequeryStage,
  renderRetainedStage,
  renderCertificateStage,
];


function stageLabel(step) {
  if (!labResolverEnabled) return step.label;
  if (step.id === "resolve-proposal") {
    return `Show the ${labConfig.resolver.mode} server resolver proposal`;
  }
  if (step.id === "forget") {
    return "Show the committed recorded deletion";
  }
  return step.label;
}


function renderStageList() {
  stageList.replaceChildren();
  const activeStage = awaitingConfirmation ? nextStage - 1 : nextStage;
  payload.demo.sequence.forEach((step, index) => {
    const item = createElement("li", "", stageLabel(step));
    item.dataset.stage = step.id;
    if (index < activeStage) item.classList.add("done");
    if (index === activeStage) item.classList.add("active");
    stageList.append(item);
  });
}


function updateControls() {
  renderStageList();
  nextButton.disabled =
    requestInFlight || awaitingConfirmation || nextStage >= renderers.length;
  resetButton.disabled = requestInFlight || nextStage === 0;
  confirmButton.disabled =
    requestInFlight || !awaitingConfirmation || deletionConfirmed;
  if (requestInFlight) {
    nextButton.textContent = "Waiting for lab server…";
  } else if (awaitingConfirmation) {
    nextButton.textContent = "Confirmation required";
  } else if (nextStage === 0) {
    nextButton.textContent = "Start replay";
  } else if (nextStage < renderers.length) {
    nextButton.textContent = `Next: ${payload.demo.sequence[nextStage].id}`;
  } else {
    nextButton.textContent = "Replay complete";
  }
}


function resetReplay() {
  transcript.replaceChildren();
  labResolution = null;
  labConfirmation = null;
  nextStage = 0;
  awaitingConfirmation = false;
  deletionConfirmed = false;
  confirmButton.hidden = true;
  confirmButton.disabled = true;
  updateControls();
}


async function advanceReplay() {
  if (
    requestInFlight ||
    awaitingConfirmation ||
    nextStage >= renderers.length
  ) return;
  if (STAGE_IDS[nextStage] === "forget" && !deletionConfirmed) return;
  requestInFlight = true;
  updateControls();
  try {
    await renderers[nextStage]();
    nextStage += 1;
  } catch (error) {
    appendMessage(
      "system error",
      "stage failed",
      `${error.message}. No recorded result was substituted for this stage.`,
    );
    banner.textContent = `Lab stage unavailable: ${error.message}`;
    banner.classList.add("preview");
  } finally {
    requestInFlight = false;
    updateControls();
  }
}


async function confirmAndDelete() {
  if (requestInFlight || !awaitingConfirmation || deletionConfirmed) return;
  const selected = selectedResolverCandidate();
  requestInFlight = true;
  updateControls();
  let confirmed = false;
  try {
    if (labResolverEnabled) {
      labConfirmation = await fetchJson(LAB_CONFIRM_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          proposal_id: labResolution.proposal_id,
          confirmed: true,
        }),
      });
      assert(
        labConfirmation.confirmed === true &&
          labConfirmation.selected_record_id === selected.record_id &&
          labConfirmation.confirmed_record?.record_id === selected.record_id &&
          labConfirmation.deletion_executed === false &&
          labConfirmation.subsequent_deletion_evidence?.kind ===
            "recorded_replay" &&
          labConfirmation.subsequent_deletion_evidence
            .live_model_deletion_run === false,
        "lab server confirmation boundary differs",
      );
    }
    deletionConfirmed = true;
    awaitingConfirmation = false;
    confirmButton.hidden = true;
    appendMessage(
      "system",
      "explicit confirmation recorded",
      labResolverEnabled
        ? `Confirmed ${selected.record_id} on the server. Advancing to the existing recorded deletion stages; no live deletion model ran.`
        : `Confirmed ${selected.record_id}. Advancing to the recorded deletion operation.`,
      "confirmation-accepted",
    );
    confirmed = true;
  } catch (error) {
    appendMessage(
      "system error",
      "confirmation failed",
      `${error.message}. The recorded deletion stages remain locked.`,
    );
  } finally {
    requestInFlight = false;
    updateControls();
  }
  if (confirmed) await advanceReplay();
}


async function initialize() {
  payload = validatePayload(await fetchJson(PAYLOAD_URL));
  if (labResolverEnabled) {
    labConfig = validateLabConfig(await fetchJson(LAB_CONFIG_URL));
    const live = labConfig.resolver.mode === "gemini";
    banner.textContent = live
      ? `LIVE RESOLVER · ${labConfig.resolver.model_id} · RECORDED DELETION EVIDENCE`
      : "LAB REHEARSAL · RECORDED RESOLVER · RECORDED DELETION EVIDENCE";
    subtitle.textContent = live
      ? "One live server-side Gemini resolution · committed LongMemEval deletion replay"
      : "Server-backed recorded resolver rehearsal · no provider call";
    resolverNote.textContent = live
      ? "The resolver proposal is a live Gemini outcome. Its identification and confirmation remain outside the certificate; every deletion result below is the committed replay."
      : "The resolver proposal is a server-side recorded fixture with no provider call. Identification and confirmation remain outside the certificate.";
  } else {
    banner.textContent =
      payload.status === "incomplete_case_study_preview"
        ? PREVIEW_LABEL
        : "RECORDED FINAL 16/16 CASE STUDY";
  }
  banner.classList.toggle(
    "preview",
    payload.status === "incomplete_case_study_preview",
  );
  const source = payload.provenance.source_artifacts.method_report;
  provenance.textContent =
    `record ${payload.case.record_id} · report sha256 ${source.file_sha256.slice(0, 16)}… · ` +
    `payload sha256 ${payload.integrity.sha256.slice(0, 16)}…` +
    (labResolverEnabled
      ? ` · resolver ${labConfig.resolver.mode}/${labConfig.resolver.evidence}`
      : "");
  nextButton.disabled = false;
  resetButton.disabled = true;
  updateControls();
}


nextButton.addEventListener("click", advanceReplay);
resetButton.addEventListener("click", resetReplay);
confirmButton.addEventListener("click", confirmAndDelete);

initialize().catch((error) => {
  banner.textContent = `Replay unavailable: ${error.message}`;
  banner.classList.add("preview");
  appendMessage(
    "system error",
    "load failure",
    "The recorded payload or requested lab server failed validation. No fallback values were fabricated.",
  );
});
