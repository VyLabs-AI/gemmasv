import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { once } from "node:events";

import { Resvg } from "@resvg/resvg-js";
import PDFDocument from "pdfkit";
import SVGtoPDF from "svg-to-pdfkit";

import { normalizePngSrgb } from "./png_srgb.mjs";


const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_PAYLOAD = path.join(
  ROOT,
  "gemma_sv",
  "demo_site",
  "assets",
  "longmemeval_forgetting_final.json",
);
const DEFAULT_OUTPUT_BASE = path.join(
  HERE,
  "previews",
  "longmemeval_chat_forgetting_final",
);

const PAYLOAD_SCHEMA = "gemma-sv-longmemeval-chat-forgetting-payload-v2";
const PREVIEW_LABEL = "INCOMPLETE CASE STUDY PREVIEW";
const FINAL_REPORT_SHA256 =
  "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6";
const CASE_RECORD_ID =
  "longmemeval-constrained-chat-v1-c8276e265e3db489c090cdcc";
const OUTCOME_IDS = [
  "present",
  "fresh_raw_omission",
  "exact_decrement",
  "fixed_c_refit",
];
const DEMO_SEQUENCE = [
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

// Dissertation-style restraint: black type, white space, hairline rules, and
// color only where it carries semantic meaning.
const COLORS = {
  ink: "#171717",
  muted: "#62676D",
  line: "#D4D7DA",
  surface: "#F7F7F5",
  userBubble: "#EEF3F7",
  blue: "#2F6F9F",
  sky: "#6D9EBE",
  green: "#2E7D5B",
  orange: "#A56A16",
  red: "#A34A42",
  previewInk: "#8F3D3D",
  previewFill: "#F8ECEC",
  white: "#FFFFFF",
};

const DIALOGUE_SHAPE = {
  target: ["user", "assistant"],
  retained: ["user", "assistant"],
  intervening: ["user", "assistant"],
};

const OUTCOME_STYLES = {
  present: {
    label: "Present",
    note: "before forget request",
    color: COLORS.blue,
  },
  fresh_raw_omission: {
    label: "Raw omission",
    note: "fresh behavioral reference",
    color: "#333333",
  },
  exact_decrement: {
    label: "Executed mixed policy",
    note: "decrement or fixed-C refit fallback",
    color: COLORS.sky,
  },
  fixed_c_refit: {
    label: "Fixed-C refit",
    note: "certificate reference",
    color: COLORS.green,
  },
};


function fail(message) {
  throw new Error(`LongMemEval forgetting payload: ${message}`);
}


function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}


function measured(value, label) {
  if (
    !value ||
    value.artifact !== "method_report" &&
      value.artifact !== "admission_report" ||
    typeof value.source_pointer !== "string" ||
    !value.source_pointer.startsWith("/records/")
  ) {
    fail(`${label} lacks a report source pointer`);
  }
  if (typeof value.value === "number" && !Number.isFinite(value.value)) {
    fail(`${label} is not finite`);
  }
  return value.value;
}


export function validateForgettingPayload(payload) {
  if (
    !payload ||
    payload.schema !== PAYLOAD_SCHEMA ||
    payload.schema_version !== 2
  ) {
    fail("schema differs");
  }
  if (
    !payload.integrity ||
    payload.integrity.algorithm !== "sha256" ||
    !isSha256(payload.integrity.sha256)
  ) {
    fail("integrity descriptor is invalid");
  }
  if (
    payload.contains_model_generated_text !== false ||
    payload.contains_unquoted_source_text !== false ||
    payload.raw_context_text_present !== false ||
    payload.demo?.model_outcome_text_present !== false
  ) {
    fail("text-boundary disclosure differs");
  }
  if (
    payload.case?.record_id !== CASE_RECORD_ID ||
    payload.case?.selection?.selected_from_method_outcomes !== false ||
    payload.case?.selection?.selected_before_method_scoring !== true
  ) {
    fail("predeclared case selection differs");
  }
  if (
    !Array.isArray(payload.scope?.aggregate_claims) ||
    payload.scope.aggregate_claims.length !== 0 ||
    payload.scope.aggregate_claims_allowed !== false ||
    payload.scope.full_repack_certificate_claimed !== false
  ) {
    fail("case-study scope contains an aggregate or full-repack claim");
  }
  if (
    payload.status === "incomplete_case_study_preview" &&
    (payload.preview_label !== PREVIEW_LABEL ||
      payload.scope.final_all16_complete !== false)
  ) {
    fail("partial15 payload is not visibly marked preview");
  }
  if (
    payload.status !== "incomplete_case_study_preview" &&
    (payload.status !== "final_case_study" ||
      payload.preview_label !== null ||
      payload.scope.final_all16_complete !== true ||
      payload.provenance?.source_artifacts?.method_report?.file_sha256 !==
        FINAL_REPORT_SHA256)
  ) {
    fail("final payload is not a validated all16 case study");
  }

  const outcomes = payload.case?.outcomes;
  if (
    !Array.isArray(outcomes) ||
    outcomes.map((row) => row.condition_id).join("|") !== OUTCOME_IDS.join("|")
  ) {
    fail("required condition ordering differs");
  }
  for (const outcome of outcomes) {
    if (outcome.status !== "completed") fail(`${outcome.condition_id} incomplete`);
    measured(
      outcome.target?.geometric_mean_probability,
      `${outcome.condition_id} target probability`,
    );
    measured(
      outcome.target?.mean_log_probability,
      `${outcome.condition_id} target mean log probability`,
    );
    measured(
      outcome.target?.first_target_token_rank,
      `${outcome.condition_id} target rank`,
    );
    measured(
      outcome.target?.full_vocabulary_kl_to_raw_repack_nats,
      `${outcome.condition_id} target KL`,
    );
    measured(
      outcome.retained?.mean_log_probability_drift_from_raw_repack_nats,
      `${outcome.condition_id} retained drift`,
    );
  }
  const exact = outcomes[2];
  if (
    exact.execution?.executed_method?.value !== "fixed_c_refit_fallback" ||
    exact.execution?.fixed_c_refit_fallback?.value !== true ||
    exact.execution?.full_repack_fallback?.value !== false
  ) {
    fail("exact-path fallback disclosure differs");
  }

  const dialogue = payload.case?.dialogue;
  for (const [section, expectedRoles] of Object.entries(DIALOGUE_SHAPE)) {
    const quotes = dialogue?.[section]?.quotes;
    if (
      !Array.isArray(quotes) ||
      quotes.map((quote) => quote.role).join("|") !== expectedRoles.join("|")
    ) {
      fail(`${section} public dialogue shape differs`);
    }
    for (const quote of quotes) {
      if (
        quote.origin !== "public_longmemeval_dialogue" ||
        typeof quote.display_text !== "string" ||
        !quote.display_text ||
        quote.display_text.includes("...") ||
        quote.display_text.includes("…") ||
        quote.truncated !== false ||
        quote.excerpt_policy?.selection !== "exact UTF-8 source substring" ||
        quote.excerpt_policy?.normalization !== "none" ||
        !Number.isInteger(quote.source_span_start) ||
        !Number.isInteger(quote.source_span_end) ||
        !isSha256(quote.source_turn_sha256) ||
        !isSha256(quote.display_text_sha256)
      ) {
        fail(`${section} quote is not source-bound`);
      }
    }
  }
  for (const query of Object.values(payload.case?.queries ?? {})) {
    if (
      query.origin !== "public_longmemeval_query" ||
      typeof query.display_text !== "string" ||
      query.official_longmemeval_question !== true ||
      query.evaluation_use !== "official question replayed by evaluator" ||
      query.evaluator_added_text !== false ||
      !isSha256(query.display_text_sha256)
    ) {
      fail("registered query is not source-bound");
    }
  }
  if (
    payload.case?.deletion_action?.origin !==
      "evaluator_added_deletion_action" ||
    payload.case?.deletion_action?.public_longmemeval_source !== false ||
    payload.case?.deletion_action?.out_of_band !== true ||
    payload.case?.deletion_action?.model_generated !== false
  ) {
    fail("evaluation deletion action provenance differs");
  }
  const resolution = payload.case?.memory_resolution;
  const catalog = resolution?.candidate_catalog;
  const selectedRecordId = resolution?.decision?.selected_record_id;
  if (
    resolution?.schema !== "gemma-sv-recorded-memory-resolver-fixture-v1" ||
    resolution?.mode !== "deterministic_recorded_fixture" ||
    resolution?.network_request_performed !== false ||
    resolution?.provider?.call_performed !== false ||
    resolution?.decision?.status !== "resolved" ||
    resolution?.decision?.deletion_executed !== false ||
    resolution?.decision?.requires_confirmation !== true ||
    !Array.isArray(catalog) ||
    catalog.length !== 2 ||
    !catalog.some((record) => record.record_id === selectedRecordId) ||
    catalog.find((record) => record.record_id === selectedRecordId)?.label !==
      "Current number of bicycles" ||
    resolution?.confirmation?.selected_record_id !== selectedRecordId ||
    resolution?.confirmation?.requires_explicit_click !== true ||
    !Array.isArray(resolution?.confirmation?.owned_exchange) ||
    resolution.confirmation.owned_exchange.length !== 2 ||
    resolution?.outside_certificate_boundary !== true
  ) {
    fail("recorded resolver fixture differs");
  }
  if (
    payload.provenance?.public_dialogue?.assembly_disclosure !==
    "target, retained-control, and gap excerpts come from separate public LongMemEval examples selected and assembled by the benchmark; they are not one organic continuous conversation"
  ) {
    fail("benchmark assembly provenance differs");
  }
  measured(
    payload.case?.chronology?.tokens_strictly_after_owned,
    "gap token count",
  );
  if (payload.case?.chronology?.owned_round_outside_local_window?.value !== true) {
    fail("owned round is not outside the locked local window");
  }
  measured(payload.case?.certificate?.max_output_kl_nats, "certificate KL");
  measured(payload.case?.certificate?.decrement_fallbacks, "fallback count");
  measured(payload.case?.certificate?.head_gate_solves, "attempted head-gate solves");
  measured(payload.case?.certificate?.head_gates, "head-gate count");
  if (payload.case?.certificate?.full_repack_certificate !== false) {
    fail("certificate is mislabeled as full repack");
  }
  if (
    !Array.isArray(payload.demo?.sequence) ||
    payload.demo.sequence.map((step) => step.id).join("|") !==
      DEMO_SEQUENCE.join("|") ||
    payload.demo.live_compute !== false
  ) {
    fail("recorded demo sequence differs");
  }
  return payload;
}


function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}


function wrapText(value, maximumCharacters) {
  const words = String(value).trim().split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length <= maximumCharacters || !line) {
      line = candidate;
    } else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  return lines.length ? lines : [""];
}


function textLines(lines, x, y, options = {}) {
  const {
    size = 14,
    color = COLORS.ink,
    weight = 400,
    lineHeight = Math.round(size * 1.35),
    family = "Liberation Sans, Arial, sans-serif",
    anchor = "start",
  } = options;
  const effectiveSize =
    size === 20 ? 24 : size >= 17 && size < 19 ? 22 : size < 17 ? 21 : size;
  const spans = lines
    .map(
      (line, index) =>
        `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`,
    )
    .join("");
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${effectiveSize}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}">${spans}</text>`;
}


function roundedRect(x, y, width, height, options = {}) {
  const {
    fill = COLORS.white,
    stroke = COLORS.line,
    strokeWidth = 1,
    radius = 9,
  } = options;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
}


function bubble(quote, x, y, width, kind = "source") {
  const maxCharacters = Math.max(28, Math.floor(width / 7.4));
  const lines = wrapText(quote.display_text, maxCharacters);
  const lineHeight = 17;
  const height = 38 + lines.length * lineHeight;
  const user = quote.role === "user";
  const fill = user ? COLORS.userBubble : COLORS.surface;
  const stroke =
    kind === "target"
      ? COLORS.blue
      : kind === "request"
        ? COLORS.previewInk
        : COLORS.line;
  const role = user ? "USER" : "ASSISTANT";
  const sourceLabel =
    kind === "request" ? "EVALUATOR ACTION" : "PUBLIC LONGMEMEVAL";
  const pieces = [
    roundedRect(x, y, width, height, {
      fill,
      stroke,
      strokeWidth: kind === "target" ? 1.4 : 1,
      radius: 10,
    }),
    textLines([role], x + 14, y + 18, {
      size: 10,
      color: COLORS.muted,
      weight: 700,
      family: "Liberation Mono, monospace",
    }),
    textLines([sourceLabel], x + width - 14, y + 18, {
      size: 9,
      color: COLORS.muted,
      weight: 600,
      family: "Liberation Mono, monospace",
      anchor: "end",
    }),
    textLines(lines, x + 14, y + 40, {
      size: 13,
      lineHeight,
    }),
  ];
  return { svg: pieces.join(""), height };
}


function outcomeCard(outcome, x, y, width) {
  const style = OUTCOME_STYLES[outcome.condition_id];
  const probability = measured(
    outcome.target.geometric_mean_probability,
    `${outcome.condition_id} probability`,
  );
  const meanLogProbability = measured(
    outcome.target.mean_log_probability,
    `${outcome.condition_id} mean log probability`,
  );
  const rank = measured(
    outcome.target.first_target_token_rank,
    `${outcome.condition_id} rank`,
  );
  const rawKl = measured(
    outcome.target.full_vocabulary_kl_to_raw_repack_nats,
    `${outcome.condition_id} raw KL`,
  );
  const height = 78;
  const probabilityLabel = formatProbability(probability);
  const klLabel = formatScientific(rawKl);
  return [
    roundedRect(x, y, width, height, {
      fill: COLORS.white,
      stroke: COLORS.line,
      radius: 8,
    }),
    `<rect x="${x}" y="${y}" width="5" height="${height}" rx="2.5" fill="${style.color}"/>`,
    textLines([style.label], x + 17, y + 23, {
      size: 14,
      weight: 700,
      color: COLORS.ink,
    }),
    textLines([style.note], x + width - 14, y + 22, {
      size: 10,
      color: style.color,
      weight: 700,
      family: "Liberation Mono, monospace",
      anchor: "end",
    }),
    textLines([`P(target) ${probabilityLabel}`], x + 17, y + 51, {
      size: 12,
      color: COLORS.ink,
      weight: 600,
      family: "Liberation Mono, monospace",
    }),
    textLines([`rank ${rank}`], x + 205, y + 51, {
      size: 12,
      color: COLORS.ink,
      family: "Liberation Mono, monospace",
    }),
    textLines([`mean log P ${meanLogProbability.toFixed(2)}`], x + 292, y + 51, {
      size: 12,
      color: COLORS.ink,
      family: "Liberation Mono, monospace",
    }),
    textLines([`KL(raw‖method) ${klLabel}`], x + 17, y + 69, {
      size: 10,
      color: COLORS.muted,
      family: "Liberation Mono, monospace",
    }),
  ].join("");
}


function storyTurn(quote, x, y, width, options = {}) {
  const { accent = COLORS.line, synthetic = false } = options;
  const lines = wrapText(
    quote.display_text,
    Math.max(34, Math.floor(width / 5.7)),
  );
  const lineHeight = 13.5;
  const height = 28 + lines.length * lineHeight;
  const user = quote.role === "user";
  const sourceLabel = synthetic
    ? "evaluator-added intervention"
    : `public LongMemEval · ${quote.source_turn_index} · ${quote.source_turn_sha256.slice(0, 8)}`;
  return {
    height,
    svg: [
      roundedRect(x, y, width, height, {
        fill: user ? COLORS.userBubble : COLORS.white,
        stroke: synthetic ? COLORS.previewInk : COLORS.line,
        strokeWidth: synthetic ? 1.1 : 0.7,
        radius: 5,
      }),
      `<line x1="${x}" y1="${y + 4}" x2="${x}" y2="${y + height - 4}" stroke="${accent}" stroke-width="2"/>`,
      textLines([user ? "User" : "Assistant"], x + 10, y + 15, {
        size: 8.5,
        weight: 700,
      }),
      textLines([sourceLabel], x + width - 9, y + 15, {
        size: 7.2,
        color: COLORS.muted,
        anchor: "end",
      }),
      textLines(lines, x + 10, y + 31, {
        size: 11.2,
        lineHeight,
      }),
    ].join(""),
  };
}


function storyHeading(label, title, x, y) {
  return [
    textLines([`(${label})`], x, y, {
      size: 12,
      weight: 700,
    }),
    textLines([title], x + 27, y, {
      size: 12,
      weight: 700,
    }),
  ].join("");
}


function formatScientific(value, digits = 2) {
  if (value === 0) return "0";
  return Number(value)
    .toExponential(digits)
    .replace("e+", "e");
}


function formatProbability(value) {
  if (value >= 0.001) return Number(value).toPrecision(3);
  return formatScientific(value);
}


function renderLegacyLongMemEvalForgettingSvg(payload) {
  validateForgettingPayload(payload);
  const width = 1000;
  const height = 1100;
  const phoneX = 34;
  const phoneY = 25;
  const phoneWidth = 535;
  const phoneHeight = 1040;
  const before = payload.case.outcomes[0];
  const raw = payload.case.outcomes[1];
  const exact = payload.case.outcomes[2];
  const beforeProbability = measured(
    before.target.geometric_mean_probability,
    "before target probability",
  );
  const rawProbability = measured(
    raw.target.geometric_mean_probability,
    "raw target probability",
  );
  const exactProbability = measured(
    exact.target.geometric_mean_probability,
    "exact target probability",
  );
  const exactRank = measured(
    exact.target.first_target_token_rank,
    "exact target rank",
  );
  const retainedProbability = measured(
    exact.retained.geometric_mean_probability,
    "retained probability",
  );
  const retainedRank = measured(
    exact.retained.first_target_token_rank,
    "retained rank",
  );
  const maxKl = measured(
    payload.case.certificate.max_output_kl_nats,
    "certificate KL",
  );
  const gap = measured(
    payload.case.chronology.tokens_strictly_after_owned,
    "gap tokens",
  );
  const localWindow = measured(
    payload.case.chronology.local_window_tokens,
    "local window",
  );
  const pieces = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">`,
    `<title id="title">Measured LongMemEval forgetting audit</title>`,
    `<desc id="description">A phone-shaped vertical conversation replay beside log-scale target evidence, a retained-memory gauge, and a fixed-C numerical certificate.</desc>`,
    `<metadata>${escapeXml(
      JSON.stringify({
        schema: payload.schema,
        record_id: payload.case.record_id,
        payload_integrity_sha256: payload.integrity.sha256,
        method_report_sha256:
          payload.provenance.source_artifacts.method_report.file_sha256,
      }),
    )}</metadata>`,
    `<rect width="${width}" height="${height}" fill="${COLORS.white}"/>`,
  ];

  if (payload.status === "incomplete_case_study_preview") {
    pieces.push(
      `<rect x="0" y="0" width="${width}" height="22" fill="${COLORS.previewFill}"/>`,
      textLines([PREVIEW_LABEL], width / 2, 15, {
        size: 8.5,
        color: COLORS.previewInk,
        weight: 700,
        anchor: "middle",
      }),
    );
  }

  pieces.push(
    roundedRect(phoneX, phoneY, phoneWidth, phoneHeight, {
      fill: "#FBFBFA",
      stroke: COLORS.ink,
      strokeWidth: 2,
      radius: 34,
    }),
    `<rect x="${phoneX + 190}" y="${phoneY + 12}" width="155" height="17" rx="8.5" fill="${COLORS.ink}"/>`,
    textLines(["9:41"], phoneX + 32, phoneY + 35, {
      size: 10,
      weight: 700,
    }),
    textLines(["Assistant memory"], phoneX + phoneWidth / 2, phoneY + 58, {
      size: 13,
      weight: 700,
      anchor: "middle",
    }),
    `<line x1="${phoneX + 18}" y1="${phoneY + 70}" x2="${phoneX + phoneWidth - 18}" y2="${phoneY + 70}" stroke="${COLORS.line}" stroke-width="0.8"/>`,
  );

  function phoneBubble(text, label, y, options = {}) {
    const {
      side = "left",
      fill = COLORS.surface,
      stroke = COLORS.line,
      labelColor = COLORS.muted,
      width: bubbleWidth = 410,
      weight = 400,
    } = options;
    const lines = wrapText(text, Math.max(32, Math.floor(bubbleWidth / 7.1)));
    const lineHeight = 14;
    const bubbleHeight = 31 + lines.length * lineHeight;
    const x =
      side === "right"
        ? phoneX + phoneWidth - 24 - bubbleWidth
        : phoneX + 24;
    return {
      height: bubbleHeight,
      svg: [
        roundedRect(x, y, bubbleWidth, bubbleHeight, {
          fill,
          stroke,
          strokeWidth: 0.9,
          radius: 13,
        }),
        textLines([label], x + 12, y + 15, {
          size: 7.5,
          color: labelColor,
          weight: 700,
        }),
        textLines(lines, x + 12, y + 33, {
          size: 10.8,
          lineHeight,
          weight,
        }),
      ].join(""),
    };
  }

  let chatY = phoneY + 82;
  const targetQuotes = payload.case.dialogue.target.quotes;
  const retainedQuotes = payload.case.dialogue.retained.quotes;
  const interveningQuotes = payload.case.dialogue.intervening.quotes;
  const chatRows = [
    [
      targetQuotes[0].display_text,
      "USER · TARGET MEMORY",
      { side: "right", fill: COLORS.userBubble, stroke: COLORS.blue },
    ],
    [
      targetQuotes[1].display_text,
      "ASSISTANT · TARGET MEMORY",
      { side: "left", fill: COLORS.white, stroke: COLORS.blue },
    ],
    [
      retainedQuotes[0].display_text,
      "USER · RETAINED MEMORY",
      { side: "right", fill: "#EEF7F2", stroke: COLORS.green },
    ],
    [
      retainedQuotes[1].display_text,
      "ASSISTANT · RETAINED MEMORY",
      { side: "left", fill: COLORS.white, stroke: COLORS.green },
    ],
  ];
  for (const [message, label, options] of chatRows) {
    const bubbleResult = phoneBubble(message, label, chatY, options);
    pieces.push(bubbleResult.svg);
    chatY += bubbleResult.height + 6;
  }

  pieces.push(
    `<line x1="${phoneX + 42}" y1="${chatY + 4}" x2="${phoneX + phoneWidth - 42}" y2="${chatY + 4}" stroke="${COLORS.orange}" stroke-width="0.8" stroke-dasharray="4 4"/>`,
    textLines(
      [`${gap.toLocaleString("en-US")} tokens later · beyond ${localWindow.toLocaleString("en-US")}-token window`],
      phoneX + phoneWidth / 2,
      chatY + 18,
      {
        size: 7.5,
        color: COLORS.orange,
        weight: 700,
        anchor: "middle",
      },
    ),
  );
  chatY += 27;

  for (const quote of interveningQuotes) {
    const bubbleResult = phoneBubble(
      quote.display_text,
      `${quote.role.toUpperCase()} · LATER TURN`,
      chatY,
      {
        side: quote.role === "user" ? "right" : "left",
        fill: quote.role === "user" ? "#FFF7E8" : COLORS.white,
        stroke: COLORS.orange,
      },
    );
    pieces.push(bubbleResult.svg);
    chatY += bubbleResult.height + 6;
  }

  const actionBubble = phoneBubble(
    payload.case.deletion_action.display_text,
    "EVALUATOR ACTION · FORGET",
    chatY,
    {
      side: "right",
      fill: COLORS.previewFill,
      stroke: COLORS.previewInk,
      labelColor: COLORS.previewInk,
      weight: 600,
    },
  );
  pieces.push(actionBubble.svg);
  chatY += actionBubble.height + 7;

  const resolution = payload.case.memory_resolution;
  const selectedRecord = resolution.candidate_catalog.find(
    (record) => record.record_id === resolution.decision.selected_record_id,
  );
  const proposalBubble = phoneBubble(
    `Proposed memory: ${selectedRecord.label}`,
    "RESOLVER FIXTURE · NO PROVIDER CALL",
    chatY,
    {
      side: "left",
      fill: "#F4F1FA",
      stroke: "#7656A8",
      labelColor: "#7656A8",
    },
  );
  pieces.push(proposalBubble.svg);
  chatY += proposalBubble.height + 5;
  const confirmationBubble = phoneBubble(
    "Confirm this complete two-turn exchange before deletion",
    "USER CONFIRMATION REQUIRED",
    chatY,
    {
      side: "right",
      fill: "#F1F8F5",
      stroke: COLORS.green,
      labelColor: COLORS.green,
      weight: 600,
    },
  );
  pieces.push(confirmationBubble.svg);
  chatY += confirmationBubble.height + 7;

  const targetQuery = phoneBubble(
    payload.case.queries.target.display_text,
    "USER · OFFICIAL TARGET QUERY",
    chatY,
    { side: "right", fill: COLORS.userBubble, stroke: COLORS.blue },
  );
  pieces.push(targetQuery.svg);
  chatY += targetQuery.height + 5;
  const targetMeasure = phoneBubble(
    `Recorded score: P(target answer) ${formatProbability(exactProbability)} · rank ${exactRank}`,
    "MEASUREMENT · NOT A DECODED REPLY",
    chatY,
    { side: "left", fill: "#F5F8FC", stroke: COLORS.blue },
  );
  pieces.push(targetMeasure.svg);
  chatY += targetMeasure.height + 5;

  const retainedQuery = phoneBubble(
    payload.case.queries.retained.display_text,
    "USER · OFFICIAL RETAINED QUERY",
    chatY,
    { side: "right", fill: "#EEF7F2", stroke: COLORS.green },
  );
  pieces.push(retainedQuery.svg);
  chatY += retainedQuery.height + 5;
  const retainedMeasure = phoneBubble(
    `Recorded score: P(retained answer) ${retainedProbability.toFixed(6)} · rank ${retainedRank}`,
    "MEASUREMENT · NOT A DECODED REPLY",
    chatY,
    { side: "left", fill: "#F1F8F5", stroke: COLORS.green },
  );
  pieces.push(retainedMeasure.svg);

  pieces.push(
    textLines(["TARGET ANSWER PROBABILITY · LOG SCALE"], 620, 75, {
      size: 10,
      color: COLORS.blue,
      weight: 700,
    }),
  );
  const plotLeft = 625;
  const plotRight = 955;
  const plotTop = 112;
  const plotBottom = 352;
  function logPosition(probability) {
    const exponent = Math.max(-12, Math.min(0, Math.log10(probability)));
    return plotLeft + ((exponent + 12) / 12) * (plotRight - plotLeft);
  }
  for (const exponent of [-12, -9, -6, -3, 0]) {
    const x = plotLeft + ((exponent + 12) / 12) * (plotRight - plotLeft);
    pieces.push(
      `<line x1="${x}" y1="${plotTop}" x2="${x}" y2="${plotBottom}" stroke="${COLORS.line}" stroke-width="0.7"/>`,
      textLines([`10^${exponent}`], x, plotBottom + 19, {
        size: 8,
        color: COLORS.muted,
        anchor: "middle",
      }),
    );
  }
  const targetRows = [
    ["BEFORE", beforeProbability, COLORS.blue],
    ["RAW OMITTED", rawProbability, COLORS.ink],
    ["DELETE / REFIT", exactProbability, COLORS.blue],
  ];
  targetRows.forEach(([label, probability, color], index) => {
    const y = plotTop + 48 + index * 72;
    const x = logPosition(probability);
    pieces.push(
      textLines([label], plotLeft, y - 18, {
        size: 8.5,
        color,
        weight: 700,
      }),
      `<line x1="${plotLeft}" y1="${y}" x2="${x}" y2="${y}" stroke="${color}" stroke-width="2"/>`,
      `<circle cx="${x}" cy="${y}" r="7" fill="${color}"/>`,
      textLines([`P ${formatProbability(probability)}`], x, y + 24, {
        size: 10,
        weight: 700,
        family: "Liberation Mono, monospace",
        anchor: "middle",
      }),
    );
  });

  pieces.push(
    `<line x1="610" y1="403" x2="970" y2="403" stroke="${COLORS.line}" stroke-width="0.9"/>`,
    textLines(["RETAINED MEMORY"], 620, 438, {
      size: 10,
      color: COLORS.green,
      weight: 700,
    }),
    `<line x1="625" y1="491" x2="955" y2="491" stroke="${COLORS.line}" stroke-width="8" stroke-linecap="round"/>`,
    `<line x1="625" y1="491" x2="${625 + 330 * retainedProbability}" y2="491" stroke="${COLORS.green}" stroke-width="8" stroke-linecap="round"/>`,
    `<circle cx="${625 + 330 * retainedProbability}" cy="491" r="9" fill="${COLORS.green}"/>`,
    textLines(["0"], 625, 518, { size: 8, color: COLORS.muted, anchor: "middle" }),
    textLines(["1"], 955, 518, { size: 8, color: COLORS.muted, anchor: "middle" }),
    textLines([`P ${retainedProbability.toFixed(6)} · rank ${retainedRank}`], 790, 553, {
      size: 18,
      weight: 700,
      family: "Liberation Mono, monospace",
      anchor: "middle",
    }),
    `<line x1="610" y1="599" x2="970" y2="599" stroke="${COLORS.line}" stroke-width="0.9"/>`,
    textLines(["NUMERICAL CERTIFICATE"], 620, 637, {
      size: 10,
      color: COLORS.green,
      weight: 700,
    }),
    `<circle cx="790" cy="752" r="94" fill="#F1F8F5" stroke="${COLORS.green}" stroke-width="2"/>`,
    textLines(["POLICY ≈ FIXED-C REFIT"], 790, 716, {
      size: 9,
      color: COLORS.green,
      weight: 700,
      anchor: "middle",
    }),
    textLines([`KL ${formatScientific(maxKl)}`], 790, 762, {
      size: 24,
      weight: 700,
      family: "Liberation Mono, monospace",
      anchor: "middle",
    }),
    textLines(["raw omission is separate"], 790, 793, {
      size: 9,
      color: COLORS.muted,
      anchor: "middle",
    }),
    `<line x1="610" y1="874" x2="970" y2="874" stroke="${COLORS.line}" stroke-width="0.9"/>`,
    textLines(["WHAT THE FIGURE ESTABLISHES"], 620, 913, {
      size: 10,
      color: COLORS.ink,
      weight: 700,
    }),
    textLines(["target answer ↓"], 620, 956, {
      size: 17,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["retained answer stays"], 620, 991, {
      size: 17,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["certificate names its reference"], 620, 1026, {
      size: 12,
      color: COLORS.muted,
    }),
    textLines(
      [
        payload.status === "incomplete_case_study_preview"
          ? "15/16 preview"
          : "16/16 report · one predeclared case",
      ],
      970,
      1070,
      {
        size: 9,
        color:
          payload.status === "incomplete_case_study_preview"
            ? COLORS.previewInk
            : COLORS.green,
        weight: 700,
        anchor: "end",
      },
    ),
    "</svg>",
  );
  return pieces.join("");
}


function renderThreeColumnLongMemEvalForgettingSvg(payload) {
  validateForgettingPayload(payload);
  const width = 1000;
  const height = 680;
  const before = payload.case.outcomes[0];
  const raw = payload.case.outcomes[1];
  const exact = payload.case.outcomes[2];
  const beforeProbability = measured(
    before.target.geometric_mean_probability,
    "before target probability",
  );
  const rawProbability = measured(
    raw.target.geometric_mean_probability,
    "raw target probability",
  );
  const exactProbability = measured(
    exact.target.geometric_mean_probability,
    "exact target probability",
  );
  const retainedProbability = measured(
    exact.retained.geometric_mean_probability,
    "retained probability",
  );
  const maxKl = measured(
    payload.case.certificate.max_output_kl_nats,
    "certificate KL",
  );
  const status =
    payload.status === "incomplete_case_study_preview"
      ? PREVIEW_LABEL
      : "MEASURED CASE · 16/16 REPORT";

  const pieces = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="longmem-title" aria-describedby="longmem-desc">`,
    `<title id="longmem-title">One LongMemEval case separates behavioral suppression from a numerical certificate</title>`,
    `<desc id="longmem-desc">A measured teacher-forced case moves from a confirmed record-level forget request to an evaluated fixed-C policy. The behavioral panel compares target probability with fresh raw omission. The numerical panel compares policy output with the fixed-C retained-key refit and states that decoded responses, state equality, and causal regeneration are not covered.</desc>`,
    `<metadata>${escapeXml(
      JSON.stringify({
        schema: payload.schema,
        record_id: payload.case.record_id,
        payload_integrity_sha256: payload.integrity.sha256,
        method_report_sha256:
          payload.provenance.source_artifacts.method_report.file_sha256,
      }),
    )}</metadata>`,
    `<rect width="${width}" height="${height}" fill="${COLORS.white}"/>`,
    textLines([status], 500, 25, {
      size: 18,
      color:
        payload.status === "incomplete_case_study_preview"
          ? COLORS.previewInk
          : COLORS.green,
      weight: 700,
      anchor: "middle",
    }),
  ];

  function stage(x, y, number, heading, detail, options = {}) {
    const {
      fill = COLORS.surface,
      stroke = COLORS.line,
      color = COLORS.ink,
    } = options;
    pieces.push(
      roundedRect(x, y, 270, 92, {
        fill,
        stroke,
        strokeWidth: 1.5,
        radius: 12,
      }),
      textLines([`${number}  ${heading}`], x + 16, y + 28, {
        size: 18,
        color,
        weight: 700,
      }),
      textLines(detail.split("\n"), x + 16, y + 61, {
        size: 16,
        lineHeight: 24,
        color: COLORS.muted,
      }),
    );
  }

  pieces.push(
    textLines(["(a)"], 32, 66, { size: 20, weight: 700 }),
    textLines(["Evaluated record-level workflow"], 72, 66, {
      size: 18,
      weight: 700,
    }),
  );
  stage(32, 88, "1", "SELECTED UNIT", "complete two-turn exchange", {
    fill: COLORS.userBubble,
    stroke: COLORS.blue,
    color: COLORS.blue,
  });
  stage(32, 205, "2", "OPERATION", "confirmed forget request", {
    fill: COLORS.previewFill,
    stroke: COLORS.red,
    color: COLORS.red,
  });
  stage(32, 322, "3", "EXECUTED POLICY", "fixed-C refit path", {
    fill: "#F1F8F5",
    stroke: COLORS.green,
    color: COLORS.green,
  });
  stage(32, 439, "4", "RECORDED CHECK", "teacher-forced scores", {
    fill: "#F5F8FC",
    stroke: COLORS.blue,
    color: COLORS.blue,
  });
  for (const y of [180, 297, 414]) {
    pieces.push(
      `<line x1="167" y1="${y}" x2="167" y2="${y + 20}" stroke="${COLORS.orange}" stroke-width="2"/>`,
      `<path d="M167 ${y + 27} l-6 -9 h12 z" fill="${COLORS.orange}"/>`,
    );
  }

  pieces.push(
    textLines(["(b)"], 340, 66, { size: 20, weight: 700 }),
    textLines(["BEHAVIORAL REFERENCE: fresh raw omission"], 380, 66, {
      size: 18,
      color: COLORS.ink,
      weight: 700,
    }),
    roundedRect(340, 88, 300, 420, {
      fill: "#FBFBFA",
      stroke: COLORS.line,
      strokeWidth: 1.5,
      radius: 12,
    }),
    textLines(["TARGET ANSWER PROBABILITY"], 362, 126, {
      size: 17,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["record present"], 362, 174, {
      size: 16,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines([formatProbability(beforeProbability)], 610, 174, {
      size: 18,
      color: COLORS.blue,
      weight: 700,
      anchor: "end",
      family: "Liberation Mono, monospace",
    }),
    textLines(["fresh raw omission"], 362, 224, {
      size: 16,
      color: COLORS.ink,
      weight: 700,
    }),
    textLines([formatProbability(rawProbability)], 610, 224, {
      size: 18,
      weight: 700,
      anchor: "end",
      family: "Liberation Mono, monospace",
    }),
    textLines(["evaluated policy"], 362, 274, {
      size: 16,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines([formatProbability(exactProbability)], 610, 274, {
      size: 18,
      color: COLORS.blue,
      weight: 700,
      anchor: "end",
      family: "Liberation Mono, monospace",
    }),
    `<line x1="362" y1="308" x2="618" y2="308" stroke="${COLORS.line}" stroke-width="1.2"/>`,
    textLines(["CHECK"], 362, 346, {
      size: 17,
      color: COLORS.orange,
      weight: 700,
    }),
    textLines(["policy scored on registered target"], 362, 378, {
      size: 16,
      color: COLORS.muted,
    }),
    textLines(["ESTABLISHES"], 362, 425, {
      size: 17,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["target suppression in this case"], 362, 458, {
      size: 16,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["NOT COVERED: decoded response"], 362, 492, {
      size: 16,
      color: COLORS.muted,
      weight: 700,
    }),
  );

  pieces.push(
    textLines(["(c)"], 678, 66, { size: 20, weight: 700 }),
    textLines(["REFERENCE: fixed-C refit"], 718, 66, {
      size: 18,
      color: COLORS.green,
      weight: 700,
    }),
    roundedRect(678, 88, 290, 420, {
      fill: "#F7FBF9",
      stroke: COLORS.green,
      strokeWidth: 1.5,
      radius: 12,
    }),
    textLines(["REFERENCE"], 700, 126, {
      size: 17,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["fixed-C retained-key refit"], 700, 159, {
      size: 16,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["CHECK"], 700, 210, {
      size: 17,
      color: COLORS.orange,
      weight: 700,
    }),
    textLines(["KL(policy ‖ refit)"], 700, 243, {
      size: 16,
      color: COLORS.muted,
    }),
    textLines([formatScientific(maxKl)], 823, 300, {
      size: 28,
      color: COLORS.green,
      weight: 700,
      anchor: "middle",
      family: "Liberation Mono, monospace",
    }),
    textLines(["ESTABLISHES"], 700, 354, {
      size: 17,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["registered-output agreement"], 700, 387, {
      size: 16,
      color: COLORS.green,
      weight: 700,
    }),
    textLines([`retained P ${retainedProbability.toFixed(6)}`], 700, 421, {
      size: 16,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["NOT COVERED"], 700, 466, {
      size: 17,
      color: COLORS.muted,
      weight: 700,
    }),
    textLines(["state equality", "causal regeneration"], 700, 492, {
      size: 16,
      lineHeight: 22,
      color: COLORS.muted,
    }),
  );

  pieces.push(
    roundedRect(32, 555, 936, 90, {
      fill: "#EFF1F2",
      stroke: COLORS.line,
      strokeWidth: 1.2,
      radius: 12,
    }),
    textLines(["BOUNDARY"], 54, 590, {
      size: 17,
      color: COLORS.muted,
      weight: 700,
    }),
    textLines(
      [
        "One predeclared case; aggregate efficacy uses the separate frozen 10/16 subset.",
        "The v1 decoded-response attempt terminated without a report and supplies no outcome evidence.",
      ],
      190,
      584,
      {
        size: 16,
        lineHeight: 28,
        color: COLORS.muted,
      },
    ),
    "</svg>",
  );
  return pieces.join("");
}


export function renderLongMemEvalForgettingSvg(payload) {
  validateForgettingPayload(payload);
  const width = 1000;
  const height = 1060;
  const before = payload.case.outcomes[0];
  const raw = payload.case.outcomes[1];
  const exact = payload.case.outcomes[2];
  const beforeProbability = measured(
    before.target.geometric_mean_probability,
    "before target probability",
  );
  const rawProbability = measured(
    raw.target.geometric_mean_probability,
    "raw target probability",
  );
  const exactProbability = measured(
    exact.target.geometric_mean_probability,
    "exact target probability",
  );
  const retainedProbability = measured(
    exact.retained.geometric_mean_probability,
    "retained probability",
  );
  const maxKl = measured(
    payload.case.certificate.max_output_kl_nats,
    "certificate KL",
  );
  const fallbackSolves = measured(
    payload.case.certificate.decrement_fallbacks,
    "fallback count",
  );
  const affectedSolves = measured(
    payload.case.certificate.head_gate_solves,
    "affected solve count",
  );
  const incrementalSolves = affectedSolves - fallbackSolves;
  const pieces = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="longmem-title" aria-describedby="longmem-desc">`,
    `<title id="longmem-title">One LongMemEval case separates behavioral suppression from an implementation check</title>`,
    `<desc id="longmem-desc">A measured teacher-forced case moves from a confirmed record-level forget request to an executed mixed policy. The behavioral panel compares target probability with fresh raw omission. The implementation panel compares the executed path with an independent fixed-C retained-key refit and discloses that 522 of 560 affected solves used fallback while 38 completed incrementally.</desc>`,
    `<metadata>${escapeXml(
      JSON.stringify({
        schema: payload.schema,
        record_id: payload.case.record_id,
        payload_integrity_sha256: payload.integrity.sha256,
      }),
    )}</metadata>`,
    `<rect width="${width}" height="${height}" fill="${COLORS.white}"/>`,
    textLines(
      [
        payload.status === "incomplete_case_study_preview"
          ? PREVIEW_LABEL
          : "MEASURED CASE · 16/16 REPORT",
      ],
      500,
      28,
      {
        size: 22,
        color:
          payload.status === "incomplete_case_study_preview"
            ? COLORS.previewInk
            : COLORS.green,
        weight: 700,
        anchor: "middle",
      },
    ),
    textLines(["(a)"], 32, 72, { size: 24, weight: 700 }),
    textLines(["Evaluated record-level workflow"], 84, 72, {
      size: 22,
      weight: 700,
    }),
  ];

  const workflow = [
    ["1  SELECTED UNIT", "two-turn exchange", COLORS.blue, COLORS.userBubble],
    ["2  OPERATION", "confirmed forget", COLORS.red, COLORS.previewFill],
    ["3  POLICY", "executed mixed path", COLORS.green, "#F1F8F5"],
    ["4  CHECK", "teacher forcing", COLORS.blue, "#F5F8FC"],
  ];
  workflow.forEach(([heading, detail, color, fill], index) => {
    const x = 32 + index * 241;
    pieces.push(
      roundedRect(x, 96, 210, 128, {
        fill,
        stroke: color,
        strokeWidth: 1.6,
        radius: 12,
      }),
      textLines([heading], x + 15, 136, {
        size: 22,
        color,
        weight: 700,
      }),
      textLines([detail], x + 15, 184, {
        size: 21,
        color: COLORS.muted,
      }),
    );
    if (index < workflow.length - 1) {
      pieces.push(
        `<line x1="${x + 214}" y1="160" x2="${x + 232}" y2="160" stroke="${COLORS.orange}" stroke-width="2"/>`,
        `<path d="M${x + 239} 160 l-9 -6 v12 z" fill="${COLORS.orange}"/>`,
      );
    }
  });

  pieces.push(
    textLines(["(b)"], 32, 302, { size: 24, weight: 700 }),
    textLines(["Comparison A — policy vs fresh omission"], 84, 302, {
      size: 22,
      weight: 700,
    }),
    roundedRect(32, 326, 936, 290, {
      fill: "#FBFBFA",
      stroke: COLORS.line,
      strokeWidth: 1.5,
      radius: 12,
    }),
    textLines(["TARGET ANSWER PROBABILITY"], 58, 370, {
      size: 22,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["record present", "fresh omission", "evaluated policy"], 58, 420, {
      size: 21,
      lineHeight: 52,
      weight: 700,
    }),
    textLines(
      [
        formatProbability(beforeProbability),
        formatProbability(rawProbability),
        formatProbability(exactProbability),
      ],
      430,
      420,
      {
        size: 22,
        lineHeight: 52,
        family: "Liberation Mono, monospace",
        weight: 700,
        anchor: "end",
      },
    ),
    `<line x1="482" y1="350" x2="482" y2="592" stroke="${COLORS.line}" stroke-width="1.2"/>`,
    textLines(["MEASURED"], 520, 370, {
      size: 22,
      color: COLORS.orange,
      weight: 700,
    }),
    textLines(["registered target score"], 520, 408, {
      size: 21,
      color: COLORS.muted,
    }),
    textLines(["ESTABLISHES"], 520, 468, {
      size: 22,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["target suppression in this case"], 520, 506, {
      size: 21,
      color: COLORS.blue,
      weight: 700,
    }),
    textLines(["NOT COVERED"], 520, 562, {
      size: 22,
      color: COLORS.muted,
      weight: 700,
    }),
    textLines(["decoded response"], 718, 562, {
      size: 21,
      color: COLORS.muted,
    }),
  );

  pieces.push(
    textLines(["(c)"], 32, 680, { size: 24, weight: 700 }),
    textLines(["Implementation check — mixed path vs independent refit"], 84, 680, {
      size: 22,
      weight: 700,
    }),
    roundedRect(32, 704, 936, 230, {
      fill: "#F7FBF9",
      stroke: COLORS.green,
      strokeWidth: 1.5,
      radius: 12,
    }),
    textLines(["INDEPENDENT REFERENCE"], 58, 750, {
      size: 22,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["fixed-C retained-key", "refit"], 58, 790, {
      size: 21,
      lineHeight: 30,
      color: COLORS.green,
      weight: 700,
    }),
    `<line x1="300" y1="816" x2="344" y2="816" stroke="${COLORS.orange}" stroke-width="2"/>`,
    `<path d="M354 816 l-10 -7 v14 z" fill="${COLORS.orange}"/>`,
    textLines(["MEASURED"], 380, 750, {
      size: 22,
      color: COLORS.orange,
      weight: 700,
    }),
    textLines(["KL(executed path ‖ refit)"], 380, 790, {
      size: 21,
      color: COLORS.muted,
    }),
    textLines([formatScientific(maxKl)], 380, 845, {
      size: 28,
      color: COLORS.green,
      weight: 700,
      family: "Liberation Mono, monospace",
    }),
    `<line x1="606" y1="816" x2="650" y2="816" stroke="${COLORS.orange}" stroke-width="2"/>`,
    `<path d="M660 816 l-10 -7 v14 z" fill="${COLORS.orange}"/>`,
    textLines(["CHECKS"], 686, 750, {
      size: 22,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(["implementation conformance"], 686, 790, {
      size: 21,
      color: COLORS.green,
      weight: 700,
    }),
    textLines(
      [
        `${fallbackSolves}/${affectedSolves} fallback`,
        `${incrementalSolves} incremental`,
      ],
      686,
      816,
      {
        size: 19,
        lineHeight: 24,
        color: COLORS.green,
        weight: 700,
      },
    ),
    textLines(["NOT COVERED"], 686, 872, {
      size: 22,
      color: COLORS.muted,
      weight: 700,
    }),
    textLines(["state equality", "causal regeneration"], 686, 902, {
      size: 19,
      lineHeight: 20,
      color: COLORS.muted,
    }),
    roundedRect(32, 966, 936, 70, {
      fill: "#EFF1F2",
      stroke: COLORS.line,
      strokeWidth: 1.2,
      radius: 12,
    }),
    textLines(["BOUNDARY"], 52, 1008, {
      size: 22,
      color: COLORS.muted,
      weight: 700,
    }),
    textLines(
      [
        "Historical case only; decoded K=32 clusters, n=96 histories are separate.",
      ],
      220,
      1008,
      {
        size: 21,
        color: COLORS.muted,
      },
    ),
    "</svg>",
  );
  return pieces.join("");
}


function svgDimensions(svg) {
  const width = Number(svg.match(/\bwidth="([0-9.]+)"/)?.[1]);
  const height = Number(svg.match(/\bheight="([0-9.]+)"/)?.[1]);
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error("rendered SVG has no numeric dimensions");
  }
  return { width, height };
}


async function writePdf(svg, filename) {
  const { width, height } = svgDimensions(svg);
  await new Promise((resolve, reject) => {
    const stream = createWriteStream(filename);
    const document = new PDFDocument({
      size: [width, height],
      margin: 0,
      compress: true,
      info: {
        Title: "Measured LongMemEval forgetting case study",
        Author: "Anonymous research artifact",
        Subject: "Source-bound recorded final case study",
        CreationDate: new Date("2026-08-24T00:00:00.000Z"),
        ModDate: new Date("2026-08-24T00:00:00.000Z"),
      },
    });
    document.pipe(stream);
    SVGtoPDF(document, svg, 0, 0, {
      width,
      height,
      preserveAspectRatio: "xMinYMin meet",
      assumePt: true,
    });
    document.end();
    stream.on("error", reject);
    once(stream, "finish").then(resolve, reject);
  });
}


export async function writeLongMemEvalForgettingFigure(payload, outputBase) {
  const svg = renderLongMemEvalForgettingSvg(payload);
  await fs.mkdir(path.dirname(outputBase), { recursive: true });
  await fs.writeFile(`${outputBase}.svg`, `${svg}\n`, "utf8");
  await writePdf(svg, `${outputBase}.pdf`);
  const png = new Resvg(svg, {
    background: "white",
    fitTo: { mode: "zoom", value: 3 },
  })
    .render()
    .asPng();
  await fs.writeFile(`${outputBase}.png`, normalizePngSrgb(png));
  return {
    svg: `${outputBase}.svg`,
    pdf: `${outputBase}.pdf`,
    png: `${outputBase}.png`,
  };
}


export async function renderLongMemEvalForgettingFigure(
  payloadPath = DEFAULT_PAYLOAD,
  outputBase = DEFAULT_OUTPUT_BASE,
) {
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  return writeLongMemEvalForgettingFigure(payload, outputBase);
}


if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const payloadPath = process.argv[2]
    ? path.resolve(process.argv[2])
    : DEFAULT_PAYLOAD;
  const outputBase = process.argv[3]
    ? path.resolve(process.argv[3])
    : DEFAULT_OUTPUT_BASE;
  const written = await renderLongMemEvalForgettingFigure(payloadPath, outputBase);
  console.log(`wrote ${written.svg}`);
  console.log(`wrote ${written.pdf}`);
  console.log(`wrote ${written.png}`);
}
