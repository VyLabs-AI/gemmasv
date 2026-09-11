import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  renderLongMemEvalForgettingSvg,
  validateForgettingPayload,
  writeLongMemEvalForgettingFigure,
} from "./render_longmemeval_chat_forgetting.mjs";
import { parsePngChunks } from "./png_srgb.mjs";


const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAYLOAD_PATH = path.join(
  HERE,
  "..",
  "gemma_sv",
  "demo_site",
  "assets",
  "longmemeval_forgetting_final.json",
);
const CAPTION_PATH = path.join(
  HERE,
  "..",
  "..",
  "paper",
  "appendix.tex",
);


async function fixture() {
  return JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
}


test("chat-forgetting SVG is deterministic and visibly final", async () => {
  const payload = await fixture();
  validateForgettingPayload(payload);
  const first = renderLongMemEvalForgettingSvg(payload);
  const second = renderLongMemEvalForgettingSvg(payload);
  assert.equal(first, second);
  assert.match(first, /^<svg/);
  assert.doesNotMatch(first, /INCOMPLETE CASE STUDY PREVIEW/);
  assert.match(first, /16\/16 REPORT/);
  assert.match(first, /width="1000" height="1060"/);
  assert.match(first, /SELECTED UNIT/);
  assert.match(first, /Comparison A — policy vs fresh omission/);
  assert.match(first, /registered target score/);
  assert.match(first, /1\.40e-11/);
  assert.match(first, /4\.33e-17/);
  assert.match(first, /Implementation check — mixed path vs independent refit/);
  assert.match(first, /KL\(executed path ‖ refit\)/);
  assert.match(first, /implementation conformance/);
  assert.match(first, /522\/560 fallback/);
  assert.match(first, /38 incremental/);
  assert.match(first, /decoded response/);
  assert.match(first, /aria-describedby="longmem-desc"/);
  assert.doesNotMatch(first, /Since I&apos;ll have four bikes with me/);
  assert.equal(payload.status, "final_case_study");
  assert.equal(payload.preview_label, null);
  assert.equal(payload.scope.final_all16_complete, true);
});


test("chat-forgetting renderer is deterministic for SVG PDF and sRGB PNG", async () => {
  const payload = await fixture();
  const directory = await fs.mkdtemp(
    path.join(os.tmpdir(), "longmemeval-forgetting-"),
  );
  const written = await writeLongMemEvalForgettingFigure(
    payload,
    path.join(directory, "figure-a"),
  );
  const repeated = await writeLongMemEvalForgettingFigure(
    payload,
    path.join(directory, "figure-b"),
  );
  const [svg, pdf, png] = await Promise.all([
    fs.readFile(written.svg, "utf8"),
    fs.readFile(written.pdf),
    fs.readFile(written.png),
  ]);
  for (const format of ["svg", "pdf", "png"]) {
    assert.deepEqual(
      await fs.readFile(written[format]),
      await fs.readFile(repeated[format]),
    );
  }
  assert.match(svg, /^<svg/);
  assert.equal(pdf.subarray(0, 4).toString("ascii"), "%PDF");
  assert.deepEqual(
    parsePngChunks(png)
      .filter(({ type }) => ["sRGB", "iCCP"].includes(type))
      .map(({ type, payload: chunk }) => [type, [...chunk]]),
    [["sRGB", [0]]],
  );
});


test("renderer rejects aggregate claims and outcome relabeling", async () => {
  const payload = await fixture();
  const aggregate = structuredClone(payload);
  aggregate.scope.aggregate_claims.push("fabricated aggregate");
  assert.throws(
    () => validateForgettingPayload(aggregate),
    /aggregate or full-repack claim/,
  );

  const relabeled = structuredClone(payload);
  relabeled.case.outcomes[2].execution.executed_method.value =
    "incremental_float64_fixed_c_decrement";
  assert.throws(
    () => validateForgettingPayload(relabeled),
    /fallback disclosure differs/,
  );

  const wrongReport = structuredClone(payload);
  wrongReport.provenance.source_artifacts.method_report.file_sha256 = "0".repeat(64);
  assert.throws(
    () => validateForgettingPayload(wrongReport),
    /validated all16 case study/,
  );

  const wrongResolver = structuredClone(payload);
  wrongResolver.case.memory_resolution.decision.selected_record_id =
    wrongResolver.case.memory_resolution.candidate_catalog[1].record_id;
  assert.throws(
    () => validateForgettingPayload(wrongResolver),
    /resolver fixture differs/,
  );
});


test("recorded figure carries no model response text", async () => {
  const payload = await fixture();
  const serialized = JSON.stringify(payload);
  assert.equal(payload.contains_model_generated_text, false);
  assert.equal(payload.demo.model_outcome_text_present, false);
  assert.doesNotMatch(serialized, /"generated_text"/);
  assert.doesNotMatch(serialized, /"model_output_text"/);
  assert.equal(payload.scope.aggregate_claims.length, 0);
});


test("compact figure omits source dialogue while payload retains it", async () => {
  const payload = await fixture();
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(payload.case.dialogue).map(([section, value]) => [
        section,
        value.quotes.length,
      ]),
    ),
    { intervening: 2, retained: 2, target: 2 },
  );
  const svg = renderLongMemEvalForgettingSvg(payload);
  assert.doesNotMatch(svg, /Since I&apos;ll have four bikes with me/);
  assert.doesNotMatch(svg, /Phone: \+49 \(0\) 62 32/);
  for (const quote of [
    ...payload.case.dialogue.target.quotes,
    ...payload.case.dialogue.retained.quotes,
    ...payload.case.dialogue.intervening.quotes,
  ]) {
    assert.doesNotMatch(svg, new RegExp(quote.source_turn_sha256.slice(0, 8)));
  }
  assert.doesNotMatch(svg, /MEASURED REGISTERED-PROBE OUTCOMES/);
});


test("displayed chat has exact complete source units and no ellipses", async () => {
  const payload = await fixture();
  for (const section of ["target", "retained", "intervening"]) {
    for (const quote of payload.case.dialogue[section].quotes) {
      assert.equal(quote.truncated, false);
      assert.equal(
        quote.excerpt_policy.selection,
        "exact UTF-8 source substring",
      );
      assert.equal(quote.excerpt_policy.normalization, "none");
      assert.doesNotMatch(quote.display_text, /\.{3}|…/u);
    }
  }
  assert.doesNotMatch(renderLongMemEvalForgettingSvg(payload), /\.{3}|…/u);
});


test("paper caption names both references and scope", async () => {
  const caption = await fs.readFile(CAPTION_PATH, "utf8");
  assert.match(caption, /teacher-forced scores/);
  assert.match(caption, /fresh raw omission/);
  assert.match(caption, /fixed-\\\(C\\\) retained-key refit/);
  assert.match(caption, /One historical LongMemEval pilot case/);
  assert.match(caption, /Implementation check---against the independent/);
  assert.match(caption, /executed mixed path reaches KL/);
  assert.match(caption, /522\/560/);
});
