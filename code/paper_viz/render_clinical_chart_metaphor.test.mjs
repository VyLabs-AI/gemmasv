import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  renderClinicalChartMetaphorSvg,
  writeClinicalChartMetaphorFigure,
} from "./render_clinical_chart_metaphor.mjs";
import { parsePngChunks } from "./png_srgb.mjs";


test("clinical-chart metaphor is deterministic and scoped", () => {
  const first = renderClinicalChartMetaphorSvg();
  const second = renderClinicalChartMetaphorSvg();
  assert.equal(first, second);
  assert.match(first, /^<svg/);
  assert.match(first, /FICTIONAL ANALOGY — NOT MEASURED EVIDENCE/);
  assert.match(first, /aria-labelledby="clinical-title"/);
  assert.match(first, /aria-describedby="clinical-desc"/);
  assert.match(first, /RECORD PRESENT/);
  assert.match(first, /PROMPT SUPPRESSION/);
  assert.match(first, /MEMORY EDIT \/ REPLAY/);
  assert.match(first, /COMPARISON REFERENCE/);
  assert.match(first, /Incorrect fact stored/);
  assert.match(first, /Allergy: latex/);
  assert.match(first, /Fact hidden from answer/);
  assert.match(first, />HIDE</);
  assert.match(first, /REMOVE CARD/);
  assert.match(first, /OR REWIND \+ REPLAY/);
  assert.match(first, /COMPARE/);
  assert.match(first, /OUTSIDE CHECK/);
  assert.doesNotMatch(first, /HIPAA|GDPR|compliant|all information erased/i);
});


test("clinical renderer is byte-deterministic for SVG PDF and sRGB PNG", async () => {
  const directory = await fs.mkdtemp(
    path.join(os.tmpdir(), "clinical-chart-metaphor-"),
  );
  const written = await writeClinicalChartMetaphorFigure(
    path.join(directory, "figure-a"),
  );
  const repeated = await writeClinicalChartMetaphorFigure(
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
      .map(({ type, payload }) => [type, [...payload]]),
    [["sRGB", [0]]],
  );
});
