import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { compile } from "vega-lite";

import {
  buildDeletionFrontierSpec,
  buildUtilitySpec,
  renderGemmaFigures,
} from "./render_gemma_iclr.mjs";
import { parsePngChunks } from "./png_srgb.mjs";

function interval(mean, halfWidth = 0.2) {
  return {
    n: 3,
    mean,
    lower: mean - halfWidth,
    upper: mean + halfWidth,
  };
}

function fixture() {
  const methods = {};
  const methodIds = [
    "full_repack",
    "exact_decrement",
    "fixed_c_refit",
    "fp32_proxy",
    "cache_delete_shift",
    "decay_0_01",
    "icul_4",
  ];
  methodIds.forEach((method, index) => {
    methods[method] = {
      metrics: {
        mean_end_to_end_median_seconds: interval(0.4 + index * 0.25, 0.04),
        mean_deleted_behavioral_kl_to_repack_nats: interval(
          index === 0 ? 0 : 10 ** (-10 + index),
          index === 0 ? 0 : 10 ** (-11 + index),
        ),
        mean_incremental_tensor_storage_bytes: interval(
          (40 + index * 8) * 2 ** 20,
          2 ** 20,
        ),
      },
    };
  });
  return {
    seed_rows: [
      {
        seed: 0,
        recovered_ppl: 21.4,
        control_ppl: 19.0,
        paired_utility_cost_percent: 12.8,
      },
      {
        seed: 1,
        recovered_ppl: 20.6,
        control_ppl: 19.1,
        paired_utility_cost_percent: 7.6,
      },
      {
        seed: 2,
        recovered_ppl: 19.6,
        control_ppl: 18.8,
        paired_utility_cost_percent: 4.3,
      },
    ],
    perplexity: {
      recovered: interval(20.5, 2.3),
      control: interval(19.0, 0.4),
      paired_utility_cost_percent: interval(8.2, 5.1),
    },
    deletion_baselines: { methods },
  };
}

test("Flint pilot specs compile to Vega-Lite", async () => {
  const summary = fixture();
  const utility = await buildUtilitySpec(summary);
  const deletion = await buildDeletionFrontierSpec(summary);
  assert.ok(compile(utility).spec);
  assert.ok(compile(deletion).spec);
  assert.equal(utility.vconcat.length, 2);
  assert.equal(deletion.layer.length, 3);
  assert.equal(deletion.layer[2].mark.type, "circle");
});

test("pilot renderers are deterministic for SVG PDF and sRGB PNG", async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "paper-viz-"));
  const summaryPath = path.join(directory, "summary.json");
  const outputDirectory = path.join(directory, "figs-a");
  const repeatedDirectory = path.join(directory, "figs-b");
  await fs.writeFile(summaryPath, JSON.stringify(fixture()));
  const written = await renderGemmaFigures(summaryPath, outputDirectory);
  const repeated = await renderGemmaFigures(summaryPath, repeatedDirectory);
  for (const key of ["utility", "deletion"]) {
    for (const extension of [".svg", ".pdf", ".png"]) {
      const firstPath = written[key].replace(/\.pdf$/, extension);
      const repeatedPath = repeated[key].replace(/\.pdf$/, extension);
      assert.deepEqual(
        await fs.readFile(firstPath),
        await fs.readFile(repeatedPath),
      );
    }
    const svg = await fs.readFile(
      written[key].replace(/\.pdf$/, ".svg"),
      "utf8",
    );
    const pdf = await fs.readFile(written[key]);
    const png = await fs.readFile(written[key].replace(/\.pdf$/, ".png"));
    assert.match(svg, /^<svg/);
    assert.equal(pdf.subarray(0, 4).toString("ascii"), "%PDF");
    assert.deepEqual(
      parsePngChunks(png)
        .filter(({ type }) => ["sRGB", "iCCP"].includes(type))
        .map(({ type, payload }) => [type, [...payload]]),
      [["sRGB", [0]]],
    );
  }
});
