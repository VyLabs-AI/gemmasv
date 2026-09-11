import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { once } from "node:events";

import { assembleVegaLite } from "flint-chart";
import { stripPrivateKeys } from "flint-chart-mcp/render";
import { Resvg } from "@resvg/resvg-js";
import PDFDocument from "pdfkit";
import SVGtoPDF from "svg-to-pdfkit";
import * as vega from "vega";
import { compile } from "vega-lite";

import { normalizePngSrgb } from "./png_srgb.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RELEASE_ROOT = path.resolve(HERE, "..", "..");
const DEFAULT_SUMMARY = path.join(
  RELEASE_ROOT,
  "code",
  "gemma_sv",
  "benchmarks",
  "iclr_multiseed_v1.json",
);
const DEFAULT_OUTPUT = path.join(RELEASE_ROOT, "paper", "figs");
const NUMERICAL_FLOOR = 1e-14;

const ARM_COLORS = {
  Recovered: "#0072B2",
  Control: "#E69F00",
};
const METHOD_LABELS = {
  full_repack: "Full repack",
  exact_decrement: "Exact decrement",
  fixed_c_refit: "Fixed-C refit",
  fp32_proxy: "FP32 proxy",
  cache_delete_shift: "KV delete/shift",
  decay_0_01: "Decay",
  icul_4: "ICUL",
};
const METHOD_COLORS = {
  "Full repack": "#000000",
  "Exact decrement": "#0072B2",
  "Fixed-C refit": "#56B4E9",
  "FP32 proxy": "#009E73",
  "KV delete/shift": "#E69F00",
  Decay: "#D55E00",
  ICUL: "#CC79A7",
};

async function readJson(filename) {
  return JSON.parse(await fs.readFile(filename, "utf8"));
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${label} must be finite`);
  }
  return number;
}

function ci(metric, label) {
  if (!metric || metric.lower == null || metric.upper == null) {
    throw new Error(`${label} requires a defined confidence interval`);
  }
  return {
    mean: finite(metric.mean, `${label}.mean`),
    lower: finite(metric.lower, `${label}.lower`),
    upper: finite(metric.upper, `${label}.upper`),
  };
}

export function flintSpec(template, values) {
  const assembled = assembleVegaLite({
    data: { values },
    semantic_types: template.semantic_types,
    chart_spec: template.chart_spec,
  });
  const width = assembled._width;
  const height = assembled._height;
  const spec = stripPrivateKeys(assembled);
  delete spec.config;
  if (width) spec.width = width;
  if (height && typeof spec.height !== "object") spec.height = height;
  return spec;
}

function colorEncoding(field, domain, range, legend = null) {
  return {
    field,
    type: "nominal",
    scale: { domain, range },
    legend,
  };
}

function cleanPanel(spec, title, width, height) {
  spec.title = {
    text: title,
    anchor: "start",
    fontSize: 25,
    fontWeight: 600,
    offset: 10,
  };
  spec.width = width;
  spec.height = height;
  return spec;
}

export async function buildUtilitySpec(summary, templatePath = null) {
  const templates = await readJson(
    templatePath ?? path.join(HERE, "specs", "gemma-utility-ci.json"),
  );
  const recovered = ci(summary?.perplexity?.recovered, "recovered PPL");
  const control = ci(summary?.perplexity?.control, "control PPL");
  const cost = ci(
    summary?.perplexity?.paired_utility_cost_percent,
    "paired utility cost",
  );
  const seedRows = summary?.seed_rows;
  const minimumSeeds = summary?.preview ? 2 : 3;
  if (!Array.isArray(seedRows) || seedRows.length < minimumSeeds) {
    throw new Error(
      `utility figure requires at least ${minimumSeeds} seed rows`,
    );
  }

  const pplEndpoints = [
    { Arm: "Recovered", Perplexity: recovered.lower },
    { Arm: "Recovered", Perplexity: recovered.upper },
    { Arm: "Control", Perplexity: control.lower },
    { Arm: "Control", Perplexity: control.upper },
  ];
  const pplMeans = [
    {
      Arm: "Recovered",
      Perplexity: recovered.mean,
      Label: recovered.mean.toFixed(2),
    },
    {
      Arm: "Control",
      Perplexity: control.mean,
      Label: control.mean.toFixed(2),
    },
  ];
  const pplSeeds = seedRows.flatMap((row) => [
    {
      Arm: "Recovered",
      Perplexity: finite(row.recovered_ppl, "seed recovered PPL"),
    },
    {
      Arm: "Control",
      Perplexity: finite(row.control_ppl, "seed control PPL"),
    },
  ]);
  const armDomain = Object.keys(ARM_COLORS);
  const armRange = Object.values(ARM_COLORS);
  const ppl = cleanPanel(
    flintSpec(templates.ppl, pplEndpoints),
    "(a) Perplexity across training seeds",
    390,
    105,
  );
  ppl.encoding.x.axis = {
    title: "WikiText-103 perplexity (lower is better)",
    format: ".2f",
    grid: true,
  };
  ppl.encoding.y.axis = { title: null };
  ppl.layer[0].mark = { type: "line", strokeWidth: 2.2 };
  ppl.layer[0].encoding.color = colorEncoding(
    "Arm",
    armDomain,
    armRange,
  );
  ppl.layer[1].mark = {
    type: "point",
    filled: true,
    size: 34,
    opacity: 0.65,
  };
  ppl.layer[1].encoding.color = colorEncoding(
    "Arm",
    armDomain,
    armRange,
  );
  ppl.layer.push(
    {
      data: { values: pplSeeds },
      mark: {
        type: "point",
        filled: true,
        size: 42,
        opacity: 0.5,
        stroke: "white",
        strokeWidth: 0.6,
      },
      encoding: {
        x: { field: "Perplexity", type: "quantitative" },
        y: { field: "Arm", type: "nominal" },
        color: colorEncoding("Arm", armDomain, armRange),
      },
    },
    {
      data: { values: pplMeans },
      mark: {
        type: "point",
        shape: "diamond",
        filled: true,
        size: 120,
        stroke: "white",
        strokeWidth: 1,
      },
      encoding: {
        x: { field: "Perplexity", type: "quantitative" },
        y: { field: "Arm", type: "nominal" },
        color: colorEncoding("Arm", armDomain, armRange),
      },
    },
    {
      data: { values: pplMeans },
      mark: {
        type: "text",
        align: "left",
        baseline: "middle",
        dx: 9,
        fontSize: 22,
        fontWeight: 600,
      },
      encoding: {
        x: { field: "Perplexity", type: "quantitative" },
        y: { field: "Arm", type: "nominal" },
        text: { field: "Label" },
      },
    },
  );

  const costEndpoints = [
    { Estimate: "Paired cost", "Utility cost (%)": cost.lower },
    { Estimate: "Paired cost", "Utility cost (%)": cost.upper },
  ];
  const costSeeds = seedRows.map((row) => ({
    Estimate: "Paired cost",
    "Utility cost (%)": finite(
      row.paired_utility_cost_percent,
      "seed utility cost",
    ),
  }));
  const costMeans = [
    {
      Estimate: "Paired cost",
      "Utility cost (%)": cost.mean,
      Label: `${cost.mean.toFixed(1)}%`,
    },
  ];
  const utilityCost = cleanPanel(
    flintSpec(templates.cost, costEndpoints),
    "(b) Paired gate cost",
    280,
    105,
  );
  utilityCost.encoding.x.axis = {
    title: "Recovered / control − 1 (%)",
    format: ".1f",
    grid: true,
  };
  utilityCost.encoding.y.axis = { title: null, labels: false, ticks: false };
  utilityCost.layer[0].mark = {
    type: "line",
    strokeWidth: 2.2,
    color: "#6F4E7C",
  };
  utilityCost.layer[1].mark = {
    type: "point",
    filled: true,
    size: 34,
    opacity: 0.65,
    color: "#6F4E7C",
  };
  delete utilityCost.layer[1].encoding.color;
  utilityCost.layer.push(
    {
      mark: { type: "rule", color: "#777777", strokeDash: [3, 3] },
      encoding: { x: { datum: 0 } },
    },
    {
      data: { values: costSeeds },
      mark: {
        type: "point",
        filled: true,
        size: 42,
        opacity: 0.5,
        color: "#6F4E7C",
        stroke: "white",
        strokeWidth: 0.6,
      },
      encoding: {
        x: { field: "Utility cost (%)", type: "quantitative" },
        y: { field: "Estimate", type: "nominal" },
      },
    },
    {
      data: { values: costMeans },
      mark: {
        type: "point",
        shape: "diamond",
        filled: true,
        size: 120,
        color: "#6F4E7C",
        stroke: "white",
        strokeWidth: 1,
      },
      encoding: {
        x: { field: "Utility cost (%)", type: "quantitative" },
        y: { field: "Estimate", type: "nominal" },
      },
    },
    {
      data: { values: costMeans },
      mark: {
        type: "text",
        align: "left",
        baseline: "middle",
        dx: 9,
        fontSize: 22,
        fontWeight: 600,
      },
      encoding: {
        x: { field: "Utility cost (%)", type: "quantitative" },
        y: { field: "Estimate", type: "nominal" },
        text: { field: "Label" },
      },
    },
  );

  const result = {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    vconcat: [ppl, utilityCost],
    spacing: 28,
    config: paperConfig(),
    resolve: { scale: { color: "independent" } },
  };
  if (summary.preview) {
    result.title = {
      text: `STYLE PREVIEW — ${summary.preview_note ?? "provisional data"}`,
      anchor: "start",
      color: "#9C2F2F",
      fontSize: 11,
      fontWeight: 600,
      offset: 12,
    };
  }
  return result;
}

function metric(method, name, label) {
  return ci(method?.metrics?.[name], label);
}

export async function buildDeletionFrontierSpec(summary, templatePath = null) {
  const template = await readJson(
    templatePath ?? path.join(HERE, "specs", "gemma-deletion-frontier.json"),
  );
  const methods = summary?.deletion_baselines?.methods;
  if (!methods || typeof methods !== "object") {
    throw new Error("deletion frontier requires deletion_baselines.methods");
  }
  const rows = Object.entries(METHOD_LABELS).map(([methodId, display]) => {
    const method = methods[methodId];
    if (!method) throw new Error(`missing deletion method ${methodId}`);
    const latency = metric(
      method,
      "mean_end_to_end_median_seconds",
      `${methodId} latency`,
    );
    const kl = metric(
      method,
      "mean_deleted_behavioral_kl_to_repack_nats",
      `${methodId} KL`,
    );
    const storage = metric(
      method,
      "mean_incremental_tensor_storage_bytes",
      `${methodId} storage`,
    );
    const plottedKl = Math.max(NUMERICAL_FLOOR, kl.mean);
    const labelFactor = {
      full_repack: 2.2,
      exact_decrement: 5.0,
      fixed_c_refit: 0.2,
      fp32_proxy: 0.28,
      cache_delete_shift: 1.5,
      decay_0_01: 5.0,
      icul_4: 1.45,
    }[methodId];
    const labelLatency = latency.mean + {
      full_repack: 0.7,
      exact_decrement: -3.0,
      fixed_c_refit: -5.0,
      fp32_proxy: 2.8,
      cache_delete_shift: 0.8,
      decay_0_01: 2.2,
      icul_4: 1.0,
    }[methodId];
    return {
      Method: display,
      "End-to-end latency (s)": latency.mean,
      "Label latency": Math.max(0, labelLatency),
      "Latency lower": Math.max(0, latency.lower),
      "Latency upper": latency.upper,
      "Output KL to full repack (nats)": plottedKl,
      "Label KL": Math.max(NUMERICAL_FLOOR, plottedKl * labelFactor),
      "KL lower": Math.max(NUMERICAL_FLOOR, kl.lower),
      "KL upper": Math.max(NUMERICAL_FLOOR, kl.upper),
      "Extra state (MiB)": storage.mean / 2 ** 20,
    };
  });
  const assembled = flintSpec(template, rows);
  const pointEncoding = assembled.encoding;
  pointEncoding.x.axis = {
    title: "Deletion + four-probe query latency (s, lower is better)",
    format: ".2~f",
    grid: true,
  };
  pointEncoding.y.axis = {
    title: "Behavioral KL to full repack (nats, lower is better)",
    format: ".1e",
    grid: true,
  };
  pointEncoding.color = colorEncoding(
    "Method",
    Object.values(METHOD_LABELS),
    Object.values(METHOD_LABELS).map((label) => METHOD_COLORS[label]),
    { orient: "right", title: null, columns: 1 },
  );
  delete pointEncoding.size;
  const result = {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    width: 640,
    height: 300,
    data: { values: rows },
    layer: [
      {
        mark: { type: "rule", strokeWidth: 1, opacity: 0.55 },
        encoding: {
          x: { field: "Latency lower", type: "quantitative" },
          x2: { field: "Latency upper" },
          y: pointEncoding.y,
          color: pointEncoding.color,
        },
      },
      {
        mark: { type: "rule", strokeWidth: 1, opacity: 0.55 },
        encoding: {
          x: pointEncoding.x,
          y: {
            field: "KL lower",
            type: "quantitative",
            scale: { type: "log" },
          },
          y2: { field: "KL upper" },
          color: pointEncoding.color,
        },
      },
      {
        mark: {
          type: "circle",
          filled: true,
          opacity: 0.92,
          stroke: "white",
          strokeWidth: 1,
          size: 125,
        },
        encoding: pointEncoding,
      },
    ],
    config: paperConfig(),
  };
  if (summary.preview) {
    result.title = {
      text: "STYLE PREVIEW — one-record smoke data; not paper evidence",
      anchor: "start",
      color: "#9C2F2F",
      fontSize: 11,
      fontWeight: 600,
      offset: 12,
    };
  }
  return result;
}

export function paperConfig() {
  return {
    font: "Liberation Sans",
    background: "white",
    view: { stroke: null },
    axis: {
      domainColor: "#444444",
      domainWidth: 0.8,
      gridColor: "#E5E5E5",
      gridOpacity: 0.7,
      gridWidth: 0.7,
      labelColor: "#222222",
      labelFontSize: 21,
      tickColor: "#666666",
      tickSize: 3,
      titleColor: "#111111",
      titleFontSize: 23,
      titleFontWeight: 500,
      titlePadding: 8,
    },
    legend: {
      labelFontSize: 21,
      labelLimit: 300,
      symbolStrokeWidth: 1,
      titleFontSize: 23,
    },
  };
}

async function renderVegaLite(spec) {
  const compiled = compile(spec, { config: spec.config }).spec;
  const view = new vega.View(vega.parse(compiled), {
    renderer: "none",
  }).initialize();
  const svg = await view.toSVG();
  await view.finalize();
  return svg;
}

function svgDimensions(svg) {
  const width = Number(svg.match(/\bwidth="([0-9.]+)"/)?.[1]);
  const height = Number(svg.match(/\bheight="([0-9.]+)"/)?.[1]);
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error("rendered SVG has no numeric dimensions");
  }
  return { width, height };
}

const FIGURE_METADATA = {
  utility_multiseed: {
    title: "Recovered and control utility remain close across three seeds",
    description:
      "Three measured seed points and mean confidence intervals compare recovered and control WikiText perplexity and the paired utility-cost estimate.",
  },
  deletion_frontier: {
    title: "Legacy deletion methods trade latency against behavioral KL",
    description:
      "Measured method points compare end-to-end latency with behavioral KL to full repack, with seed-level uncertainty intervals and no universal-frontier claim.",
  },
};


function accessibleSvg(svg, figureId, metadata) {
  return svg.replace(
    /<svg\b([^>]*)>/,
    `<svg role="img" aria-labelledby="${figureId}-title" aria-describedby="${figureId}-desc"$1>` +
      `<title id="${figureId}-title">${metadata.title}</title>` +
      `<desc id="${figureId}-desc">${metadata.description}</desc>`,
  );
}


async function writePdf(svg, filename, metadata) {
  const { width, height } = svgDimensions(svg);
  await new Promise((resolve, reject) => {
    const stream = createWriteStream(filename);
    const document = new PDFDocument({
      size: [width, height],
      margin: 0,
      compress: true,
      info: {
        Title: metadata.title,
        Subject: metadata.description,
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

export async function writeFigure(spec, outputBase) {
  const figureId = path.basename(outputBase).replaceAll("_", "-");
  const metadata = FIGURE_METADATA[path.basename(outputBase)];
  if (!metadata) throw new Error(`missing figure metadata for ${outputBase}`);
  let rendered = await renderVegaLite(spec);
  if (path.basename(outputBase) === "deletion_frontier") {
    rendered = rendered.replace(
      /width="[0-9]+" height="966" viewBox="0 0 [0-9]+ 966"/,
      'width="1097" height="470" viewBox="20 300 1097 470"',
    );
  }
  const svg = accessibleSvg(rendered, figureId, metadata);
  await fs.writeFile(`${outputBase}.svg`, `${svg}\n`);
  await writePdf(svg, `${outputBase}.pdf`, metadata);
  const png = new Resvg(svg, {
    background: "white",
    fitTo: { mode: "zoom", value: 3 },
  })
    .render()
    .asPng();
  await fs.writeFile(`${outputBase}.png`, normalizePngSrgb(png));
}

export async function renderGemmaFigures(
  summaryPath = DEFAULT_SUMMARY,
  outputDirectory = DEFAULT_OUTPUT,
) {
  const summary = await readJson(summaryPath);
  await fs.mkdir(outputDirectory, { recursive: true });
  const utility = await buildUtilitySpec(summary);
  const deletion = await buildDeletionFrontierSpec(summary);
  await Promise.all([
    writeFigure(
      utility,
      path.join(outputDirectory, "utility_multiseed"),
    ),
    writeFigure(
      deletion,
      path.join(outputDirectory, "deletion_frontier"),
    ),
  ]);
  return {
    utility: path.join(outputDirectory, "utility_multiseed.pdf"),
    deletion: path.join(outputDirectory, "deletion_frontier.pdf"),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const summaryPath = process.argv[2]
    ? path.resolve(process.argv[2])
    : DEFAULT_SUMMARY;
  const outputDirectory = process.argv[3]
    ? path.resolve(process.argv[3])
    : DEFAULT_OUTPUT;
  const written = await renderGemmaFigures(summaryPath, outputDirectory);
  console.log(`wrote ${written.utility}`);
  console.log(`wrote ${written.deletion}`);
}
