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
const DEFAULT_OUTPUT_BASE = path.join(
  HERE,
  "previews",
  "clinical_chart_metaphor",
);

const COLORS = {
  ink: "#171717",
  muted: "#62676D",
  line: "#C8CFD5",
  paper: "#FCFCFB",
  target: "#A84B42",
  targetFill: "#F8ECEA",
  retain: "#2F805D",
  retainFill: "#EDF7F1",
  recompute: "#3878B8",
  recomputeFill: "#EDF4FA",
  reference: "#725A9D",
  referenceFill: "#F4F0FA",
  white: "#FFFFFF",
};


function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}


function text(value, x, y, options = {}) {
  const {
    size = 21,
    color = COLORS.ink,
    weight = 400,
    anchor = "start",
    family = "Liberation Sans, Arial, sans-serif",
  } = options;
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}">${escapeXml(value)}</text>`;
}


function rect(x, y, width, height, options = {}) {
  const {
    fill = COLORS.white,
    stroke = COLORS.line,
    strokeWidth = 1.4,
    radius = 8,
    dash = "",
  } = options;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
}


function arrow(x1, y1, x2, y2, options = {}) {
  const {
    color = COLORS.ink,
    width = 2.3,
    dash = "",
    double = false,
  } = options;
  const head = 10;
  const pieces = [
    `<line x1="${x1 + (double ? head : 0)}" y1="${y1}" x2="${x2 - head}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`,
    `<path d="M${x2} ${y2} l-${head} -6 v12 z" fill="${color}"/>`,
  ];
  if (double) {
    pieces.push(
      `<path d="M${x1} ${y1} l${head} -6 v12 z" fill="${color}"/>`,
    );
  }
  return pieces.join("");
}


const NOTE_ROWS = {
  before: [
    ["S", "Incorrect fact stored", "target"],
    ["O", "Later correction retained", "neutral"],
    ["A", "Assessment affected", "target"],
    ["P", "Plan affected", "target"],
  ],
  prompt: [
    ["S", "Fact hidden from answer", "hidden"],
    ["O", "Correction unchanged", "neutral"],
    ["A", "Assessment still affected", "target"],
    ["P", "Plan still affected", "target"],
  ],
  edited: [
    ["S", "Incorrect fact removed", "retain"],
    ["O", "Correction retained", "retain"],
    ["A", "Assessment recomputed", "recompute"],
    ["P", "Plan recomputed", "recompute"],
  ],
  reference: [
    ["S", "Incorrect fact never stored", "retain"],
    ["O", "Correction retained", "retain"],
    ["A", "Assessment recomputed", "recompute"],
    ["P", "Plan recomputed", "recompute"],
  ],
};


function rowPalette(kind) {
  if (kind === "target") return [COLORS.targetFill, COLORS.target];
  if (kind === "retain") return [COLORS.retainFill, COLORS.retain];
  if (kind === "recompute") return [COLORS.recomputeFill, COLORS.recompute];
  if (kind === "hidden") return ["#F3F3F2", COLORS.muted];
  return [COLORS.white, COLORS.muted];
}


function soapNote(x, y, variant, heading) {
  const width = 420;
  const height = 282;
  const pieces = [
    rect(x, y, width, height, {
      fill: COLORS.paper,
      stroke:
        variant === "reference" ? COLORS.reference : COLORS.muted,
      strokeWidth: 1.8,
      radius: 11,
    }),
    `<rect x="${x}" y="${y}" width="${width}" height="42" rx="11" fill="${variant === "reference" ? COLORS.referenceFill : "#EFF1F2"}"/>`,
    `<rect x="${x}" y="${y + 31}" width="${width}" height="11" fill="${variant === "reference" ? COLORS.referenceFill : "#EFF1F2"}"/>`,
    text(heading, x + 16, y + 27, {
      size: 22,
      color: variant === "reference" ? COLORS.reference : COLORS.ink,
      weight: 700,
    }),
    text("Patient: Mira Voss", x + 16, y + 62, {
      size: 21,
      color: COLORS.muted,
    }),
    text("Allergy: latex", x + width - 16, y + 62, {
      size: 21,
      color: COLORS.retain,
      weight: 700,
      anchor: "end",
    }),
    `<line x1="${x + 14}" y1="${y + 74}" x2="${x + width - 14}" y2="${y + 74}" stroke="${COLORS.line}" stroke-width="1"/>`,
  ];

  NOTE_ROWS[variant].forEach(([section, value, kind], index) => {
    const rowY = y + 81 + index * 47;
    const [fill, color] = rowPalette(kind);
    pieces.push(
      `<rect x="${x + 14}" y="${rowY}" width="${width - 28}" height="39" rx="5" fill="${fill}"/>`,
      text(section, x + 29, rowY + 26, {
        size: 22,
        color,
        weight: 700,
      }),
      text(value, x + 62, rowY + 25, {
        size: 21,
        color,
        weight: kind === "neutral" ? 400 : 600,
      }),
    );
    if (kind === "hidden") {
      pieces.push(
        `<line x1="${x + 64}" y1="${rowY + 20}" x2="${x + width - 27}" y2="${rowY + 20}" stroke="${COLORS.muted}" stroke-width="5" stroke-linecap="round"/>`,
      );
    }
  });

  return pieces.join("");
}


export function renderClinicalChartMetaphorSvg() {
  const width = 1000;
  const height = 760;
  const pieces = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="clinical-title" aria-describedby="clinical-desc">`,
    `<title id="clinical-title">Fictional SOAP-note analogy for bounded memory checks</title>`,
    `<desc id="clinical-desc">Four fictional notes distinguish prompt-only hiding from editing active memory and comparing it with an independently built note. The analogy maps a note field to the selected memory unit, the active note to the edited state surface, and the note comparison to the named reference check. It is not measured clinical or deletion evidence.</desc>`,
    `<rect width="${width}" height="${height}" fill="${COLORS.white}"/>`,
    text("FICTIONAL ANALOGY — NOT MEASURED EVIDENCE", 500, 22, {
      size: 22,
      color: COLORS.muted,
      weight: 700,
      anchor: "middle",
    }),

    soapNote(20, 34, "before", "RECORD PRESENT"),
    soapNote(560, 34, "prompt", "PROMPT SUPPRESSION"),
    soapNote(20, 372, "edited", "MEMORY EDIT / REPLAY"),
    soapNote(560, 372, "reference", "COMPARISON REFERENCE"),

    arrow(444, 170, 556, 170, {
      color: COLORS.target,
      dash: "6 5",
    }),
    text("HIDE", 500, 145, {
      size: 21,
      color: COLORS.target,
      weight: 700,
      anchor: "middle",
    }),

    arrow(230, 318, 230, 368, {
      color: COLORS.recompute,
    }),
    text("REMOVE CARD", 255, 337, {
      size: 21,
      color: COLORS.retain,
      weight: 700,
    }),
    text("OR REWIND + REPLAY", 255, 357, {
      size: 21,
      color: COLORS.recompute,
      weight: 700,
    }),

    arrow(444, 514, 556, 514, {
      color: COLORS.reference,
      double: true,
    }),
    text("COMPARE", 500, 489, {
      size: 21,
      color: COLORS.reference,
      weight: 700,
      anchor: "middle",
    }),
    rect(30, 707, 940, 38, {
      fill: "#EFF1F2",
      stroke: COLORS.line,
      radius: 19,
    }),
    text(
      "OUTSIDE CHECK: prior outputs · actions · treatments · external logs",
      500,
      731,
      {
        size: 21,
        color: COLORS.muted,
        weight: 700,
        anchor: "middle",
      },
    ),
    "</svg>",
  ];
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
        Title: "Fictional SOAP-note metaphor for memory deletion",
        Author: "Anonymous research artifact",
        Subject: "ELI5 schematic; not clinical validation",
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


export async function writeClinicalChartMetaphorFigure(outputBase) {
  const svg = renderClinicalChartMetaphorSvg();
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


if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const outputBase = process.argv[2]
    ? path.resolve(process.argv[2])
    : DEFAULT_OUTPUT_BASE;
  const written = await writeClinicalChartMetaphorFigure(outputBase);
  console.log(`wrote ${written.svg}`);
  console.log(`wrote ${written.pdf}`);
  console.log(`wrote ${written.png}`);
}
