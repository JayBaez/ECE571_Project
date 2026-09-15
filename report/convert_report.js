// convert_report.js — converts report/FINAL_REPORT.md to report/FINAL_REPORT.docx
// Parses markdown line-by-line: headings, paragraphs (with **bold**/*italic*),
// tables, and ![](path) images with a following "**Figure N.** caption" paragraph.
// Built as a script (not manual retyping) specifically so every number in the
// verified markdown report carries over to the docx without transcription risk.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const REPORT_DIR = __dirname;
const PROJECT_ROOT = path.resolve(REPORT_DIR, "..");
const md = fs.readFileSync(path.join(REPORT_DIR, "FINAL_REPORT.md"), "utf8");
const lines = md.split("\n");

// ---- inline markdown (**bold**, *italic*) -> TextRun[] ----
function parseInline(text, forceBold = false) {
  const runs = [];
  const re = /(\*\*.+?\*\*|\*[^*]+?\*)/g;
  let lastIndex = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > lastIndex) runs.push(new TextRun({ text: text.slice(lastIndex, m.index), bold: forceBold }));
    const token = m[0];
    if (token.startsWith("**")) {
      runs.push(new TextRun({ text: token.slice(2, -2), bold: true }));
    } else {
      runs.push(new TextRun({ text: token.slice(1, -1), italics: true, bold: forceBold }));
    }
    lastIndex = re.lastIndex;
  }
  if (lastIndex < text.length) runs.push(new TextRun({ text: text.slice(lastIndex), bold: forceBold }));
  return runs.length ? runs : [new TextRun({ text, bold: forceBold })];
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    margins: { top: 20, bottom: 20, left: 80, right: 80 },
    children: [new Paragraph({ children: parseInline(text, !!opts.header), spacing: { after: 0, line: 240, lineRule: "auto" } })],
  });
}

function buildTable(rows) {
  const nCols = rows[0].length;
  const colWidth = Math.floor(9000 / nCols);
  const columnWidths = new Array(nCols).fill(colWidth);
  const tableRows = rows.map((row, i) =>
    new TableRow({
      children: row.map((c) => cell(c, { header: i === 0, width: colWidth })),
    })
  );
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths, rows: tableRows });
}

const children = [];
let i = 0;

// Title (first line, starts with "# ")
if (lines[0].startsWith("# ")) {
  children.push(new Paragraph({
    children: [new TextRun({ text: lines[0].slice(2), bold: true, size: 40 })],
    heading: HeadingLevel.TITLE,
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  }));
  i = 1;
}

while (i < lines.length) {
  const line = lines[i];

  if (line.trim() === "" ) { i++; continue; }
  if (line.trim() === "---") { i++; continue; }

  // Subtitle italic line right after title, e.g. "*ECE571 Machine Learning Course Project*"
  if (i <= 3 && line.startsWith("*") && line.endsWith("*") && !line.startsWith("**")) {
    children.push(new Paragraph({
      children: [new TextRun({ text: line.slice(1, -1), italics: true, size: 24 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }));
    i++; continue;
  }

  // Headings
  const h2 = line.match(/^## (.+)/);
  const h3 = line.match(/^### (.+)/);
  if (h2) {
    children.push(new Paragraph({ text: h2[1], heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 100 } }));
    i++; continue;
  }
  if (h3) {
    children.push(new Paragraph({ text: h3[1], heading: HeadingLevel.HEADING_2, spacing: { before: 140, after: 60 } }));
    i++; continue;
  }

  // Images: ![Figure N](path)
  const imgMatch = line.match(/^!\[.*?\]\((.+?)\)/);
  if (imgMatch) {
    const relPath = imgMatch[1].replace(/^\.\.\//, "");
    const imgPath = path.join(PROJECT_ROOT, relPath);
    const imgBuffer = fs.readFileSync(imgPath);
    children.push(new Paragraph({
      children: [new ImageRun({ type: "png", data: imgBuffer, transformation: { width: 340, height: 243 } })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 60 },
    }));
    i++;
    // Next non-empty line is expected to be the bold "**Figure N.** caption" paragraph - handled by normal paragraph logic below
    continue;
  }

  // Tables: consecutive lines starting with "|"
  if (line.trim().startsWith("|")) {
    const tableLines = [];
    while (i < lines.length && lines[i].trim().startsWith("|")) {
      tableLines.push(lines[i]);
      i++;
    }
    // Remove the "|---|---|" separator row (row index 1)
    const rows = tableLines
      .filter((l, idx) => idx !== 1)
      .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
    children.push(buildTable(rows));
    children.push(new Paragraph({ text: "", spacing: { after: 150 } }));
    continue;
  }

  // Numbered list items (Introduction's 5 paradigms, References section): "1. text"
  const numMatch = line.match(/^(\d+)\.\s(.+)/);
  if (numMatch) {
    let refBuffer = [numMatch[2]];
    i++;
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].match(/^##|^###|^!\[|^\|/) && !lines[i].match(/^\d+\.\s/)) {
      refBuffer.push(lines[i].trim());
      i++;
    }
    const refText = refBuffer.join(" ");
    children.push(new Paragraph({
      children: [new TextRun({ text: `${numMatch[1]}. `, bold: false }), ...parseInline(refText)],
      spacing: { after: 80, line: 252, lineRule: "auto" },
      indent: { left: 360, hanging: 360 },
    }));
    continue;
  }

  // Plain paragraph text: accumulate consecutive non-blank, non-special
  // lines (the markdown source hard-wraps prose at ~70 chars, so a
  // single paragraph spans several source lines) until a blank line.
  let buffer = [line];
  i++;
  while (i < lines.length && lines[i].trim() !== "" && !lines[i].match(/^##|^###|^!\[|^\|/) && !lines[i].match(/^\d+\.\s/)) {
    buffer.push(lines[i].trim());
    i++;
  }
  const paragraphText = buffer.join(" ");
  children.push(new Paragraph({ children: parseInline(paragraphText), spacing: { after: 100, line: 252, lineRule: "auto" }, alignment: AlignmentType.JUSTIFIED }));
}

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
      },
    },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 } }, // 10.5pt
    },
  },
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(REPORT_DIR, "FINAL_REPORT.docx"), buffer);
  console.log("Wrote FINAL_REPORT.docx");
});
