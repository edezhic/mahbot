// Document kit — the one script the `document` tool runs on the managed bun
// runtime. It is deliberately a closed set of operations over data the caller
// passes: no expression is evaluated, no code is accepted, and every path in a
// request was chosen by the caller (the Rust tool).
import fs from "node:fs";
import nodePath from "node:path";
import PizZip from "pizzip";
import Docxtemplater from "docxtemplater";
import PptxGenJS from "pptxgenjs";
import {
  Document as DocxDocument, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, AlignmentType, NumberFormat,
} from "docx";
import { PDFDocument, rgb, degrees } from "pdf-lib";
import fontkit from "@pdf-lib/fontkit";
// rules.json is the single statement of the input rules both the `document`
// tool and this kit enforce — a set or a bound is written there once and read
// from both sides; bun inlines it into the committed bundle.
import RULES from "./rules.json";

const A4 = [595.28, 841.89];
const MARGIN = 56;
const BODY_SIZE = 12;
// A table's cells: ten-point text on a twelve-point leading, and a gap under
// the row that makes a one-line row twenty points tall.
const CELL_SIZE = 10;
const CELL_LEADING = 12;
const CELL_ROW_GAP = 8;
// A content block states its image size in pixels (the unit the docx and pptx
// arms take, and what pdf-lib reports for an embedded image), while a PDF page
// is laid out in points.
const PX_TO_POINTS = 72 / 96;
const DOCX_ORDERED = "document-kit-ordered";

const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
function writeOut(p, bytes) {
  fs.mkdirSync(nodePath.dirname(p), { recursive: true });
  fs.writeFileSync(p, bytes);
}
// A file the kit cannot read at all — gone, or locked by another process, in the
// window between the tool's own check and this read — is the caller's own input
// rather than a kit fault: the tool validates that an input exists, not that it
// stays readable across spawning this runtime.
const readInput = (p) => {
  try {
    return fs.readFileSync(p);
  } catch {
    throw new UsageError(`cannot read ${nodePath.basename(p)} — the file is not readable`);
  }
};

// A rejection of the REQUEST's own data — a page the document does not have, a
// field it does not declare, a colour that is not a hex one, a file that is not
// readable. The caller can fix it, so the result carries `class: "usage"` and
// the tool reports it as a usage error; any other failure is the kit's own and
// is reported as a product fault.
// The tool checks every one of those shapes at its own boundary and refuses
// before the kit is spawned, so these are the last line rather than the only
// one — and what the two sides share is stated once, in rules.json.
class UsageError extends Error {}

// ── images ─────────────────────────────────────────────────────
// Only the extensions the shared rules name are accepted: all three writers
// embed PNG and JPEG and nothing else, and one rule is what the tool's prompt
// can state honestly.
function imageKind(file) {
  const lower = file.toLowerCase();
  const extension = RULES.image_extensions.find((candidate) => lower.endsWith(`.${candidate}`));
  if (!extension) throw new UsageError(`only PNG and JPEG images can be embedded, got: ${nodePath.basename(file)}`);
  return extension === "png" ? "png" : "jpg";
}
// The shapes the tool's own boundary guarantees, as the kit's last line: a text
// field is a string and a bullet or a cell is a scalar. A structure reaching a
// writer is either thrown as a fault of the kit's own or, worse, silently
// written as "[object Object]" — and either way the request is what is wrong.
function textOf(value, what) {
  if (typeof value !== "string") throw new UsageError(`${what} must be text`);
  return value;
}
function scalarOf(value, what) {
  if (RULES.scalar_kinds.includes(typeof value)) return value;
  throw new UsageError(`${what} must be text, a number or a boolean`);
}
// `1, 2 or 3` — how these messages name the values a rule allows.
function listed(values) {
  if (values.length < 2) return String(values[0] ?? "");
  return `${values.slice(0, -1).join(", ")} or ${values.at(-1)}`;
}
// One of the heading levels the writers have, refused rather than clamped: a
// silent clamp delivers a document that differs from the request. A content
// block reaches the kit as the model wrote it, so an absent level is spelled
// `undefined` or `null` here; the tool never writes a `null` into an argument it
// builds itself.
function headingLevelNumber(level) {
  const value = level === undefined || level === null ? 1 : level;
  if (!RULES.heading_levels.includes(value)) throw new UsageError(`a heading's level must be ${listed(RULES.heading_levels)}, got: ${value}`);
  return value;
}
function headingLevel(level) {
  return [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][headingLevelNumber(level) - 1];
}

// An image's bytes, checked against the kind its name claims: every writer
// embeds them as that type and two of the three do not decode what they embed,
// so a mislabelled or truncated file would become content no reader can open —
// in a document the caller receives as a success. The check walks the format's
// own structure and yields the image's own pixel size with it; a whole image of
// a flavour pdf-lib's decoder rejects is still that arm's to refuse, one step
// later.
function readImage(file) {
  const kind = imageKind(file);
  const bytes = readInput(file);
  const size = kind === "png" ? pngSize(bytes) : jpegSize(bytes);
  // A declared size of nothing is a degenerate image, not a small one: the
  // proportions every other size is scaled from would not exist.
  if (!size || size.width < 1 || size.height < 1) {
    throw new UsageError(`cannot embed ${nodePath.basename(file)} as a ${kind === "png" ? "PNG" : "JPEG"} image: the file is not a whole one`);
  }
  return { kind, bytes, width: size.width, height: size.height };
}
// A PNG: the signature, then chunks of declared length whose first is a 13-byte
// IHDR, at least one of which is image data, closed by an IEND. Bytes after the
// IEND are allowed — the readers embed the primary image and ignore them. The
// IHDR carries the pixel size, which is what a size the caller leaves out is
// scaled from.
function pngSize(bytes) {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24 || !signature.every((byte, index) => bytes[index] === byte)) return null;
  let offset = 8;
  let pixels = false;
  // Chunks are length(4) + type(4) + data(length) + crc(4).
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString("latin1", offset + 4, offset + 8);
    if (offset === 8 && (type !== "IHDR" || length !== 13)) return null;
    if (type === "IDAT") pixels = true;
    if (type === "IEND") {
      if (!pixels) return null;
      return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
    }
    offset += 12 + length;
  }
  return null;
}
// The end-of-image marker: an FFD9 in the scan is what ends the image.
const JPEG_EOI = Buffer.from([0xff, 0xd9]);
// A JPEG: SOI, marker segments of declared length, a frame header, and an EOI
// after the scan. Only the marker segments are walked — past the start-of-scan
// marker the image data is entropy-coded and not length-prefixed, and an FFD9
// inside it is the EOI, so the scan is where the walk stops looking. Trailing
// bytes are allowed: a multi-picture (MPO) or padded camera file carries them,
// and any number of FF fill bytes may precede a marker. The frame header carries
// the pixel size.
function jpegSize(bytes) {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  let offset = 2;
  let size = null;
  while (offset + 4 <= bytes.length) {
    if (bytes[offset] !== 0xff) return null;
    while (bytes[offset] === 0xff) offset += 1;
    if (offset + 4 > bytes.length) return null;
    const marker = bytes[offset];
    if (marker === 0xda) return size && bytes.indexOf(JPEG_EOI, offset) >= 0 ? size : null;
    if (marker === 0xd9) return null;
    // A restart or a temporary marker carries no payload and no length.
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 1;
      continue;
    }
    const length = bytes.readUInt16BE(offset + 1);
    if (length < 2) return null;
    // The frame headers carry the image's dimensions, and are read only once
    // their own bytes are there: a file cut right after the marker is a
    // truncated image, not a fault of the kit's. The tables and the
    // arithmetic-coding markers share the range but are not a frame.
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
      if (offset + 8 > bytes.length) return null;
      size = { width: bytes.readUInt16BE(offset + 6), height: bytes.readUInt16BE(offset + 4) };
    }
    offset += 1 + length;
  }
  return null;
}
// The size a content image is drawn at, in the unit its request states (pixels).
// A request that gives one side only scales the image's own proportions instead
// of stretching it, and one that gives neither keeps the image's own size, capped
// to a default width a page's text column comfortably holds: the same numbers for
// every arm, so one block draws one picture whatever the format. `span` is the
// range the shared rules allow a stated side (pixels for a content image, points
// for a placed PDF one) and `maxWidth` is that cap.
const DEFAULT_IMAGE_WIDTH = 400;
// Whether `value` is a finite number inside `span`, a range from rules.json
// (`min_exclusive` marks a low end the value must exceed rather than reach).
function inSpan(value, span) {
  if (typeof value !== "number" || !Number.isFinite(value)) return false;
  if (span.min_exclusive ? value <= span.min : value < span.min) return false;
  return value <= span.max;
}
// The phrase a refusal names `span` by. One helper for every arm, so a rule that
// flips `min_exclusive` cannot leave one message saying "between" and another
// "greater than" about the same range.
function spanBounds(span) {
  return span.min_exclusive
    ? `greater than ${span.min} and at most ${span.max}`
    : `between ${span.min} and ${span.max}`;
}
function imageSize(natural, width, height, span, maxWidth = DEFAULT_IMAGE_WIDTH) {
  // A side the block left out is no side at all: `null` counts as absent, since
  // a content block reaches the kit as the model wrote it.
  for (const side of [width, height]) {
    if (side === undefined || side === null) continue;
    if (!inSpan(side, span)) {
      // The refusal names the range it enforces: the shared span has an upper
      // bound, and a low end a placed image's points must exceed rather than
      // reach, so "positive" would understate both.
      throw new UsageError(`an image's width and height must be ${spanBounds(span)}, got: ${side}`);
    }
  }
  const round = (side) => Math.max(1, Math.round(side));
  if (width && height) return { width, height };
  if (width) return { width, height: round(width * natural.height / natural.width) };
  if (height) return { width: round(height * natural.width / natural.height), height };
  const fit = Math.min(1, maxWidth / natural.width);
  return { width: round(natural.width * fit), height: round(natural.height * fit) };
}
// Embed `file` into a PDF document. A file the decoder rejects (truncated, or a
// kind the magic bytes did not settle) is the request's own input.
async function embedImage(pdf, file) {
  const image = readImage(file);
  try {
    return image.kind === "png" ? await pdf.embedPng(image.bytes) : await pdf.embedJpg(image.bytes);
  } catch (error) {
    throw new UsageError(`cannot read ${nodePath.basename(file)} as an image: ${(error && error.message) || error}`);
  }
}

// ── create: docx ───────────────────────────────────────────────
function docxImage(block) {
  const image = readImage(block.path);
  return new ImageRun({
    type: image.kind,
    data: image.bytes,
    transformation: imageSize(image, block.width, block.height, RULES.image_side_px),
  });
}
function docxTableRow(cells, header) {
  const what = header ? "a table heading" : "a table cell";
  return new TableRow({
    children: cells.map((cell) => new TableCell({
      children: [new Paragraph({ children: [new TextRun({ text: String(scalarOf(cell, what)), bold: header })] })],
    })),
  });
}
function createDocx(req) {
  const children = [];
  for (const block of req.content) {
    if (block.type === "heading") {
      children.push(new Paragraph({ text: textOf(block.text, "a heading's text"), heading: headingLevel(block.level) }));
    } else if (block.type === "paragraph") {
      children.push(new Paragraph({ children: [new TextRun(textOf(block.text, "a paragraph's text"))] }));
    } else if (block.type === "list") {
      for (const item of block.items) {
        // A bullet is text as far as the writer is concerned: a number or a
        // boolean left to the library becomes an empty paragraph.
        const text = String(scalarOf(item, "a bullet"));
        children.push(new Paragraph(block.ordered
          ? { text, numbering: { reference: DOCX_ORDERED, level: 0 } }
          : { text, bullet: { level: 0 } }));
      }
    } else if (block.type === "table") {
      const rows = [];
      // A row with no cells is a `<w:tr/>`, which Word refuses to open, so an
      // empty header or an empty row is never written.
      if (block.headers && block.headers.length) rows.push(docxTableRow(block.headers, true));
      // `rows` is optional: a header alone is a table.
      for (const row of block.rows || []) {
        if (row.length) rows.push(docxTableRow(row, false));
      }
      children.push(new Table({ rows, width: { size: 100, type: WidthType.PERCENTAGE } }));
    } else if (block.type === "image") {
      children.push(new Paragraph({ children: [docxImage(block)], spacing: { after: 200 } }));
    }
  }
  const numbering = {
    config: [{
      reference: DOCX_ORDERED,
      levels: [{ level: 0, format: NumberFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START }],
    }],
  };
  return Packer.toBuffer(new DocxDocument({ numbering, sections: [{ children }] }));
}

// ── create: xlsx ───────────────────────────────────────────────
// Written by hand: the package is small, and owning the writer is what keeps a
// text cell that starts with "=" a text cell and a formula cell a formula.
const xmlEscape = (value) => String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
const colName = (index) => {
  let name = "";
  let n = index + 1;
  while (n > 0) { const rest = (n - 1) % 26; name = String.fromCharCode(65 + rest) + name; n = Math.floor((n - 1) / 26); }
  return name;
};
function xlsxCell(reference, value) {
  // A null is neither a scalar nor a formula cell: the docx/pptx/pdf arms
  // refuse one through `scalarOf`, and writing the literal "null" into a cell
  // would be wrong content rather than an empty cell.
  if (value === null) throw new UsageError(`cell ${reference} must be text, a number or a boolean`);
  // A request is JSON, so a number reaching here is finite by construction.
  if (typeof value === "number") return `<c r="${reference}"><v>${value}</v></c>`;
  if (typeof value === "boolean") return `<c r="${reference}" t="b"><v>${value ? 1 : 0}</v></c>`;
  if (value && typeof value === "object") {
    // The rule the tool's boundary states, as the kit's own last line: the
    // object holds `formula` alone, and the text — with its one optional leading
    // `=` and the spacing removed — is a formula, so a second `=` is refused
    // rather than written into the `<f>` element as text. The tool's paired
    // refusal test drives both sides, keeping the two aligned.
    const formula = typeof value.formula === "string" ? value.formula.trim().replace(/^=/, "").trim() : "";
    if (Object.keys(value).length === 1 && formula && !formula.startsWith("=")) return `<c r="${reference}"><f>${xmlEscape(formula)}</f></c>`;
    throw new UsageError(`cell ${reference}: an object value must be {"formula": "SUM(A1:A2)"}`);
  }
  const text = String(value);
  return text === "" ? "" : `<c r="${reference}" t="inlineStr"><is><t xml:space="preserve">${xmlEscape(text)}</t></is></c>`;
}
// A name written here reaches `xl/workbook.xml` verbatim, and a name ECMA-376
// refuses makes the workbook one an office application offers to repair — while
// the call reports success. The tool's boundary sends every name explicitly and
// refuses first; this is the writer's own last line, over the names it will
// actually write (a defaulted one included). Length is counted in code points,
// the unit the rule states.
function checkSheetNames(sheets) {
  const seen = new Set();
  sheets.forEach((sheet, i) => {
    const where = `sheet ${i + 1}`;
    const name = sheet.name;
    if (typeof name !== "string") {
      throw new UsageError(`${where}: a sheet name must be text, got: ${String(name)}`);
    }
    const length = [...name].length;
    if (length < 1 || length > RULES.sheet_name_max) {
      throw new UsageError(`${where}: a sheet name must be 1 to ${RULES.sheet_name_max} characters, got: ${name}`);
    }
    if ([...RULES.sheet_name_forbidden].some((character) => name.includes(character))) {
      throw new UsageError(`${where}: a sheet name must not contain any of ${JSON.stringify(RULES.sheet_name_forbidden)}, got: ${name}`);
    }
    const lower = name.toLowerCase();
    if (seen.has(lower)) throw new UsageError(`${where}: a sheet name must be unique, ignoring case, got: ${name}`);
    seen.add(lower);
  });
}
function createXlsx(req) {
  const zip = new PizZip();
  // A name reaches the writer only when it is there: an absent one defaults to
  // its position, while a name of any other kind — null or empty included — is
  // left for `checkSheetNames` to refuse rather than silently defaulted over. A
  // sheet that is not an object is refused here, before a property of it is read.
  const sheets = req.sheets.map((sheet, i) => {
    if (typeof sheet !== "object" || sheet === null) {
      throw new UsageError(`sheet ${i + 1} must be an object, got: ${JSON.stringify(sheet)}`);
    }
    return { name: sheet.name === undefined ? `Sheet${i + 1}` : sheet.name, rows: sheet.rows || [] };
  });
  checkSheetNames(sheets);
  const contentTypes = [
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
    '<Default Extension="xml" ContentType="application/xml"/>',
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ...sheets.map((_, i) => `<Override PartName="/xl/worksheets/sheet${i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`),
    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
  ];
  zip.file("[Content_Types].xml", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">${contentTypes.join("")}</Types>`);
  zip.file("_rels/.rels", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>`);
  zip.file("xl/workbook.xml", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${sheets.map((sheet, i) => `<sheet name="${xmlEscape(sheet.name)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`).join("")}</sheets></workbook>`);
  const workbookRels = [
    ...sheets.map((_, i) => `<Relationship Id="rId${i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${i + 1}.xml"/>`),
    `<Relationship Id="rId${sheets.length + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>`,
  ];
  zip.file("xl/_rels/workbook.xml.rels", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${workbookRels.join("")}</Relationships>`);
  zip.file("xl/styles.xml", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>`);
  sheets.forEach((sheet, i) => {
    // A worksheet's cells nest inside their `<row>`; a sheet whose cells sit
    // directly under `<sheetData>` is schema-invalid and opens empty in real
    // readers (openpyxl among them), however readable it looks.
    const rows = [];
    sheet.rows.forEach((row, r) => {
      const cells = [];
      row.forEach((value, c) => {
        const cell = xlsxCell(`${colName(c)}${r + 1}`, value);
        if (cell) cells.push(cell);
      });
      if (cells.length) rows.push(`<row r="${r + 1}">${cells.join("")}</row>`);
    });
    zip.file(`xl/worksheets/sheet${i + 1}.xml`, `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>${rows.join("")}</sheetData></worksheet>`);
  });
  return zip.generate({ type: "nodebuffer", compression: "DEFLATE" });
}

// ── create: pptx ───────────────────────────────────────────────
function createPptx(req) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_16x9";
  let slide = null;
  let y = 1.3;
  const fresh = () => { slide = pptx.addSlide(); y = 1.3; };
  const room = (needed) => { if (y + needed > 5.2) fresh(); };
  for (const block of req.content) {
    if (block.type === "heading") {
      if (headingLevelNumber(block.level) === 1 || !slide) {
        fresh();
        slide.addText(textOf(block.text, "a heading's text"), { x: 0.4, y: 0.3, w: 9.2, h: 0.8, fontSize: 28, bold: true });
      } else {
        room(0.7);
        slide.addText(textOf(block.text, "a heading's text"), { x: 0.4, y, w: 9.2, h: 0.5, fontSize: 20, bold: true });
        y += 0.7;
      }
    } else if (block.type === "paragraph") {
      if (!slide) fresh();
      room(1.1);
      slide.addText(textOf(block.text, "a paragraph's text"), { x: 0.4, y, w: 9.2, h: 1, fontSize: 16 });
      y += 1.1;
    } else if (block.type === "list") {
      if (!slide) fresh();
      const bullet = block.ordered ? { type: "number" } : true;
      room(0.5 * block.items.length + 0.3);
      // A bullet is text to the writer, so a number or a boolean is stringified
      // here rather than handed to a library that draws nothing for one.
      slide.addText(block.items.map((item) => ({ text: String(scalarOf(item, "a bullet")), options: { bullet, breakLine: true } })), { x: 0.4, y, w: 9.2, h: Math.max(0.5, 0.5 * block.items.length), fontSize: 16 });
      y += 0.5 * block.items.length + 0.3;
    } else if (block.type === "table") {
      // A table row is an array of cells: a header row is one cell per heading,
      // each carrying its own bold option (one cell holding all the headings is
      // rejected by the writer).
      const header = block.headers && block.headers.length ? [block.headers.map((h) => ({ text: String(scalarOf(h, "a table heading")), options: { bold: true } }))] : [];
      const body = (block.rows || []).map((row) => row.map((cell) => ({ text: String(scalarOf(cell, "a table cell")) }))).filter((row) => row.length);
      // A cell-less row is dropped above, so a table left with nothing to draw
      // is one the tool refuses; this only keeps the writer from throwing on
      // one.
      if (!header.length && !body.length) continue;
      if (!slide) fresh();
      room(0.45 * (body.length + header.length) + 0.3);
      slide.addTable([...header, ...body], { x: 0.4, y, w: 9.2, fontSize: 14 });
      y += 0.45 * (body.length + header.length) + 0.3;
    } else if (block.type === "image") {
      if (!slide) fresh();
      const image = readImage(block.path);
      const size = imageSize(image, block.width, block.height, RULES.image_side_px);
      const w = size.width / 96;
      const h = size.height / 96;
      room(h + 0.2);
      slide.addImage({ data: `data:image/${image.kind === "png" ? "png" : "jpeg"};base64,${image.bytes.toString("base64")}`, x: 0.4, y, w, h });
      y += h + 0.2;
    } else if (block.type === "notes") {
      if (!slide) fresh();
      slide.addNotes(textOf(block.text, "speaker notes"));
    }
  }
  if (!slide) pptx.addSlide();
  return pptx.write({ outputType: "nodebuffer" });
}

// ── create: pdf ────────────────────────────────────────────────
// A streaming layout: blocks follow one another down the page and a new page
// starts when the current one is full. Columns, floats and exact page
// placement are deliberately not attempted.
//
// pdf-lib writes a character the embedded font has no glyph for as the notdef
// glyph — a blank box — so a document that draws e.g. CJK with the bundled
// NotoSans is delivered as a success that silently lost the text. Every string
// drawn with that font is offered to `noteGlyphs`, and whatever it cannot draw
// is reported in the result's `unsupported`.
const missingGlyphs = new Set();
// Code points pdf-lib's own text pipeline rewrites before the font is ever
// consulted: `cleanText` turns U+0009, U+0085, U+2028 and U+2029 into four
// spaces, and `lineSplit`/`mergeLines` turn U+000A, U+000C and U+000D into a
// line break or a space. Reporting one would name a loss that did not happen, so
// they are skipped rather than offered to the coverage check. A code point the
// pipeline DROPS instead (backspace U+0008, vertical tab U+000B) really does
// leave the file and stays reported — where the string reaches the font as it
// was written (`pdf_text`, `pdf_form_fill`); `create pdf`'s own word splitter
// turns a vertical tab into a space and reports nothing.
const LAYOUT_CODE_POINTS = new Set([0x09, 0x0a, 0x0c, 0x0d, 0x85, 0x2028, 0x2029]);
function noteGlyphs(glyphs, text) {
  // By code point, not UTF-16 unit: an astral character is one glyph.
  for (const character of text) {
    const codePoint = character.codePointAt(0);
    if (LAYOUT_CODE_POINTS.has(codePoint)) continue;
    if (!glyphs.hasGlyphForCodePoint(codePoint)) missingGlyphs.add(character);
  }
}
async function pdfWithFont(req) {
  const pdf = await PDFDocument.create();
  pdf.registerFontkit(fontkit);
  const bytes = fs.readFileSync(req.font);
  const font = await pdf.embedFont(bytes, { subset: true });
  // The same bytes, read by fontkit itself: what pdf-lib embeds is a subset, so
  // the outline table of the input is what tells an unrenderable character.
  const glyphs = fontkit.create(bytes);
  return { pdf, font, glyphs };
}

function wrapText(font, text, size, indent, maxWidth) {
  const lines = [];
  for (const paragraph of String(text).split("\n")) {
    let current = "";
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      const candidate = current ? `${current} ${word}` : word;
      if (font.widthOfTextAtSize(candidate, size) + indent <= maxWidth) { current = candidate; continue; }
      if (current) lines.push(current);
      let rest = word;
      while (font.widthOfTextAtSize(rest, size) + indent > maxWidth && rest.length > 1) {
        let cut = rest.length - 1;
        while (cut > 1 && font.widthOfTextAtSize(rest.slice(0, cut), size) + indent > maxWidth) cut -= 1;
        lines.push(rest.slice(0, cut));
        rest = rest.slice(cut);
      }
      current = rest;
    }
    lines.push(current);
  }
  return lines;
}

async function createPdf(req) {
  const { pdf, font, glyphs } = await pdfWithFont(req);
  const width = A4[0] - 2 * MARGIN;
  let page = pdf.addPage(A4);
  let y = A4[1] - MARGIN;
  const fresh = () => { page = pdf.addPage(A4); y = A4[1] - MARGIN; };
  const draw = (text, size, leading, indent) => {
    if (y - leading < MARGIN) fresh();
    noteGlyphs(glyphs, text);
    page.drawText(text, { x: MARGIN + indent, y, size, font, color: rgb(0, 0, 0) });
    y -= leading;
  };
  for (const block of req.content) {
    if (block.type === "heading") {
      const size = [22, 17, 14][headingLevelNumber(block.level) - 1];
      y -= size * 0.4;
      for (const line of wrapText(font, textOf(block.text, "a heading's text"), size, 0, width)) draw(line, size, size * 1.25, 0);
      y -= size * 0.2;
    } else if (block.type === "paragraph") {
      for (const line of wrapText(font, textOf(block.text, "a paragraph's text"), BODY_SIZE, 0, width)) draw(line, BODY_SIZE, BODY_SIZE * 1.35, 0);
      y -= BODY_SIZE * 0.4;
    } else if (block.type === "list") {
      block.items.forEach((item, index) => {
        const marker = block.ordered ? `${index + 1}. ` : "• ";
        for (const line of wrapText(font, marker + String(scalarOf(item, "a bullet")), BODY_SIZE, 12, width)) draw(line, BODY_SIZE, BODY_SIZE * 1.35, 12);
      });
      y -= BODY_SIZE * 0.4;
    } else if (block.type === "table") {
      // A cell-less row draws nothing but its rule, so it is dropped rather
      // than left as a blank band.
      const rows = [...(block.headers ? [block.headers] : []), ...(block.rows || [])].filter((row) => row.length);
      const cols = Math.max(1, ...rows.map((row) => row.length));
      const colWidth = width / cols;
      for (const row of rows) {
        // Every wrapped line of a cell is drawn, and the row is broken across
        // pages line by line as a paragraph is: a row taller than the text
        // column would otherwise draw past the page bottom, where a viewer clips
        // it while the call still reported success.
        const cells = [];
        for (let c = 0; c < cols; c += 1) {
          const cell = row[c] === undefined ? "" : String(scalarOf(row[c], "a table cell"));
          cells.push(wrapText(font, cell, CELL_SIZE, 3, colWidth));
        }
        const lines = Math.max(...cells.map((cell) => cell.length));
        if (y - 13 < MARGIN) fresh();
        page.drawLine({ start: { x: MARGIN, y }, end: { x: MARGIN + width, y }, thickness: 0.5, color: rgb(0.6, 0.6, 0.6) });
        for (let i = 0; i < lines; i += 1) {
          if (y - 13 < MARGIN) fresh();
          cells.forEach((cell, c) => {
            const line = cell[i];
            if (line === undefined) return;
            noteGlyphs(glyphs, line);
            page.drawText(line, { x: MARGIN + c * colWidth + 3, y: y - 13, size: CELL_SIZE, font });
          });
          y -= CELL_LEADING;
        }
        y -= CELL_ROW_GAP;
      }
      y -= BODY_SIZE;
    } else if (block.type === "image") {
      const image = await embedImage(pdf, block.path);
      const size = imageSize(image, block.width, block.height, RULES.image_side_px);
      let targetWidth = size.width * PX_TO_POINTS;
      let targetHeight = size.height * PX_TO_POINTS;
      // The streaming layout keeps what it draws inside the text column.
      if (targetWidth > width) {
        targetHeight *= width / targetWidth;
        targetWidth = width;
      }
      if (y - targetHeight < MARGIN) fresh();
      page.drawImage(image, { x: MARGIN, y: y - targetHeight, width: targetWidth, height: targetHeight });
      y -= targetHeight + BODY_SIZE;
    }
  }
  return pdf.save();
}

// ── fill_template ──────────────────────────────────────────────
// Substitution runs over JOINED text runs, never over a run on its own: the
// editor that wrote the sample may have split `{name}` across runs, and only the
// joined text is the one placeholder the user sees.
const PLACEHOLDER = /\{([^{}]+)\}/g;
// The run element of a part, capturing its text. `prefix` is the run element's
// name up to the tag itself: `a:` in a drawing part, empty in a workbook part.
const runPattern = (prefix) => new RegExp(`<${prefix}t(?:\\s[^>]*)?>([\\s\\S]*?)</${prefix}t>`, "g");
// The workbook's shared-string table and one `<si>` of it (empty or not).
const SHARED_STRINGS_PART = "xl/sharedStrings.xml";
const SHARED_STRING = "<si(?:\\s[^>]*)?>[\\s\\S]*?</si>|<si(?:\\s[^>]*)?/>";
// One `<a:p>` paragraph of a drawing part.
const DRAWING_PARAGRAPH = "<a:p(?:\\s[^>]*)?>[\\s\\S]*?</a:p>";

// Replace `text`'s placeholders with their values, recording every name the
// sample declared (`seen`) and every one no value was given for (`missing`). The
// lookup is own-property: a placeholder named after an Object.prototype member
// (`{__proto__}`, `{toString}`) is one the caller gave no value for, not the
// prototype's own member rendered into the document.
function substituteText(text, values, seen, missing) {
  return text.replace(PLACEHOLDER, (whole, name) => {
    seen.add(name);
    const value = Object.hasOwn(values, name) ? values[name] : undefined;
    if (value === undefined) { missing.add(name); return whole; }
    return String(value);
  });
}

// XML text with its entity references resolved, so a placeholder is matched
// against the text a reader sees rather than against the escaping the writer
// chose — openpyxl writes every Cyrillic letter as a numeric reference, and
// re-escaping one would turn it into the literal `&#1054;` a user then reads.
// The digits are matched as digits and only as digits: a looser pattern reads
// `&#123abc;` as the number 123 followed by junk and injects a brace.
const xmlUnescape = (text) => text.replace(/&(#[0-9]+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/g, (whole, body) => {
  if (body[0] !== "#") return { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" }[body] ?? whole;
  const code = body[1] === "x" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
  return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
});

// The joined text of `element`'s runs, unescaped, or `null` when it has none or
// none of them holds a placeholder.
function placeholderText(element, runs) {
  const text = runText(element, runs);
  // A literal of its own: `PLACEHOLDER` carries the `g` flag `replace` needs.
  return /\{[^{}]+\}/.test(text) ? text : null;
}

// The joined, unescaped text of every run of `element` matching `runs`.
function runText(element, runs) {
  return xmlUnescape([...element.matchAll(runs)].map((match) => match[1]).join(""));
}

// `element` with its first run holding `text` and every further run emptied;
// the formatting around those runs (their run properties) is kept.
function rewriteRuns(element, runs, prefix, text) {
  let first = true;
  return element.replace(runs, () => {
    const content = first ? text : "";
    first = false;
    return `<${prefix}t xml:space="preserve">${xmlEscape(content)}</${prefix}t>`;
  });
}

// Substitute the placeholders of every `container` of `xml`, one at a time: the
// container's runs are joined, substituted once, and written back into its
// first run.
function fillTextContainers(xml, container, prefix, values, seen, missing) {
  const runs = runPattern(prefix);
  return xml.replace(new RegExp(container, "g"), (element) => {
    const text = placeholderText(element, runs);
    if (text === null) return element;
    return rewriteRuns(element, runs, prefix, substituteText(text, values, seen, missing));
  });
}

// Open a sample package, blaming the REQUEST when it cannot be opened at all —
// an encrypted or truncated file is a bad input the caller can do something
// about, not a kit fault.
function openPackage(file, kind) {
  try {
    return new PizZip(fs.readFileSync(file));
  } catch (error) {
    throw new UsageError(`cannot open ${nodePath.basename(file)} as a ${kind} package: ${(error && error.message) || error}`);
  }
}

// The library's own resolution of a tag (a plain lookup in the current scope),
// with every tag it compiles recorded: what a sample declared is then a fact the
// kit reports instead of one the caller re-derives from the output bytes.
function recordingParser(tags) {
  return (tag) => {
    tags.add(tag);
    return {
      get(scope) {
        if (tag === ".") return scope;
        return scope && Object.hasOwn(scope, tag) ? scope[tag] : undefined;
      },
    };
  };
}

// A substitute is text, a number or a boolean: a structure has no place in a
// document, and writing one out would put `[object Object]` into it, so it is
// refused once here — for the name-keyed maps both `fill_template` and
// `pdf_form_fill` take.
function checkValues(values) {
  for (const [name, value] of Object.entries(values)) {
    if (!RULES.scalar_kinds.includes(typeof value)) {
      throw new UsageError(`the value for "${name}" must be text, a number or a boolean, not a structure`);
    }
  }
}

// The template library reports a compile or render failure as a `Multi error`
// wrapping one sub-error per problem, each naming the offending tag, the part
// and the surrounding text: the sub-errors are what tells the caller how to fix
// their own sample, while the wrapper's own message says nothing. Such a failure
// is the sample's, so it is a usage error.
function templateError(error) {
  const details = ((error && error.properties && error.properties.errors) || []).map((sub) => {
    const where = (sub && sub.properties) || {};
    return [where.explanation || sub.message, where.context && `…${where.context}…`, where.file].filter(Boolean).join(" ");
  });
  const reason = details.length ? details.join("; ") : String((error && error.message) || error);
  return new UsageError(`the sample cannot be filled: ${reason}`);
}

// The family a package's own PARTS declare, whatever the name says: the three
// OOXML families each own a part the other two do not have. `null` when the
// archive declares none of them — a plain zip, or a package built on another
// convention, which the reader that does know it can still accept.
function packageFamily(zip) {
  if (zip.file("word/document.xml")) return "docx";
  if (zip.file("xl/workbook.xml")) return "xlsx";
  if (zip.file("ppt/presentation.xml")) return "pptx";
  return null;
}

// A package whose parts are another family's cannot be filled as this one: the
// mismatch is named here rather than left to the library, whose message for it
// talks about its own paid modules instead of the file.
function ensurePackageFamily(zip, expected) {
  const family = packageFamily(zip);
  if (family && family !== expected) {
    throw new UsageError(`the sample is really a ${family} package, not the ${expected} one its name says — hint: rename it to match its content`);
  }
}

function fillDocxOrPptx(req) {
  const missing = new Set();
  const tags = new Set();
  // The request's `format` is the sample's family; without one it is a
  // document, which the library reads on its own.
  const expected = req.format === "pptx" ? "pptx" : "docx";
  const zip = openPackage(req.template, expected);
  ensurePackageFamily(zip, expected);
  let doc;
  try {
    doc = new Docxtemplater(zip, {
      paragraphLoop: true,
      linebreaks: true,
      // The library logs a failure as JSON on stdout by default, which would
      // bury the kit's own result channel in noise.
      errorLogging: false,
      parser: recordingParser(tags),
      nullGetter: (part) => { missing.add(part.value); return `{${part.value}}`; },
    });
    doc.render(req.values);
  } catch (error) {
    // A sample the library cannot compile — an unbalanced `{`, a file that is
    // not the package its name claims — is the caller's own input.
    throw templateError(error);
  }
  const out = doc.getZip();
  // Presenter notes are not a part docxtemplater renders, so their placeholders
  // are substituted here instead of being left in the delivered file.
  for (const name of Object.keys(out.files)) {
    if (!/^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(name)) continue;
    const xml = out.file(name).asText();
    const filled = fillTextContainers(xml, DRAWING_PARAGRAPH, "a:", req.values, tags, missing);
    if (filled !== xml) out.file(name, filled);
  }
  return { buffer: out.generate({ type: "nodebuffer", compression: "DEFLATE" }), missing: [...missing], placeholders: tags.size };
}

// The shared string a `t="s"` cell indexes into, joined and unescaped, or `null`
// when the cell is not a shared-string cell (or the table does not have that
// index). A workbook written by Excel, Sheets or LibreOffice keeps EVERY string
// here and writes only an index into the cell, so this is the text such a cell
// displays.
function sharedStringOf(cell, shared) {
  if (!/\bt="s"/.test(cell)) return null;
  const index = Number((cell.match(/<v>\s*(\d+)\s*<\/v>/) || [])[1]);
  return shared[index] ?? null;
}

// The joined, unescaped text of every `<si>` of the shared-string table, in
// index order: what a `t="s"` cell's index refers to (`""` for an `<si/>` with
// no runs, which still occupies its index).
function sharedStringTexts(xml) {
  return [...xml.matchAll(new RegExp(SHARED_STRING, "g"))].map((match) => runText(match[0], runPattern("")));
}

// The zero-based column a cell reference's letters name (`"B12"` -> 1).
const columnOf = (reference) => [...reference.replace(/\d+/g, "")].reduce((index, letter) => index * 26 + letter.toUpperCase().charCodeAt(0) - 64, 0) - 1;

// One cell with its placeholders substituted, or the cell unchanged.
function fillCell(cell, reference, values, seen, missing, shared) {
  const own = placeholderText(cell, runPattern(""));
  const text = own ?? sharedStringOf(cell, shared);
  if (text === null) return cell;
  const style = (cell.match(/\bs="(\d+)"/) || [])[1];
  const attribute = style ? ` s="${style}"` : "";
  const inline = (content) => `<c r="${reference}"${attribute} t="inlineStr"><is><t xml:space="preserve">${xmlEscape(content)}</t></is></c>`;
  const whole = text.match(/^\{([^{}]+)\}$/);
  // A cell whose text lives in the shared table is only rewritten when the
  // whole cell is one placeholder (below); a partial one stays a shared-string
  // cell and is substituted in the table by its own pass.
  if (!whole) return own === null ? cell : inline(substituteText(text, values, seen, missing));
  seen.add(whole[1]);
  const value = Object.hasOwn(values, whole[1]) ? values[whole[1]] : undefined;
  if (value === undefined) { missing.add(whole[1]); return cell; }
  // A number in a cell that is exactly one placeholder becomes a real number
  // cell, whatever shape the sample kept its strings in.
  if (typeof value === "number") return `<c r="${reference}"${attribute}><v>${value}</v></c>`;
  return own === null ? cell : inline(String(value));
}

// A table sample is edited cell by cell: everything else in the package —
// charts, comments, media, other sheets — is copied through untouched, and a
// whole-cell placeholder becomes a number when the value is one. A cell is
// addressed by its own `r` or, when a writer left it out, by its position in the
// row — the rule the reader uses, so a sample whose cells the reader can show is
// one this filler can fill.
function fillXlsxSheet(xml, values, seen, missing, shared) {
  let rowNumber = 0;
  let column = 0;
  return xml.replace(/<row\b[^>]*\/>|<row\b[^>]*>|<c\b[^>]*\/>|<c\b[^>]*>[\s\S]*?<\/c>/g, (part) => {
    if (part.startsWith("<row")) {
      rowNumber = Number((part.match(/\br="(\d+)"/) || [])[1]) || rowNumber + 1;
      column = 0;
      return part;
    }
    const address = (part.match(/\br="([A-Z]+\d+)"/) || [])[1];
    const reference = address || `${colName(column)}${rowNumber}`;
    column = address ? columnOf(address) + 1 : column + 1;
    return fillCell(part, reference, values, seen, missing, shared);
  });
}

function fillXlsx(req) {
  const missing = new Set();
  const seen = new Set();
  const zip = openPackage(req.template, "xlsx");
  ensurePackageFamily(zip, "xlsx");
  const rewrite = (name, fill) => {
    const part = zip.file(name);
    if (!part) return;
    const xml = part.asText();
    const filled = fill(xml);
    if (filled !== xml) zip.file(name, filled);
  };
  const sharedPart = zip.file(SHARED_STRINGS_PART);
  const shared = sharedPart ? sharedStringTexts(sharedPart.asText()) : [];
  for (const name of Object.keys(zip.files)) {
    if (/^xl\/worksheets\/sheet\d+\.xml$/.test(name)) rewrite(name, (xml) => fillXlsxSheet(xml, req.values, seen, missing, shared));
  }
  // A real Excel file keeps its strings in the shared table and writes only an
  // index into the cell, so a placeholder there is invisible to the cell pass.
  rewrite(SHARED_STRINGS_PART, (xml) => fillTextContainers(xml, SHARED_STRING, "", req.values, seen, missing));
  return { buffer: zip.generate({ type: "nodebuffer", compression: "DEFLATE" }), missing: [...missing], placeholders: seen.size };
}

// ── pdf page operations ────────────────────────────────────────
// A PDF the library cannot parse is the request's own input, not a kit fault.
const loadPdf = async (p) => {
  const bytes = readInput(p);
  let pdf;
  try {
    // `ignoreEncryption` is what makes an encrypted document load far enough to
    // be NAMED as one: the library refuses it before this side can see why, and
    // a cause the caller recognises is worth more than a library message.
    pdf = await PDFDocument.load(bytes, { ignoreEncryption: true });
    // The page tree must resolve too: a header-valid file whose body the
    // library cannot read would otherwise fail deep inside an operation, with a
    // raw library message. Touching it here refuses such a file in this same
    // place, as the request's own input.
    pdf.getPageIndices();
  } catch {
    // The library's own text for a body it cannot read ("undefined is not an
    // object (evaluating '_this.catalog.Pages')") names nothing the caller can
    // act on, so the refusal states only what is known here.
    throw new UsageError(`cannot read ${nodePath.basename(p)} as a PDF — its bytes do not form a PDF this tool can read`);
  }
  if (pdf.isEncrypted) {
    throw new UsageError(`${nodePath.basename(p)} is password-protected — the tool cannot open an encrypted PDF`);
  }
  return pdf;
};
// The zero-based page indices `pages` names, rejecting one the document does
// not have: the library's own message for that names neither the page nor the
// document's length.
function pageIndices(pdf, pages) {
  const indices = !pages || pages === "all" ? pdf.getPageIndices() : pages.map((n) => n - 1);
  const count = pdf.getPageCount();
  for (const index of indices) {
    if (index < 0 || index >= count) throw new UsageError(`no page ${index + 1}: this document has ${count} page(s)`);
  }
  return indices;
}
// A PDF position or angle a page operation may state: a finite number whose
// magnitude no page can carry (a page is laid out in points). One helper, so
// the coordinates and the rotation are refused the same way rather than by a
// pdf-lib fault further in.
function pointNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > RULES.pdf_point_abs_max) {
    throw new UsageError(`"${name}" must be within ±${RULES.pdf_point_abs_max} points, got ${value}`);
  }
  return value;
}
// The digits a colour must have, taken once from the shared rules.
const HEX_COLOR = new RegExp(`^[0-9a-fA-F]{${RULES.color_digits}}$`);
const hexColor = (hex) => {
  // A colour the tool accepts is a string; a JSON number is refused here too
  // rather than coerced into digits.
  const digits = typeof hex === "string" ? hex.replace(/^#/, "") : "";
  if (!HEX_COLOR.test(digits)) throw new UsageError(`color must be ${RULES.color_digits} hex digits, with an optional leading "#", got: ${hex}`);
  const per = RULES.color_digits / 3;
  const channel = (index) => parseInt(digits.slice(index * per, (index + 1) * per), 16) / 255;
  return rgb(channel(0), channel(1), channel(2));
};

async function pdfMerge(req) {
  const out = await PDFDocument.create();
  for (const input of req.inputs) {
    const source = await loadPdf(input);
    for (const page of await out.copyPages(source, source.getPageIndices())) out.addPage(page);
  }
  return out.save();
}

async function pdfSplit(req) {
  const source = await loadPdf(req.input);
  const outputs = [];
  for (const group of req.groups) {
    const out = await PDFDocument.create();
    for (const page of await out.copyPages(source, pageIndices(source, group))) out.addPage(page);
    outputs.push(await out.save());
  }
  return outputs;
}

async function pdfRotate(req) {
  // A rotation the pdf-lib writer cannot take is refused here, in the request's
  // own terms, rather than surfacing as an internal fault of the library.
  if (typeof req.degrees !== "number" || !Number.isFinite(req.degrees) || req.degrees % RULES.degrees_step !== 0) {
    throw new UsageError(`"degrees" must be a multiple of ${RULES.degrees_step}, got: ${req.degrees}`);
  }
  const pdf = await loadPdf(req.input);
  for (const index of pageIndices(pdf, req.pages)) {
    const page = pdf.getPage(index);
    page.setRotation(degrees((((page.getRotation().angle + req.degrees) % 360) + 360) % 360));
  }
  return pdf.save();
}

async function pdfText(req) {
  const pdf = await loadPdf(req.input);
  pdf.registerFontkit(fontkit);
  const bytes = fs.readFileSync(req.font);
  const font = await pdf.embedFont(bytes, { subset: true });
  const glyphs = fontkit.create(bytes);
  // An absent size takes the body default; a stated one is a measure the page
  // must be able to show, so a zero is refused rather than read as "absent".
  const size = req.size === undefined ? 12 : req.size;
  if (!inSpan(size, RULES.pdf_size_points)) {
    throw new UsageError(`"size" must be ${spanBounds(RULES.pdf_size_points)} points, got ${size}`);
  }
  // A colour that is present but not a hex string (an empty string, a number,
  // `null`) is refused by `hexColor`; only an absent one takes the default.
  const color = hexColor(req.color === undefined ? "#000000" : req.color);
  const rotate = req.rotate === undefined ? 0 : pointNumber(req.rotate, "rotate");
  for (const index of pageIndices(pdf, req.pages)) {
    const page = pdf.getPage(index);
    const { height } = page.getSize();
    const x = req.x === undefined ? MARGIN : pointNumber(req.x, "x");
    const y = req.y === undefined ? height / 2 : pointNumber(req.y, "y");
    if (req.stamp) {
      const textWidth = font.widthOfTextAtSize(req.text, size);
      page.drawRectangle({ x: x - 8, y: y - 8, width: textWidth + 16, height: size + 16, borderWidth: 1.5, borderColor: rgb(0.7, 0, 0), rotate: degrees(rotate) });
    }
    page.drawText(req.text, { x, y, size, font, color, rotate: degrees(rotate) });
    noteGlyphs(glyphs, req.text);
  }
  return pdf.save();
}

async function pdfImage(req) {
  const pdf = await loadPdf(req.input);
  const image = await embedImage(pdf, req.image);
  // The action's own sides are points, so the natural size is stated in points
  // too (and the cap with it) — one rule, in the unit the caller is writing in.
  const size = imageSize(
    { width: image.width * PX_TO_POINTS, height: image.height * PX_TO_POINTS },
    req.width,
    req.height,
    RULES.pdf_size_points,
    DEFAULT_IMAGE_WIDTH * PX_TO_POINTS,
  );
  for (const index of pageIndices(pdf, req.pages)) {
    const page = pdf.getPage(index);
    const { height } = page.getSize();
    const x = req.x === undefined ? MARGIN : pointNumber(req.x, "x");
    const y = req.y === undefined ? height / 2 : pointNumber(req.y, "y");
    page.drawImage(image, { x, y, width: size.width, height: size.height });
  }
  return pdf.save();
}

async function pdfFormFill(req) {
  const pdf = await loadPdf(req.input);
  pdf.registerFontkit(fontkit);
  const bytes = fs.readFileSync(req.font);
  const font = await pdf.embedFont(bytes, { subset: true });
  const glyphs = fontkit.create(bytes);
  const form = pdf.getForm();
  const names = form.getFields().map((field) => field.getName());
  for (const [name, value] of Object.entries(req.values)) {
    const field = form.getFieldMaybe(name);
    if (!field) throw new UsageError(`no such form field: ${name}; this document has: ${names.join(", ") || "no fillable fields"}`);
    if (typeof value === "boolean" && field.check) { value ? field.check() : field.uncheck(); continue; }
    if (field.setText) { field.setText(String(value)); noteGlyphs(glyphs, String(value)); continue; }
    if (field.select) { field.select(String(value)); noteGlyphs(glyphs, String(value)); continue; }
    throw new UsageError(`form field "${name}" cannot take a value`);
  }
  form.updateFieldAppearances(font);
  if (req.flatten) form.flatten();
  return pdf.save();
}

// ── the closed operation set ───────────────────────────────────
const operations = {
  probe: async (req) => {
    const scratch = req.scratch;
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8z8Dwn4GBgYGJAQoAHgQCAZ1n4sEAAAAASUVORK5CYII=", "base64");
    writeOut(`${scratch}/probe.png`, png);
    // Each operation the tool exposes, on the smallest input that exercises it:
    // a kit that loads but fails an operation is what the tool must not offer.
    // The step's name rides the error, since a failure is reported as the probe's
    // own message.
    const step = async (name, task) => {
      try { await task(); } catch (error) { throw new Error(`${name}: ${(error && error.message) || error}`); }
    };
    await step("create_docx", async () => writeOut(`${scratch}/probe.docx`, await createDocx({ content: [{ type: "heading", level: 1, text: "Проверка" }, { type: "paragraph", text: "текст" }] })));
    await step("create_xlsx", () => writeOut(`${scratch}/probe.xlsx`, createXlsx({ sheets: [{ rows: [["Проверка {name}", 1, { formula: "B1" }]] }] })));
    await step("create_pptx", async () => writeOut(`${scratch}/probe.pptx`, await createPptx({ content: [{ type: "heading", level: 1, text: "Проверка" }, { type: "list", items: ["раз", "два"] }] })));
    await step("create_pdf", async () => writeOut(`${scratch}/probe.pdf`, await createPdf({ font: req.font, content: [{ type: "paragraph", text: "Проверка" }] })));
    await step("merge", async () => writeOut(`${scratch}/probe_merged.pdf`, await pdfMerge({ inputs: [`${scratch}/probe.pdf`, `${scratch}/probe.pdf`] })));
    await step("split", async () => {
      const parts = await pdfSplit({ input: `${scratch}/probe_merged.pdf`, groups: [[1], [2]] });
      parts.forEach((bytes, i) => writeOut(`${scratch}/probe_part${i + 1}.pdf`, bytes));
    });
    await step("rotate", async () => writeOut(`${scratch}/probe_rotated.pdf`, await pdfRotate({ input: `${scratch}/probe.pdf`, pages: "all", degrees: RULES.degrees_step })));
    await step("text", async () => writeOut(`${scratch}/probe_text.pdf`, await pdfText({ input: `${scratch}/probe.pdf`, pages: "all", text: "Проверка", font: req.font })));
    await step("image", async () => writeOut(`${scratch}/probe_image.pdf`, await pdfImage({ input: `${scratch}/probe.pdf`, pages: "all", image: `${scratch}/probe.png` })));
    // A form to fill, authored here: the tool has no operation that writes one,
    // and a kit broken only in form filling must not stay advertised.
    await step("form_fill", async () => {
      const form = await PDFDocument.create();
      const field = form.getForm().createTextField("name");
      field.addToPage(form.addPage(A4), { x: MARGIN, y: MARGIN, width: 200, height: 20 });
      writeOut(`${scratch}/probe_form.pdf`, await form.save());
      writeOut(`${scratch}/probe_form_filled.pdf`, await pdfFormFill({ input: `${scratch}/probe_form.pdf`, values: { name: "Проверка" }, font: req.font, flatten: true }));
    });
    // The fill path needs a sample that really declares a placeholder, so the
    // substitution runs rather than finding nothing to do; the presentation's
    // notes slide exercises the part the library does not render itself.
    await step("template_docx", async () => {
      writeOut(`${scratch}/probe_template.docx`, await createDocx({ content: [{ type: "paragraph", text: "Проверка {name}" }] }));
      const filled = fillDocxOrPptx({ template: `${scratch}/probe_template.docx`, format: "docx", values: { name: "Проверка" } });
      writeOut(`${scratch}/probe_filled.docx`, filled.buffer);
    });
    await step("template_pptx", async () => {
      const template = await createPptx({ content: [{ type: "paragraph", text: "Проверка {name}" }, { type: "notes", text: "Заметка {name}" }] });
      writeOut(`${scratch}/probe_template.pptx`, template);
      const filled = fillDocxOrPptx({ template: `${scratch}/probe_template.pptx`, format: "pptx", values: { name: "Проверка" } });
      writeOut(`${scratch}/probe_filled.pptx`, filled.buffer);
    });
    await step("template_xlsx", () => {
      const filled = fillXlsx({ template: `${scratch}/probe.xlsx`, values: { name: "Проверка" } });
      writeOut(`${scratch}/probe_filled.xlsx`, filled.buffer);
    });
    return {};
  },
  create: async (req) => {
    if (req.format === "docx") writeOut(req.output, await createDocx(req));
    else if (req.format === "xlsx") writeOut(req.output, createXlsx(req));
    else if (req.format === "pptx") writeOut(req.output, await createPptx(req));
    else if (req.format === "pdf") writeOut(req.output, await createPdf(req));
    else throw new UsageError(`unknown format: ${req.format}`);
    return { outputs: [req.output] };
  },
  fill_template: async (req) => {
    // `format` is the sample's own family: a workbook has its own cell-wise
    // filler, while a document and a presentation both go through the template
    // library (which reads the package's file type itself).
    checkValues(req.values);
    const filled = req.format === "xlsx" ? fillXlsx(req) : fillDocxOrPptx(req);
    writeOut(req.output, filled.buffer);
    return { outputs: [req.output], missing: filled.missing, placeholders: filled.placeholders };
  },
  pdf_merge: async (req) => { writeOut(req.output, await pdfMerge(req)); return { outputs: [req.output] }; },
  pdf_split: async (req) => {
    const pages = await pdfSplit(req);
    pages.forEach((bytes, i) => writeOut(req.outputs[i], bytes));
    return { outputs: req.outputs };
  },
  pdf_rotate: async (req) => { writeOut(req.output, await pdfRotate(req)); return { outputs: [req.output] }; },
  pdf_text: async (req) => { writeOut(req.output, await pdfText(req)); return { outputs: [req.output] }; },
  pdf_image: async (req) => { writeOut(req.output, await pdfImage(req)); return { outputs: [req.output] }; },
  pdf_form_fill: async (req) => {
    checkValues(req.values);
    writeOut(req.output, await pdfFormFill(req));
    return { outputs: [req.output] };
  },
};

async function main() {
  const request = readJson(process.argv[2]);
  const result = { ok: false };
  try {
    // An OWN-property lookup: the operation set is closed, so a name from the
    // prototype chain ("constructor", "__proto__") is not an operation and is
    // reported as the unknown name it is.
    const operation = Object.hasOwn(operations, request.op) ? operations[request.op] : null;
    if (!operation) throw new Error(`unknown document kit operation: ${request.op}`);
    Object.assign(result, await operation(request), { ok: true });
    if (missingGlyphs.size) result.unsupported = [...missingGlyphs];
  } catch (error) {
    result.ok = false;
    result.error = String((error && error.message) || error);
    if (error instanceof UsageError) result.class = "usage";
  }
  const text = JSON.stringify(result);
  if (request.result) fs.writeFileSync(request.result, text);
  else process.stdout.write(text);
}

await main();
