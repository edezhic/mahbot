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
function imageType(file) {
  const lower = file.toLowerCase();
  const extension = RULES.image_extensions.find((candidate) => lower.endsWith(`.${candidate}`));
  if (!extension) throw new UsageError(`only PNG and JPEG images can be embedded, got: ${nodePath.basename(file)}`);
  return extension === "png"
    ? { kind: "png", contentType: "image/png" }
    : { kind: "jpg", contentType: "image/jpeg" };
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
// later. The media part the image becomes is named with the kind and declared
// with the content type. The size is the frame's own, with no EXIF orientation
// applied: a viewer that honours an orientation tag draws such a JPEG turned.
function readImage(file) {
  const { kind, contentType } = imageType(file);
  const bytes = readInput(file);
  const size = kind === "png" ? pngSize(bytes) : jpegSize(bytes);
  // A declared size of nothing means the bytes are not a whole image of the kind
  // the name claims — truncated, or another format under that extension — not
  // that the image is small: the proportions every other size is scaled from
  // would not exist.
  if (!size || size.width < 1 || size.height < 1) {
    const label = kind === "png" ? "PNG" : "JPEG";
    throw new UsageError(`cannot embed ${nodePath.basename(file)} as a ${label} image: the file's bytes are not a whole ${label} image`);
  }
  return { kind, contentType, bytes, width: size.width, height: size.height };
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
//
// The parts a workbook names its cell styles by, and the content type and
// relationship type OPC declares one with. A workbook that never saved styles
// carries no styles part at all; an edit that brings formatting to it writes the
// same part `createXlsx` would have (see `stylesText`, and the entries and blocks
// it shares with the style machinery under `STYLE_BLOCKS`). The name below is the
// one every writer uses and the one a workbook this kit creates gets; an edit
// writes the part the workbook's OWN relationships name, which is this one unless
// the package saved its styles elsewhere (`stylesPart`).
const XLSX_STYLES = "xl/styles.xml";
const XLSX_WORKBOOK = "xl/workbook.xml";
const XLSX_WORKBOOK_RELS = "xl/_rels/workbook.xml.rels";
const STYLES_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml";
const STYLES_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles";
const xmlEscape = (value) => String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
const colName = (index) => {
  let name = "";
  let n = index + 1;
  while (n > 0) { const rest = (n - 1) % 26; name = String.fromCharCode(65 + rest) + name; n = Math.floor((n - 1) / 26); }
  return name;
};
// The three spellings a tag of `name` is written in — an open tag, a close tag and
// one that closes itself — stated once, so every pattern that reads a part's tags is
// built from one rule and no two readers can disagree about which shape a part is:
// such a disagreement has read an element as absent and made a writer place a second
// one beside it. `middle` is the attributes and whitespace between a name and its
// tag's end, stopping at the first `>` so no read runs past the tag it is on; the
// close tag and a self-closing tag take the whitespace XML allows before their `>` or
// `/>`. A POSITION taken from a match built here comes from the match's own span or
// from `elementEdges`, never from a close tag spelled out again to measure it.
const TAG_MIDDLE = "(?:\\s[^>]*)?";
const openTag = (name, middle = TAG_MIDDLE) => `<${name}${middle}>`;
const selfClosingTag = (name, middle = TAG_MIDDLE) => `<${name}${middle}\\s*/>`;
const closeTag = (name) => `</${name}\\s*>`;
// The text a formula cell holds: the object names `formula` alone, and the text
// — with its one optional leading `=` and the spacing removed — is a formula, so
// a second `=` is refused rather than written into the `<f>` element as text.
// The rule the tool's boundary states, as the kit's own last line; the tool's
// paired refusal test drives both sides, keeping the two aligned.
function formulaOf(reference, value) {
  const formula = typeof value.formula === "string" ? value.formula.trim().replace(/^=/, "").trim() : "";
  if (Object.keys(value).length === 1 && formula && !formula.startsWith("=")) return formula;
  throw new UsageError(`cell ${reference}: an object value must be {"formula": "SUM(A1:A2)"}`);
}
function xlsxCell(reference, value) {
  // A null is neither a scalar nor a formula cell: the docx/pptx/pdf arms
  // refuse one through `scalarOf`, and writing the literal "null" into a cell
  // would be wrong content rather than an empty cell.
  if (value === null) throw new UsageError(`cell ${reference} must be text, a number or a boolean`);
  // A request is JSON, so a number reaching here is finite by construction.
  if (typeof value === "number") return `<c r="${reference}"><v>${value}</v></c>`;
  if (typeof value === "boolean") return `<c r="${reference}" t="b"><v>${value ? 1 : 0}</v></c>`;
  if (value && typeof value === "object") return `<c r="${reference}"><f>${xmlEscape(formulaOf(reference, value))}</f></c>`;
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
    `<Override PartName="/${XLSX_WORKBOOK}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>`,
    ...sheets.map((_, i) => `<Override PartName="/xl/worksheets/sheet${i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`),
    `<Override PartName="/${XLSX_STYLES}" ContentType="${STYLES_CONTENT_TYPE}"/>`,
  ];
  zip.file("[Content_Types].xml", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">${contentTypes.join("")}</Types>`);
  zip.file("_rels/.rels", `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="${XLSX_WORKBOOK}"/></Relationships>`);
  zip.file(XLSX_WORKBOOK, `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${sheets.map((sheet, i) => `<sheet name="${xmlEscape(sheet.name)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`).join("")}</sheets></workbook>`);
  const workbookRels = [
    ...sheets.map((_, i) => `<Relationship Id="rId${i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${i + 1}.xml"/>`),
    `<Relationship Id="rId${sheets.length + 1}" Type="${STYLES_REL_TYPE}" Target="styles.xml"/>`,
  ];
  zip.file(XLSX_WORKBOOK_RELS, `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${workbookRels.join("")}</Relationships>`);
  zip.file(XLSX_STYLES, MINIMAL_STYLES);
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
      slide.addImage({ data: `data:${image.contentType};base64,${image.bytes.toString("base64")}`, x: 0.4, y, w, h });
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
const runPattern = (prefix) => new RegExp(`${openTag(`${prefix}t`)}([\\s\\S]*?)${closeTag(`${prefix}t`)}`, "g");
// The workbook's shared-string table, its `<si>` entries read as elements.
const SHARED_STRINGS_PART = "xl/sharedStrings.xml";

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

// The element holding one run's text, in the namespace `prefix` names. A
// WordprocessingML `w:t` and a SpreadsheetML `<t>` declare `xml:space`, and both
// get it so a significant space survives; DrawingML's `<a:t>` declares no
// attributes at all, so it is written bare — the attribute is not part of that
// format, and a presentation holding it offers to repair the file. Element text
// reaches a reader whole either way, so no significant space is lost without it.
const textElement = (prefix, text) => (prefix === "a:" ? `<a:t>${text}</a:t>` : `<${prefix}t xml:space="preserve">${text}</${prefix}t>`);

// `element` with its first run holding `text` and every further run emptied;
// the formatting around those runs (their run properties) is kept.
function rewriteRuns(element, runs, prefix, text) {
  let first = true;
  return element.replace(runs, () => {
    const content = first ? text : "";
    first = false;
    return textElement(prefix, xmlEscape(content));
  });
}

// Substitute the placeholders of every `container` element of `xml`, one at a
// time: the container's runs are joined, substituted once, and written back into
// its first run. `container` is a complete-element pattern (see
// `elementPattern`), so both shapes a file may write the element in are matched.
function fillTextContainers(xml, container, prefix, values, seen, missing) {
  const runs = runPattern(prefix);
  return xml.replace(container, (element) => {
    const text = placeholderText(element, runs);
    if (text === null) return element;
    return rewriteRuns(element, runs, prefix, substituteText(text, values, seen, missing));
  });
}

// Open a sample package, blaming the REQUEST when it cannot be opened at all —
// an encrypted or truncated file is a bad input the caller can do something
// about, not a kit fault. The family is named as the extension it was read as,
// so the sentence reads the same for every one of them.
function openPackage(file, kind) {
  try {
    return new PizZip(fs.readFileSync(file));
  } catch (error) {
    throw new UsageError(`${nodePath.basename(file)} is not a readable .${kind} package: ${(error && error.message) || error}`);
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
  if (zip.file(XLSX_WORKBOOK)) return "xlsx";
  if (zip.file("ppt/presentation.xml")) return "pptx";
  return null;
}

// A package whose parts are another family's cannot be used as this one: the
// mismatch is named here rather than left to the library, whose message for it
// talks about its own paid modules instead of the file. The caller passes its
// own voice: `noun` names the file it handed over ("sample" for a fill, "input"
// for an edit) and `hint` builds the advice that fits its own action from the
// family the package really is, so the check itself stays one. The family is
// named as the extension it is, the way the file is named.
function ensurePackageFamily(zip, expected, noun, hint) {
  const family = packageFamily(zip);
  if (!family || family === expected) return;
  throw new UsageError(`the ${noun} is really a .${family} package, not the .${expected} one its name says — hint: ${hint(family)}`);
}

function fillDocxOrPptx(req) {
  const missing = new Set();
  const tags = new Set();
  // The request's `format` is the sample's family; without one it is a
  // document, which the library reads on its own.
  const expected = req.format === "pptx" ? "pptx" : "docx";
  const zip = openPackage(req.template, expected);
  ensurePackageFamily(zip, expected, "sample", () => "rename it to match its content");
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
    const filled = fillTextContainers(xml, elementPattern("a:p"), "a:", req.values, tags, missing);
    if (filled !== xml) out.file(name, filled);
  }
  return { buffer: out.generate({ type: "nodebuffer", compression: "DEFLATE" }), missing: [...missing], placeholders: tags.size };
}

// A cell's `<v>` element, read with the shared spellings (`openTag`/`closeTag`), so a
// `<v >` is the element a `<v>` is, like every other element this kit reads.
const CELL_VALUE = new RegExp(`${openTag("v")}\\s*(\\d+)\\s*${closeTag("v")}`);

// The shared string a `t="s"` cell indexes into, joined and unescaped, or `null`
// when the cell is not a shared-string cell (or the table does not have that
// index). A workbook written by Excel, Sheets or LibreOffice keeps EVERY string
// here and writes only an index into the cell, so this is the text such a cell
// displays.
function sharedStringOf(cell, shared) {
  if (!/\bt="s"/.test(cell)) return null;
  const index = Number((cell.match(CELL_VALUE) || [])[1]);
  return shared[index] ?? null;
}

// The joined, unescaped text of every `<si>` of the shared-string table, in
// index order: what a `t="s"` cell's index refers to (`""` for an `<si/>` with
// no runs, which still occupies its index).
function sharedStringTexts(xml) {
  return [...xml.matchAll(elementPattern("si"))].map((match) => runText(match[0], runPattern("")));
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
  return xml.replace(new RegExp(`${selfClosingTag("row")}|${openTag("row")}|${selfClosingTag("c")}|${openTag("c")}[\\s\\S]*?${closeTag("c")}`, "g"), (part) => {
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
  ensurePackageFamily(zip, "xlsx", "sample", () => "rename it to match its content");
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
  rewrite(SHARED_STRINGS_PART, (xml) => fillTextContainers(xml, elementPattern("si"), "", req.values, seen, missing));
  return { buffer: zip.generate({ type: "nodebuffer", compression: "DEFLATE" }), missing: [...missing], placeholders: seen.size };
}

// ── surgical package edits ─────────────────────────────────────
// `docx_edit`, `xlsx_edit` and `pptx_edit` change an existing package part by
// part instead of rewriting it through a writer library: every part an edit does
// not name is carried into the output with the exact content it had, which keeps
// a chart, a comment or a media file the caller never mentioned intact. The raw
// archive bytes are not identical — a fresh DEFLATE stream re-compresses every
// part — but the CONTENT of every part no edit touched is.
//
// Each edit walks the part it applies to — a `part` call charges the text it
// opens, and an arm that folds a whole edit list over one part charges each edit's
// own walk (the docx arm) — so a call costs the part text its edits walk, added
// up. Past this much the run would reach its own time limit, which the caller
// reads as a product fault: refusing here instead names a call the caller can fix
// by splitting it. The unit is the part text's own length, which is its byte count
// for the XML these parts hold.
const EDIT_WORK_MAX = 4 * 1024 ** 3;

// The refusal a call past that budget states, naming the size it walked and the
// fix. One place, so the arms that charge work cannot word it differently.
const editWorkRefusal = (work) => new UsageError(`the edits rewrite about ${Math.round(work / 1024 / 1024)} MB of the document's parts, more than one call applies (limit ${Math.round(EDIT_WORK_MAX / 1024 / 1024)} MB) — hint: split the edits into several calls`);

function openEdit(req, family) {
  const zip = openPackage(req.input, family);
  ensurePackageFamily(zip, family, "input", (real) => `use ${real}_edit to edit it`);
  // The family's own part is what an edit rewrites, so a package that declares
  // no family at all — `ensurePackageFamily` refuses another family, not the
  // absence of one — is refused here rather than left to a no-op. The sentence
  // names what the file lacks, since a caller can act on that.
  if (!packageFamily(zip)) throw new UsageError(`cannot edit ${nodePath.basename(req.input)}: it holds no document, workbook or presentation part, so it is not a .${family} package`);
  // The parts the input held, for the signature check below. A folder entry ends
  // with `/` and `zip.file` does not resolve one, so the names kept are the
  // parts.
  const baseline = Object.keys(zip.files).filter((name) => !name.endsWith("/"));
  // A package that carries a digital signature part cannot be edited into a file
  // the signature still matches: the parts it signs change while the signature
  // does not, so a changed package says so in one note rather than staying
  // silent about it.
  const signed = baseline.some((name) => name.startsWith("_xmlsignatures/"));
  const notes = [];
  let changed = false;
  let touched = false;
  let work = 0;
  // Charge one walk over a part's text against the call's budget. The budget bounds
  // the work a call does rather than the distinct bytes it touches, so a part taken
  // into hand twice is charged twice: `part`, `sheet` and `read` charge the pass each
  // makes, `rewrite` none, and the arms that walk a part of their own charge theirs
  // with this (the docx arm, `formatCells`' index cache, `addRelationship`). The pptx
  // arms that read a slide, a notes part or the presentation off the archive with
  // `zip.file(…).asText()` walk their parts uncharged, so the bound is short of the
  // work they do.
  const charge = (length) => {
    work += length;
    if (work > EDIT_WORK_MAX) throw editWorkRefusal(work);
  };
  return {
    zip,
    charge,
    // A user-facing note, already bracketed: the Rust side appends it verbatim.
    // Kept once each, in the order the edits first raised it.
    note: (message) => { if (!notes.includes(message)) notes.push(message); },
    // Whether the call changed the sheet's CONTENT — a value written or an address
    // moved — rather than only the markup around it: an insert past the used range
    // adds a line without moving anything and a repeated write states what the cell
    // already held, and the caveats about a workbook's charts and pivots are owed
    // for a change of content, not for a rewrite. Recorded by the edits themselves
    // (`markContent`), because only they know what their write meant.
    content: () => touched,
    // Record a change of content (see `content`).
    markContent: () => { touched = true; },
    // Rewrite a part only when the change really changed it; a part the family
    // does not have is left alone.
    part(name, change) {
      const file = zip.file(name);
      if (!file) return;
      const xml = file.asText();
      charge(xml.length);
      const written = change(xml);
      if (written !== xml) { zip.file(name, written); changed = true; }
    },
    // `part`'s own rule (only a real change is written), for the SHEET parts, plus
    // the one refusal this family needs: the written part must not hold two elements
    // at an address the part arrived as holding fewer of — a `<row>` number or a `<c>`
    // address two of them share, which a reader offers to repair and the reply would
    // pass off as a success. An edit places its content by the address it reads, so a
    // row it removes or adds can move an element that states no `r` onto the address
    // another one already holds. A repeat the part arrived with is passed over where
    // it stands and a part whose rows cannot be read is not refused for the check's
    // own sake (see `addedRepeat`); the two readings are charged like every other
    // walk, and a change that changed nothing is not read a third time.
    sheet(name, change) {
      const file = zip.file(name);
      if (!file) return;
      const before = file.asText();
      charge(before.length);
      const after = change(before);
      if (after === before) return;
      charge(after.length);
      const repeated = addedRepeat(before, after);
      if (repeated) throw new UsageError(`${addressLabel(repeated)} would be stated twice in the sheet part — two elements claiming one address is a part a reader offers to repair, so this edit was refused rather than written`);
      zip.file(name, after);
      changed = true;
    },
    // Read a part's text and charge the walk over it, for an arm that reads a part
    // outside one `part` call (see `charge`). `undefined` for a part the package
    // does not hold, so a caller can tell "no part" from "an empty one".
    read(name) {
      const file = zip.file(name);
      if (!file) return undefined;
      const xml = file.asText();
      charge(xml.length);
      return xml;
    },
    // Write a part back as the text the caller read and charged, `from`, compared
    // against it: `part`/`sheet` would read and charge the same text again, counting
    // one call's work twice (`formatCells` and `setCell` grow the styles part across
    // a whole walk and write it once, this way).
    rewrite(name, from, to) {
      if (from === to) return;
      zip.file(name, to);
      changed = true;
    },
    // Add a part the input did not have, or put one back with new content —
    // `pptx_edit`'s slide add is the operation that means to. A name the package
    // already holds with the same content is not a change.
    add(name, content) {
      const held = zip.file(name);
      if (held && held.asText() === content) return;
      zip.file(name, content);
      changed = true;
    },
    remove(name) {
      if (!zip.file(name)) return;
      zip.remove(name);
      changed = true;
    },
    // The result an edit hands back.
    finish() {
      if (changed && signed) notes.push("[the input carries a digital signature, which was carried through unchanged — the file changed, so the signature no longer matches it]");
      return { buffer: zip.generate({ type: "nodebuffer", compression: "DEFLATE" }), notes };
    },
  };
}

// The edits one call carries, bounded before any of them runs.
function editList(req) {
  if (!Array.isArray(req.edits)) throw new UsageError("edits must be a list");
  if (req.edits.length > RULES.edits_max) throw new UsageError(`a call may carry at most ${RULES.edits_max} edits, got: ${req.edits.length}`);
  return req.edits;
}

// A package with no `[Content_Types].xml` is not one this kit can edit: it could
// not declare a part it adds. The sentence names the file the way the pptx media
// path names it, so both families refuse it the same way.
const missingContentTypes = (input) => new UsageError(`cannot edit ${nodePath.basename(input)}: it has no [Content_Types].xml part`);

// One string an edit names — a find, a replacement, an inserted text — as the
// kit's last line after the tool's own boundary: text, and inside the cap the
// two sides share.
function editText(value, what) {
  if (typeof value !== "string") throw new UsageError(`${what} must be text`);
  if ([...value].length > RULES.edit_text_max) throw new UsageError(`${what} must be at most ${RULES.edit_text_max} characters`);
  return value;
}

// Every span `find` occurs at in `text`, left to right and non-overlapping. The
// offsets are code points, the unit `locateSlot` counts in: a raw `indexOf`
// offset is a UTF-16 code unit, so a supplementary character before a match
// would shift it by one and make the edit rewrite the wrong slot.
function occurrences(text, find) {
  const spans = [];
  const width = [...find].length;
  let from = 0;
  // The code points of `text` before `from`, kept in step with the UTF-16 cursor
  // `indexOf` advances.
  let walked = 0;
  while (true) {
    const at = text.indexOf(find, from);
    if (at < 0) break;
    walked += [...text.slice(from, at)].length;
    spans.push({ start: walked, end: walked + width });
    from = at + find.length;
    walked += width;
  }
  return spans;
}

// ── docx_edit ──────────────────────────────────────────────────
// The three spellings a tag is written in (`selfClosingTag`/`openTag`/`closeTag`),
// as the balanced walker `tagSpans` reads them. One definition of the rule, so a
// family walked later cannot be handed a pattern missing a spelling.
const tagPattern = (name) => new RegExp(`${selfClosingTag(name)}|${openTag(name)}|${closeTag(name)}`, "g");

// A text edit addresses the joined, unescaped text of a `<w:p>` paragraph's
// `<w:t>` runs. A field's cached result is not body text, and a run inside a
// text box belongs to its own paragraph, so the runs of a field region, of a
// `<w:fldSimple>`, and of a nested paragraph are excluded from the text an edit
// addresses. The header, footer, footnote and endnote parts are not read at
// all: only `word/document.xml` is edited.
const DOCX_PARAGRAPH = tagPattern("w:p");
const DOCX_RUN = tagPattern("w:r");
const DOCX_FIELD = tagPattern("w:fldSimple");
const DOCX_CELL = tagPattern("w:tc");
const DOCX_TXBX = tagPattern("w:txbxContent");
const DOCX_TABLE = tagPattern("w:tbl");
// A run's text, the same pattern the filler substitutes through.
const DOCX_TEXT = runPattern("w:");
// The refusal every op that searched and found nothing states: the parts a body
// edit does not look at are named, because the text there is real and the
// caller may well expect it found.
const missingFind = (find) => `the text ${JSON.stringify(find)} is not in the document's body (headers, footers, footnotes and fields are not searched)`;

// The one balanced scan every "walk the elements of one tag family" caller here
// is built on. It returns one entry per COMPLETE element of `pattern`'s family:
// `start`/`end` are byte offsets, `closeAt` is where the element's own close tag
// begins (the place a child added "at the end" goes), `depth` is the element's own
// nesting level among its family (`1` for a top-level one) and `selfClosing` says
// whether it was written `<name/>`. A self-closing element is complete where it
// opens, a child-bearing one where its own close balances it, so the entries come
// out in the order the scan finishes them — the order each walk this replaced
// produced.
function tagSpans(xml, pattern) {
  const spans = [];
  const stack = [];
  for (const match of xml.matchAll(pattern)) {
    const tag = match[0];
    if (tag.endsWith("/>")) {
      spans.push({ start: match.index, end: match.index + tag.length, depth: stack.length + 1, selfClosing: true });
      continue;
    }
    if (tag.startsWith("</")) {
      const open = stack.pop();
      if (open) spans.push({ start: open.start, end: match.index + tag.length, closeAt: match.index, depth: open.depth, selfClosing: false });
    } else {
      stack.push({ start: match.index, depth: stack.length + 1 });
    }
  }
  return spans;
}

// The `[start, end]` span of every complete `<w:p>` of a fragment, in the order
// the scan reaches their closing tags (a nested one before the one that holds
// it). A self-closing `<w:p/>` holds nothing an edit can address and is dropped.
function docxParagraphs(xml) {
  return tagSpans(xml, DOCX_PARAGRAPH)
    .filter((span) => !span.selfClosing)
    .map(({ start, end }) => ({ start, end }));
}

// The spans of the paragraphs nested in `fragment`, i.e. every complete `<w:p>`
// that is not `fragment`'s own element (the only one at depth 1), each a
// `[start, end]` pair.
function nestedParagraphSpans(fragment) {
  return tagSpans(fragment, DOCX_PARAGRAPH)
    .filter((span) => !span.selfClosing && span.depth > 1)
    .map(({ start, end }) => [start, end]);
}

// The nested paragraphs that are not themselves inside another nested one,
// i.e. `fragment`'s direct child paragraphs — the ones a recursive edit walks.
function childParagraphSpans(fragment) {
  const nested = nestedParagraphSpans(fragment);
  return nested.filter(([start, end]) => !nested.some(([outerStart, outerEnd]) => outerStart < start && outerEnd > end));
}

// The spans of the `<w:fldSimple>` elements of a fragment: everything inside one
// is a field, which a text edit does not address.
function fieldSimpleSpans(fragment) {
  return tagSpans(fragment, DOCX_FIELD).map(({ start, end }) => [start, end]);
}

// The `[start, end]` spans of every field of a whole part: each `<w:fldSimple>`,
// and the region from a `<w:fldChar begin>` to its `<w:fldChar end>`. Computed
// over the part rather than per paragraph, because a field's begin and end may
// sit in different paragraphs — a per-paragraph counter would leave the second
// paragraph's cached result looking like body text and make the "fields are not
// searched" promise false.
function fieldRegions(xml) {
  const regions = [];
  const stack = [];
  for (const match of xml.matchAll(tagPattern("w:fldChar"))) {
    const type = (match[0].match(/\bw:fldCharType="([^"]*)"/) || [])[1];
    if (type === "begin") stack.push(match.index);
    else if (type === "end" && stack.length) regions.push([stack.pop(), match.index + match[0].length]);
  }
  return [...fieldSimpleSpans(xml), ...regions];
}

// The part of each excluded span that falls inside `[start, end]`, moved to that
// slice's own origin — how a span found over a whole part reaches the fragment a
// recursive edit works on.
const clipSpans = (spans, start, end) => spans
  .map(([from, to]) => [Math.max(from, start), Math.min(to, end)])
  .filter(([from, to]) => from < to)
  .map(([from, to]) => [from - start, to - start]);

// The text a docx edit ADDRESSES in a fragment: the joined text of its own runs,
// with the field regions that reach into it excluded. The fragments a text edit
// matches, and the text a removal tests `find` against.
function addressedText(fragment, excluded = []) {
  return docxRuns(fragment, excluded).slots.map((slot) => xmlUnescape(slot.raw)).join("");
}

// The text containers that must keep at least one block-level child — ECMA-376
// requires one in `CT_Tc` and in `CT_TxbxContent` — with the name the refusal
// gives them.
const DOCX_BLOCK_CONTAINERS = [
  { pattern: DOCX_CELL, name: "table cell" },
  { pattern: DOCX_TXBX, name: "text box" },
];

// The paragraphs a container holds of its OWN — the ones that keep it valid. A
// paragraph of a table nested in it is that table's cell's, not this container's,
// and one drawn in a text box sits inside a paragraph (paragraph depth 2), which
// `topParagraphs` already leaves out.
function ownParagraphSpans(container) {
  const tables = tagSpans(container, DOCX_TABLE).filter((span) => !span.selfClosing);
  return topParagraphs(container).filter((paragraph) => !tables.some((span) => span.start < paragraph.start && paragraph.end <= span.end));
}

// Refuse a removal that would leave a container ECMA-376 requires a paragraph in
// with none of its own. A container loses all of its own paragraphs exactly when
// every one of them holds `find`, so the check runs once over the part before the
// removal — the form that asked each removed paragraph for its enclosing
// container re-walked the whole part once per paragraph, which made a
// `remove_paragraph` matching many paragraphs quadratic in the part's size.
function requireParagraphLeft(xml, find) {
  const regions = fieldRegions(xml);
  for (const { pattern, name } of DOCX_BLOCK_CONTAINERS) {
    for (const span of tagSpans(xml, pattern)) {
      if (span.selfClosing) continue;
      const own = ownParagraphSpans(xml.slice(span.start, span.end));
      if (!own.length) continue;
      const everyOneGoes = own.every((paragraph) => {
        const start = span.start + paragraph.start;
        const end = span.start + paragraph.end;
        return addressedText(xml.slice(start, end), clipSpans(regions, start, end)).includes(find);
      });
      if (everyOneGoes) throw new UsageError(`removing every paragraph holding ${JSON.stringify(find)} would leave a ${name} with no paragraph — a ${name} must keep one`);
    }
  }
}

// The `[start, end]` spans of a fragment's `<w:r>` elements, each with the depth
// it sits at: a run that draws a text box holds the text box's runs.
function runSpans(fragment) {
  return tagSpans(fragment, DOCX_RUN).map(({ start, end, depth }) => ({ start, end, depth }));
}

// The `[start, end]` span of the `<name>…</name>` element — self-closing or
// child-bearing — that starts exactly at `at`, or `undefined` when there is
// none. Found by balancing, because the first `</name>` a lazy match reaches may
// close a nested element of the same name (a `<w:rPrChange>` holds its own
// `<w:rPr>`). Deliberately NOT a `tagSpans` call: the tag name is known only at
// run time and the scan starts at `at`, while `tagSpans` walks a fixed family
// over the whole part — this runs once per run, so re-scanning the whole part
// each time would be quadratic.
function elementSpan(xml, at, name) {
  const open = new RegExp(`${selfClosingTag(name)}|${openTag(name)}`, "g");
  open.lastIndex = at;
  const first = open.exec(xml);
  if (!first || first.index !== at) return undefined;
  if (first[0].endsWith("/>")) return { start: at, end: at + first[0].length };
  const token = tagPattern(name);
  token.lastIndex = at + first[0].length;
  let depth = 1;
  let match;
  while ((match = token.exec(xml)) !== null) {
    if (match[0].startsWith("</")) { depth -= 1; if (depth === 0) return { start: at, end: match.index + match[0].length }; }
    else if (!match[0].endsWith("/>")) depth += 1;
  }
  return undefined;
}

// The `<w:rPr>` a run holds as its OWN first child, or `undefined` when it has
// none. A run's properties are the element directly inside `<w:r>`, never one a
// nested `<w:rPrChange>` holds (a tracked formatting revision would otherwise
// take the edit meant for the run).
function runProperties(runXml) {
  const open = runXml.match(new RegExp(openTag("w:r")));
  if (!open) return undefined;
  const rest = runXml.slice(open.index + open[0].length);
  const lead = rest.length - rest.trimStart().length;
  if (!new RegExp(`^(?:${openTag("w:rPr")}|${selfClosingTag("w:rPr")})`).test(rest.trimStart())) return undefined;
  const span = elementSpan(runXml, open.index + open[0].length + lead, "w:rPr");
  return span ? runXml.slice(span.start, span.end) : undefined;
}

// A paragraph's own runs and text slots. `slots` are the `<w:t>` elements whose
// joined text an edit addresses, each with its span, its raw text, and whether
// the raw text holds an entity — a slot whose text does is written back through
// `xmlEscape`, one whose text does not keeps its own bytes. A run is skipped
// when it carries the paragraph-mark property (a `<w:rPr>` with a `<w:sectPr>`,
// where a Word section break sits), when it is not a depth-1 run, or when it
// lies inside a nested paragraph or a field. `extra` are the field regions that
// reach into the fragment, relative to it, for a field whose begin and end sit
// in different paragraphs (see `fieldRegions`).
function docxRuns(fragment, extra = []) {
  const excluded = [...nestedParagraphSpans(fragment), ...fieldSimpleSpans(fragment), ...extra];
  const inExcluded = (at) => excluded.some(([start, end]) => at >= start && at < end);
  const runs = [];
  const slots = [];
  for (const span of runSpans(fragment)) {
    if (span.depth !== 1 || inExcluded(span.start)) continue;
    const runXml = fragment.slice(span.start, span.end);
    if (new RegExp(`${openTag("w:rPr")}[\\s\\S]*?<w:sectPr\\b`).test(runXml)) continue;
    const index = runs.length;
    runs.push({ start: span.start, end: span.end, rpr: runProperties(runXml) });
    for (const match of runXml.matchAll(DOCX_TEXT)) {
      const at = span.start + match.index;
      if (inExcluded(at)) continue;
      slots.push({ start: at, end: at + match[0].length, raw: match[1], unescaped: /&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/.test(match[1]), run: index });
    }
  }
  return { runs, slots };
}

// The paragraph elements no other paragraph contains: the roots a recursive edit
// starts from, so a nested paragraph is edited once and not a second time as
// part of its ancestor. Read from the scan's own nesting depth rather than by
// asking every paragraph whether another one holds it, which is quadratic in the
// part's paragraph count.
function topParagraphs(xml) {
  return tagSpans(xml, DOCX_PARAGRAPH)
    .filter((span) => !span.selfClosing && span.depth === 1)
    .map(({ start, end }) => ({ start, end }));
}

// `element` with `change` applied to it and, first, to each of its nested
// paragraphs. A nested edit rewrites its bytes, so both the paragraph's own runs
// and its field regions are located only after the nested edits are in: the runs
// are re-found in the rewritten text, and a region that lies entirely after a
// nested edit moves by that edit's byte delta. (A region is never inside the
// edited span: an edit addresses body text, and a field's text is not
// addressable.) `excluded` are the field regions that reach into the element,
// relative to it.
function editParagraph(element, excluded, change) {
  const children = childParagraphSpans(element);
  let out = element;
  let regions = excluded;
  for (let i = children.length - 1; i >= 0; i -= 1) {
    const [start, end] = children[i];
    const before = out.slice(start, end);
    const edited = editParagraph(before, clipSpans(regions, start, end), change);
    out = out.slice(0, start) + edited + out.slice(end);
    const delta = edited.length - before.length;
    if (delta) regions = regions.map(([from, to]) => (from >= end ? [from + delta, to + delta] : [from, to]));
  }
  return change(out, regions) ?? out;
}

// `xml` with every top-level paragraph put through `change`, from the last to
// the first so an earlier paragraph's byte positions stay valid. The field
// regions are found over the whole part once, then clipped to each paragraph.
// The result is assembled as the pieces of the part between and around the
// rewritten paragraphs and joined once: rewriting `xml` in place on every
// paragraph would copy the whole part once per paragraph.
function mapParagraphs(xml, change) {
  const regions = fieldRegions(xml);
  const tops = topParagraphs(xml);
  const pieces = [];
  let end = xml.length;
  for (let i = tops.length - 1; i >= 0; i -= 1) {
    const top = tops[i];
    const element = xml.slice(top.start, top.end);
    const edited = editParagraph(element, clipSpans(regions, top.start, top.end), change);
    pieces.push(xml.slice(top.end, end), edited);
    end = top.start;
  }
  pieces.push(xml.slice(0, end));
  return pieces.reverse().join("");
}

// The slot a character offset in the joined text falls in, with the offset from
// that slot's start. An offset exactly at a slot's end belongs to the next slot
// when it OPENS a fragment (a fragment that begins where one run's text ends
// begins in the next run) and to the slot it ends when it CLOSES one (the run a
// fragment ends in is the one whose text ends at that offset).
function locateSlot(slots, offset, atEnd = false) {
  let walked = 0;
  for (let index = 0; index < slots.length; index += 1) {
    const length = [...xmlUnescape(slots[index].raw)].length;
    if (offset < walked + length || (atEnd && offset === walked + length) || index === slots.length - 1) return { index, offset: offset - walked };
    walked += length;
  }
  return { index: 0, offset: 0 };
}

// A slot's own text, written back the way it was read: a slot whose text held
// an entity is escaped again, one that held none keeps its own bytes — escaping
// a raw `&` that was already an entity would double it.
const slotText = (slot, text) => (slot.unescaped ? xmlEscape(text) : text);

// One slot's `<t>`, rewritten from the sub-edits the occurrences in it produced.
// Each edit is `[from, to, text]` over the slot's own code points, replacing
// `[from, to)` with the already-escaped `text`; they are non-overlapping and
// ascending, so they are applied from the last to the first, where an earlier
// offset is still the one the slot's own text has. The characters no edit
// touches keep their own bytes (re-escaped only when the slot's raw text held an
// entity, see `slotText`), while every replacement is already `xmlEscape`d.
function rewrittenSlot(slot, edits, prefix) {
  const points = [...xmlUnescape(slot.raw)];
  let at = points.length;
  let text = "";
  for (let i = edits.length - 1; i >= 0; i -= 1) {
    const [from, to, replacement] = edits[i];
    text = replacement + slotText(slot, points.slice(to, at).join("")) + text;
    at = from;
  }
  return textElement(prefix, slotText(slot, points.slice(0, at).join("")) + text);
}

// `xml` with every one of `offsets` rewritten in a single pass. `offsets` is the
// WHOLE list of code-point `{start, end}` spans for one paragraph's joined text,
// taken together with the `runs`/`slots` of that same text, so no occurrence's
// positions can go stale against another's. For each span, the replacement is
// written into the slot it starts in — the fragment's first run is the one whose
// text takes the replacement, and the one whose formatting it keeps — every slot
// strictly between the first and the last is emptied, and the slot it ends in
// keeps the characters after it. The replacement takes the formatting of the run
// its span starts in, so a span covering any run with different formatting raises
// a note saying so; a removal substitutes nothing, takes no formatting with it,
// and raises nothing. `prefix` is the run/text namespace (`w:` for the docx edit,
// `a:` for the pptx one).
function replaceOccurrences(xml, runs, slots, offsets, replacement, note, prefix) {
  const escaped = xmlEscape(replacement);
  // Whether the span's slots cover a run whose own properties differ from the run
  // the span starts in: the replacement is written into the first run, so every
  // such run loses the formatting it had. Every covered slot is asked rather than
  // the two ends alone, because a differently formatted run in the middle is lost
  // exactly the same way.
  const foldsFormatting = (first, last) => {
    for (let index = first; index <= last; index += 1) {
      if (runs[slots[index].run].rpr !== runs[slots[first].run].rpr) return true;
    }
    return false;
  };
  // The sub-edits each slot takes, keyed by its index, added in the order the
  // occurrences are found, which keeps each slot's own list ascending.
  const edits = new Map();
  const add = (index, from, to, text) => {
    const list = edits.get(index);
    if (list) list.push([from, to, text]);
    else edits.set(index, [[from, to, text]]);
  };
  for (const { start, end } of offsets) {
    const first = locateSlot(slots, start);
    const last = locateSlot(slots, end, true);
    if (replacement && foldsFormatting(first.index, last.index)) {
      note("[the replaced text spanned several differently formatted runs — it took the formatting of the first one]");
    }
    if (first.index === last.index) {
      add(first.index, first.offset, last.offset, escaped);
      continue;
    }
    add(first.index, first.offset, [...xmlUnescape(slots[first.index].raw)].length, escaped);
    add(last.index, 0, last.offset, "");
    for (let index = first.index + 1; index < last.index; index += 1) {
      add(index, 0, [...xmlUnescape(slots[index].raw)].length, "");
    }
  }
  // Each slot is written once, from the last to the first so an earlier slot's
  // position in `xml` is the one it still has.
  let out = xml;
  for (const index of [...edits.keys()].sort((a, b) => b - a)) {
    const slot = slots[index];
    out = out.slice(0, slot.start) + rewrittenSlot(slot, edits.get(index), prefix) + out.slice(slot.end);
  }
  return out;
}

// The pattern of an element of `name` in BOTH shapes a file may write it: a file
// may spell it `<name/>` or `<name>…</name>`, and a pattern keeping only one
// shape drops the other — the defect every hand-spelled copy of this rule has
// invited (an `<Override>` that survived a removal, a `<dimension>` that stopped
// being shifted, an old `<w:sz>` left beside the new one). `middle` is what the
// open tag holds between the name and the tag's own close (`>` or `/>`); the
// default states nothing but whatever attributes the tag carries. Built per call,
// so no `lastIndex` of a global pattern is carried from one use to the next. The
// two shapes are the shared spellings (`selfClosingTag`/`openTag`/`closeTag`), so
// this reader, `tagPattern`'s walk and `elementEdges`' own read cannot disagree
// about what a tag of `name` looks like.
const elementForms = (name, middle = TAG_MIDDLE) => new RegExp(`${selfClosingTag(name, middle)}|${openTag(name, middle)}[\\s\\S]*?${closeTag(name)}`, "g");
const elementPattern = (name) => elementForms(name);

// The FIRST `<name>` element of `xml`, as a match object with its own `index`:
// `elementPattern` is global, and `String.match` with a global pattern returns
// the matched TEXTS rather than a match with an index, so anything that needs
// the position of an element uses this. Only the first match is taken, so no
// caller materializes the rest of the part.
const firstElement = (xml, name) => xml.matchAll(elementPattern(name)).next().value;

// The `<name>` element a caller is about to write BESIDE — `firstElement`'s own
// answer, with a part that states the element's open tag and no close refused rather
// than read as absent: `elementForms` reads the closed shapes alone, and a second
// `<sheetData>`, `<cols>` or styles block written beside an unclosed one is a part a
// reader offers to repair, the one outcome an edit must not report as a success.
// `what` names the part the caller is writing into.
const writableElement = (xml, name, what) => {
  const found = firstElement(xml, name);
  if (found) return found;
  if (openedNotClosed(xml, name)) {
    throw new UsageError(`${what} holds <${name}> opened and never closed, so a second <${name}> cannot be written beside it`);
  }
  return undefined;
};

// ECMA-376's `EG_RPrBase`: the order a run's own `<w:rPr>` children must come in.
// Word drops a child written out of sequence, so a property appended at the end
// of the body would be reported as applied while the run renders as it did.
// `<w:rPrChange>` is not part of the sequence; the caller keeps it last. The body
// is flat — that one child, which holds others, is split off before a body reaches
// the writer — so `insertOrderedChild` writes each child as one complete element.
const RUN_PROPERTY_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"];

// The run-properties body with a bold/italic toggle set or removed. Setting one
// writes it at its place in the sequence, so a toggle added to a body that
// already holds a later property (`<w:sz>`, `<w:u>`) is not written where Word
// would drop it.
function runToggle(body, name, on) {
  const without = body.replace(elementPattern(`w:${name}`), "");
  return on ? insertOrderedChild(without, name, `<w:${name}/>`, RUN_PROPERTY_ORDER) : without;
}

// The `<w:rPr>` the format request needs, from the run's own: a boolean writes
// or drops a `<w:b>`/`<w:i>`, and `size` replaces the run's own `<w:sz>`/
// `<w:szCs>` — in either shape — with the half-point value, the unit `<w:sz>`
// counts in. Each property is written at its own place in `EG_RPrBase` (see
// `insertOrderedChild`). A `null` property counts as absent — the caller's own
// boundary treats a null-valued key that way — so only a stated one is written.
//
// A `<w:rPrChange>` holds the properties a tracked formatting revision recorded
// — the state before that revision, not the run's own formatting — so only the
// elements before it are read and rewritten: a revision keeps its own bytes (a
// request is never answered by changing one), and the properties this writes go
// in front of it, which is the one place `<w:rPr>` accepts it, `<w:rPrChange>`
// being the child that must come last. A request that leaves no property at all
// writes nothing: an empty `<w:rPr>` is a change that states nothing. A run whose
// own body the request emptied but which records a revision keeps that revision:
// it is content the request never named, and removing it would be the request
// answering itself by deleting a recorded change.
function formattedRunProperties(runXml, edit) {
  const current = runProperties(runXml);
  const inner = current ? current.replace(new RegExp(`^(?:${selfClosingTag("w:rPr")}|${openTag("w:rPr")})`), "").replace(new RegExp(closeTag("w:rPr") + "$"), "") : "";
  const revisionAt = inner.search(new RegExp(`${selfClosingTag("w:rPrChange")}|${openTag("w:rPrChange")}`));
  const revision = revisionAt < 0 ? undefined : elementSpan(inner, revisionAt, "w:rPrChange");
  let body = revision ? inner.slice(0, revision.start) : inner;
  if (edit.bold != null) body = runToggle(body, "b", edit.bold);
  if (edit.italic != null) body = runToggle(body, "i", edit.italic);
  if (edit.size != null) {
    const without = body.replace(elementPattern("w:sz"), "").replace(elementPattern("w:szCs"), "");
    body = insertOrderedChild(without, "szCs", `<w:szCs w:val="${edit.size}"/>`, RUN_PROPERTY_ORDER);
    body = insertOrderedChild(body, "sz", `<w:sz w:val="${edit.size}"/>`, RUN_PROPERTY_ORDER);
  }
  if (!body && !revision) return "";
  return `<w:rPr>${body}${revision ? inner.slice(revision.start) : ""}</w:rPr>`;
}

// `xml` with `edit`'s formatting applied to the runs a matched fragment covers.
// Run properties live on the run, not on a character, so a fragment covering
// part of a run formats that whole run — splitting a run to format half of it
// would rewrite bytes the preservation promise is about. Nothing outside those
// runs changes.
function formatRuns(xml, runs, runIndices, edit) {
  const pieces = [];
  for (const index of [...runIndices].sort((a, b) => a - b)) {
    const run = runs[index];
    const runXml = xml.slice(run.start, run.end);
    const open = runXml.match(new RegExp(openTag("w:r")))[0];
    const next = formattedRunProperties(runXml, edit);
    // A run's properties sit immediately after its open tag, before its content,
    // and are replaced by their own balanced span: a lazy regex would stop at the
    // close of the `<w:rPr>` a nested `<w:rPrChange>` holds, writing the toggle
    // into that nested element instead of the run's own.
    const lead = runXml.slice(open.length).length - runXml.slice(open.length).trimStart().length;
    const own = run.rpr ? elementSpan(runXml, open.length + lead, "w:rPr") : undefined;
    pieces.push({
      start: run.start,
      end: run.end,
      xml: own ? runXml.slice(0, own.start) + next + runXml.slice(own.end) : open + next + runXml.slice(open.length),
    });
  }
  pieces.sort((a, b) => b.start - a.start);
  let out = xml;
  for (const piece of pieces) out = out.slice(0, piece.start) + piece.xml + out.slice(piece.end);
  return out;
}

// The `<w:body>` position a paragraph added "at the end" goes to: before the
// body's OWN `<w:sectPr>` section properties when it has them, since those must
// stay the body's last element, else at the very end of the body. The first
// `<w:sectPr>` in the body is not necessarily the body's own — a section break
// sits inside a paragraph's `<w:pPr>` — so only a `<w:sectPr>` outside every
// paragraph counts, and the body's own is the last of those.
function bodyEnd(xml) {
  const body = xml.match(new RegExp(`${openTag("w:body")}([\\s\\S]*)${closeTag("w:body")}`));
  if (!body) throw new UsageError("the document has no body to add a paragraph to");
  // The content is the match's own middle, not a slice up to a close tag spelled
  // out again: a body whose close carries whitespace (`</w:body >`) is read by the
  // pattern above and by nothing else.
  const content = body[1];
  const open = body[0].indexOf(">") + 1;
  const paragraphs = docxParagraphs(content);
  const insideParagraph = (at) => paragraphs.some((span) => at >= span.start && at < span.end);
  let at = content.length;
  for (const match of content.matchAll(elementPattern("w:sectPr"))) {
    if (!insideParagraph(match.index)) at = match.index;
  }
  return body.index + open + at;
}

// `xml` with a paragraph holding `text` added: after the FIRST paragraph whose
// text holds `after`, or at the end of the body. The first match in part order
// is the anchor on purpose — the same rule the presentation's `add_paragraph`
// uses (see `addSlideParagraph`) — so a fragment that matches several paragraphs
// always lands the new one in the same place. The set searched is the document's
// OWN paragraphs (`topParagraphs`): a table cell's is one of them, a text box's
// is not — it is a paragraph nested inside another — and an `after` naming text
// there is refused with a message saying where the text lies rather than
// anchoring the new paragraph inside the box.
function addParagraph(xml, edit) {
  const text = editText(edit.text, "text");
  const paragraph = `<w:p><w:r><w:t xml:space="preserve">${xmlEscape(text)}</w:t></w:r></w:p>`;
  if (edit.after === undefined) {
    const at = bodyEnd(xml);
    return xml.slice(0, at) + paragraph + xml.slice(at);
  }
  const after = editText(edit.after, "after");
  if (!after) throw new UsageError("after must not be empty");
  const regions = fieldRegions(xml);
  const textOf = (candidate) => {
    const { slots } = docxRuns(xml.slice(candidate.start, candidate.end), clipSpans(regions, candidate.start, candidate.end));
    return slots.map((slot) => xmlUnescape(slot.raw)).join("");
  };
  const span = topParagraphs(xml).find((candidate) => textOf(candidate).includes(after));
  if (!span) {
    // The fragment may still be text this document holds and the text ops can
    // edit — a text box's own paragraph — so the refusal says where it is
    // instead of the body's "not searched" wording, which reads as "not in the
    // file".
    const buried = docxParagraphs(xml).find((candidate) => textOf(candidate).includes(after));
    if (buried) throw new UsageError(`the text ${JSON.stringify(after)} is inside a text box: an add_paragraph after anchors in the document's own paragraphs (a table cell's counts), and a text box's paragraph is a nested one`);
    throw new UsageError(missingFind(after));
  }
  return xml.slice(0, span.end) + paragraph + xml.slice(span.end);
}

// `element` with every paragraph — itself or a nested one — whose joined text
// holds `find` removed, `null` when the whole element goes. A nested paragraph
// is removed on its own: it is only the paragraph the text is in that goes, not
// the one that draws the text box around it. `excluded` are the field regions
// that reach into the element, relative to it. A nested edit rewrites bytes, so
// the field regions and the element's own runs are located only after it —
// exactly as `editParagraph` does.
function removeParagraph(element, excluded, find, found) {
  const children = childParagraphSpans(element);
  let out = element;
  let regions = excluded;
  for (let i = children.length - 1; i >= 0; i -= 1) {
    const [start, end] = children[i];
    const before = out.slice(start, end);
    const removed = removeParagraph(before, clipSpans(regions, start, end), find, found);
    if (removed === null) out = out.slice(0, start) + out.slice(end);
    else if (removed !== before) out = out.slice(0, start) + removed + out.slice(end);
    const delta = (removed === null ? 0 : removed.length) - before.length;
    if (delta) regions = regions.map(([from, to]) => (from >= end ? [from + delta, to + delta] : [from, to]));
  }
  if (!addressedText(out, regions).includes(find)) return out;
  found.value = true;
  return null;
}

// `xml` with the top-level paragraphs put through `removeParagraph`, from the
// last to the first so an earlier one's byte positions stay valid. The result is
// assembled as the pieces of the part around the removed paragraphs and joined
// once, the way `mapParagraphs` does.
function removeParagraphs(xml, find, found) {
  // A container a paragraph is removed from must keep a paragraph — ECMA-376
  // requires one in a `<w:tc>` and in a `<w:txbxContent>` — and the removal is
  // refused rather than written out as a part no reader may hold. The check runs
  // over the whole part once, before anything goes.
  requireParagraphLeft(xml, find);
  const regions = fieldRegions(xml);
  const tops = topParagraphs(xml);
  const pieces = [];
  let end = xml.length;
  for (let i = tops.length - 1; i >= 0; i -= 1) {
    const top = tops[i];
    const element = xml.slice(top.start, top.end);
    const removed = removeParagraph(element, clipSpans(regions, top.start, top.end), find, found);
    pieces.push(xml.slice(top.end, end));
    if (removed !== null) pieces.push(removed);
    end = top.start;
  }
  pieces.push(xml.slice(0, end));
  return pieces.reverse().join("");
}

// One docx edit applied to `xml`: the edit addresses the joined text of a
// paragraph's runs, and every occurrence of `find` is rewritten.
function applyDocxEdit(xml, edit, note) {
  const op = edit && edit.op;
  if (op === "add_paragraph") return addParagraph(xml, edit);
  const find = editText(edit && edit.find, "find");
  if (!find) throw new UsageError("find must not be empty");
  if (op === "remove_paragraph") {
    const found = { value: false };
    const out = removeParagraphs(xml, find, found);
    if (!found.value) throw new UsageError(missingFind(find));
    return out;
  }
  if (!["replace_text", "insert_text", "remove_text", "format_text"].includes(op)) throw new UsageError(`unknown docx edit: ${JSON.stringify(op)}`);
  const format = { bold: edit.bold, italic: edit.italic };
  if (op === "format_text") {
    if (edit.bold == null && edit.italic == null && edit.size == null) throw new UsageError("format_text needs at least one of bold, italic or size");
    if (edit.size != null) {
      if (!inSpan(edit.size, RULES.text_size_points)) throw new UsageError(`size must be ${spanBounds(RULES.text_size_points)} points, got: ${JSON.stringify(edit.size)}`);
      // `<w:sz>` counts half-points, and the shared bound's ends are the one and
      // the 1638 half-points a `<w:sz>` can hold, so rounding a size the rule
      // allows never reaches the zero a reader cannot draw.
      format.size = String(Math.round(edit.size * 2));
    }
  }
  let replacement = "";
  if (op === "replace_text") replacement = editText(edit.replace, "replace");
  else if (op === "insert_text") {
    const insert = editText(edit.insert, "insert");
    if (!insert) throw new UsageError("insert must not be empty");
    const position = edit.position === undefined ? "after" : edit.position;
    if (position !== "after" && position !== "before") throw new UsageError(`position must be "after" or "before", got: ${JSON.stringify(position)}`);
    replacement = position === "before" ? insert + find : find + insert;
  }
  let inserted = false;
  const out = mapParagraphs(xml, (element, excluded) => {
    const { runs, slots } = docxRuns(element, excluded);
    const text = slots.map((slot) => xmlUnescape(slot.raw)).join("");
    const offsets = occurrences(text, find);
    if (!offsets.length) return null;
    inserted = true;
    if (op === "format_text") {
      // The runs the matched fragments cover, found once and formatted together:
      // a later edit could not move an earlier one's run.
      const runIndices = new Set();
      for (const span of offsets) {
        const first = locateSlot(slots, span.start);
        const last = locateSlot(slots, span.end, true);
        for (let i = first.index; i <= last.index; i += 1) runIndices.add(slots[i].run);
      }
      return formatRuns(element, runs, runIndices, format);
    }
    // Every occurrence is written in one pass over the same runs and slots, so
    // no occurrence's offsets can go stale against another's.
    return replaceOccurrences(element, runs, slots, offsets, replacement, note, "w:");
  });
  if (!inserted) throw new UsageError(missingFind(find));
  return out;
}

function docxEdit(req) {
  const editor = openEdit(req, "docx");
  const edits = editList(req);
  editor.part("word/document.xml", (xml) => {
    for (const edit of edits) {
      // Each edit walks the whole body text, and what it walks is a part it does
      // not open through `part` — so it charges its own walk, and the budget
      // covers a long edit list over a large body rather than one document
      // length.
      editor.charge(xml.length);
      xml = applyDocxEdit(xml, edit, (message) => editor.note(message));
    }
    return xml;
  });
  return editor.finish();
}

// ── xlsx_edit ──────────────────────────────────────────────────
// A cell's address is its own `r` or, when a writer left it out, its position in
// the row — the rule the reader and the filler already use.
const ROW_ELEMENT = elementPattern("row");
const CELL_ELEMENT = elementPattern("c");
const styleOf = (cell) => (cell.match(/\bs="(\d+)"/) || [])[1];

// Whether a part opens a `<name>` element and never closes it. `elementPattern` reads
// the closed shapes alone, so such an element is invisible to every reader here, and a
// writer places its content relative to what it cannot see — a second `<row>` beside
// the one it missed, a `<c>` nested inside it. A sheet's rows, and a row's cells, hold
// no elements of their own kind, so the tag spellings are counted rather than parsed —
// the shared ones (`tagPattern`), so this test and the walk agree about every shape.
const openedNotClosed = (xml, name) => {
  let opens = 0;
  let closes = 0;
  for (const tag of xml.matchAll(tagPattern(name))) {
    if (tag[0].startsWith("</")) closes += 1;
    else if (!tag[0].endsWith("/>")) opens += 1;
  }
  return opens > closes;
};

// `items` in ascending order of the number `at` reads off one: a list already
// ascending is returned as it stands — so a caller can keep the text between the
// items of a part it does not reorder — while one no writer produces is put in
// address order, the way a reader expects them rather than the way they arrived. An
// item that states no address of its own IS the position it sits in, so such a list
// is refused (`where` names it) rather than silently relocated.
const ascending = (items, at, where) => {
  if (items.every((item, index) => index === 0 || at(item) > at(items[index - 1]))) return items;
  if (items.some((item) => !item.stated)) throw new UsageError(`${where} are not in address order and one of them states no address of its own, so reordering them would move what it holds`);
  return [...items].sort((a, b) => at(a) - at(b));
};

// The `<row>` elements of a sheet, each with the number it holds, its span and whether
// it states that number itself: a row that left `r` out stands for the one after its
// predecessor.
function sheetRows(xml) {
  if (openedNotClosed(xml, "row")) throw new UsageError("the sheet part holds a <row> opened and never closed, so its rows cannot be read");
  const rows = [];
  let number = 0;
  for (const match of xml.matchAll(ROW_ELEMENT)) {
    const attribute = (match[0].match(/\br="(\d+)"/) || [])[1];
    number = attribute ? Number(attribute) : number + 1;
    rows.push({ start: match.index, end: match.index + match[0].length, number, stated: attribute !== undefined, xml: match[0] });
  }
  return rows;
}

// The `<c>` elements of one row, each with the zero-based column it addresses and
// whether it states that address itself: a cell that left `r` out stands for the
// position it sits in.
function rowCells(rowXml) {
  if (openedNotClosed(rowXml, "c")) throw new UsageError("the sheet part holds a <c> opened and never closed, so its cells cannot be read");
  const cells = [];
  let column = 0;
  for (const match of rowXml.matchAll(CELL_ELEMENT)) {
    const attribute = (match[0].match(/\br="([A-Za-z]+\d+)"/) || [])[1];
    const index = attribute ? columnOf(attribute) : column;
    column = index + 1;
    cells.push({ start: match.index, end: match.index + match[0].length, column: index, address: attribute, stated: attribute !== undefined, xml: match[0] });
  }
  return cells;
}

// The largest row number and column the sheet's cells use, for the reach of a
// row or column operation.
function usedRange(xml) {
  let rowMax = 0;
  let columnMax = 0;
  for (const row of sheetRows(xml)) {
    rowMax = Math.max(rowMax, row.number);
    for (const cell of rowCells(row.xml)) columnMax = Math.max(columnMax, cell.column + 1);
  }
  return { rowMax, columnMax };
}

// How many elements a sheet part holds at each address, or `undefined` for a part
// whose rows or cells a reader refuses (see `openedNotClosed`) — a reading with
// nothing to compare against is skipped rather than turned into a refusal an edit
// that never needed the rows (a column width) does not owe. An address is keyed by
// itself — a row as `#3`, a cell by its reference folded to upper case, so two
// spellings of one address are one key — and an element that states no `r` of its own
// counts at the position `sheetRows`/`rowCells` resolve for it, the format making `r`
// optional and inferred. A count above one is two elements claiming one address.
const rowKey = (number) => `#${number}`;
function addressCounts(xml) {
  const counts = new Map();
  const bump = (key) => counts.set(key, (counts.get(key) ?? 0) + 1);
  try {
    for (const row of sheetRows(xml)) {
      bump(rowKey(row.number));
      for (const cell of rowCells(row.xml)) bump((cell.address ?? `${colName(cell.column)}${row.number}`).toUpperCase());
    }
  } catch (error) {
    if (error instanceof UsageError) return undefined;
    throw error;
  }
  return counts;
}

// The words an answer names one of those addresses by.
const addressLabel = (key) => (key.startsWith("#") ? `the row ${key.slice(1)}` : `the cell ${key}`);

// The first address `after` holds more elements at than `before` did, or `undefined`
// when there is none to tell. A repeat counts only where the part the edit was built
// from held fewer there, so a repeat that part arrived with is passed over where it
// stands and refused only once an edit carries it onto an address the part held one
// element at; a part whose rows cannot be read is not refused for the check's own
// sake.
function addedRepeat(before, after) {
  const was = addressCounts(before);
  if (!was) return undefined;
  for (const [address, count] of addressCounts(after) ?? []) {
    if (count > 1 && count > (was.get(address) ?? 0)) return address;
  }
  return undefined;
}

// A sheet's `<col>` entries in the file's own order: the text each is spelled with,
// its 1-based `min`/`max` span (`NaN` for one a reader cannot place, which therefore
// covers no column) and the `style` it states. ONE reading of the block for the three
// callers that each had their own copy of it — the style a cell inherits
// (`columnStyleEntries`), the entry a `format_cells` covers (`formatColumn`) and the
// entry it splits (`splitColumn`) — and every entry's own text is kept, so a block
// rewritten from these keeps the entries it does not name byte for byte. `[]` for a
// sheet with no `<cols>` block at all.
const colEntries = (sheet) => {
  const cols = firstElement(sheet, "cols");
  if (!cols) return [];
  return [...cols[0].matchAll(elementPattern("col"))].map((match) => ({
    text: match[0],
    min: Number((match[0].match(/\bmin="(\d+)"/) || [])[1]),
    max: Number((match[0].match(/\bmax="(\d+)"/) || [])[1]),
    style: (match[0].match(/\bstyle="(\d+)"/) || [])[1],
  }));
};

// The workbook's sheets in its own order, each `<sheet>` element's name beside the
// part it addresses: the target of the relationship its `r:id` names, resolved by
// the one OPC resolver `resolvePart` — an absolute `/xl/worksheets/sheet1.xml`, a
// relative `worksheets/sheet1.xml` and a target spelled with `..` all name the part
// the resolver folds them to — and, for a relationship the workbook does not
// declare, the conventional part name for that position, the same fallback the
// reader makes. `named` says which of the two it was. Both parts are read through
// the editor (`read`), so the workbook and its relationships are charged like every
// other part an edit takes into hand: reading them is work a check can spend the
// call's budget on, and a read that skipped the charge would be a hole in the bound
// the caller is refused by.
function workbookSheets(editor) {
  const workbook = editor.read(XLSX_WORKBOOK);
  if (workbook === undefined) return [];
  const targets = relationshipMap(editor.read(XLSX_WORKBOOK_RELS) ?? "");
  return [...workbook.matchAll(new RegExp(`${selfClosingTag("sheet")}|${openTag("sheet")}`, "g"))].map((match, index) => {
    const id = xmlAttribute(match[0], "r:id");
    const target = id === undefined ? undefined : targets.get(id);
    return {
      name: xmlUnescape(xmlAttribute(match[0], "name") ?? ""),
      part: target ? resolvePart("xl/", target) : `xl/worksheets/sheet${index + 1}.xml`,
      named: target !== undefined,
    };
  });
}

// The part a sheet NAME addresses, through `xl/workbook.xml` and its
// relationships. The name is matched the way Excel matches one — ignoring case,
// in the workbook's own order — while a refusal keeps the file's spelling. The
// part's own presence is checked on the archive (`zip.file`), which decompresses
// nothing; the sheet's text is the caller's own read, charged once there.
function sheetPart(editor, name) {
  const sheets = workbookSheets(editor);
  const wanted = typeof name === "string" ? name.toLowerCase() : null;
  const index = wanted === null ? -1 : sheets.findIndex((sheet) => sheet.name.toLowerCase() === wanted);
  if (index < 0) throw new UsageError(`there is no sheet named ${JSON.stringify(name)}: this workbook has ${listed(sheets.map((sheet) => sheet.name)) || "no sheets"}`);
  const { part, named } = sheets[index];
  if (!editor.zip.file(part)) {
    throw new UsageError(named
      ? `the sheet ${JSON.stringify(sheets[index].name)} names the part ${part}, which the package does not have`
      : `the sheet ${JSON.stringify(sheets[index].name)} has no worksheet part: the workbook's relationships do not name one and the package has no ${part}`);
  }
  return { part, name: sheets[index].name };
}

// The row and zero-based column an A1 address names, refusing a reference to a
// cell no workbook's grid holds.
function cellAddress(value, what = "cell") {
  const match = typeof value === "string" ? value.match(/^([A-Za-z]+)(\d+)$/) : null;
  if (!match) throw new UsageError(`${what} must be an A1 address like B7, got: ${JSON.stringify(value)}`);
  const row = Number(match[2]);
  if (row < 1 || row > RULES.sheet_row_max) throw new UsageError(`${what} ${value}: the row must be between 1 and ${RULES.sheet_row_max}`);
  const column = columnOf(match[1]);
  if (column + 1 > RULES.sheet_column_max) throw new UsageError(`${what} ${value}: the column must be between A and ${colName(RULES.sheet_column_max - 1)}`);
  return { row, column };
}

function rowNumber(value) {
  if (!Number.isInteger(value) || value < 1 || value > RULES.sheet_row_max) throw new UsageError(`row must be a whole number between 1 and ${RULES.sheet_row_max}, got: ${JSON.stringify(value)}`);
  return value;
}

// The 1-based column a caller's own `column` field names: one to three UPPERCASE
// letters, exactly the shape the Rust boundary's `column_number` accepts, so the
// kit cannot take a spelling the boundary would already have refused. The
// cell-address reader (`cellAddress`) stays tolerant of either case because it
// also reads the refs a file states of its own — a `<mergeCell ref>`, an
// `<autoFilter ref>` — where a lowercase letter is a writer's spelling rather
// than a request this kit gets to judge.
function columnNumber(value) {
  if (typeof value !== "string" || !/^[A-Z]{1,3}$/.test(value) || columnOf(value) + 1 > RULES.sheet_column_max) {
    throw new UsageError(`column must be column letters between A and ${colName(RULES.sheet_column_max - 1)}, got: ${JSON.stringify(value)}`);
  }
  return columnOf(value) + 1;
}

// The `<c>` a value makes at `reference`, with the style a `number_format` chose
// or the cell's own. Every attribute but the style is dropped: a `cm`/`vm`
// metadata reference names a value the cell no longer holds and the reader
// rebuilds it, so keeping it would point at nothing.
function valueCell(reference, value, style) {
  const attribute = style === null || style === undefined ? "" : ` s="${style}"`;
  if (value && typeof value === "object") return `<c r="${reference}"${attribute}><f>${xmlEscape(formulaOf(reference, value))}</f></c>`;
  // The scalar rule the tool's boundary states, as the kit's own last line.
  const scalar = scalarOf(value, `cell ${reference}`);
  if (typeof scalar === "number") return `<c r="${reference}"${attribute}><v>${scalar}</v></c>`;
  if (typeof scalar === "boolean") return `<c r="${reference}"${attribute} t="b"><v>${scalar ? 1 : 0}</v></c>`;
  return `<c r="${reference}"${attribute} t="inlineStr"><is><t xml:space="preserve">${xmlEscape(scalar)}</t></is></c>`;
}

// The blocks of a `<styleSheet>` an edit reads or grows, and the element name one
// entry of each takes: they are read and grown by one rule rather than one per
// block. Each entry is matched in either shape — an `<xf>`, say, may hold an
// `<alignment>` or a `<protection>` (both legal `CT_Xf`), and a matcher that kept
// only the self-closing ones would drop the entry and shift every cell's `s=`
// index. `<cellStyleXfs>` is where an id a cell xf omits resolves from and
// `<cellStyles>` where a workbook's named styles live: both are read wherever the
// part states them, and written with the default entry only when it states none —
// a cell xf names a style in `<cellStyleXfs>` and a workbook states its default
// style in `<cellStyles>`, so a part grown without them would point at blocks it
// does not hold (see `STYLE_REFERENCES`).
const STYLE_BLOCKS = { numFmts: "numFmt", fonts: "font", fills: "fill", borders: "border", cellXfs: "xf", cellStyleXfs: "xf", cellStyles: "cellStyle" };

// `CT_Stylesheet`'s child order. A block the part lacks is created at its own
// place in it: a `<fonts>` written after `<cellXfs>`, say, is a styles part a
// reader may drop.
const STYLESHEET_ORDER = ["numFmts", "fonts", "fills", "borders", "cellStyleXfs", "cellXfs", "cellStyles", "dxfs", "tableStyles", "colors", "extLst"];
const stylesheetRank = (name) => {
  const rank = STYLESHEET_ORDER.indexOf(name);
  return rank < 0 ? STYLESHEET_ORDER.length : rank;
};

// The neutral `<xf>` the two xf blocks' index-0 entries are built from: it names
// every format as 0 — no number format, font, fill or border of its own — so it
// restyles nothing. The cell xf adds the `xfId="0"` that points it at cell style
// 0; the cell-style xf states none, since it is the style an id resolves from.
const NEUTRAL_XF = '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"';

// The entry each `<styleSheet>` block states when it holds one, and the block a
// part states with it: ONE statement of the defaults, read by `MINIMAL_STYLES`
// (the part a workbook this kit creates gets) and by the entries a part that never
// stated a block is grown with (`putStyleEntry`), so a created workbook and a grown
// part state the same font 0, fill 0 and `<xf>` 0.
const STYLE_PART_ENTRIES = {
  fonts: '<font><sz val="11"/><name val="Calibri"/></font>',
  fills: '<fill><patternFill patternType="none"/></fill>',
  borders: "<border/>",
  cellStyleXfs: `${NEUTRAL_XF}/>`,
  cellXfs: `${NEUTRAL_XF} xfId="0"/>`,
  cellStyles: '<cellStyle name="Normal" xfId="0" builtinId="0"/>',
};
// The block a styles part states with one entry in it.
const styleBlock = (block) => `<${block} count="1">${STYLE_PART_ENTRIES[block]}</${block}>`;

// The styles part a workbook this kit creates is written with, in `CT_Stylesheet`'s own
// order: one font, the "none" fill, no border and the `<cellStyleXfs>`/`<cellXfs>` pair
// every `<styleSheet>` states — the blocks and entries `STYLE_PART_ENTRIES` states once,
// so a created workbook and a part an edit grows state the same font 0, fill 0 and `<xf>`
// 0. The default style `<cellStyles>` holds is not among them: creating a workbook has
// never written one (this op leaves creation as it was), and a part grown for a style
// entry is given the blocks that entry's ids resolve into (`STYLE_REFERENCES`),
// `<cellStyles>` included.
const MINIMAL_STYLES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">${Object.keys(STYLE_PART_ENTRIES).filter((block) => block !== "cellStyles").sort((a, b) => stylesheetRank(a) - stylesheetRank(b)).map(styleBlock).join("")}</styleSheet>`;

// The `<styleSheet>` root of a styles part, in the one reading of an element's shape
// (`elementEdges`), and `undefined` for a part whose root is missing or never closed —
// `firstElement` reads closed elements alone — so an opening check and a writer reading
// through this cannot disagree about what "writable" means: a part spelled
// `</styleSheet >` is sound, and a writer matching the close tag literally would write
// past it and report a success over a part that never took the child.
function stylesRoot(styles) {
  const found = firstElement(styles, "styleSheet");
  if (!found) return undefined;
  const edges = elementEdges(found[0], "styleSheet");
  return { open: edges.open, selfClosing: edges.selfClosing, close: found.index + edges.close };
}

// `xml` with a whole `<name>` block inserted where `CT_Stylesheet` puts it: after every
// block the schema orders before it and before the first it orders after. The scan is
// over the schema's own block names, since an element scan would stop on an entry
// nested in a block — a `<numFmt>` inside `<numFmts>` — instead of on the block itself.
// A block written here is one its caller read through `writableElement` and found
// absent, so a part that states its open tag without its close is already refused; a
// self-closing root is opened around the element and a closed one takes it before its
// own close.
function insertStylesChild(xml, name, element) {
  const root = stylesRoot(xml);
  // An internal invariant rather than a caller's fault: `requireStylesRoot` refuses
  // a part without a root before any edit runs.
  if (!root) throw new Error(`there is no <styleSheet> root to write a <${name}> block into`);
  if (root.selfClosing) return xml.slice(0, root.close - 2) + `>${element}</styleSheet>` + xml.slice(root.close);
  const rank = stylesheetRank(name);
  for (const later of STYLESHEET_ORDER.slice(rank + 1)) {
    const found = firstElement(xml, later);
    if (found) return xml.slice(0, found.index) + element + xml.slice(found.index);
  }
  return xml.slice(0, root.close) + element + xml.slice(root.close);
}

// The entries of a styles part's block — `<fonts>`, `<fills>`, `<borders>`,
// `<cellXfs>`, `<cellStyleXfs>`, `<cellStyles>` — in the file's own order; `[]`
// when the part has no such block at all. `putStyleEntry` is what adds one.
function styleBlockEntries(styles, block) {
  // A block the table states no entry element for is a rule this kit forgot, not
  // a part with no entries: `elementPattern(undefined)` matches nothing, so the
  // caller reads "this block states no entry" and writes its default over every
  // byte the block really held. That is how a part whose `<cellStyles>` stated
  // three named styles was replaced by the one default style.
  if (!STYLE_BLOCKS[block]) throw new Error(`STYLE_BLOCKS states no entry element for the ${block} block`);
  const found = firstElement(styles, block);
  return found ? [...found[0].matchAll(elementPattern(STYLE_BLOCKS[block]))].map((match) => match[0]) : [];
}

// The blocks a style entry's ids resolve into, and the block a workbook states
// its default style in. The entries this kit writes carry ids of their own: the
// index-0 entry `STYLE_PART_ENTRIES` states names font 0, fill 0, border 0 and
// style 0, and an `<xf>` omits what it does not state, which means id 0 too — so
// a part that states none of those blocks would hold entries naming an id nothing
// resolves, the dangling reference a reader refuses the whole styles part over.
const STYLE_REFERENCES = { cellStyleXfs: ["fonts", "fills", "borders"], cellXfs: ["fonts", "fills", "borders", "cellStyleXfs", "cellStyles"] };

// `styles` with `name` holding at least one entry, so the ids an entry names
// resolve. A block the part does not state at all is written at its own
// `CT_Stylesheet` place (`insertStylesChild`); a block the part states holding no
// entry — a `<fills count="0"/>`, the shape a stripped workbook leaves — is
// written over the block element itself, since there is no body to append an entry
// to. A block that already states one keeps its own bytes.
function ensureStyleBlock(styles, name) {
  const found = writableElement(styles, name, "the styles part");
  if (found && styleBlockEntries(styles, name).length) return styles;
  return found ? styles.replace(found[0], () => styleBlock(name)) : insertStylesChild(styles, name, styleBlock(name));
}

// `styles` with every block `STYLE_REFERENCES` says `block`'s entries resolve into
// grown, and each of those grown in turn: a `cellXfs` brings its `cellStyleXfs`,
// whose own entry needs fonts, fills and borders. A no-op for a block that resolves
// nothing — `fonts`, `fills`, `borders`, `numFmts` — so it runs unconditionally.
function ensureStyleBlocks(styles, block) {
  return (STYLE_REFERENCES[block] ?? []).reduce(
    (out, name) => ensureStyleBlocks(ensureStyleBlock(out, name), name),
    styles,
  );
}

// The index `entry` takes in `block`'s list: an identical entry already there is reused
// rather than added twice, else `entry` is appended and the block's `count` recomputed.
// Entries keep their raw spelling, so adding one never rewrites the others. A block with
// no entries takes the index-0 entry `STYLE_PART_ENTRIES` states first and `entry` at
// index 1, so the entry never lands on the slot that restyles every cell naming no style
// — and the blocks its ids resolve into are ensured first (`ensureStyleBlocks`).
function putStyleEntry(styles, block, entry) {
  const found = writableElement(styles, block, "the styles part");
  const entries = found ? styleBlockEntries(styles, block) : [];
  const existing = entries.indexOf(entry);
  if (existing >= 0) return { styles, index: existing };
  const written = entries.length ? [...entries, entry] : [STYLE_PART_ENTRIES[block], entry];
  const body = written.join("");
  const index = written.length - 1;
  // The referenced blocks go into the text this call inserts into, so the block
  // lands after them; the string match below still finds the element itself.
  const base = ensureStyleBlocks(styles, block);
  if (!found) return { styles: insertStylesChild(base, block, `<${block} count="${written.length}">${body}</${block}>`), index };
  // The block's own open tag is kept — a `<cellXfs>`'s attributes, whatever it
  // spells — and read through `elementParts`, so a close tag spelled with the
  // whitespace XML allows does not make a closed block read as self-closing.
  const { open } = elementParts(found[0], block);
  return { styles: base.replace(found[0], () => `${setXmlAttribute(open, "count", String(written.length))}${body}</${block}>`), index };
}

// The open-tag/body/close shape of an element a part states: its own tag byte for byte
// — a handler that rewrites a child keeps a `<border diagonalUp="1">` or a `<font>`'s
// own attributes, which a rebuild from the tag name alone would drop — where its body
// starts, whether it is self-closing and where its close tag starts. ONE reading of
// the shape, shared by the element handlers (`elementParts`), the styles root
// (`stylesRoot`) and a sheet's body (`sheetBody`). The close tag is read with the
// whitespace XML allows before its `>` (`</name >`), the tolerance `elementForms`
// states: a literal `</name>` test reads a sound element as self-closing and splices
// its own open tag's close into the middle of it. This assumes `element` starts at the
// element, which every caller's own match does. `close` is `undefined` for an element
// a part opens and never closes.
function elementEdges(element, name) {
  // The open-tag pattern reads a self-closing tag too, whether or not the tag states
  // an attribute before its `/` (`<border/>`): such a tag is the self-closing form and
  // has no body.
  const open = element.match(new RegExp(`^(?:${selfClosingTag(name)}|${openTag(name)})`))[0];
  const bodyStart = open.length;
  if (open.endsWith("/>")) return { open, bodyStart, selfClosing: true, close: bodyStart };
  const close = element.slice(bodyStart).match(new RegExp(closeTag(name)));
  return { open, bodyStart, selfClosing: false, close: close ? bodyStart + close.index : undefined };
}

// An element's open tag — the self-closing form opened, so a caller can write children
// into it — and the text of its body, `""` for a self-closing element.
function elementParts(element, name) {
  const edges = elementEdges(element, name);
  return edges.selfClosing
    ? { open: `${edges.open.slice(0, -2)}>`, body: "" }
    : { open: edges.open, body: element.slice(edges.bodyStart, edges.close) };
}

// `element` with `name="value"` set on its OWN open tag — the text up to and
// including the first `>` — or, when it states no such attribute, ` name="value"`
// inserted just before that tag's close, so a self-closing tag stays
// self-closing; `null` removes the attribute instead. Only the open tag is
// touched, so an attribute of the same name on a child element is the child's own
// and the text around the tag is byte-identical. A function replacement
// throughout, so a `$` in `value` is never read as a substitution pattern.
function setXmlAttribute(element, name, value) {
  const close = element.indexOf(">");
  const open = element.slice(0, close + 1);
  const pattern = new RegExp(`\\s${name}="[^"]*"`);
  let written;
  if (value === null) written = open.replace(pattern, "");
  else if (pattern.test(open)) written = open.replace(pattern, () => ` ${name}="${value}"`);
  else written = open.replace(/(\s*\/?>)$/, (end) => ` ${name}="${value}"${end}`);
  return written + element.slice(close + 1);
}

// ECMA-376's `EG_Font` order: a `<font>`'s children must come in it, or a reader
// may drop one written out of sequence, and a property appended at the end would
// be reported as applied while the font renders as it did.
const FONT_PROPERTY_ORDER = ["b", "i", "strike", "condense", "extend", "outline", "shadow", "u", "vertAlign", "sz", "color", "name", "family", "charset", "scheme"];

// An element whose name the pattern itself captures: the self-closing shape
// first, then the open/close shape whose close carries the name back. The readers
// that walk children whose names are not known ahead build on this, so such a
// child is read with the same tag spellings every other reader here uses.
const ANY_ELEMENT = new RegExp(
  `${selfClosingTag("([A-Za-z][A-Za-z0-9:]*)")}|${openTag("([A-Za-z][A-Za-z0-9:]*)")}[\\s\\S]*?${closeTag("\\2")}`,
  "g",
);

// `body` with `element` written where `order` puts it: after every child the
// sequence puts before it and before the first it puts after it. A body is flat —
// the children of a `<font>` and the sides of a `<border>` hold no elements of
// their own, and the docx arm's `<w:rPr>` one that does is split off before the
// body arrives — so each child is one complete element. The docx arm's order is
// `EG_RPrBase` (`RUN_PROPERTY_ORDER`); a child is ranked by its own local name,
// since such a body spells its children with a namespace prefix (`<w:sz>`).
function insertOrderedChild(body, name, element, order) {
  const rank = (found) => {
    const at = order.indexOf(found.slice(found.indexOf(":") + 1));
    return at < 0 ? order.length : at;
  };
  const placed = rank(name);
  for (const match of body.matchAll(ANY_ELEMENT)) {
    if (rank(match[1] ?? match[2]) > placed) return body.slice(0, match.index) + element + body.slice(match.index);
  }
  return body + element;
}

// The workbook's own `<numFmts>` id for `formatCode`: an existing `<numFmt>` with
// the same code is reused, else the id after the highest in use — at least the
// first custom id, 164 — is allocated and its `<numFmts>` block written back. The
// lookup is the block's own entries, not every `<numFmt>` in the part: a
// `<numFmt>` a `<dxf>` holds is not a member of the workbook's table, and reading
// one would reuse its id for a cell format the block never states. A built-in id
// the file does not spell out resolves to whatever code the reader's locale gives
// it, so it is not guessed at, and a format code is compared as text.
function numberFormatId(styles, formatCode) {
  const length = [...formatCode].length;
  if (length > RULES.number_format_max) throw new UsageError(`a number format must be at most ${RULES.number_format_max} characters, got: ${length}`);
  const entries = styleBlockEntries(styles, "numFmts");
  const formats = new Map();
  for (const entry of entries) {
    const id = Number((entry.match(/\bnumFmtId="(\d+)"/) || [])[1]);
    const code = (entry.match(/\bformatCode="([^"]*)"/) || [])[1];
    if (Number.isFinite(id) && code !== undefined) formats.set(id, xmlUnescape(code));
  }
  let edited = styles;
  let id = [...formats.entries()].find(([, code]) => code === formatCode)?.[0];
  if (id === undefined) {
    id = Math.max(163, ...formats.keys()) + 1;
    // The entries keep their raw spelling, so a format the file already had is
    // not rewritten by this one being added. `insertStylesChild` writes the block
    // at its own `CT_Stylesheet` place — `<numFmts>` directly after
    // `<styleSheet>`, before `<fonts>` — and opens a self-closing root so the
    // block lands inside it rather than beside a second top-level element.
    // Everything rewritten in is the file's own text or the caller's code, so a
    // function replacement is used: a `$` in either would otherwise be a
    // substitution pattern (`$&`, `$$`, `` $` ``, `$'`) and corrupt the part.
    const block = `<numFmts count="${entries.length + 1}">${entries.join("")}<numFmt numFmtId="${id}" formatCode="${xmlEscape(formatCode)}"/></numFmts>`;
    const numFmts = writableElement(styles, "numFmts", "the styles part");
    edited = numFmts
      ? edited.replace(numFmts[0], () => block)
      : insertStylesChild(edited, "numFmts", block);
  }
  return { styles: edited, id };
}

// The `<font>` a request needs, from the cell's own: a toggle writes or drops
// `<b/>`/`<i/>`, `size` replaces `<sz>`, `color` replaces `<color rgb="FF……"/>`
// (the digits uppercased under a full alpha, the spelling Excel writes) and `font`
// replaces `<name>` — and takes the font out of the theme's scheme with it: a
// `<scheme val="minor"/>` beside a `<name>` says the face is the one the workbook's
// THEME states, so a requested face kept beside it would still render as the
// theme's and be rewritten by a theme change (the scheme element is what tells a
// reader the font IS the theme's — ISO/IEC 29500 §18.8.35). `<family>`, `<charset>`
// and anything the request did not name keep their own bytes, `<scheme>` among them
// when the face is not named; a child written back goes at its place in `EG_Font`
// (`insertOrderedChild`), so it is not written where a reader would drop it.
function formatFont(font, edit) {
  const { open, body } = elementParts(font, "font");
  let out = body;
  for (const [name, toggle] of [["b", "bold"], ["i", "italic"]]) {
    if (edit[toggle] === undefined) continue;
    const without = out.replace(elementPattern(name), "");
    out = edit[toggle] ? insertOrderedChild(without, name, `<${name}/>`, FONT_PROPERTY_ORDER) : without;
  }
  if (edit.size !== undefined) out = replaceFontChild(out, "sz", `<sz val="${edit.size}"/>`);
  if (edit.color !== undefined) out = replaceFontChild(out, "color", `<color rgb="FF${edit.color}"/>`);
  if (edit.font !== undefined) {
    out = replaceFontChild(out.replace(elementPattern("scheme"), ""), "name", `<name val="${xmlEscape(edit.font)}"/>`);
  }
  return `${open}${out}</font>`;
}

// `body` with the `<name>` child it states replaced by `element`, or `element`
// written at that child's place in `EG_Font` when it states none.
function replaceFontChild(body, name, element) {
  const held = firstElement(body, name);
  if (held) return body.slice(0, held.index) + element + body.slice(held.index + held[0].length);
  return insertOrderedChild(body, name, element, FONT_PROPERTY_ORDER);
}

// `CT_Border`'s own child order: `start`/`end` are the table-border sides a
// reader would drop a child written before, and a request names only the sides a
// cell draws.
const BORDER_SIDES = ["start", "end", "left", "right", "top", "bottom", "diagonal", "vertical", "horizontal"];
// The sides a `border` request may name, from the shared rules (rules.json).
const CELL_BORDER_SIDES = RULES.cell_border_sides;

// The `<border>` a request needs, from the cell's own: a named side `true` is a
// thin default-coloured border (the spelling Excel itself uses) and `false` an
// empty side that draws nothing. A side the request did not name — `<diagonal>`
// included — keeps its own bytes, and a side written back goes at its place in
// `CT_Border`.
function formatBorder(border, edit) {
  const { open, body } = elementParts(border, "border");
  let out = body;
  for (const side of CELL_BORDER_SIDES) {
    if (edit.border[side] === undefined) continue;
    const element = edit.border[side] ? `<${side} style="thin"><color indexed="64"/></${side}>` : `<${side}/>`;
    const held = firstElement(out, side);
    out = held ? out.slice(0, held.index) + element + out.slice(held.index + held[0].length) : insertOrderedChild(out, side, element, BORDER_SIDES);
  }
  return `${open}${out}</border>`;
}

// `CT_Xf`'s children after the entry's own attributes, in the order the schema
// writes them: a child a request adds goes at its place, never before one that
// must come first.
const XF_CHILD_ORDER = ["alignment", "protection"];

// The `<cellStyleXfs>` entry a cell xf's own `xfId` names — the named style it
// takes anything it omits from — or `undefined` when the id names no entry.
const namedStyleXf = (styles, xf) => styleBlockEntries(styles, "cellStyleXfs")[Number((xf.match(/\bxfId="(\d+)"/) || [])[1])];

// The `<xf>` with its `<alignment>` changed by the named `align`/`vertical`/`wrap`
// properties: each writes or replaces one attribute and every attribute the
// element states keeps its value, so a cell with its own `vertical` that is asked
// only for `align` keeps it. The base element is the xf's own `<alignment>` when it
// states one or claims to apply its own alignment (`applyAlignment="1"`), else the
// one the named style the xf's `xfId` references carries — the rule
// `xfAttributeId` applies to a font, fill or border id, so naming only `align`
// does not drop the `vertical`/`wrapText` the cell renders through its named
// style; an inherited element is materialized onto the cell at the xf's own child
// place, as a newly added one is. An `<alignment>` left stating nothing at all is
// removed rather than written empty.
function withAlignment(styles, xf, edit) {
  const { open, body } = elementParts(xf, "xf");
  const held = firstElement(body, "alignment");
  const inherited = held || /\bapplyAlignment="1"/.test(xf) ? null : firstElement(namedStyleXf(styles, xf) ?? "", "alignment");
  let element = held ? held[0] : inherited ? inherited[0] : "<alignment/>";
  if (edit.align !== undefined) element = setXmlAttribute(element, "horizontal", edit.align);
  if (edit.vertical !== undefined) element = setXmlAttribute(element, "vertical", edit.vertical);
  if (edit.wrap !== undefined) element = edit.wrap ? setXmlAttribute(element, "wrapText", "1") : setXmlAttribute(element, "wrapText", null);
  const bare = !/<alignment\b[^>]*[A-Za-z]+=/.test(element);
  let out;
  if (held) out = body.slice(0, held.index) + (bare ? "" : element) + body.slice(held.index + held[0].length);
  else out = bare ? body : insertOrderedChild(body, "alignment", element, XF_CHILD_ORDER);
  return `${open}${out}</xf>`;
}

// The id an `<xf>` states for `attribute`, or the one the named style it
// references carries: a cell xf that omits an id takes it from the
// `<cellStyleXfs>` entry its own `xfId` names (see `namedStyleXf`), so resolving
// such an id to 0 would base the new entry on a look the cell never had. An entry
// that claims to apply its own formatting (`applied`) yet states no id is the
// schema default 0 instead — it inherits nothing.
function xfAttributeId(styles, xf, attribute, applied) {
  const pattern = new RegExp(`\\b${attribute}="(\\d+)"`);
  const own = (xf.match(pattern) || [])[1];
  if (own !== undefined) return Number(own);
  if (new RegExp(`\\b${applied}="1"`).test(xf)) return 0;
  const inherited = (namedStyleXf(styles, xf)?.match(pattern) || [])[1];
  return inherited === undefined ? 0 : Number(inherited);
}

// The `<cellXfs>` entry the named properties need, from `xf` — the cell's own, or
// the minimal default for a cell that names no style — with ONLY those properties
// changed: everything unnamed (the number format, the fill, the borders, the
// wrap, the vertical alignment, the protection) is the entry's own bytes. The
// font, fill and border a named property needs are added to `styles` and
// referenced by the entry's own ids, and the `apply…` flag is stated for every
// property the request NAMED — the entry is cloned for what the caller asked for,
// and an id written without its flag would leave the format inert.
function styledXf(styles, xf, edit) {
  let current = styles;
  let out = xf;
  if (["font", "size", "bold", "italic", "color"].some((name) => edit[name] !== undefined)) {
    const fontId = xfAttributeId(current, out, "fontId", "applyFont");
    const put = putStyleEntry(current, "fonts", formatFont(styleBlockEntries(current, "fonts")[fontId] ?? STYLE_PART_ENTRIES.fonts, edit));
    current = put.styles;
    out = setXmlAttribute(setXmlAttribute(out, "fontId", String(put.index)), "applyFont", "1");
  }
  if (edit.fill !== undefined) {
    const fill = edit.fill === false
      ? '<fill><patternFill patternType="none"/></fill>'
      : `<fill><patternFill patternType="solid"><fgColor rgb="FF${edit.fill}"/><bgColor indexed="64"/></patternFill></fill>`;
    const put = putStyleEntry(current, "fills", fill);
    current = put.styles;
    out = setXmlAttribute(setXmlAttribute(out, "fillId", String(put.index)), "applyFill", "1");
  }
  if (edit.border !== undefined) {
    const borderId = xfAttributeId(current, out, "borderId", "applyBorder");
    const put = putStyleEntry(current, "borders", formatBorder(styleBlockEntries(current, "borders")[borderId] ?? STYLE_PART_ENTRIES.borders, edit));
    current = put.styles;
    out = setXmlAttribute(setXmlAttribute(out, "borderId", String(put.index)), "applyBorder", "1");
  }
  if (edit.align !== undefined || edit.vertical !== undefined || edit.wrap !== undefined) {
    // The flag is stated even when `withAlignment` left no `<alignment>` child —
    // turning every named property off means the cell renders no alignment of its
    // own, and without the flag a reader would fall back to the one the named style
    // the xf's `xfId` references carries.
    out = setXmlAttribute(withAlignment(current, out, edit), "applyAlignment", "1");
  }
  if (edit.number_format !== undefined) {
    const format = numberFormatId(current, edit.number_format);
    current = format.styles;
    out = setXmlAttribute(setXmlAttribute(out, "numFmtId", String(format.id)), "applyNumberFormat", "1");
  }
  return { styles: current, xf: out };
}

// The `<cellXfs>` index a cell takes for the properties `edit` names: the base
// style changed by exactly those properties — the cell's own `<cellXfs>` entry,
// or the list's first for a cell that names no style of its own. An identical
// entry already in the list is reused rather than appended twice, and the index
// returned is the entry's real position in the list.
function cellStyleIndex(styles, edit, cellStyle) {
  const xfs = styleBlockEntries(styles, "cellXfs");
  const index = Number(cellStyle);
  // The cell's own entry, or — for a cell that names no `s=` — the list's first,
  // which is the base style such a cell renders as: a hard-coded default instead
  // would take away a font, fill or alignment the entry states.
  const own = Number.isInteger(index) && xfs[index] !== undefined ? xfs[index] : xfs[0];
  const styled = styledXf(styles, own ?? STYLE_PART_ENTRIES.cellXfs, edit);
  return putStyleEntry(styled.styles, "cellXfs", styled.xf);
}

// Whether a line moves for an insert (a line at or after the point) or a delete
// (only the lines after it — the deleted line itself is gone).
const moves = (line, at, delta) => (delta > 0 ? line >= at : line > at);

// A moved line number, kept inside the sheet's grid: a line at the sheet's own
// last row or column stays there instead of naming one past the edge, which no
// reader of the sheet has. The two caps are the shared rules the tool's own
// boundary validates a caller's row and column against.
const shiftNumber = (value, at, delta, max) => (moves(value, at, delta) ? Math.min(value + delta, max) : value);
const shiftColumn = (column, at, delta) => shiftNumber(column + 1, at, delta, RULES.sheet_column_max) - 1;

// One token of a range — `$A$1`, `B:D`, `3:7`, a single cell — moved; a token in
// another shape (a defined name, `Sheet1!A1`) is left exactly as it is.
function shiftToken(token, kind, at, delta) {
  const cellToken = token.match(/^(\$?)([A-Za-z]{1,3})(\$?)(\d+)$/);
  if (cellToken) {
    const column = kind === "column" ? shiftColumn(columnOf(cellToken[2]), at, delta) : columnOf(cellToken[2]);
    const row = kind === "row" ? shiftNumber(Number(cellToken[4]), at, delta, RULES.sheet_row_max) : Number(cellToken[4]);
    return `${cellToken[1]}${colName(column)}${cellToken[3]}${row}`;
  }
  const columnToken = token.match(/^(\$?)([A-Za-z]{1,3})$/);
  if (columnToken) {
    if (kind !== "column") return token;
    return `${columnToken[1]}${colName(shiftColumn(columnOf(columnToken[2]), at, delta))}`;
  }
  const rowToken = token.match(/^(\$?)(\d+)$/);
  if (rowToken) return kind === "row" ? `${rowToken[1]}${shiftNumber(Number(rowToken[2]), at, delta, RULES.sheet_row_max)}` : token;
  return token;
}

// A range or an address (`A1:C5`, `$A$1:$B$2`, `B:D`, `3:7`), both ends moved;
// a value in another shape is returned byte-identical.
function shiftRef(reference, kind, at, delta) {
  const parts = reference.split(":");
  const shifted = parts.map((part) => shiftToken(part, kind, at, delta));
  return shifted.every((part, index) => part === parts[index]) ? reference : shifted.join(":");
}

// A `[low, high]` line range with the line `at` deleted: the deleted line is
// gone from the range when it was inside it, so the range shrinks by one, and a
// range that was that line alone vanishes. A range entirely after the point
// moves down. `null` when nothing is left.
function deleteRange(low, high, at) {
  const nextLow = low > at ? low - 1 : low;
  const nextHigh = high >= at ? high - 1 : high;
  return nextHigh < nextLow ? null : { low: nextLow, high: nextHigh };
}

// A token naming whole lines (`B:D` on a column shift, `3:7` on a row shift, with
// the `$` markers either end may carry): its line range with one line deleted —
// the deleted line leaves the range, a range entirely after the point moves down,
// a range the delete covered whole is `null` (nothing of it is left), and one the
// delete does not reach comes back byte-identical. `undefined` for any other
// token, so the caller keeps its own handling of cells and of names.
function deleteLineRef(reference, kind, at) {
  const pattern = kind === "column" ? /^(\$?)([A-Za-z]{1,3})$/ : /^(\$?)(\d+)$/;
  const parts = reference.split(":");
  const lines = parts.map((part) => part.match(pattern));
  if (parts.length > 2 || lines.some((line) => line === null)) return undefined;
  const numbers = lines.map((line) => (kind === "column" ? columnOf(line[2]) + 1 : Number(line[2])));
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const span = deleteRange(low, high, at);
  if (span === null) return null;
  if (span.low === low && span.high === high) return reference;
  const highest = numbers.indexOf(high);
  return lines
    .map((line, index) => {
      const number = index === highest ? span.high : span.low;
      return `${line[1]}${kind === "column" ? colName(number - 1) : number}`;
    })
    .join(":");
}

// A range with one line deleted on the shift's axis. A merge shrinks the way a
// spreadsheet's merged cells do: a range the delete narrowed to one cell is no
// longer a merge and is dropped, but one that was ALREADY a single cell and the
// delete merely moved keeps its own spelling — dropping it would lose content
// the edit did not name. A region (a `<dimension>`, an `sqref` token, an
// autofilter) keeps the narrowed cell spelled once instead. A range the delete
// does not move is returned byte-identical, each end's `$` markers and the
// higher end kept; an insert shifts both ends the ordinary way. `null` when
// nothing of the range is left, or, for a merge, when it no longer merges.
function deleteRangeRef(reference, kind, at, delta, merge) {
  if (delta > 0) return shiftRef(reference, kind, at, delta);
  // A whole-line token is a range of the axis' own lines, not of cells, so it is
  // moved by the line rules rather than the cell ones below. A merge is always a
  // cell range.
  if (!merge) {
    const line = deleteLineRef(reference, kind, at);
    if (line !== undefined) return line;
  }
  const parts = reference.split(":");
  const cells = parts.map((part) => part.match(/^(\$?)([A-Za-z]{1,3})(\$?)(\d+)$/));
  if (cells.some((cell) => cell === null) || (merge ? parts.length !== 2 : parts.length > 2)) {
    return merge ? shiftRef(reference, kind, at, delta) : reference;
  }
  const columns = cells.map((cell) => columnOf(cell[2]) + 1);
  const rows = cells.map((cell) => Number(cell[4]));
  const lowColumn = Math.min(...columns);
  const highColumnValue = Math.max(...columns);
  const lowRow = Math.min(...rows);
  const highRowValue = Math.max(...rows);
  const columnSpan = kind === "column" ? deleteRange(lowColumn, highColumnValue, at) : { low: lowColumn, high: highColumnValue };
  const rowSpan = kind === "row" ? deleteRange(lowRow, highRowValue, at) : { low: lowRow, high: highRowValue };
  if (columnSpan === null || rowSpan === null) return null;
  if (columnSpan.low === lowColumn && columnSpan.high === highColumnValue && rowSpan.low === lowRow && rowSpan.high === highRowValue) return reference;
  const highColumn = columns.indexOf(highColumnValue);
  const highRow = rows.indexOf(highRowValue);
  const rebuilt = cells.map((cell, index) => {
    const column = index === highColumn ? columnSpan.high : columnSpan.low;
    const row = index === highRow ? rowSpan.high : rowSpan.low;
    return `${cell[1]}${colName(column - 1)}${cell[3]}${row}`;
  });
  if (columnSpan.high === columnSpan.low && rowSpan.high === rowSpan.low) {
    if (!merge) return rebuilt[0];
    // A merge that no longer merges is dropped; one that was already a single
    // cell and is merely moved keeps its own two-end spelling.
    return lowColumn === highColumnValue && lowRow === highRowValue ? rebuilt.join(":") : null;
  }
  return rebuilt.join(":");
}

// A single-cell reference with one line deleted: the delete removes the element
// when it names the very cell that goes (a hyperlink addresses one cell, so a
// delete of that cell's row or column drops the link rather than leaving it to
// re-attach to whatever cell shifts into its address), and shifts it otherwise.
// An insert only shifts.
function deleteCellRef(reference, kind, at, delta) {
  if (delta > 0) return shiftRef(reference, kind, at, delta);
  const cell = reference.match(/^(\$?)([A-Za-z]{1,3})(\$?)(\d+)$/);
  if (cell === null) return shiftRef(reference, kind, at, delta);
  if ((kind === "row" && Number(cell[4]) === at) || (kind === "column" && columnOf(cell[2]) + 1 === at)) return null;
  return shiftRef(reference, kind, at, delta);
}

// The children the counted containers below are walked with: one element of the
// container's own name, self-closing or child-bearing.
const MERGE_CELL = elementPattern("mergeCell");
const HYPERLINK = elementPattern("hyperlink");
const COL_ENTRY = elementPattern("col");
const DATA_VALIDATION = elementPattern("dataValidation");
const PROTECTED_RANGE = elementPattern("protectedRange");
const IGNORED_ERROR = elementPattern("ignoredError");

// A container of counted children (`<mergeCells>`, `<hyperlinks>`, `<cols>`,
// `<dataValidations>`, `<protectedRanges>`, `<ignoredErrors>`): each child's own
// text is passed through `change`, which returns the child's new text or `null`
// to drop it. An emptied container is dropped whole, because every one of these
// elements requires a child, and a `count` the file stated follows the survivors.
// A count is never INVENTED: `CT_Cols`, `CT_Hyperlinks`, `CT_ProtectedRanges` and
// `CT_IgnoredErrors` declare no attributes at all, so one written for them would
// make the part invalid, and an element the file wrote without a count keeps its
// open tag exactly as it was written.
//
// The children are rewritten where they stand and nothing else in the block is
// touched: a member this pass does not name — the `extLst` the schema's own
// `CT_IgnoredErrors` allows, say — and the text between children are content the
// edit never named, so they stay exactly as they were written. A block that held
// none of its own children is returned untouched rather than dropped: it held
// nothing already, so removing it is not this pass's to do. A block whose own
// children ALL went with the edit is dropped whole, and a member it happened to
// hold goes with it: the schema requires a child, so an emptied container cannot
// stay at all, and this is the one case where an unnamed member cannot.
function countedContainer(block, child, change) {
  let held = 0;
  let kept = 0;
  const body = block.replace(child, (entry) => {
    held += 1;
    const next = change(entry);
    if (next === null) return "";
    kept += 1;
    return next;
  });
  if (!held) return block;
  if (!kept) return "";
  const open = block.slice(0, block.indexOf(">") + 1);
  return `${open.replace(/\bcount="\d*"/, `count="${kept}"`)}${body.slice(open.length)}`;
}

// What an element carrying an `sqref` is, in the words a reader of the answer
// knows it by: the note says a conditional format, a data validation, a protected
// range or a selection is gone, not the XML element's own name.
function describeRange(name) {
  if (name === "conditionalFormatting") return "conditional format's range";
  if (name === "dataValidation") return "data validation's range";
  if (name === "protectedRange") return "protected range";
  if (name === "ignoredError") return "ignored error";
  if (name === "selection") return "selection";
  // A carrier outside these is named by its range alone: a raw element name is
  // markup, not something a reader of the answer knows the sheet by.
  return "range";
}

// The whitespace-separated range tokens of a value — what an `sqref` attribute and
// an extended `<xm:sqref>` child both hold — each one moved by the shift the way
// `deleteRangeRef` moves a region: a token the delete covered whole is dropped, so
// an empty result means nothing of the range is left. `null` says the same to the
// caller, which drops the element, and the note such a drop owes is written here,
// once, for every element that carries a range — `what` naming it in the words a
// reader of the answer knows it by. A value holding no token at all named no range,
// so there was nothing to move and nothing the edit covered: `[]` says that, and
// the element stays as it is — one the edit never named is not the edit's to drop.
const movedRange = (range, what, kind, at, delta, note) => {
  const tokens = range.split(/\s+/).filter(Boolean);
  if (!tokens.length) return [];
  const kept = tokens.map((token) => deleteRangeRef(token, kind, at, delta, false)).filter((token) => token !== null);
  if (kept.length) return kept;
  note(`[the deleted ${kind} covered the whole ${what}, ${JSON.stringify(range)}, which is gone with it]`);
  return null;
};

// One element carrying an `sqref` attribute, its range moved: the element with
// its range rewritten, or `null` to drop it — the note a full-cover delete owes
// is written once, in `movedRange`. A value holding no token named no range, so
// there was nothing to move and the element stays exactly as it is — one the
// edit never named is not the edit's to drop.
function shiftedSqref(whole, name, value, kind, at, delta, note) {
  const kept = movedRange(value, describeRange(name), kind, at, delta, note);
  if (kept === null) return null;
  if (!kept.length) return whole;
  return whole.replace(/\bsqref="[^"]*"/, () => `sqref="${kept.join(" ")}"`);
}

// The counted containers whose children carry an `sqref` themselves, with the
// name each child is known by. Their children are moved inside the container's
// own pass, NEVER by the bare-element pass above: that is what lets a container
// the delete emptied go with its last child, exactly the way the merge, column
// and hyperlink containers are handled — an emptied one is schema-invalid
// (ECMA-376 requires a child) and an answer that kept it as an empty group with
// a stale count would be reporting a corrupt file as a success.
const SQUREF_CONTAINERS = [
  ["dataValidations", DATA_VALIDATION, "dataValidation"],
  ["protectedRanges", PROTECTED_RANGE, "protectedRange"],
  ["ignoredErrors", IGNORED_ERROR, "ignoredError"],
];

// A whole element that carries an `sqref` of addresses, with one line deleted:
// each token shrinks the way a merge range does (`deleteRangeRef` in region mode —
// a rectangle narrows on the deleted axis, a token the delete covered whole is
// gone) and the rewritten element keeps every surviving token. When no token is
// left the element goes with them, and `note` says what was dropped, because an
// edit that removes a conditional-format range, a data validation, a protected
// range or a pane selection must not do it silently. Anchored on the open tag
// (`[^<>]*` never crosses a `<` or `>`), so a cell whose own TEXT reads
// `sqref="…"` is content and stays as it is.
function shiftSqrefs(xml, kind, at, delta, note) {
  const name = "([A-Za-z0-9_]+(?::[A-Za-z0-9_]+)?)";
  const middle = "\\b[^<>]*\\bsqref=\"([^\"]*)\"[^<>]*";
  const element = new RegExp(`${selfClosingTag(name, middle)}|${openTag(name, middle)}[\\s\\S]*?${closeTag("\\3")}`, "g");
  const inContainer = new Set(SQUREF_CONTAINERS.map(([, , name]) => name));
  let moved = xml.replace(element, (whole, selfName, selfValue, name, value) => {
    const elementName = selfName ?? name;
    // A counted container's child is moved by that container's own pass below.
    if (inContainer.has(elementName)) return whole;
    const next = shiftedSqref(whole, elementName, selfValue ?? value, kind, at, delta, note);
    return next === null ? "" : next;
  });
  for (const [container, child, name] of SQUREF_CONTAINERS) {
    moved = moved.replace(elementPattern(container), (block) => countedContainer(block, child, (entry) => {
      const value = (entry.match(/\bsqref="([^"]*)"/) || [])[1];
      return value === undefined ? entry : shiftedSqref(entry, name, value, kind, at, delta, note);
    }));
  }
  return moved;
}

// One `<col>` entry with a column operation applied: an insert shifts `min`/`max`
// the way a column address moves — a range covering the sheet's whole grid keeps
// its last column — a delete shrinks the range and drops the entry when its own
// column is the deleted one, leaving it would give two entries the same range.
function columnEntry(tag, at, delta) {
  const min = Number((tag.match(/\bmin="(\d+)"/) || [])[1]);
  const max = Number((tag.match(/\bmax="(\d+)"/) || [])[1]);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return tag;
  if (delta > 0) {
    const moved = (value) => shiftNumber(value, at, delta, RULES.sheet_column_max);
    return tag.replace(/\b(min|max)="(\d+)"/g, (whole, name, value) => `${name}="${moved(Number(value))}"`);
  }
  const next = deleteRange(min, max, at);
  if (next === null) return null;
  return tag.replace(/\bmin="\d+"/, () => `min="${next.low}"`).replace(/\bmax="\d+"/, () => `max="${next.high}"`);
}

// The `<brk id>` page breaks of a `<rowBreaks>`/`<colBreaks>` block. A break's id
// is a BOUNDARY — the number of lines above it — not a line, so a break at or below
// the line that went moves up with it instead of keeping a number the data below
// moved out of, and one at the sheet's top cannot go past it. `max` is the axis'
// own last line, so an insert pushing a break down keeps it inside the grid. The
// rewrite is anchored on the `<brk>` element's own tag, so an unrelated `id=` the
// block happens to hold is left as it is.
const shiftBreaks = (block, at, delta, max) => block.replace(/<brk\b[^<>]*>/g, (tag) => tag.replace(/(\bid=")(\d+)(")/g, (whole, before, value, after) => {
  const id = Number(value);
  if (id < at) return whole;
  const next = delta > 0 ? Math.min(id + delta, max) : Math.max(id + delta, 1);
  return `${before}${next}${after}`;
}));

// A cell range (`A1`, `A1:B2`) or a whole-line range (`B:D`, `3:7`), each end with
// or without its `$` markers.
const A1_OR_LINE = "(?:\\$?[A-Za-z]{1,3}\\$?\\d+(?::\\$?[A-Za-z]{1,3}\\$?\\d+)?|\\$?[A-Za-z]{1,3}:\\$?[A-Za-z]{1,3}|\\$?\\d+:\\$?\\d+)";

// An A1 reference or whole-line range inside a conditional-format or
// data-validation formula, with the sheet qualifier such a reference may carry.
// Moved the way the range the rule names is moved: a reference to a line the
// delete took is gone, so a range shrinks to the end that survives and a
// reference to the cell itself becomes Excel's own `#REF!` instead of naming
// whatever cell shifts into its place, while an insert moves both ends the
// ordinary way. Anything that is not a reference is left as it is: a quoted
// string is a list of values, a defined name is not an address, and a reference
// qualified with ANOTHER sheet's name (`Sheet2!A1`) names a range this shift does
// not move — only a reference to the sheet being shifted, bare or under its own
// name, moves. A sheet name needs no quoting when it carries no space or
// punctuation, so the bare qualifier takes letters of ANY script: a workbook's
// `Данные!$B$2:$B$4` is as ordinary a reference as `Sheet1!A1`, and an
// ASCII-only class would leave the whole range invisible to the shift. The
// qualifier is part of the match, so a qualified range cannot be
// half-rewritten by its tail matching alone, and a `:` in front of a match (a 3-D
// `Sheet1:Sheet3!A1`) blocks it the way it blocked the tail before. A whole-line
// range takes the same guard on its own tail (`!`): a 3-D reference spelled with
// sheet names that carry no digits (`Jan:Mar!A1`) reads as one, and the `!` is
// what tells it apart.
const A1_REFERENCE = new RegExp([
  '"(?:[^"]|"")*"',
  `(?<![\\w.!$[:])(?:'((?:[^']|'')*)'|([\\p{L}_][\\p{L}\\p{N}_.]*))!(${A1_OR_LINE})(?![\\w(!])`,
  `(?<![\\w.!$[:])(${A1_OR_LINE})(?![\\w(!])`,
].join("|"), "gu");

// One match of `A1_REFERENCE` split into what it names: the reference itself, and
// the sheet its qualifier names — `undefined` for a bare reference, which belongs
// to the sheet the formula sits on — with a quoted name's doubled quotes folded.
// The qualifier is XML markup (`R&amp;D`) while the name it stands for is not, so
// it is unescaped the way the workbook's own name is before the two are compared.
const refMatch = (whole, quotedSheet, bareSheet, qualified, local) => ({
  reference: qualified ?? local,
  sheet: qualified === undefined ? undefined : xmlUnescape(quotedSheet ?? bareSheet).replace(/''/g, "'"),
});

// `text` with every reference to the sheet named `sheet` moved by the shift.
// `sheet` is the name the workbook spells for the sheet being edited, so a
// reference the file qualified with its own name (`Sheet1!$A$1:$A$5`, which is
// how Excel writes a validation list on the sheet itself) moves with it, while a
// reference to any other sheet keeps its text.
function shiftFormula(text, kind, at, delta, sheet) {
  const wanted = typeof sheet === "string" ? sheet.toLowerCase() : null;
  return text.replace(A1_REFERENCE, (whole, ...groups) => {
    const { reference, sheet: qualifier } = refMatch(whole, ...groups);
    // A quoted string is a list of values rather than a reference: it matches with
    // the whole text and nothing else.
    if (reference === undefined) return whole;
    if (qualifier !== undefined && qualifier.toLowerCase() !== wanted) return whole;
    const next = deleteRangeRef(reference, kind, at, delta, false);
    if (next === null) return "#REF!";
    return whole.slice(0, whole.length - reference.length) + next;
  });
}

// The references `text` spells under a qualifier naming the sheet `sheet` — what a
// formula held by ANOTHER part writes about this one, where a bare reference
// belongs to the sheet the formula itself sits on.
function qualifiedRefs(text, sheet) {
  const wanted = sheet.toLowerCase();
  const out = [];
  for (const match of text.matchAll(A1_REFERENCE)) {
    const { reference, sheet: qualifier } = refMatch(...match);
    if (qualifier !== undefined && qualifier.toLowerCase() === wanted) out.push(reference);
  }
  return out;
}

// The extended twins Excel writes beside a base conditional format or data
// validation in an `<extLst>`: the same rule in the `x14:`/`xm:` spelling, whose
// range sits in an `<xm:sqref>` CHILD element where the base form has an `sqref`
// attribute and whose formula sits in an `<xm:f>` child (inside an
// `<x14:formulaN>` wrapper). They mirror the base rules the attribute pass moved,
// so they move with them — and a rule whose range the delete took goes with it
// rather than staying as one that applies to nothing.
const XM_SQREF = elementPattern("xm:sqref");
const X14_CONDITIONAL_FORMATTING = elementPattern("x14:conditionalFormatting");
const X14_DATA_VALIDATION = elementPattern("x14:dataValidation");
// Only `<xm:f>` holds formula text: an `<x14:formula1>` merely WRAPS one, so
// matching the wrapper too would hand the reference rewriter its child markup and
// let it read a tag name as a range. The element's name is the pattern's own
// capture and both its tags are the shared spellings, so the close tag's
// whitespace is read here as everywhere else.
const EXTENDED_FORMULA = new RegExp(`${openTag("(xm:f)")}([\\s\\S]*?)${closeTag("\\1")}`, "g");

// One extended rule, or `null` to drop it when the delete took its whole range.
// `what` is the rule in the words a reader of the answer knows it by.
function extendedRule(entry, what, kind, at, delta, sheet, note) {
  const sqref = entry.match(XM_SQREF);
  if (sqref) {
    // The element's own body is the range list, as the shared element reading gives
    // it: `""` for a self-closing element.
    const range = elementParts(sqref[0], "xm:sqref").body;
    const kept = movedRange(range, what, kind, at, delta, note);
    if (kept === null) return null;
    if (kept.length) entry = entry.replace(XM_SQREF, () => `<xm:sqref>${kept.join(" ")}</xm:sqref>`);
  }
  return entry.replace(EXTENDED_FORMULA, (whole, name, text) => {
    const moved = shiftFormula(text, kind, at, delta, sheet);
    return moved === text ? whole : `<${name}>${moved}</${name}>`;
  });
}

// The names of the two containers an extended rule lives in: what says a wrapper
// held a rule this pass moved, and so may have been emptied by it.
const X14_RULE_CONTAINER = /<x14:(?:conditionalFormattings|dataValidations)\b/;

// `xml` with the extended rules above moved. An extended conditional format
// keeps its range beside its own `<x14:cfRule>` children, an extended data
// validation holds its own: each goes whole when the range is gone, and a
// container the delete emptied goes with it — both containers require a child, so
// an emptied `<x14:conditionalFormattings>`/`<x14:dataValidations>` is not left
// behind, and neither is the `null` a dropped rule would otherwise be written as.
//
// The containers are shifted inside the wrapper that holds them, so only a
// wrapper that really held one of the two can have been emptied, and only such a
// wrapper goes with its container: a payload of another kind — a plain-text
// extension, one this pass never named — is left exactly as it stands.
//
// The wrappers are moved inside the `<extLst>` that holds them, so the list can
// tell whether the pass emptied it: an `<extLst>` that held an `<ext>` and now
// holds none goes with its last wrapper (the schema requires a child), while one
// that held none to begin with — a self-closing `<extLst/>` or a paired empty
// one — is not this pass's to remove. The open-tag alternation states a
// self-closing list first, so it never pairs with a later `</extLst>`.
// `elementPattern("ext")` cannot match `<extLst` (the name must be followed by a
// space or `>`), and the ordered alternation keeps a self-closing `<ext .../>`
// from being read as the open tag of a pair.
function extendedRules(xml, kind, at, delta, sheet, note) {
  const rule = (what) => (entry) => extendedRule(entry, what, kind, at, delta, sheet, note);
  const shift = (block) => block
    .replace(elementPattern("x14:conditionalFormattings"), (container) =>
      countedContainer(container, X14_CONDITIONAL_FORMATTING, rule("extended conditional format's range")))
    .replace(elementPattern("x14:dataValidations"), (container) =>
      countedContainer(container, X14_DATA_VALIDATION, rule("extended data validation's range")));
  const wrapper = (block) => {
    const shifted = shift(block);
    if (!X14_RULE_CONTAINER.test(block)) return shifted;
    // The wrapper's own open tag is not payload: a wrapper whose shift left no
    // element inside it held one of the two containers and nothing else, so it
    // goes with them. The close tag reads `</…` to the test, never `<…`.
    return /<[A-Za-z]/.test(shifted.slice(shifted.indexOf(">") + 1)) ? shifted : "";
  };
  return xml.replace(elementForms("extLst", "\\b[^<>]*"), (block) => {
    if (!/<ext\b/.test(block)) return block;
    const shifted = block.replace(elementPattern("ext"), wrapper);
    return /<ext\b/.test(shifted) ? shifted : "";
  });
}

// A whole element that names a range in its own `ref` — the sheet's own
// `<dimension>` extent, its `<autoFilter>` range — with the ref narrowed on the
// deleted axis the way the merges are; a ref the delete covered whole takes the
// element with it rather than keeping the removed line or leaving an empty `ref`.
// Neither element is required by the format, so a dropped one is owed no note.
const refElement = (kind, at, delta) => (element) => {
  const ref = (element.match(/\bref="([^"]*)"/) || [])[1];
  if (ref === undefined) return element;
  const next = deleteRangeRef(ref, kind, at, delta, false);
  return next === null ? "" : element.replace(/\bref="[^"]*"/, () => `ref="${next}"`);
};

// Move every address one row or column insert/delete moved, across a sheet part.
// Formulas, defined names, charts and pivot caches keep their text on purpose:
// rewriting a formula's references is a spreadsheet engine's job, and a wrong
// rewrite is worse than a stale one — the notes this raises say so. The notes are
// raised here rather than at each call site, so no row or column op can forget
// them, and only when the shift really moved a row or cell address: an insert
// past the used range moves nothing, and the formulas, comments, shapes and
// drawing anchors the part holds still name the cells they did. `stale` is what
// `staleRefs` collected for this part, and `sheet` the name the workbook spells
// for the sheet being shifted, which is what tells a formula's reference to this
// sheet from one to another.
//
// A shift that moved nothing returns the part it was handed: making the addresses
// a writer left implicit explicit is the only thing it would have written, and a
// part rewritten for nothing is a change the caller would see and a chart caveat
// nobody is owed. `removed` is the caller's own delete having taken cells out
// before this call: a delete of the sheet's last line moves no address, yet it
// changed the sheet's content and leaves the same parts naming old cells.
// `addressed` is for a caller whose own delete had to materialize the addresses
// first: it hands the part over already addressed, so the walk is not made twice.
function shiftSheet(editor, xml, kind, at, delta, { stale, sheet, removed, addressed }) {
  const note = editor.note;
  const materialized = addressed ? xml : materializeAddresses(xml);
  const data = shiftSheetData(materialized, kind, at, delta);
  const shifted = data
    .replace(elementPattern("dimension"), refElement(kind, at, delta))
    .replace(elementPattern("mergeCells"), (block) =>
      countedContainer(block, MERGE_CELL, (cell) => {
        const ref = (cell.match(/\bref="([^"]*)"/) || [])[1];
        if (ref === undefined) return cell;
        const next = deleteRangeRef(ref, kind, at, delta, true);
        return next === null ? null : cell.replace(/\bref="[^"]*"/, () => `ref="${next}"`);
      }))
    // An `<autoFilter ref>` names the range the filter covers, so it narrows or
    // goes with a delete exactly as the merges above do (see `refElement`).
    .replace(elementPattern("autoFilter"), refElement(kind, at, delta))
    // A pane's `topLeftCell` and a selection's `activeCell` name a cursor
    // POSITION in the grid rather than a range of content: a position after the
    // deleted line moves up, the deleted line's own number stays and now points at
    // the line that shifted into it, and an insert pushes a position at or after
    // the point down — what a spreadsheet does with a selected cell. The `sqref`
    // sibling is a range the rule named, so it shrinks or goes with the delete
    // (see `shiftSqrefs`), and the element goes with it when nothing of the range
    // is left: that is why a `selection` can vanish while the pane beside it keeps
    // its cursor. The match is anchored on the tag itself, so a cell whose own text
    // reads `activeCell="B2"` is content.
    .replace(/<(?:pane|selection)\b[^<>]*>/g, (tag) => tag
      .replace(/(\btopLeftCell=")([^"]*)(")/, (whole, before, value, after) => before + shiftRef(value, kind, at, delta) + after)
      .replace(/(\bactiveCell=")([^"]*)(")/, (whole, before, value, after) => before + shiftRef(value, kind, at, delta) + after))
    .replace(elementPattern("hyperlinks"), (block) =>
      countedContainer(block, HYPERLINK, (link) => {
        const ref = (link.match(/\bref="([^"]*)"/) || [])[1];
        if (ref === undefined) return link;
        const next = deleteCellRef(ref, kind, at, delta);
        return next === null ? null : link.replace(/\bref="[^"]*"/, () => `ref="${next}"`);
      }));
  // `conditionalFormatting`, `dataValidation` and `pane/selection` all carry a
  // `sqref` of addresses; each token shrinks or drops the way the merges above
  // do, and an element left with no token goes with them (see `shiftSqrefs`).
  const sqrefs = shiftSqrefs(shifted, kind, at, delta, note)
    .replace(elementPattern("cols"), (block) =>
      kind === "column" ? countedContainer(block, COL_ENTRY, (entry) => columnEntry(entry, at, delta)) : block)
    .replace(elementPattern("rowBreaks"), (block) => kind === "row" ? shiftBreaks(block, at, delta, RULES.sheet_row_max) : block)
    .replace(elementPattern("colBreaks"), (block) => kind === "column" ? shiftBreaks(block, at, delta, RULES.sheet_column_max) : block)
    // The ranges conditional formatting and data validation name are moved above
    // with their `sqref`; the references their own formulas carry move with them —
    // including the ranges of whole columns or rows and a reference the file
    // qualified with this sheet's own name — so a rule keeps testing the cells it
    // was written for. A cell's `<f>` is not touched — see `formulaNote`.
    .replace(elementPattern("conditionalFormatting"), (block) =>
      block.replace(new RegExp(`(${openTag("formula")})([\\s\\S]*?)(${closeTag("formula")})`, "g"), (whole, open, text, close) => open + shiftFormula(text, kind, at, delta, sheet) + close))
    .replace(DATA_VALIDATION, (entry) =>
      entry.replace(new RegExp(`(${openTag("formula1")}|${openTag("formula2")})([\\s\\S]*?)(${closeTag("formula1")}|${closeTag("formula2")})`, "g"), (whole, open, text, close) => open + shiftFormula(text, kind, at, delta, sheet) + close));
  // The extended forms of the same two rules live in an `<extLst>` beside them
  // and move the same way.
  const moved = extendedRules(sqrefs, kind, at, delta, sheet, note);
  // What owes a content change and a note is the sheet having changed: a shift that
  // moved a surviving address, or a delete that took cells without moving one (the
  // last line of a sheet). An insert past the used range did neither — it has
  // nothing to warn about and writes nothing.
  const shiftedPart = moved !== materialized;
  if (!shiftedPart && !removed) return xml;
  editor.markContent();
  formulaNote(editor, moved, kind, at, delta, sheet);
  if (stale.size) note(`[the shift left ${[...stale].join(" and ")} pointing at the old cells — they were not rewritten]`);
  return shiftedPart ? moved : xml;
}

// Whether an element's own open tag states an `r` at all: an address this kit's
// reader can place (`A1`) and one it cannot (`$A$1`) are both addresses the element
// names, and writing a second `r` beside either would state the attribute twice.
const statesAddress = (element) => /\sr\s*=/.test(element.slice(0, element.indexOf(">") + 1));

// The sheet part with the addresses its writer left implicit made explicit: each
// `<row>` gets the computed `r` it stands for, each addressless `<c>` its
// computed `r`. The product's own reader already numbers them that way — an
// addressless cell is its position in the row, an addressless row is the one
// after its predecessor — so a shift that moved only written addresses would
// move the wrong cells, or none. This rewrites the part the shift already
// rewrites and no other, and an element that already states an address of its own
// keeps its bytes (see `statesAddress`).
function materializeAddresses(xml) {
  let number = 0;
  return xml.replace(ROW_ELEMENT, (rowXml) => {
    const attribute = (rowXml.match(/\br="(\d+)"/) || [])[1];
    number = attribute ? Number(attribute) : number + 1;
    const row = statesAddress(rowXml) ? rowXml : rowXml.replace(/^<row\b/, (open) => `${open} r="${number}"`);
    const cells = rowCells(row);
    let out = row;
    for (let i = cells.length - 1; i >= 0; i -= 1) {
      const cell = cells[i];
      if (statesAddress(cell.xml)) continue;
      const addressed = cell.xml.replace(/^<c\b/, (open) => `${open} r="${colName(cell.column)}${number}"`);
      out = out.slice(0, cell.start) + addressed + out.slice(cell.end);
    }
    return out;
  });
}

// The sheet's own rows and cells, with every address the shift moved. The caller
// materializes the addresses first (see `materializeAddresses`), so a cell or row
// with no `r` moves by the position that addresses it rather than staying put —
// and so a shift can tell whether it really moved one.
function shiftSheetData(xml, kind, at, delta) {
  return xml.replace(ROW_ELEMENT, (rowXml) => {
    const cells = rowXml.replace(CELL_ELEMENT, (cellXml) => cellXml.replace(/\br="([A-Za-z]+\d+)"/, (whole, reference) => `r="${shiftToken(reference, kind, at, delta)}"`));
    if (kind !== "row") return cells;
    return cells.replace(/(<row\b[^>]*\br=")(\d+)(")/, (whole, before, value, after) => `${before}${shiftNumber(Number(value), at, delta, RULES.sheet_row_max)}${after}`);
  });
}

// Insert a new cell into its row at the address-ordered position.
function insertCell(rowXml, cells, column, element) {
  const after = cells.find((cell) => cell.column > column);
  if (after) return rowXml.slice(0, after.start) + element + rowXml.slice(after.start);
  // A self-closing `<row r="1"/>` is opened to hold the new cell; one already open
  // takes it before its close, which `elementParts` reads in either spelling.
  const { open, body } = elementParts(rowXml, "row");
  return `${open}${body}${element}</row>`;
}

// `CT_Worksheet`'s children from `<sheetData>` on, in the sequence ISO/IEC 29500
// states: a sheet part that states no `<sheetData>` gets one written before the
// first child the sequence puts after it, so the part stays one a reader accepts.
// The tail is the schema's own, `drawingHF` included — a sheet's header/footer
// images are declared there — which some published copies of the type leave out.
const WORKSHEET_AFTER_SHEET_DATA = ["sheetCalcPr", "sheetProtection", "protectedRanges", "scenarios", "autoFilter", "sortState", "dataConsolidate", "customSheetViews", "mergeCells", "phoneticPr", "conditionalFormatting", "dataValidations", "hyperlinks", "printOptions", "pageMargins", "pageSetup", "headerFooter", "rowBreaks", "colBreaks", "customProperties", "cellWatches", "ignoredErrors", "smartTags", "drawing", "legacyDrawing", "legacyDrawingHF", "drawingHF", "picture", "oleObjects", "controls", "webPublishItems", "tableParts", "extLst"];

// The worksheet root's two tags, in the shared spellings (`openTag`/`closeTag`), for
// the two places that need a position in the root rather than the elements it holds.
const WORKSHEET_OPEN = new RegExp(openTag("worksheet"));
const WORKSHEET_CLOSE = new RegExp(closeTag("worksheet"));

// `xml` with a `<sheetData/>` written at its `CT_Worksheet` place when the sheet
// part states none: an edit that brings a value or formatting to such a sheet
// writes the element it needs rather than refusing a legal part.
function ensureSheetData(xml) {
  // The element a written `<sheetData/>` would go beside, or none: one the part leaves
  // open is refused (`writableElement`).
  if (writableElement(xml, "sheetData", "the sheet part")) return xml;
  for (const later of WORKSHEET_AFTER_SHEET_DATA) {
    const found = firstElement(xml, later);
    if (found) return xml.slice(0, found.index) + "<sheetData/>" + xml.slice(found.index);
  }
  const close = xml.match(WORKSHEET_CLOSE);
  if (!close) throw new UsageError("the sheet part has no </worksheet> to write a <sheetData> into");
  return xml.slice(0, close.index) + "<sheetData/>" + xml.slice(close.index);
}

// The `<sheetData>` a row- or cell-writing edit works in: the index its open tag starts
// at, the tag it is spelled with, where its body ends and whether it is self-closing —
// ONE reading of the element (`writableElement`, which refuses a part that leaves it
// open), and `undefined` for a part that states none (`ensureSheetData` writes one). A
// worksheet holds its rows in ONE `<sheetData>` (`CT_Worksheet` states exactly one), so
// a part that states a second, or a `<row>` outside this one, is a shape no writer
// produces and one no edit can write into without guessing which body is the sheet's:
// refused by the callers that write into the body — a row created in it, the body
// rebuilt — since a success over a part a reader offers to repair is the one outcome
// an edit must not report. A caller that places a block BESIDE the body (a column
// width's `<cols>`) needs the element's own index and reads no rows. An edit that
// rewrites a cell inside a row it found changes nothing structural and leaves the
// part as healthy as it arrived.
function sheetBody(xml) {
  const found = writableElement(xml, "sheetData", "the sheet part");
  if (!found) return undefined;
  const edges = elementEdges(found[0], "sheetData");
  const bodyStart = found.index + edges.bodyStart;
  const bodyEnd = found.index + edges.close;
  const outside = sheetRows(xml).some((row) => row.start < bodyStart || row.end > bodyEnd);
  if (outside || xml.slice(bodyEnd).includes("<sheetData")) {
    throw new UsageError("the sheet part states a <row> outside its <sheetData>, or a second <sheetData>: a worksheet holds its rows in one body, which this tool does not edit around");
  }
  return { index: found.index, open: edges.open, selfClosing: edges.selfClosing, bodyStart, bodyEnd };
}

// Insert a `<row>` into the sheet's own `<sheetData>` in row order.
function insertSheetRow(xml, rows, number, element) {
  const source = ensureSheetData(xml);
  // A part whose rows stand outside its `<sheetData>` is refused here, and a part
  // that states none has no rows for `after` to name — so the positions `rows`
  // carries are positions in `source` whenever `after` is found.
  const body = sheetBody(source);
  const after = rows.find((row) => row.number > number);
  if (after) return source.slice(0, after.start) + element + source.slice(after.start);
  // No row to order after: the element is the sheet's first, so the body holding
  // the rows is opened when the one it states is self-closing.
  if (!body.selfClosing) return source.slice(0, body.bodyEnd) + element + source.slice(body.bodyEnd);
  return source.slice(0, body.index) + `${body.open.slice(0, -2)}>${element}</sheetData>` + source.slice(body.bodyStart);
}

// The cells a deleted column held, removed from their rows — the row itself
// stays, so the sheet keeps a body to address. The caller materializes the
// addresses first, so every cell here carries the `r` it sits at.
function removeColumn(xml, column) {
  return xml.replace(ROW_ELEMENT, (rowXml) => {
    const cells = rowCells(rowXml).filter((cell) => cell.column + 1 === column);
    let out = rowXml;
    for (let i = cells.length - 1; i >= 0; i -= 1) out = out.slice(0, cells[i].start) + out.slice(cells[i].end);
    return out;
  });
}

// The `<f>` elements a part holds: a cell's formula, with the `ref` a shared
// formula states and the formula's own text. A formula is the only place a
// reference lives, so this is what a scan for one reads — a cell's own text is
// content, and one that merely reads like a reference names nothing.
const cellFormulas = (text) => [...text.matchAll(new RegExp(`${selfClosingTag("f", "\\b([^<>]*)")}|${openTag("f", "\\b([^<>]*)")}([\\s\\S]*?)${closeTag("f")}`, "g"))]
  .map((match) => ({ range: ((match[1] ?? match[2]).match(/\bref="([^"]*)"/) || [])[1], text: match[3] ?? "" }));

// One note when a sheet a row or column shift touched holds cell formulas naming a
// cell the shift moved: their own text is not rewritten — rewriting references is a
// spreadsheet engine's job — so a reference may now mean a different cell. A
// formula naming nothing the shift moved is left out rather than named: its text
// means exactly what it did, and a note that stated a loss which did not happen
// would be as wrong as a silent one. A cell's `<f>` is what counts — a
// conditional-format or data-validation formula is not left behind (the shift
// moves its references with the range it belongs to) — and a shared formula's own
// covered range counts too, since the shift does not rewrite it either. `xml` is
// the part as it stands AFTER the shift, so a formula the shift removed is not
// counted as one left in place.
function formulaNote(editor, xml, kind, at, delta, sheet) {
  const movesReference = (text) => shiftFormula(text, kind, at, delta, sheet) !== text;
  let count = 0;
  for (const formula of cellFormulas(xml)) {
    if (movesReference(formula.text) || (formula.range !== undefined && movesReference(formula.range))) count += 1;
  }
  if (!count) return;
  const one = count === 1;
  editor.note(`[the sheet's ${count} formula${one ? " was" : "s were"} left as ${one ? "it is" : "they are"} while ${one ? "a cell it names" : "cells they name"} moved — check the references]`);
}

// The addresses a cell- or range-anchored part names in its own text, which is
// what a shift has to move for the part to be left behind: the `ref` of a comment,
// of a table and of its autofilter, the range a `<definedName>` states, and the row
// and column a floating drawing anchors its object at (0-based in the markup, so
// one is added).
const partRefs = (text) => [...text.matchAll(/\bref="([^"]*)"/g)].map((match) => match[1]);
const definedNameTexts = (text) => [...text.matchAll(new RegExp(`${openTag("definedName")}([\\s\\S]*?)${closeTag("definedName")}`, "g"))].map((match) => xmlUnescape(match[1]));
const anchoredCells = (text) => [...text.matchAll(new RegExp(`${openTag("xdr:(?:from|to)", "\\b[\\s\\S]*?")}${closeTag("xdr:(?:from|to)")}`, "g"))].flatMap((anchor) => {
  const column = (anchor[0].match(new RegExp(`${openTag("xdr:col")}(\\d+)${closeTag("xdr:col")}`)) || [])[1];
  const row = (anchor[0].match(new RegExp(`${openTag("xdr:row")}(\\d+)${closeTag("xdr:row")}`)) || [])[1];
  return column === undefined || row === undefined ? [] : [`${colName(Number(column))}${Number(row) + 1}`];
});

// The parts a row/column shift leaves naming the cells it moved, because their
// own text is not rewritten: the workbook's defined names, the shifted sheet's
// own table ranges (a table is declared either by the sheet's `<tableParts>` or
// by a `<.../table>` relationship of its `.rels`) and the sheet's cell-anchored
// parts — a comment's own `ref`, the VML shape a comment is drawn with, and a
// floating drawing's cell anchor all name the cell they were written for while
// the data under them moves. A part is collected only when the shift really moves
// one of the addresses it names: one whose own cells the shift moved with is as
// valid as it was, and naming it would state a loss that did not happen. A part
// the sheet declares that the package cannot read keeps its caveat — one that
// cannot be inspected is not one known to be fine. The formulas held by OTHER
// sheets of the workbook belong here too: a reference such a formula qualifies
// with this sheet's own name names cells this shift moved and its part is not the
// one being edited. What is collected, the reply names, the way it names formulas,
// charts and pivots.
function staleRefs(editor, part, xml, stale, kind, at, delta, sheet) {
  const base = part.slice(0, part.lastIndexOf("/") + 1);
  // Every part this check reads goes through the editor (`read`), so the walks it
  // makes are charged like the ones the edits themselves make: a check that read
  // whole parts around the budget would be the very hole the bound exists to close.
  const relations = relationships(editor.read(relsPartFor(part)) ?? "");
  // Whether the shift changes what an address names. The test is the shift's own
  // reference rule, so a part anchored on a line the delete took is stale like one
  // below it.
  const movesAddress = (address) => shiftFormula(address, kind, at, delta, sheet) !== address;
  const anyMoves = (addresses) => addresses.some(movesAddress);
  const declares = (what) => relations.some((rel) => rel.type.endsWith(`/${what}`));
  // Whether a part of this type the sheet declares names an address the shift moves.
  const movedPart = (what, addresses) => relations
    .filter((rel) => rel.type.endsWith(`/${what}`))
    .some((rel) => {
      const text = editor.read(resolvePart(base, rel.target));
      return text === undefined || anyMoves(addresses(text));
    });
  const workbook = editor.read(XLSX_WORKBOOK);
  if (workbook !== undefined && anyMoves(definedNameTexts(workbook))) stale.add("the workbook's defined names");
  // A table declared by the sheet's `<tableParts>` but named by no relationship of
  // its own can be neither read nor ruled out.
  if (movedPart("table", partRefs) || (/<tableParts\b/.test(xml) && !declares("table"))) stale.add("the sheet's table ranges");
  if (movedPart("comments", partRefs)) {
    stale.add("the sheet's comments");
    // The shape a comment is drawn with follows the comment's own cell.
    if (declares("vmlDrawing")) stale.add("the sheet's comment shapes");
  }
  if (movedPart("drawing", anchoredCells)) stale.add("the sheet's drawing anchors");
  // A formula held by ANOTHER sheet is not this shift's to rewrite — that part is
  // not the one being edited — so a reference it qualifies with this sheet's name
  // keeps naming cells that moved. A bare reference belongs to the sheet the
  // formula sits on, which is what the qualifier tells apart. The scan reads the
  // `<f>` elements the part holds rather than its whole text: a cell's own text
  // can read like a reference and is content, not a formula, and naming it would
  // state a loss that did not happen.
  const others = workbookSheets(editor).map((entry) => entry.part).filter((name) => name !== part);
  const staleElsewhere = others.some((name) => {
    const text = editor.read(name);
    return text !== undefined && cellFormulas(text).some((formula) => qualifiedRefs(formula.text, sheet).some(movesAddress));
  });
  if (staleElsewhere) stale.add("the other sheets' formulas naming this sheet");
}

// `xml` with a `<dimension>` added for a workbook that states none, at
// `CT_Worksheet`'s own place for it: directly after `<sheetPr>` when the sheet
// has one, else as the worksheet's first child. A created workbook's sheet holds
// only `<sheetData>`, so this is the sheet an edit widens for the first time.
function insertDimension(xml, ref) {
  const element = `<dimension ref="${ref}"/>`;
  const sheetPr = firstElement(xml, "sheetPr");
  const at = sheetPr ? sheetPr.index + sheetPr[0].length : (() => {
    const open = xml.match(WORKSHEET_OPEN);
    return open ? open.index + open[0].length : -1;
  })();
  if (at < 0) return xml;
  return xml.slice(0, at) + element + xml.slice(at);
}

// The `<dimension ref>` a sheet that states none gets, so a write into it can
// extend a declared reach it did not have: the smallest rectangle covering every
// cell the part holds AND the address a write names, because a ref naming the
// written cell alone would describe a part holding more than the ref says. The
// cells' own addresses (`sheetRows`/`rowCells`) already resolve what a writer left
// implicit, so the computed numbers are the ones a reader uses.
function dimensionRef(xml, row, column) {
  let firstRow = row;
  let lastRow = row;
  let firstColumn = column;
  let lastColumn = column;
  for (const entry of sheetRows(xml)) {
    for (const cell of rowCells(entry.xml)) {
      firstRow = Math.min(firstRow, entry.number);
      lastRow = Math.max(lastRow, entry.number);
      firstColumn = Math.min(firstColumn, cell.column);
      lastColumn = Math.max(lastColumn, cell.column);
    }
  }
  const low = `${colName(firstColumn)}${firstRow}`;
  const high = `${colName(lastColumn)}${lastRow}`;
  return low === high ? low : `${low}:${high}`;
}

// A worksheet's `<dimension ref>` widened so it covers a cell a `set_cell` wrote:
// the cell may lie outside the sheet's declared reach, and a ref that does not
// name it describes a part holding more than the ref says. The ref's own bounds
// are the extent a write extends; a sheet that states no `<dimension>` at all
// gets one covering its cells and the written address, since a part with no
// declared reach is what an edit first brings formatting to, and one that states
// a `<dimension>` with no `ref` (a part no writer would produce) takes the ref on
// that element rather than a second element beside it. One the reader cannot
// parse is left alone, as a row or column shift updates the ref the ordinary way.
function widenDimension(xml, row, column) {
  // The `<dimension>` element the ref goes on, or none: one the part leaves open is
  // refused (`writableElement`).
  const dimension = writableElement(xml, "dimension", "the sheet part");
  if (!dimension) return insertDimension(xml, dimensionRef(xml, row, column));
  if (!/\bref="/.test(dimension[0])) {
    // `setXmlAttribute` keeps a self-closing tag self-closing, so `<dimension/>`
    // becomes `<dimension ref="A1:B2"/>` rather than gaining a second element.
    const covering = setXmlAttribute(dimension[0], "ref", dimensionRef(xml, row, column));
    return xml.slice(0, dimension.index) + covering + xml.slice(dimension.index + dimension[0].length);
  }
  return xml.replace(/(<dimension\b[^>]*\bref=")([^"]*)(")/, (whole, before, value, after) => {
    const cells = value.split(":").map((part) => part.match(/^([A-Za-z]+)(\d+)$/));
    if (cells.length > 2 || cells.some((cell) => cell === null)) return whole;
    const columns = [column + 1, ...cells.map((cell) => columnOf(cell[1]) + 1)];
    const rows = [row, ...cells.map((cell) => Number(cell[2]))];
    const low = `${colName(Math.min(...columns) - 1)}${Math.min(...rows)}`;
    const high = `${colName(Math.max(...columns) - 1)}${Math.max(...rows)}`;
    return before + (low === high ? low : `${low}:${high}`) + after;
  });
}

function setCell(editor, edit, input, stylesName) {
  const { part } = sheetPart(editor, edit.sheet);
  const address = cellAddress(edit.cell);
  const reference = colName(address.column) + address.row;
  // A `set_cell` writes a value; changing a cell's formatting without writing one
  // is the `format_cells` op's job, so there is one spelling of "format only".
  // Writing through the value path with no value would answer such a request by
  // emptying the cell it was given.
  if (edit.value === undefined) throw new UsageError("set_cell needs a value — hint: use the format_cells op to change only a cell's formatting");
  const format = edit.number_format === undefined ? null : editText(edit.number_format, "number_format");
  // A workbook that never saved styles gets the part written rather than a
  // refusal: setting a number format is exactly the edit that needs one.
  const styles = format !== null ? stylesText(editor, input, stylesName) : null;
  // The sheet is read and written by the one `sheet` call (its own duplicate check
  // included), and the styles part by `stylesText` — each read charged — with the
  // grown styles written back through `rewrite` rather than read again for the write
  // (see `charge`'s doc).
  editor.sheet(part, (sheet) => {
    let style = null;
    if (format !== null) {
      const rows = sheetRows(sheet);
      const row = rows.find((candidate) => candidate.number === address.row);
      const cell = row ? rowCells(row.xml).find((candidate) => candidate.column === address.column) : undefined;
      // The entry the number format clones is the style the cell already renders
      // as — its own `s=` when it states one, else the one it inherits from its row
      // or column — so setting a number format does not drop the cell's font, fill,
      // border or alignment.
      const base = (cell ? styleOf(cell.xml) : undefined) ?? inheritedStyle(rowStyleOf(row?.xml), columnStyleEntries(sheet), address.column);
      const applied = cellStyleIndex(styles, { number_format: format }, base);
      style = String(applied.index);
      editor.rewrite(stylesName, styles, applied.styles);
    }
    const written = writeCell(sheet, edit, address, reference, style);
    // A write that landed on the value the cell already held changed no content,
    // so the caveats keyed on content are not owed for it.
    if (written !== sheet) editor.markContent();
    return written;
  });
}

// `xml` with `edit`'s cell written: the row is added when the sheet has none, the
// cell replaced or inserted inside its row, and the sheet's extent widened to
// cover the address.
function writeCell(xml, edit, address, reference, style) {
  const rows = sheetRows(xml);
  const row = rows.find((candidate) => candidate.number === address.row);
  if (!row) {
    const written = insertSheetRow(xml, rows, address.row, `<row r="${address.row}">${valueCell(reference, edit.value, style)}</row>`);
    return widenDimension(written, address.row, address.column);
  }
  const cells = rowCells(row.xml);
  const existing = cells.find((candidate) => candidate.column === address.column);
  const cellStyle = style !== null ? style : existing ? styleOf(existing.xml) : null;
  const element = valueCell(reference, edit.value, cellStyle);
  const cellXml = existing ? row.xml.slice(0, existing.start) + element + row.xml.slice(existing.end) : insertCell(row.xml, cells, address.column, element);
  return widenDimension(xml.slice(0, row.start) + cellXml + xml.slice(row.end), address.row, address.column);
}

function clearCell(editor, edit) {
  const { part, name } = sheetPart(editor, edit.sheet);
  const address = cellAddress(edit.cell);
  editor.sheet(part, (xml) => {
    // The cell is found by the address it stands for, read the way the rest of the
    // kit reads one: `sheetRows`/`rowCells` resolve an addressless `<c>` to the
    // position it sits in, so the target is found without rewriting the part's
    // addresses. The emptied cell keeps its own address and style, and every other
    // cell and row keeps its exact bytes.
    const rows = sheetRows(xml);
    const row = rows.find((candidate) => candidate.number === address.row);
    const existing = row ? rowCells(row.xml).find((candidate) => candidate.column === address.column) : null;
    if (!existing) throw new UsageError(`the cell ${edit.cell} is not in sheet ${JSON.stringify(name)}`);
    editor.markContent();
    // The cell is emptied, not unstyled: clearing takes what the cell holds, and
    // the look it was given — the `s=` a `format_cells` wrote, an `apply…` entry's
    // own style — stays its own. The `t`/`cm`/`vm` attributes go with the value, by
    // the rule `valueCell` states, and so do the children; the address the emptied
    // element states is its own, so the emptied cell stays where it was.
    const emptied = emptiedCell(existing.xml);
    return xml.slice(0, row.start) + row.xml.slice(0, existing.start) + emptied + row.xml.slice(existing.end) + xml.slice(row.end);
  });
}

// The `<c>` element an emptied cell leaves: its address and its style index, in
// the file's own spelling — the two attributes a cell keeps when its content is
// taken out. Every other attribute described the value being removed.
function emptiedCell(cell) {
  const open = cell.slice(0, cell.indexOf(">") + 1);
  const kept = ["r", "s"].map((name) => open.match(new RegExp(`\\s${name}="[^"]*"`))).filter(Boolean);
  return `<c${kept.map((held) => held[0]).join("")}/>`;
}

function insertRow(editor, edit, stale) {
  const { part, name } = sheetPart(editor, edit.sheet);
  const row = rowNumber(edit.row);
  editor.sheet(part, (xml) => {
    const { rowMax } = usedRange(xml);
    if (rowMax === 0) throw new UsageError(`sheet ${JSON.stringify(name)} has no rows to insert into`);
    if (row > rowMax + 1) throw new UsageError(`row ${row} is past the sheet's used range (1-${rowMax})`);
    // An insert moves every line from `row` down by one, so a sheet already
    // reaching the grid's last row cannot take one: the row pushed past the edge
    // would have to keep the last row's own address. Refusing names a call the
    // caller can fix rather than writing two lines the same address.
    if (rowMax >= RULES.sheet_row_max) throw new UsageError(`sheet ${JSON.stringify(name)} reaches the last row (${RULES.sheet_row_max}), so an insert would push a row past the sheet's grid`);
    staleRefs(editor, part, xml, stale, "row", row, 1, name);
    const shifted = shiftSheet(editor, xml, "row", row, 1, { stale, sheet: name });
    return insertSheetRow(shifted, sheetRows(shifted), row, `<row r="${row}"/>`);
  });
}

function deleteRow(editor, edit, stale) {
  const { part, name } = sheetPart(editor, edit.sheet);
  const row = rowNumber(edit.row);
  editor.sheet(part, (xml) => {
    const { rowMax } = usedRange(xml);
    if (rowMax === 0) throw new UsageError(`sheet ${JSON.stringify(name)} has no rows`);
    if (row > rowMax) throw new UsageError(`row ${row} is past the sheet's used range (1-${rowMax})`);
    staleRefs(editor, part, xml, stale, "row", row, -1, name);
    // A line inside the used range the writer left no `<row>` element for is
    // still a line the delete takes: everything below it moves up by one, the
    // same rule the column arm follows. Only a line past the used range is
    // refused. The addresses a writer left implicit are materialized BEFORE the
    // line goes, the column arm's own rule: an addressless `<row>` counts as "one
    // after its predecessor", so a line taken out from above it would otherwise be
    // renumbered by that count instead of moved by the shift, and the surviving
    // row would come out stating a number its cells no longer sit at.
    const materialized = materializeAddresses(xml);
    const target = sheetRows(materialized).find((candidate) => candidate.number === row);
    const without = target ? materialized.slice(0, target.start) + materialized.slice(target.end) : materialized;
    // The delete takes the content the line held, whether or not it moves a
    // surviving address: deleting the last used line leaves nothing to shift, and
    // the caveats about a workbook's charts and pivots are owed for a change of
    // content — a line the writer left holding no cell changed none. The shift
    // raises the notes that change owes (see `shiftSheet`).
    const tookCells = Boolean(target && rowCells(target.xml).length);
    return shiftSheet(editor, without, "row", row, -1, { stale, sheet: name, removed: tookCells, addressed: true });
  });
}

function insertColumn(editor, edit, stale) {
  const { part, name } = sheetPart(editor, edit.sheet);
  const column = columnNumber(edit.column);
  editor.sheet(part, (xml) => {
    const { columnMax } = usedRange(xml);
    if (columnMax === 0) throw new UsageError(`sheet ${JSON.stringify(name)} has no columns to insert into`);
    if (column > columnMax + 1) throw new UsageError(`column ${edit.column} is past the sheet's used range (A-${colName(columnMax - 1)})`);
    // The column arm of the row guard above: a column pushed past the last one
    // would have to keep the last column's own address.
    if (columnMax >= RULES.sheet_column_max) throw new UsageError(`sheet ${JSON.stringify(name)} reaches the last column (${colName(RULES.sheet_column_max - 1)}), so an insert would push a column past the sheet's grid`);
    staleRefs(editor, part, xml, stale, "column", column, 1, name);
    return shiftSheet(editor, xml, "column", column, 1, { stale, sheet: name });
  });
}

function deleteColumn(editor, edit, stale) {
  const { part, name } = sheetPart(editor, edit.sheet);
  const column = columnNumber(edit.column);
  editor.sheet(part, (xml) => {
    const { columnMax } = usedRange(xml);
    if (columnMax === 0) throw new UsageError(`sheet ${JSON.stringify(name)} has no columns`);
    if (column > columnMax) throw new UsageError(`column ${edit.column} is past the sheet's used range (A-${colName(columnMax - 1)})`);
    staleRefs(editor, part, xml, stale, "column", column, -1, name);
    // The addresses a writer left implicit are materialized before the column is
    // removed, so the cell that sits in the deleted column by position is found
    // rather than left behind.
    const materialized = materializeAddresses(xml);
    const stripped = removeColumn(materialized, column);
    // The row arm's rule: the delete takes the cells the column held, whether or
    // not it moves a surviving address.
    return shiftSheet(editor, stripped, "column", column, -1, { stale, sheet: name, removed: stripped !== materialized, addressed: true });
  });
}

// ── xlsx_edit: format_cells ────────────────────────────────────
// `format_cells` writes only the properties a request names; everything else on
// the cell is its own `<cellXfs>` entry's bytes (see `styledXf`). The closed sets
// are the shared rules', so an alignment with no reader value — or an inherited
// name such as "constructor" — is refused rather than written as one no reader
// knows, and a border side outside the rule's list is refused the same way.
const CELL_ALIGNMENTS = new Set(RULES.cell_alignments);
const CELL_VERTICALS = new Set(RULES.cell_verticals);

// The refusal a styles part whose root cannot be written into states: a
// self-closing `<styleSheet/>` is opened by `insertStylesChild` around whatever
// goes into it while a closed one takes children before its close, so either is
// writable; a part with no root element, or one whose root never closes, has
// nothing to write into, and a change that would splice into it must not be
// reported as written. Read through the same reader the writer uses
// (`stylesRoot`), so this opening check and the write cannot disagree about the
// root's spelling.
function requireStylesRoot(styles) {
  if (!stylesRoot(styles)) throw new UsageError("the workbook's styles part cannot be written into: its <styleSheet> root is missing or never closed");
}

// The part the workbook's own relationships name for its styles, resolved like a
// sheet's part is (`resolvePart` of the relationship's target) rather than assumed to
// sit at the conventional `XLSX_STYLES`: a workbook whose styles were saved elsewhere
// is one whose formatting has to go where the relationship points, or it lands in a
// part no reader consults while the cells' own `s=` keep pointing at the part the
// relationship names. The relationship list is read whether or not it can be written
// into — the same reading `workbookSheets` makes, so sheets and styles resolve by one
// rule — and the conventional name is the fallback for a package that names no styles
// part at all: the name a relationship has to be added for, by `nameStylesPart`.
function stylesPart(editor) {
  const named = relationships(editor.read(XLSX_WORKBOOK_RELS) ?? "").find((rel) => rel.type === STYLES_REL_TYPE);
  return named ? resolvePart(partDirectory(XLSX_WORKBOOK), named.target) : XLSX_STYLES;
}

// `name` — the styles part `stylesPart` resolved — made one the workbook's own
// relationships name: a part no relationship points at is one no reader reads, so the
// formatting would sit in a part nobody consults. Says whether the part ends up named.
// `false` is a package whose relationship list cannot be written into — absent, or
// holding no close of its own — which this kit does not refuse: a reader falls back to
// the conventional part names for such a package and so does this kit (see
// `workbookSheets`), and an edit that writes no style has nothing to gain from the
// naming. An edit that DOES write styles cannot accept it — see `stylesText`.
function nameStylesPart(editor, name) {
  const rels = editor.read(XLSX_WORKBOOK_RELS);
  if (!relsWritable(rels)) return false;
  if (relationships(rels).some((rel) => rel.type === STYLES_REL_TYPE)) return true;
  addRelationship(editor, XLSX_WORKBOOK, STYLES_REL_TYPE, name);
  return true;
}

// The text of `name`, the styles part `stylesPart` resolved, added and declared when the
// workbook has none: an edit that brings formatting to a workbook that never saved styles
// writes the part `createXlsx` would have rather than refusing, and the part a reader
// reads is the one the workbook's own relationships name — a package whose relationships
// cannot name it is refused (`nameStylesPart`) rather than answered with formatting
// nobody consults. The text is the caller's own read, charged like every other walk over
// a part, and the root was checked once before any edit ran (`xlsxEdit`).
function stylesText(editor, input, name) {
  if (!editor.zip.file("[Content_Types].xml")) throw missingContentTypes(input);
  if (!editor.zip.file(name)) editor.add(name, MINIMAL_STYLES);
  editor.part("[Content_Types].xml", (xml) => withOverride(xml, name, STYLES_CONTENT_TYPE));
  if (!nameStylesPart(editor, name)) throw new UsageError(`cannot edit ${nodePath.basename(input)}: the workbook's relationships cannot name a styles part, so the formatting would sit in a part no reader reads`);
  return editor.read(name);
}

// The rectangle an A1 cell or `A1:B2` range covers, its endpoints normalized so
// `first` is the top-left: `cellAddress` reads each end (refusing a bad address or
// one past the grid) and the ends are swapped. Every cell inside is on the grid by
// construction.
function gridRect(range) {
  const parts = typeof range === "string" ? range.split(":") : [];
  if (parts.length === 0 || parts.length > 2) throw new UsageError(`range must be an A1 address or an A1:B2 range, got: ${JSON.stringify(range)}`);
  const ends = parts.map((part) => cellAddress(part, "range"));
  const [a, b] = ends.length === 1 ? [ends[0], ends[0]] : ends;
  return {
    firstRow: Math.min(a.row, b.row),
    lastRow: Math.max(a.row, b.row),
    firstColumn: Math.min(a.column, b.column),
    lastColumn: Math.max(a.column, b.column),
  };
}

// The caller's own range: `gridRect` plus the one bound on how many cells a format
// may name.
function rangeRect(range) {
  const rect = gridRect(range);
  requireRangeCap(rect, range);
  return rect;
}

// The one bound that keeps a format from walking a rectangle no call could
// finish. A rect is checked when it is read and again when a merge widens it: a
// `<mergeCell>` covering more cells than the rule allows would otherwise slip
// past the check the caller's own range passed.
const requireRangeCap = (rect, label) => {
  const cells = (rect.lastRow - rect.firstRow + 1) * (rect.lastColumn - rect.firstColumn + 1);
  if (cells > RULES.format_cells_max) throw new UsageError(`the range ${label} covers ${cells} cells, more than the ${RULES.format_cells_max} one format may name`);
};

// The `A1:C1` (or single `A1`) spelling a rectangle is named by.
const rectLabel = (rect) => {
  const first = `${colName(rect.firstColumn)}${rect.firstRow}`;
  const last = `${colName(rect.lastColumn)}${rect.lastRow}`;
  return first === last ? first : `${first}:${last}`;
};

// A rectangle widened to cover every `<mergeCell>` it touches: Excel formats a
// merged cell as one, so a format aimed at part of one must land on the whole.
// Returns the widened rectangle, the label a reply names it by, whether it
// changed, and how many merges the widened rectangle covers — an L of merges
// closes over more than one, and the note a reply carries must not call them one.
// Each `<mergeCell ref>` is read through `gridRect`, so one this reader
// cannot parse or one the grid cannot hold is skipped rather than allowed to widen
// the rectangle to nonsense; the caller's own range is grid-checked, a merge's is
// not (see `gridRect`). Widening can bring a merge the rectangle did not touch into
// reach — an L of merges whose far arm only the first widening reaches — so the
// passes repeat until the rectangle stops growing, each pass taking in at least one
// merge. A merge that widens the rectangle past the cap is NOT skipped:
// `requireRangeCap` runs on the WIDENED rect at the call site, so a merge that
// covers more cells than the rule allows still refuses the call.
function mergedRect(xml, rect) {
  const merges = [...xml.matchAll(elementPattern("mergeCell"))].flatMap((match) => {
    const ref = (match[0].match(/\bref="([^"]*)"/) || [])[1];
    if (ref === undefined) return [];
    // A ref this reader cannot parse or one the grid cannot hold is skipped: the
    // caller's own range is grid-checked, a merge's is not, and widening to a ref
    // the grid cannot hold would write an address no sheet has.
    try { return [gridRect(ref)]; } catch { return []; }
  });
  let out = { ...rect };
  let growing = true;
  while (growing) {
    growing = false;
    for (const merge of merges) {
      if (merge.lastRow < out.firstRow || merge.firstRow > out.lastRow || merge.lastColumn < out.firstColumn || merge.firstColumn > out.lastColumn) continue;
      const grown = {
        firstRow: Math.min(out.firstRow, merge.firstRow),
        lastRow: Math.max(out.lastRow, merge.lastRow),
        firstColumn: Math.min(out.firstColumn, merge.firstColumn),
        lastColumn: Math.max(out.lastColumn, merge.lastColumn),
      };
      if (grown.firstRow !== out.firstRow || grown.lastRow !== out.lastRow || grown.firstColumn !== out.firstColumn || grown.lastColumn !== out.lastColumn) growing = true;
      out = grown;
    }
  }
  const widened = out.firstRow !== rect.firstRow || out.lastRow !== rect.lastRow || out.firstColumn !== rect.firstColumn || out.lastColumn !== rect.lastColumn;
  // How many merges the widened rectangle covers, so the note can say whether the
  // format landed on one merged cell or on several: widening closes over every
  // merge a merge reaches, so a rectangle grown by an L of them covers more than
  // one and a note that called them one would be wrong.
  const covered = merges.filter((merge) => !(merge.lastRow < out.firstRow || merge.firstRow > out.lastRow || merge.lastColumn < out.firstColumn || merge.firstColumn > out.lastColumn)).length;
  return { rect: out, label: rectLabel(out), widened, covered };
}

// A `border` request: an object whose keys are the sides a cell draws and whose
// values are booleans, at least one side named. An unknown key, a non-boolean
// value or an empty object is refused rather than silently dropped, since the
// caller would otherwise be told a border was applied that no reader shows.
function rangeBorder(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new UsageError(`border must be an object naming sides (${listed(CELL_BORDER_SIDES)}), got: ${JSON.stringify(value)}`);
  const keys = Object.keys(value);
  if (!keys.length) throw new UsageError(`border must name at least one side (${listed(CELL_BORDER_SIDES)})`);
  const border = {};
  for (const key of keys) {
    if (!CELL_BORDER_SIDES.includes(key)) throw new UsageError(`border has no side ${JSON.stringify(key)} — hint: use ${listed(CELL_BORDER_SIDES)}`);
    if (typeof value[key] !== "boolean") throw new UsageError(`border.${key} must be true or false`);
    border[key] = value[key];
  }
  return border;
}

// The properties a `range` target names, normalized for `styledXf`: each one
// bounded, colours reduced to their digits and booleans checked, so a value the
// Rust boundary refused first is refused here too rather than written. A target
// that names none is refused — the op would otherwise rewrite the sheet and
// change nothing.
function rangeFormat(edit) {
  const format = {};
  let named = false;
  if (edit.font !== undefined) {
    const font = editText(edit.font, "font");
    if (!font) throw new UsageError("font must not be empty");
    if ([...font].length > RULES.font_name_max) throw new UsageError(`font must be at most ${RULES.font_name_max} characters, got: ${[...font].length}`);
    format.font = font;
    named = true;
  }
  if (edit.size !== undefined) {
    if (!inSpan(edit.size, RULES.cell_font_points)) throw new UsageError(`size must be ${spanBounds(RULES.cell_font_points)} points, got: ${JSON.stringify(edit.size)}`);
    format.size = edit.size;
    named = true;
  }
  for (const toggle of ["bold", "italic"]) {
    if (edit[toggle] === undefined) continue;
    if (typeof edit[toggle] !== "boolean") throw new UsageError(`${toggle} must be true or false`);
    format[toggle] = edit[toggle];
    named = true;
  }
  if (edit.color !== undefined) {
    format.color = hexDigits(edit.color).toUpperCase();
    named = true;
  }
  if (edit.fill !== undefined) {
    format.fill = edit.fill === false ? false : hexDigits(edit.fill).toUpperCase();
    named = true;
  }
  if (edit.border !== undefined) {
    format.border = rangeBorder(edit.border);
    named = true;
  }
  if (edit.align !== undefined) {
    if (!CELL_ALIGNMENTS.has(edit.align)) throw new UsageError(`align must be ${listed([...CELL_ALIGNMENTS])}, got: ${JSON.stringify(edit.align)}`);
    format.align = edit.align;
    named = true;
  }
  if (edit.vertical !== undefined) {
    if (!CELL_VERTICALS.has(edit.vertical)) throw new UsageError(`vertical must be ${listed([...CELL_VERTICALS])}, got: ${JSON.stringify(edit.vertical)}`);
    format.vertical = edit.vertical;
    named = true;
  }
  if (edit.wrap !== undefined) {
    if (typeof edit.wrap !== "boolean") throw new UsageError("wrap must be true or false");
    format.wrap = edit.wrap;
    named = true;
  }
  if (edit.number_format !== undefined) {
    const code = editText(edit.number_format, "number_format");
    if (!code) throw new UsageError("number_format must not be empty");
    if ([...code].length > RULES.number_format_max) throw new UsageError(`a number format must be at most ${RULES.number_format_max} characters, got: ${[...code].length}`);
    format.number_format = code;
    named = true;
  }
  if (!named) throw new UsageError("a format_cells range target names no property — hint: name at least one of font, size, bold, italic, color, fill, border, align, vertical, wrap or number_format");
  return format;
}

// The style a `<row>` itself carries, which the cells that name none inherit: it
// is stated with `customFormat="1"` — a bare `s=` a writer left on a row is not a
// style a reader applies — and a `customFormat` row with no readable `s=` takes
// the schema default, entry 0, so the row still wins over a covering `<col>`. A
// row that is not there at all has no style, so a failed lookup's `undefined`
// passes through.
const rowStyleOf = (rowXml) => {
  if (rowXml === undefined) return undefined;
  const open = elementParts(rowXml, "row").open;
  return /customFormat="1"/.test(open) ? (styleOf(open) ?? "0") : undefined;
};

// The `<col>` entries of a sheet that state a `style`, each with the span it covers:
// a cell that names no style of its own takes the last entry covering its column.
const columnStyleEntries = (sheet) => colEntries(sheet)
  .filter((entry) => entry.style !== undefined && Number.isFinite(entry.min) && Number.isFinite(entry.max));

// The style a cell that names no `s=` of its own renders by: the row's when it
// has one — the row wins, as a reader applies them — else the last `<col>` entry
// covering the column, else `undefined`, which is entry 0.
const inheritedStyle = (rowStyle, columns, column) => {
  if (rowStyle !== undefined) return rowStyle;
  let found;
  for (const entry of columns) if (column + 1 >= entry.min && column + 1 <= entry.max) found = entry.style;
  return found;
};

// The `format_cells` op on a cell range: the sheet rebuilt in ONE pass, so the
// rect's cells are rewritten and its missing rows and cells created while rows
// outside the rect stay byte-identical (a body whose rows are not in address order
// is written back in it — see the rows below). Every read is charged — the styles
// part by `stylesText`, the sheet by the caller, each new style entry's parse here —
// so a rect the rule allows but no machine could finish is refused the way any other
// call over its budget is. The rebuilt text is RETURNED rather than written, so the
// `sheet` call that read the part writes it once against the text it arrived as.
function formatCells(editor, sheet, rect, edit, input, stylesName) {
  const openedStyles = stylesText(editor, input, stylesName);
  let styles = openedStyles;
  const indices = new Map();
  // The index a cell's base style takes — its own `s=`, or the style it inherits
  // from its row or column when it names none — computed once per distinct base
  // against the styles part as it grows: a rect may hold `RULES.format_cells_max`
  // cells, and re-parsing the styles part for every cell of one base would be
  // quadratic in the worst case the rule allows. The parse each new base costs is
  // charged, so a rect of all-distinct bases is refused with a split hint rather
  // than run into the kit's own time limit.
  const indexFor = (base) => {
    const key = base ?? "";
    let index = indices.get(key);
    if (index === undefined) {
      editor.charge(styles.length);
      const applied = cellStyleIndex(styles, edit, base);
      styles = applied.styles;
      index = String(applied.index);
      indices.set(key, index);
    }
    return index;
  };
  const colStyles = columnStyleEntries(sheet);
  const present = new Set(sheetRows(sheet).map((row) => row.number));
  // The rows of the rect the sheet states no element for: the numbers come from
  // the sheet's own set rather than from a cursor over document order, so a
  // document that is not ascending cannot make one be created twice. An addressless
  // `<row>` counts as "one after its predecessor", so a row element inserted among
  // such rows renumbers every successor (see `materializeAddresses`). The addresses
  // are made explicit before any row is inserted, but only when one of these has to
  // be written.
  const missing = [];
  for (let row = rect.firstRow; row <= rect.lastRow; row += 1) {
    if (!present.has(row)) missing.push(row);
  }
  const source = ensureSheetData(missing.length ? materializeAddresses(sheet) : sheet);
  const body = sheetBody(source);
  let created = false;
  const newCell = (column, number, rowStyle) => {
    created = true;
    return `<c r="${colName(column)}${number}" s="${indexFor(inheritedStyle(rowStyle, colStyles, column))}"/>`;
  };
  // A whole row the sheet has none of: one created cell per column of the rect.
  const rowElement = (number) => {
    const cells = [];
    for (let column = rect.firstColumn; column <= rect.lastColumn; column += 1) cells.push(newCell(column, number, undefined));
    return `<row r="${number}">${cells.join("")}</row>`;
  };
  const rowBody = (row) => {
    const stated = rowCells(row.xml);
    // The row's own cells in address order: the row is rebuilt here either way, so its
    // cells go back the way a reader reads them rather than the way they arrived, each
    // address stated once (a row whose cells state none is refused — `ascending`).
    const cells = ascending(stated, (cell) => cell.column, "the cells of a sheet row");
    const rowStyle = rowStyleOf(row.xml);
    const inRect = (column) => column >= rect.firstColumn && column <= rect.lastColumn;
    // The rect's columns the row holds no cell for, in address order: emitted before
    // the first stated cell they precede, so the row is rebuilt in ONE pass rather
    // than rescanned per created cell (a rect may hold `RULES.format_cells_max` cells
    // in one row). The cursor is the next column of the rect not yet written, so a
    // cell LEFT of the rect leaves it where it is — stepping it back would create a
    // neighbouring column the caller never named.
    let column = rect.firstColumn;
    const pending = (before) => {
      let out = "";
      while (column <= rect.lastColumn && column < before) {
        out += newCell(column, row.number, rowStyle);
        column += 1;
      }
      return out;
    };
    if (!cells.length) {
      const filled = pending(Number.POSITIVE_INFINITY);
      // A self-closing `<row r="1"/>` is opened to hold them; one already open
      // takes them before its close (`elementParts` reads either spelling).
      const { open, body } = elementParts(row.xml, "row");
      return `${open}${body}${filled}</row>`;
    }
    // The row's own text around its cells — its open tag and whatever follows the
    // last one — is kept as it stands: the text BETWEEN cells is a writer's own
    // layout, and the row is written from the cells it holds.
    let out = "";
    for (const cell of cells) {
      out += pending(cell.column);
      // An existing in-rect cell has only its `s=` set — to the style it already
      // renders as, its own or the one it inherits; one outside the rect is
      // byte-identical, and the created columns go at their own place.
      out += inRect(cell.column)
        ? setXmlAttribute(cell.xml, "s", indexFor(styleOf(cell.xml) ?? inheritedStyle(rowStyle, colStyles, cell.column)))
        : cell.xml;
      column = Math.max(column, cell.column + 1);
    }
    out += pending(Number.POSITIVE_INFINITY);
    return row.xml.slice(0, stated[0].start) + out + row.xml.slice(stated[stated.length - 1].end);
  };
  // The rows of the body in the order they go back: one the sheet states in address
  // order is written where it stands, with the text between its rows kept byte for
  // byte, while a body whose rows are not in order — a shape no writer produces — is
  // rebuilt in address order, since the body is rewritten here anyway and each of the
  // rect's rows is then stated once (a row stating no number of its own cannot be
  // reordered and is refused — `ascending`).
  const stated = sheetRows(source);
  const rows = ascending(stated, (row) => row.number, "the sheet's rows");
  const kept = rows === stated;
  let text = source.slice(0, body.index) + (body.selfClosing ? `${body.open.slice(0, -2)}>` : body.open);
  let cursor = body.bodyStart;
  let next = 0;
  const absent = (before) => {
    let out = "";
    while (next < missing.length && (before === null || missing[next] < before)) {
      out += rowElement(missing[next]);
      next += 1;
    }
    return out;
  };
  for (const row of rows) {
    if (kept) text += source.slice(cursor, row.start);
    text += absent(row.number);
    if (row.number >= rect.firstRow && row.number <= rect.lastRow) {
      text += rowBody(row);
    } else {
      text += source.slice(row.start, row.end);
    }
    cursor = row.end;
  }
  text += absent(null);
  if (kept) text += source.slice(cursor, body.bodyEnd);
  text += body.selfClosing ? `</sheetData>${source.slice(body.bodyEnd)}` : source.slice(body.bodyEnd);
  // The sheet's declared reach covers the addressed rectangle only when a cell
  // was really created: rewriting an existing cell's style moves no address, and
  // a ref widened for it would be a change the caller did not ask for.
  if (created) {
    text = widenDimension(text, rect.firstRow, rect.firstColumn);
    text = widenDimension(text, rect.lastRow, rect.lastColumn);
  }
  editor.rewrite(stylesName, openedStyles, styles);
  return text;
}

// `entry` — a `colEntries` entry, so its own text and span are already read — split
// so exactly `column` names `width` while the spans on either side keep the entry's
// own attributes: each side is the same `<col>` with its `min`/`max` narrowed, and
// the named column's copy has the new width and `customWidth` stated. A span the
// named column does not touch produces no side piece.
function splitColumn(entry, column, width) {
  const span = (lo, hi) => setXmlAttribute(setXmlAttribute(entry.text, "min", String(lo)), "max", String(hi));
  const parts = [];
  if (column > entry.min) parts.push(span(entry.min, column - 1));
  parts.push(setXmlAttribute(setXmlAttribute(span(column, column), "width", String(width)), "customWidth", "1"));
  if (column < entry.max) parts.push(span(column + 1, entry.max));
  return parts;
}

// The `format_cells` op on a column: the width goes into a `<col min max width
// customWidth="1"/>` entry covering exactly that column. An entry covering a
// WIDER span is SPLIT so the columns on either side keep exactly the entry they
// had (same width, style, hidden, outlineLevel, collapsed, bestFit bytes) and
// only the named column changes; a sheet with no `<cols>` block gets one created
// directly before `<sheetData>`, the place `CT_Worksheet` requires of it.
function formatColumn(editor, part, edit) {
  const column = columnNumber(edit.column);
  const width = edit.width;
  editor.sheet(part, (xml) => {
    const source = ensureSheetData(xml);
    // The block the entry goes into, or `undefined` for a sheet that states none; one
    // the part leaves open is refused (`writableElement`).
    const cols = writableElement(source, "cols", "the sheet part");
    const entries = colEntries(source);
    const covering = entries.findIndex((entry) => Number.isFinite(entry.min) && column >= entry.min && column <= entry.max);
    const written = entries.map((entry) => entry.text);
    if (covering >= 0) {
      written[covering] = splitColumn(entries[covering], column, width).join("");
    } else {
      const placed = entries.findIndex((entry) => entry.min > column);
      // The new entry is written at its `min`'s place, so the block stays ordered
      // whether or not the sheet had a `<col>` for the column already.
      written.splice(placed < 0 ? written.length : placed, 0, `<col min="${column}" max="${column}" width="${width}" customWidth="1"/>`);
    }
    const block = `<cols>${written.join("")}</cols>`;
    if (cols) return source.slice(0, cols.index) + block + source.slice(cols.index + cols[0].length);
    // No `<cols>` to write into: the block goes at its own `CT_Worksheet` place,
    // directly before the body the sheet's rows live in. Only that element's place is
    // read (`writableElement`; `ensureSheetData` above guarantees it exists), and no
    // row is: a column width never writes into the body, so a part whose rows cannot
    // be read is no reason to refuse it.
    const body = writableElement(source, "sheetData", "the sheet part");
    return source.slice(0, body.index) + block + source.slice(body.index);
  });
}

// The `format_cells` op on a row: `ht` and `customHeight` are set on the row
// element, everything else on it (its `spans`, its `s`, its cells) kept. A sheet
// with no such row gets `<row r="N" ht="…" customHeight="1"/>` inserted in row
// order; the addresses are made explicit before the element goes in, since an
// addressless row counts as "one after its predecessor" and an inserted element
// would renumber the rows after it.
function formatRow(editor, part, edit) {
  const row = rowNumber(edit.row);
  const height = edit.height;
  editor.sheet(part, (xml) => {
    const existing = sheetRows(xml).find((candidate) => candidate.number === row);
    if (existing) {
      const element = setXmlAttribute(setXmlAttribute(existing.xml, "ht", String(height)), "customHeight", "1");
      return xml.slice(0, existing.start) + element + xml.slice(existing.end);
    }
    const source = materializeAddresses(xml);
    return insertSheetRow(source, sheetRows(source), row, `<row r="${row}" ht="${height}" customHeight="1"/>`);
  });
}

// The `format_cells` op: a cell range, a column or a row, exactly one target
// named and exactly the properties that target takes. Only the keys a target
// knows are read — a key outside its set is the Rust boundary's refusal — and
// every bound is stated here as the kit's own last line. A range target's sheet
// part is read and charged once here (`sheet`), then handed to `mergedRect` and
// `formatCells`, which returns the text the same call writes.
function xlsxFormatCells(editor, edit, input, stylesName) {
  const { part } = sheetPart(editor, edit.sheet);
  const targets = ["range", "column", "row"].filter((key) => edit[key] !== undefined);
  if (!targets.length) throw new UsageError("a format_cells edit needs one of range, column or row");
  if (targets.length > 1) throw new UsageError(`a format_cells edit names more than one target (${listed(targets)}), but takes exactly one`);
  if (targets[0] === "range") {
    const rect = rangeRect(edit.range);
    const format = rangeFormat(edit);
    editor.sheet(part, (sheet) => {
      const merged = mergedRect(sheet, rect);
      if (merged.widened) {
        requireRangeCap(merged.rect, `${merged.label} (${rectLabel(rect)} widened by a merged cell)`);
        const merge = merged.covered === 1 ? "those cells are one merged cell" : `those cells are covered by ${merged.covered} merged cells`;
        editor.note(`[the range ${rectLabel(rect)} was widened to ${merged.label}: ${merge}]`);
      }
      return formatCells(editor, sheet, merged.rect, format, input, stylesName);
    });
    return;
  }
  if (targets[0] === "column") {
    if (edit.width === undefined) throw new UsageError("a format_cells column target names no width — hint: name width");
    if (!inSpan(edit.width, RULES.column_width_chars)) throw new UsageError(`width must be ${spanBounds(RULES.column_width_chars)} characters, got: ${JSON.stringify(edit.width)}`);
    formatColumn(editor, part, edit);
    return;
  }
  if (edit.height === undefined) throw new UsageError("a format_cells row target names no height — hint: name height");
  if (!inSpan(edit.height, RULES.row_height_points)) throw new UsageError(`height must be ${spanBounds(RULES.row_height_points)} points, got: ${JSON.stringify(edit.height)}`);
  formatRow(editor, part, edit);
}

function xlsxEdit(req) {
  const editor = openEdit(req, "xlsx");
  const edits = editList(req);
  const names = Object.keys(editor.zip.files);
  // A chart or a pivot that reads the edited cells keeps the values it was
  // written with: the parts are copied through and nothing recomputes them. The
  // test is on the part names the format gives those objects — a chart's colour
  // or style sidecar (`xl/charts/colors1.xml`) is not a chart and raises no
  // caveat of its own.
  const charts = names.some((name) => /^xl\/charts\/chart[^/]*\.xml$/.test(name));
  const pivots = names.some((name) => /^xl\/pivotTables\/pivotTable[^/]*\.xml$/.test(name) || /^xl\/pivotCache\/pivotCache(?:Definition|Records)[^/]*\.xml$/.test(name));
  // The styles part is taken into hand before any edit runs, whatever the edit is: the
  // part the workbook's own relationships name (`stylesPart`), a root not one a child
  // can be written into (`requireStylesRoot`) and a part the workbook does not name
  // (`nameStylesPart` — best effort here, while a styles-writing edit refuses a
  // workbook whose relationships cannot name it, see `stylesText`). The read is charged
  // (`read`), as is the one the edit that grows the part makes itself, so the work
  // budget counts the part twice on a styles-writing call — the bound is on the work a
  // call does, not on the distinct bytes it touches. What the part holds between its
  // entries is not vetted, since an input already damaged there is no worse for this
  // edit. A workbook with no styles part has nothing to check and keeps working.
  const stylesName = stylesPart(editor);
  const styles = editor.read(stylesName);
  if (styles !== undefined) {
    requireStylesRoot(styles);
    nameStylesPart(editor, stylesName);
  }
  // The stale references a shift cannot rewrite, collected while the edits run
  // so the reply names them once.
  const stale = new Set();
  for (const edit of edits) {
    const op = edit && edit.op;
    if (op === "set_cell") setCell(editor, edit, req.input, stylesName);
    else if (op === "clear_cell") clearCell(editor, edit);
    else if (op === "insert_row") insertRow(editor, edit, stale);
    else if (op === "delete_row") deleteRow(editor, edit, stale);
    else if (op === "insert_column") insertColumn(editor, edit, stale);
    else if (op === "delete_column") deleteColumn(editor, edit, stale);
    else if (op === "format_cells") xlsxFormatCells(editor, edit, req.input, stylesName);
    else throw new UsageError(`unknown xlsx edit: ${JSON.stringify(op)}`);
  }
  // The editor knows whether the call changed the sheet's content at all — a
  // shift that moved nothing writes no part, a delete that took the last line of a
  // sheet with no extent to shrink removes cells though it moves no address, and a
  // repeated write states what the cell already held. A note about a chart or a
  // pivot is owed on that: nothing recomputes their cached values when a value or
  // an address really changed, and a call that changed neither leaves them as
  // valid as they were.
  const content = editor.content();
  if (content && charts) editor.note("[the workbook's charts were copied unchanged — nothing recomputes them, so a chart that reads the changed cells keeps the values it had]");
  if (content && pivots) editor.note("[the workbook's pivot tables were copied unchanged — nothing recomputes them, so a pivot that reads the changed cells keeps the values it had]");
  return editor.finish();
}

// ── pptx_edit ──────────────────────────────────────────────────
// A slide is addressed by its 1-based number as the reader shows it, which is
// the order `src/ooxml.rs::resolve_slide_parts` resolves the slide list in:
// `ppt/presentation.xml`'s `<p:sldIdLst>` relationship ids through
// `ppt/_rels/presentation.xml.rels`, and, when neither resolves, the
// conventionally numbered `ppt/slides/slideN.xml` parts. The model reads a deck
// with that reader and edits it here, so the two must agree on what "slide N" is
// — a change to either resolution belongs in both.
const PPT_PRESENTATION = "ppt/presentation.xml";
const PPT_RELS = "ppt/_rels/presentation.xml.rels";
const PPT_BASE = "ppt/";
const PPT_SLIDES = "ppt/slides/";
const PPT_LAYOUTS = "ppt/slideLayouts/";
const PPT_NOTES_SLIDES = "ppt/notesSlides/";
const PPT_NOTES_MASTERS = "ppt/notesMasters/";
const PPT_MEDIA = "ppt/media/";
const SLIDE_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml";
const NOTES_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml";
// The relationships namespace the parts this kit writes declare as `r`, and the
// base its relationship types are built from.
const REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const REL_BASE = `${REL_NS}/`;
const SLIDE_REL = `${REL_BASE}slide`;
const SLIDE_LAYOUT_REL = `${REL_BASE}slideLayout`;
const NOTES_REL = `${REL_BASE}notesSlide`;
const NOTES_MASTER_REL = `${REL_BASE}notesMaster`;
const IMAGE_REL = `${REL_BASE}image`;
// The run language of a slide this kit writes; a placeholder inherits its type,
// size and bullets from the layout, so nothing else about it is stated here.
const SLIDE_LANG = "en-US";
// The alignments a `format_text` may name, in the order the shared rules state
// them: each word a caller may name and the `algn` value ECMA-376 writes for it.
const SLIDE_ALIGNMENTS = new Map(RULES.slide_alignments);
// The slide size a presentation stating none is read at, in EMU: the
// 10in × 7½in one MS-OI29500 Part 1 §19.2.1.39 states PowerPoint assumes — a
// `<p:sldSz cx="9144000" cy="6858000"/>` — when the element is absent. An image
// on such a slide is placed by fraction of this.
const SLIDE_SIZE_DEFAULT = { width: 9144000, height: 6858000 };
// One CSS pixel — the unit an image's own pixel size is stated in — in EMU.
const PX_TO_EMU = 9525;

// The value of the attribute `name` of an XML tag. A longer name never matches a
// shorter one (`\bTarget=` is not `TargetMode=`, and the `Id` of a relationship
// is not the `rId` a slide points with).
const xmlAttribute = (tag, name) => (tag.match(new RegExp(`\\b${name}="([^"]*)"`)) || [])[1];
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// The relationship id of an element that refers to a part (`r:id="rId1"`): the
// prefixed `id`, never the element's own unprefixed one, the way the reader's
// `rel_id` picks it out.
const relationshipId = (tag) => (tag.match(/\b[A-Za-z0-9_]+:id="([^"]*)"/) || [])[1];

// One element of `name` whose attribute `attr` holds exactly `value`, in either
// shape it can be written in — the self-closing one and the paired one — so an
// element written the paired way goes with the self-closing one. Anchored on the
// attribute's own value, so a longer value never matches a shorter one and the
// element's own text is never rewritten. Built on the same two-shape rule as
// `elementPattern` (see `elementForms`).
const elementWithAttribute = (name, attr, value) => elementForms(name, `\\b[^<>]*\\b${attr}="${escapeRegExp(value)}"[^<>]*`);

// One `<Relationship>` of a `.rels` part; an element without an id or a target
// names no part and is dropped, the way the reader's own scan keeps only
// complete ones.
function relationships(xml) {
  const list = [];
  for (const match of xml.matchAll(/<Relationship\b[^>]*>/g)) {
    const id = xmlAttribute(match[0], "Id");
    const target = xmlAttribute(match[0], "Target");
    if (id === undefined || target === undefined) continue;
    list.push({ id, target, type: xmlAttribute(match[0], "Type") || "" });
  }
  return list;
}
const relationshipMap = (xml) => new Map(relationships(xml).map((rel) => [rel.id, rel.target]));

// A relationship target resolved against the part that owns it, the way
// `src/ooxml.rs::resolve_part` does: an absolute target drops its leading slash,
// a relative one appends to `base`, and `..` segments are folded away.
function resolvePart(base, target) {
  const path = target.startsWith("/") ? target.slice(1) : base + target;
  const parts = [];
  for (const segment of path.split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") parts.pop();
    else parts.push(segment);
  }
  return parts.join("/");
}

// The `.rels` part that belongs to `part`, by the OPC convention all three
// families follow.
function relsPartFor(part) {
  const cut = part.lastIndexOf("/");
  return `${part.slice(0, cut)}/_rels/${part.slice(cut + 1)}.rels`;
}

// The directory a part's own relationships resolve their targets against.
const partDirectory = (part) => part.slice(0, part.lastIndexOf("/") + 1);

// Whether a `.rels` part is one a relationship can be written into: a part with no
// close of its own has nowhere to insert one. The close is read with the shared
// spelling rule (see `closeTag`), so a part spelled `</Relationships >` is not read
// as one that never closes. A package whose relationships cannot be read is still
// edited — a reader falls back to the conventional part names for it and so does
// this kit (`workbookSheets`) — so this says what can be written, not what is
// refused.
const RELATIONSHIPS_CLOSE = new RegExp(closeTag("Relationships"));
const relsWritable = (rels) => rels !== undefined && RELATIONSHIPS_CLOSE.test(rels);
// Add a relationship of `type` to `part`'s own relationships, naming the
// absolute package path `target`, and return the fresh id it took. The
// relationship part is created when the package has none.
function addRelationship(editor, part, type, target) {
  const rels = relsPartFor(part);
  const file = editor.zip.file(rels);
  // Every id the owning part already holds is stepped over: the `.rels` part's own
  // `Id="rIdN"` when the package has one, else the part's own `r:id="rIdN"`
  // references — a package with no `.rels` part still spells references of its own
  // (`<sheet r:id="rId1"/>`), and an id one of those took must not be handed to the
  // new relationship, or the reference would point at it. Two relationships sharing
  // an id name each other either way, so an element this kit would not read counts.
  const holder = file ? file.asText() : editor.zip.file(part)?.asText() ?? "";
  editor.charge(holder.length);
  const pattern = file ? /\bId="rId(\d+)"/g : /\br:id="rId(\d+)"/g;
  const used = [...holder.matchAll(pattern)].map((match) => Number(match[1]));
  const id = `rId${Math.max(0, ...used) + 1}`;
  const element = `<Relationship Id="${id}" Type="${type}" Target="${relativeTarget(partDirectory(part), target)}"/>`;
  if (!file) {
    editor.add(rels, `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${element}</Relationships>`);
    return id;
  }
  editor.part(rels, (xml) => {
    const close = xml.match(RELATIONSHIPS_CLOSE);
    if (!close) throw new UsageError(`the relationships of ${part} are not readable`);
    return xml.slice(0, close.index) + element + xml.slice(close.index);
  });
  return id;
}

// The package's `<base>N.xml` parts: the number each carries and the name it
// really has. A package that numbers with a leading zero (`slideLayout01.xml`)
// holds no `slideLayout1.xml`, so a reference to a part that is already there
// must use the matched name; only a name for a part about to be written may be
// built from the number.
const numberedParts = (zip, base) => Object.keys(zip.files).flatMap((name) => {
  const match = name.match(new RegExp(`^${escapeRegExp(base)}(\\d+)\\.xml$`));
  return match ? [{ number: Number(match[1]), name }] : [];
});
// Those parts in the order the reader numbers them: by number, then by name.
const byPartNumber = (a, b) => a.number - b.number || (a.name < b.name ? -1 : 1);
// The `ppt/slides/slideN.xml` numbers the package holds, whatever order the
// presentation lists them in, and the notes-slide ones — the numbers a fresh
// part of that kind takes the next of.
const slideFileNumbers = (zip) => numberedParts(zip, `${PPT_SLIDES}slide`).map((part) => part.number);
const notesSlideNumbers = (zip) => numberedParts(zip, `${PPT_NOTES_SLIDES}notesSlide`).map((part) => part.number);
// The name of the numerically first `<base>N.xml` part a package holds, or null
// when it holds none.
function firstPartName(zip, base) {
  const parts = numberedParts(zip, base).sort(byPartNumber);
  return parts.length ? parts[0].name : null;
}

// The slide parts in the order the presentation's own `<p:sldId>` list declares
// them, or `null` when that list is not what the reader numbers by: a presentation
// without a readable list, or one whose entries resolve to nothing, is read by the
// conventional file naming instead. A slot is null when that slide's relationship
// cannot be resolved, so the slides after it keep their own number; a `<p:sldId>`
// without a relationship id is not a slide the reader numbers at all — its own
// scan keeps only elements `rel_id` returns something for — and is dropped here
// too rather than counted as an unreadable one.
function listedSlideParts(zip) {
  const presentation = zip.file(PPT_PRESENTATION);
  const rels = zip.file(PPT_RELS);
  if (!presentation || !rels) return null;
  const map = relationshipMap(rels.asText());
  const ids = [...presentation.asText().matchAll(/<p:sldId\b[^>]*>/g)].map((match) => relationshipId(match[0])).filter((id) => id !== undefined);
  const parts = ids.map((id) => {
    const target = map.get(id);
    return target === undefined ? null : resolvePart(PPT_BASE, target);
  });
  // One resolvable slide is enough to trust the declared order.
  return parts.some(Boolean) ? parts : null;
}

// The slide parts in presentation order, one slot per slide the reader would
// number. Mirrors the reader (see above).
function slideParts(zip) {
  const listed = listedSlideParts(zip);
  if (listed) return listed;
  const named = numberedParts(zip, `${PPT_SLIDES}slide`);
  named.sort(byPartNumber);
  return named.map((slide) => slide.name);
}

// The part slide number `number` addresses, refused when the deck has no such
// slide (named with the count the reader would number) or when the part it names
// is not in the package.
function addressedSlide(zip, number) {
  const parts = slideParts(zip);
  if (!Number.isInteger(number) || number < 1 || number > parts.length) {
    throw new UsageError(`no slide ${number}: this presentation has ${parts.length} slide(s)`);
  }
  const part = parts[number - 1];
  if (!part || !zip.file(part)) throw new UsageError(`slide ${number} could not be read`);
  return part;
}

// The notes part a slide declares through its own relationships, the way
// `src/ooxml.rs::slide_notes` resolves them, or null when it declares none.
function slideNotesPart(zip, slidePart) {
  const rels = zip.file(relsPartFor(slidePart));
  if (!rels) return null;
  const rel = relationships(rels.asText()).find((candidate) => candidate.type.endsWith("/notesSlide"));
  return rel ? resolvePart(PPT_SLIDES, rel.target) : null;
}

// The layout part a slide's relationship names, or null when it has none.
function slideLayoutPart(zip, slidePart) {
  const rels = zip.file(relsPartFor(slidePart));
  if (!rels) return null;
  const rel = relationships(rels.asText()).find((candidate) => candidate.type.endsWith("/slideLayout"));
  return rel ? resolvePart(PPT_SLIDES, rel.target) : null;
}

// The layout the new slide is inserted after uses, the deck's last slide's, or
// the first the package holds. A slide that names a layout the package does not
// have falls through to one that is really there.
function newSlideLayout(zip, slides, after) {
  const source = after === undefined ? slides[slides.length - 1] : slides[after - 1];
  const layout = source && slideLayoutPart(zip, source);
  if (layout && zip.file(layout)) return layout;
  const first = firstPartName(zip, `${PPT_LAYOUTS}slideLayout`);
  if (!first) throw new UsageError("the presentation has no slide layout to build a new slide on");
  return first;
}

// A target for `to` written relative to the directory `from`, e.g. the layout a
// `ppt/slides/slideN.xml.rels` points at (`../slideLayouts/slideLayout1.xml`).
function relativeTarget(from, to) {
  const fromParts = from.split("/").filter(Boolean);
  const toParts = to.split("/").filter(Boolean);
  let common = 0;
  while (common < fromParts.length && common < toParts.length - 1 && fromParts[common] === toParts[common]) common += 1;
  return [...fromParts.slice(common).map(() => ".."), ...toParts.slice(common)].join("/");
}

// A `<a:p>` paragraph and an `<a:r>` run of a slide part. `<a:pPr>` and
// `<a:rPr>` are not paragraph/run starts: the character after `a:p`/`a:r` is `P`,
// which the optional whitespace-and-attributes group does not swallow.
const SLIDE_PARAGRAPH = tagPattern("a:p");
const SLIDE_RUN = tagPattern("a:r");
const SLIDE_RUN_PROPERTIES = elementPattern("a:rPr");
const SLIDE_PPR = elementPattern("a:pPr");
// The text element of a drawing part, the same pattern the filler substitutes
// through.
const SLIDE_TEXT = runPattern("a:");

// The `[start, end]` spans of a part's `<a:p>` paragraphs, at any nesting — a
// table cell's `p:txBody`, a group shape — in part order. An empty paragraph
// written self-closing still counts, so "the last paragraph" is the last one a
// reader would see.
function slideParagraphs(xml) {
  return tagSpans(xml, SLIDE_PARAGRAPH).map(({ start, end }) => ({ start, end }));
}

// The slide's shape tree and the object shapes its own text never lives in: a
// `<p:graphicFrame>` carries a table or a chart, a `<p:grpSp>` holds other
// shapes. A top-level `<p:sp>` directly under the shape tree is a shape the
// slide's text belongs to (a title or body placeholder).
const SLIDE_SHAPE_TREE = tagPattern("p:spTree");
const SLIDE_SHAPE = tagPattern("p:sp");
const SLIDE_FRAME = tagPattern("p:graphicFrame");
const SLIDE_GROUP = tagPattern("p:grpSp");
const SLIDE_TX_BODY = new RegExp(`${openTag("p:txBody")}([\\s\\S]*?)(${closeTag("p:txBody")})`);

// The `<p:txBody>` span of a shape, or null when it holds none (a self-closing
// `<p:txBody/>` has nowhere for a paragraph to go). `closeAt` is where the body's
// own close tag begins — the place a paragraph added at its end goes — taken from
// the match rather than from a close tag spelled out again.
function shapeTextBody(xml, shape) {
  const body = xml.slice(shape.start, shape.end).match(SLIDE_TX_BODY);
  if (!body) return null;
  const start = shape.start + body.index;
  const end = start + body[0].length;
  return { start, end, closeAt: end - body[2].length };
}

// The slide's OWN text: the `<p:txBody>` bodies of its top-level `<p:sp>` shapes
// and the paragraphs inside them. A table's cells, a chart's frame and a group
// shape's children are not the slide's text, so a paragraph added here never
// lands in a frame the caller did not address. This is deliberately narrower than
// `slideParagraphs`, which numbers every paragraph (tables included) for the
// text ops that address the same text the reader shows.
function slideTextBodies(xml) {
  const tree = tagSpans(xml, SLIDE_SHAPE_TREE).find((span) => !span.selfClosing);
  if (!tree) return { bodies: [], paragraphs: [] };
  const excluded = [...tagSpans(xml, SLIDE_FRAME), ...tagSpans(xml, SLIDE_GROUP)];
  const inside = (span) => excluded.some((other) => other.start < span.start && span.end < other.end);
  const bodies = tagSpans(xml, SLIDE_SHAPE)
    .filter((span) => !span.selfClosing && span.depth === 1 && tree.start < span.start && span.end < tree.end && !inside(span))
    .map((shape) => shapeTextBody(xml, shape))
    .filter(Boolean);
  const paragraphs = slideParagraphs(xml).filter((span) => bodies.some((body) => body.start < span.start && span.end < body.end));
  return { bodies, paragraphs };
}

// A paragraph's runs and text slots, shaped like `docxRuns` so the shared
// `replaceOccurrences` rewrites one. A slot is an `<a:t>` inside a run; a
// field's text (a slide number) is not addressed, the same text the reader
// leaves out of what it shows.
function slideRuns(fragment) {
  const runs = [];
  const slots = [];
  for (const span of tagSpans(fragment, SLIDE_RUN)) {
    if (span.selfClosing) continue;
    const runXml = fragment.slice(span.start, span.end);
    const index = runs.length;
    runs.push({ start: span.start, end: span.end, rpr: runPropertiesOf(runXml) });
    for (const text of runXml.matchAll(SLIDE_TEXT)) {
      const at = span.start + text.index;
      slots.push({ start: at, end: at + text[0].length, raw: text[1], unescaped: /&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/.test(text[1]), run: index });
    }
  }
  return { runs, slots };
}

// A paragraph's joined, unescaped run text — the text an edit addresses.
const slideParagraphText = (paragraph) => slideRuns(paragraph).slots.map((slot) => xmlUnescape(slot.raw)).join("");

// `xml` with `change` applied to every paragraph in part order: `change` returns
// the paragraph's replacement text, `null` to drop it, or `undefined` to leave
// it alone. The paragraphs are written from the last to the first so an earlier
// one's byte positions stay valid.
function mapSlideParagraphs(xml, change) {
  const spans = slideParagraphs(xml);
  let out = xml;
  for (let i = spans.length - 1; i >= 0; i -= 1) {
    const span = spans[i];
    const element = xml.slice(span.start, span.end);
    const edited = change(element);
    if (edited === undefined) continue;
    out = out.slice(0, span.start) + (edited === null ? "" : edited) + out.slice(span.end);
  }
  return out;
}

// The refusal a slide edit that searched and found nothing states.
const missingSlideFind = (find, slide) => `the text ${JSON.stringify(find)} is not on slide ${slide}`;

// The pptx text ops over one part: every occurrence of `find` in a matching
// paragraph's joined text is rewritten, the replacement taking the formatting of
// the run it starts in. `replacing` is what tells a replace op from a remove one,
// and `missing` words the refusal a search that found nothing states.
function editPartText(editor, part, edit, replacing, missing) {
  const find = editText(edit && edit.find, "find");
  if (!find) throw new UsageError("find must not be empty");
  const replacement = replacing ? editText(edit.replace, "replace") : "";
  let found = false;
  editor.part(part, (xml) => {
    const out = mapSlideParagraphs(xml, (paragraph) => {
      const { runs, slots } = slideRuns(paragraph);
      const offsets = occurrences(slots.map((slot) => xmlUnescape(slot.raw)).join(""), find);
      if (!offsets.length) return undefined;
      found = true;
      return replaceOccurrences(paragraph, runs, slots, offsets, replacement, editor.note, "a:");
    });
    if (!found) throw new UsageError(missing(find));
    return out;
  });
}

// `pptx_edit`'s `replace_text`/`remove_text` on a slide's own text.
const editSlideText = (editor, edit, op) => editPartText(editor, addressedSlide(editor.zip, edit.slide), edit, op === "replace_text", (find) => missingSlideFind(find, edit.slide));

// The run markup a new paragraph takes: the anchor's own first-run properties,
// or a bare run when there is no anchor to copy.
function slideRunXml(anchor, text) {
  const rpr = anchor === undefined ? undefined : slideRuns(anchor).runs[0]?.rpr;
  return `<a:r>${rpr || ""}${textElement("a:", xmlEscape(text))}</a:r>`;
}

// The `<a:pPr>` a new paragraph takes: the anchor's own with `level` merged in,
// or a bare level when there is no anchor.
function slideParagraphProperties(anchor, level) {
  const ppr = anchor === undefined ? undefined : (anchor.match(SLIDE_PPR) || [])[0];
  if (level === undefined) return ppr || "";
  if (!ppr) return `<a:pPr lvl="${level}"/>`;
  return /\blvl="[^"]*"/.test(ppr) ? ppr.replace(/\blvl="[^"]*"/, `lvl="${level}"`) : ppr.replace(/<a:pPr/, `<a:pPr lvl="${level}"`);
}

// `pptx_edit`'s `add_paragraph`: a new `<a:p>` after the first paragraph of the
// slide's OWN text whose text holds `after`, or after its last one, carrying the
// anchor's own paragraph and run properties so the new bullet keeps the deck's
// style. "The slide's own text" is the `<p:txBody>` of its top-level `<p:sp>`
// shapes — the title and body placeholders (see `slideTextBodies`) — because a
// table's cells hold paragraphs too and a new one added there lands where the
// caller did not ask. An `after` that names text only inside such a frame is
// refused rather than placed silently, and a slide with no text body at all has
// nowhere for one to go.
function addSlideParagraph(editor, edit) {
  const part = addressedSlide(editor.zip, edit.slide);
  const text = editText(edit.text, "text");
  let level;
  if (edit.level !== undefined) {
    if (!Number.isInteger(edit.level) || edit.level < 0 || edit.level > RULES.paragraph_level_max) throw new UsageError(`level must be a whole number from 0 to ${RULES.paragraph_level_max}, got: ${JSON.stringify(edit.level)}`);
    level = edit.level;
  }
  editor.part(part, (xml) => {
    const { bodies, paragraphs } = slideTextBodies(xml);
    let anchor;
    let at;
    if (edit.after !== undefined) {
      const after = editText(edit.after, "after");
      if (!after) throw new UsageError("after must not be empty");
      const span = paragraphs.find((candidate) => slideParagraphText(xml.slice(candidate.start, candidate.end)).includes(after));
      if (!span) {
        // The text may still be on the slide inside a table. Say where it is
        // rather than writing into a shape the caller did not address.
        const frames = tagSpans(xml, SLIDE_FRAME);
        const buried = slideParagraphs(xml).find((candidate) => frames.some((frame) => frame.start < candidate.start && candidate.end < frame.end) && slideParagraphText(xml.slice(candidate.start, candidate.end)).includes(after));
        if (buried) throw new UsageError(`the text ${JSON.stringify(after)} is inside a table on slide ${edit.slide}: add_paragraph writes into the slide's own text, not into a table`);
        throw new UsageError(missingSlideFind(after, edit.slide));
      }
      anchor = xml.slice(span.start, span.end);
      at = span.end;
    } else if (paragraphs.length) {
      const span = paragraphs[paragraphs.length - 1];
      anchor = xml.slice(span.start, span.end);
      at = span.end;
    } else if (bodies.length) {
      at = bodies[bodies.length - 1].closeAt;
    } else {
      throw new UsageError("the slide has no text body to add a paragraph to");
    }
    return xml.slice(0, at) + `<a:p>${slideParagraphProperties(anchor, level)}${slideRunXml(anchor, text)}</a:p>` + xml.slice(at);
  });
}

// The text bodies of a slide part — a shape's `<p:txBody>` and a table cell's
// `<a:txBody>` — each a `CT_TextBody`, which ECMA-376 requires to hold at least
// one paragraph. A body never nests inside another, so one sweep over the bodies
// and the paragraphs in order groups the paragraphs by the body holding them.
const SLIDE_TEXT_BODIES = new RegExp(`${openTag("p:txBody")}[\\s\\S]*?${closeTag("p:txBody")}|${openTag("a:txBody")}[\\s\\S]*?${closeTag("a:txBody")}`, "g");

// `pptx_edit`'s `remove_paragraph`: every paragraph whose joined text holds
// `find` is dropped whole, so a fragment matching several paragraphs removes
// each of them. A slide part has no paragraph nesting to pick an innermost one
// from. A removal that would leave a text body with no paragraph is refused,
// since that body would be one no reader may hold.
function removeSlideParagraphs(editor, edit) {
  const part = addressedSlide(editor.zip, edit.slide);
  const find = editText(edit && edit.find, "find");
  if (!find) throw new UsageError("find must not be empty");
  editor.part(part, (xml) => {
    const spans = slideParagraphs(xml);
    const matched = new Set(spans.filter((span) => slideParagraphText(xml.slice(span.start, span.end)).includes(find)).map((span) => span.start));
    if (!matched.size) throw new UsageError(missingSlideFind(find, edit.slide));
    // The paragraphs of each text body, so a body whose every paragraph goes is
    // caught before any of them is dropped. Both lists are in part order and
    // neither kind nests, so the body holding a paragraph is the first one that
    // has not ended before it — asking each paragraph on its own would walk the
    // whole part once per paragraph.
    const bodies = [...xml.matchAll(SLIDE_TEXT_BODIES)].map((match) => ({ start: match.index, end: match.index + match[0].length, cell: match[0].startsWith("<a:") }));
    const groups = new Map();
    let body = 0;
    for (const span of spans) {
      while (body < bodies.length && bodies[body].end <= span.start) body += 1;
      const holder = body < bodies.length ? bodies[body] : null;
      const key = holder ? `${holder.start}:${holder.end}` : `at:${span.start}`;
      if (!groups.has(key)) groups.set(key, { body: holder, spans: [] });
      groups.get(key).spans.push(span);
    }
    for (const { body: holder, spans: group } of groups.values()) {
      if (holder && group.every((span) => matched.has(span.start))) {
        throw new UsageError(`removing ${JSON.stringify(find)} would leave the ${holder.cell ? "table cell" : "shape"}'s text body with no paragraph — a text body must keep one`);
      }
    }
    return mapSlideParagraphs(xml, (paragraph) => (slideParagraphText(paragraph).includes(find) ? null : undefined));
  });
}

// The minimal slide part a new slide is built from: the group shape properties,
// a title placeholder when a title was given and a body placeholder holding one
// paragraph per bullet, then the colour-map override. The slide deliberately
// carries no notes and no animation, and nothing layout-specific beyond its
// placeholders — the layout supplies their geometry.
function newSlideXml(title, bullets) {
  const shapes = [];
  if (title !== undefined) shapes.push(placeholderShape(2, "Title", `<p:ph type="title"/>`, [title]));
  if (bullets.length) shapes.push(placeholderShape(3, "Body", `<p:ph type="body" idx="1"/>`, bullets));
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n` +
    `<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="${REL_NS}" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">` +
    `<p:cSld><p:spTree>` +
    `<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>` +
    `<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>` +
    shapes.join("") +
    `</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>`;
}

// A placeholder shape holding one paragraph per text, each run stating only the
// run language.
function placeholderShape(id, name, placeholder, texts) {
  const paragraphs = texts.map((text) => `<a:p><a:r><a:rPr lang="${SLIDE_LANG}"/>${textElement("a:", xmlEscape(text))}</a:r></a:p>`).join("");
  return `<p:sp><p:nvSpPr><p:cNvPr id="${id}" name="${name}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr>${placeholder}</p:nvPr></p:nvSpPr>` +
    `<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>${paragraphs}</p:txBody></p:sp>`;
}

// A new slide's relationships part: one relationship to the layout it is built
// on.
const newSlideRelsXml = (layout) => `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
  `<Relationship Id="rId1" Type="${SLIDE_LAYOUT_REL}" Target="${relativeTarget(PPT_SLIDES, layout)}"/></Relationships>`;

// The next free `<p:sldId id>`: one past the largest the presentation uses, and
// never below 256, the value real decks start at.
function nextSldId(xml) {
  const ids = [...xml.matchAll(/<p:sldId\b[^>]*\bid="(\d+)"/g)].map((match) => Number(match[1]));
  return Math.max(255, ...ids) + 1;
}

// Add the presentation relationship naming slide `part` and return its fresh id;
// the relationships part is created when the deck has none.
const addSlideRelationship = (editor, part) => addRelationship(editor, PPT_PRESENTATION, SLIDE_REL, part);

// The presentation's `<p:sldIdLst>` and the `<p:sldId>` entries that name a
// slide, or `null` when the presentation has no such list. An entry with no
// relationship id names no slide the reader shows, so it takes no slot and no
// position counts it; the entries are matched against the list's own text, so
// each `entry.index` is relative to `text`, and `closeAt` is where the list's own
// close tag begins inside `text` — the place an entry appended at its end goes,
// taken from the match so a close tag carrying whitespace is not mis-measured.
function slideListEntries(xml) {
  const match = xml.match(new RegExp(`${openTag("p:sldIdLst")}([\\s\\S]*?)(${closeTag("p:sldIdLst")})`));
  if (!match) return null;
  const entries = [...match[0].matchAll(/<p:sldId\b[^>]*>/g)].filter((entry) => relationshipId(entry[0]) !== undefined);
  return { index: match.index, text: match[0], entries, closeAt: match[0].length - match[2].length };
}

// That list, when its entries are exactly the slides the reader numbers — one per
// slide, in that same order — or `null` when they are not. The reader numbers every
// `<p:sldId>` the presentation holds while the list holds only its own, so a
// `<p:sldId>` outside it is numbered by the reader and not by the list, and the
// entry at a position is then another slide than the reader shows there; a deck the
// reader numbers by file naming differs the same way. A slide's number is a
// position in the list only while the two agree, which is what every operation that
// reorders, copies or deletes a slide by its number asks here first. The entries
// and their positions are relative to the presentation part's text as it stands,
// which is the text an edit rewrites.
function numberedSlideList(zip) {
  const presentation = zip.file(PPT_PRESENTATION);
  const rels = zip.file(PPT_RELS);
  const list = presentation && rels ? slideListEntries(presentation.asText()) : null;
  const parts = slideParts(zip);
  if (!list || list.entries.length !== parts.length) return null;
  const map = relationshipMap(rels.asText());
  const agrees = list.entries.every((entry, index) => {
    const target = map.get(relationshipId(entry[0]));
    return (target === undefined ? null : resolvePart(PPT_BASE, target)) === parts[index];
  });
  return agrees ? list : null;
}

// The presentation part with a `<p:sldId>` for `element` inserted into its
// `<p:sldIdLst>` as its entry `before` — before the slide the reader numbers
// `before + 1`, or at the end of the list for `before` past its last entry or
// absent. A deck without a slide list is refused rather than given one that would
// renumber its slides. An entry position is a slide's number only while the list is
// the numbering the reader shows (see `numberedSlideList`), so a caller handing one
// over has asked that first; appending needs no such proof, since it makes the new
// slide the deck's last one whichever numbering it is.
function withSldId(xml, element, before) {
  const list = slideListEntries(xml);
  if (!list) throw new UsageError("the presentation has no slide list to add a slide to");
  const at = before === undefined || before >= list.entries.length
    ? list.index + list.closeAt
    : list.index + list.entries[before].index;
  return xml.slice(0, at) + element + xml.slice(at);
}

// The `<Override>` a package already declares for `part`, or `undefined` when it
// declares none. OPC gives a part one content type, so this is the declaration no
// caller may add a second of.
const overrideFor = (xml, part) => [...xml.matchAll(/<Override\b[^>]*>/g)].find((match) => (xmlAttribute(match[0], "PartName") || "").toLowerCase() === `/${part}`.toLowerCase());

// `[Content_Types].xml` declaring `part`'s `contentType`, through the `<Override>`
// written before the closing `</Types>` — after every `<Default>`, the
// schema-correct place for one. A package that already declares the part has that
// declaration given the type instead of a second one added beside it, which is also
// what leaves a stale declaration saying what the part really is. The close is
// found with the shared spelling rule (`closeTag`), never spelled out again.
const TYPES_CLOSE = new RegExp(closeTag("Types"));
function withOverride(xml, part, contentType) {
  const close = xml.match(TYPES_CLOSE);
  if (!close) throw new UsageError("[Content_Types].xml has no <Types> close to hold a part's content type");
  const declaration = `<Override PartName="/${part}" ContentType="${contentType}"/>`;
  const declared = overrideFor(xml, part);
  if (declared === undefined) return xml.slice(0, close.index) + declaration + xml.slice(close.index);
  return xml.slice(0, declared.index) + declaration + xml.slice(declared.index + declared[0].length);
}

// `pptx_edit`'s `add_slide`: a whole new slide part, wired into the presentation
// and the content types.
function addSlide(editor, edit) {
  const zip = editor.zip;
  const slides = slideParts(zip);
  const title = edit.title === undefined ? undefined : editText(edit.title, "title");
  let bullets = [];
  if (edit.bullets !== undefined) {
    if (!Array.isArray(edit.bullets)) throw new UsageError("bullets must be a list");
    if (edit.bullets.length > RULES.bullets_max) throw new UsageError(`a slide may hold at most ${RULES.bullets_max} bullets, got: ${edit.bullets.length}`);
    bullets = edit.bullets.map((bullet) => editText(bullet, "a bullet"));
  }
  let after;
  if (edit.after !== undefined) {
    after = edit.after;
    if (!Number.isInteger(after) || after < 1 || after > slides.length) throw new UsageError(`no slide ${after}: this presentation has ${slides.length} slide(s)`);
    // Asked before any edit, since the list is about to gain the new entry: the
    // named position is a slide's number only while the list is that numbering.
    if (!numberedSlideList(zip)) throw new UsageError(`a slide cannot be placed after slide ${after}: the presentation's slide list does not number its slides the way the reader shows them`);
  }
  const part = `${PPT_SLIDES}slide${Math.max(0, ...slideFileNumbers(zip)) + 1}.xml`;
  editor.add(part, newSlideXml(title, bullets));
  editor.add(relsPartFor(part), newSlideRelsXml(newSlideLayout(zip, slides, after)));
  // `openEdit` already proved the presentation part is the family's own, so it
  // is read here rather than re-checked.
  const id = addSlideRelationship(editor, part);
  const sldId = `<p:sldId id="${nextSldId(zip.file(PPT_PRESENTATION).asText())}" r:id="${id}"/>`;
  editor.part(PPT_PRESENTATION, (xml) => withSldId(xml, sldId, after));
  editor.part("[Content_Types].xml", (xml) => withOverride(xml, part, SLIDE_CONTENT_TYPE));
}

// `[Content_Types].xml` with the `<Override>` of `part` removed — the same
// declaration `overrideFor` finds, by the same rule.
function removeOverride(editor, part) {
  editor.part("[Content_Types].xml", (xml) => {
    const declared = overrideFor(xml, part);
    return declared === undefined ? xml : xml.slice(0, declared.index) + xml.slice(declared.index + declared[0].length);
  });
}

// `pptx_edit`'s `delete_slide`: the slide part, its relationships, its notes
// part and that part's relationships, its `<p:sldId>`, the matching presentation
// relationship, and the content-type overrides of both parts. The addressed
// POSITION is what goes — the reader numbers the resolved list, and two entries
// may name the same part — so exactly that `<p:sldId>` and exactly that
// relationship are removed, and a part another remaining slide still names is
// kept (its `<Override>` with it). A deck the reader could only number by file
// naming has no entry at that position: the entry whose target resolves to the
// addressed part goes instead, and when none does the deletion is refused rather
// than a `<p:sldId>` belonging to another slide going. The deck's only slide is
// refused — a presentation with no slides is not one.
function deleteSlide(editor, edit) {
  const zip = editor.zip;
  const parts = slideParts(zip);
  const at = edit.slide - 1;
  if (parts.length <= 1) throw new UsageError("the presentation's only slide cannot be deleted");
  const part = addressedSlide(zip, edit.slide);
  // The `<p:sldId>` the reader numbered as `at`. Only while the list is that
  // numbering is the entry at that position the reader's own (see
  // `numberedSlideList`): a deck whose list holds another slide there — or numbers a
  // `<p:sldId>` its list does not — has the entry whose relationship target
  // resolves to the addressed part taken instead, so the removed `<p:sldId>` is
  // always the addressed slide's own.
  const presentation = zip.file(PPT_PRESENTATION);
  const list = presentation ? slideListEntries(presentation.asText()) : null;
  const entries = list ? list.entries : [];
  const rels = zip.file(PPT_RELS);
  const targets = rels ? relationshipMap(rels.asText()) : new Map();
  const sldId = numberedSlideList(zip)
    ? entries[at]
    : entries.find((match) => {
      const target = targets.get(relationshipId(match[0]));
      return target !== undefined && resolvePart(PPT_BASE, target) === part;
    });
  if (!sldId) throw new UsageError(`slide ${edit.slide} cannot be deleted: the presentation's slide list does not name it`);
  const id = relationshipId(sldId[0]);
  const cut = list.index + sldId.index;
  const notes = slideNotesPart(zip, part);
  const others = parts.filter((_, index) => index !== at).filter(Boolean);
  // A part another remaining slide still references is not this slide's alone:
  // removing it would leave that slide dangling.
  const partShared = others.includes(part);
  const notesShared = notes !== null && others.some((other) => slideNotesPart(zip, other) === notes);
  if (!partShared) {
    editor.remove(part);
    editor.remove(relsPartFor(part));
    removeOverride(editor, part);
  }
  if (notes && !notesShared) {
    editor.remove(notes);
    editor.remove(relsPartFor(notes));
    removeOverride(editor, notes);
  }
  editor.part(PPT_PRESENTATION, (xml) => xml.slice(0, cut) + xml.slice(cut + sldId[0].length));
  if (id !== undefined) {
    editor.part(PPT_RELS, (xml) => xml.replace(elementWithAttribute("Relationship", "Id", id), ""));
  }
}

// The relationships `xml` with the `Target` of every relationship whose type
// `match` accepts replaced by `target`; every other relationship keeps its own
// bytes. A `match` compares the LAST segment of a type because a strict package
// spells the same relationship under another namespace — and the segments are
// distinct enough that `/slide` cannot match `/slideLayout`.
function withRelationshipTarget(xml, match, target) {
  return xml.replace(/<Relationship\b[^>]*>/g, (element) => (match(xmlAttribute(element, "Type") || "")
    ? element.replace(/\bTarget="[^"]*"/, () => `Target="${target}"`)
    : element));
}

// `pptx_edit`'s `move_slide`: the addressed slide's `<p:sldId>` is taken out of
// the presentation's own slide list and put back at another position of it, so
// the deck's order — the one the reader numbers and `slideParts` resolves — is
// the only thing that changes. Only a list that IS that order has the position the
// caller named (see `numberedSlideList`), and a slide already at the named position
// is refused because the move would change nothing.
function moveSlide(editor, edit) {
  const zip = editor.zip;
  addressedSlide(zip, edit.slide);
  const slides = slideParts(zip);
  if (!Number.isInteger(edit.to) || edit.to < 1 || edit.to > slides.length) throw new UsageError(`no position ${edit.to}: this presentation has ${slides.length} slide(s)`);
  if (edit.to === edit.slide) throw new UsageError(`slide ${edit.slide} is already at position ${edit.to} — moving it there would change nothing`);
  const list = numberedSlideList(zip);
  if (!list) throw new UsageError("the presentation's slide list does not number its slides the way the reader shows them, so a slide cannot be moved");
  const from = edit.slide - 1;
  const entry = list.entries[from];
  const rest = list.entries.filter((_, index) => index !== from);
  const cut = list.index + entry.index;
  // The slide takes the position the `to`-th slide of the RESULT holds: in front
  // of the entry that follows it there, or after the last one when it is that
  // deck's last slide.
  const before = edit.to - 1 < rest.length ? list.index + rest[edit.to - 1].index : list.index + list.closeAt;
  editor.part(PPT_PRESENTATION, (xml) => {
    const out = xml.slice(0, cut) + xml.slice(cut + entry[0].length);
    const at = before > cut ? before - entry[0].length : before;
    return out.slice(0, at) + entry[0] + out.slice(at);
  });
}

// The notes part of a duplicated slide: a copy of `notes` — a part the package
// holds, which is what the caller has checked — under a fresh name, with its own
// relationships (the notes master stays the shared part it was) and its slide
// back-reference pointed at the copy. The notes themselves are the copy's own, so
// editing one slide's notes leaves the other's alone.
function duplicateNotes(editor, notes, slidePart) {
  const zip = editor.zip;
  const part = `${PPT_NOTES_SLIDES}notesSlide${Math.max(0, ...notesSlideNumbers(zip)) + 1}.xml`;
  editor.add(part, zip.file(notes).asText());
  const rels = zip.file(relsPartFor(notes));
  if (rels) editor.add(relsPartFor(part), withRelationshipTarget(rels.asText(), (type) => type.endsWith("/slide"), relativeTarget(partDirectory(part), slidePart)));
  editor.part("[Content_Types].xml", (xml) => withOverride(xml, part, NOTES_CONTENT_TYPE));
  return part;
}

// `pptx_edit`'s `duplicate_slide`: a fresh slide part holding exactly what the
// addressed slide held, with a notes copy and its own relationships, inserted
// right after the source — where the reader that numbered it shows it next. The
// layout, the masters and the media the copy names stay shared parts, because no
// edit this kit makes writes to them. Only a deck whose slide list numbers its
// slides the way the reader shows them can place a copy (see `numberedSlideList`).
function duplicateSlide(editor, edit) {
  const zip = editor.zip;
  const source = addressedSlide(zip, edit.slide);
  // A copy is written among the presentation's own slides, while the source's
  // relationships are written for the folder it sits in: a slide stored elsewhere
  // is one whose links a copy in this folder would not resolve.
  if (partDirectory(source) !== PPT_SLIDES) {
    throw new UsageError(`slide ${edit.slide} is stored outside ${PPT_SLIDES}, so a copy of it cannot be placed among the presentation's other slides`);
  }
  if (!numberedSlideList(zip)) throw new UsageError("the presentation's slide list does not number its slides the way the reader shows them, so a slide cannot be duplicated");
  const notes = slideNotesPart(zip, source);
  // A copy carries the source's notes as its own, so a source declaring notes the
  // package does not hold has none to carry: a copy left naming that declaration
  // would put the two slides' notes in the one part a later edit rewrites. Both
  // checks are made before the copy is written.
  if (notes !== null && !zip.file(notes)) {
    throw new UsageError(`slide ${edit.slide} declares speaker notes (${notes}) the presentation does not hold, so a copy of it cannot have notes of its own`);
  }
  const part = `${PPT_SLIDES}slide${Math.max(0, ...slideFileNumbers(zip)) + 1}.xml`;
  editor.add(part, zip.file(source).asText());
  const notesCopy = notes === null ? null : duplicateNotes(editor, notes, part);
  const rels = zip.file(relsPartFor(source));
  if (rels) {
    // The source's own relationships with the notes relationship pointed at the
    // copy: the source's notes belong to the source.
    editor.add(relsPartFor(part), notesCopy === null ? rels.asText() : withRelationshipTarget(rels.asText(), (type) => type.endsWith("/notesSlide"), relativeTarget(partDirectory(part), notesCopy)));
  }
  editor.part("[Content_Types].xml", (xml) => withOverride(xml, part, SLIDE_CONTENT_TYPE));
  const id = addSlideRelationship(editor, part);
  const sldId = `<p:sldId id="${nextSldId(zip.file(PPT_PRESENTATION).asText())}" r:id="${id}"/>`;
  editor.part(PPT_PRESENTATION, (xml) => withSldId(xml, sldId, edit.slide));
}

// A run of speaker notes carrying `text`; notes are plain text, so the run
// states nothing but the language a reader draws it in.
const notesParagraph = (text) => `<a:p><a:r><a:rPr lang="${SLIDE_LANG}" dirty="0"/>${textElement("a:", xmlEscape(text))}</a:r></a:p>`;

// The `<p:notes>` part a slide with no notes is given: the slide image and notes
// body placeholders every notes slide holds, the body carrying `text`. Its
// relationships and content type are written by `addSlideNotes`, which is the
// only caller.
function newNotesXml(text) {
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n` +
    `<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="${REL_NS}" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">` +
    `<p:cSld><p:spTree>` +
    `<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>` +
    `<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>` +
    `<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>` +
    `<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>` +
    `<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>${notesParagraph(text)}</p:txBody></p:sp>` +
    `</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>`;
}

// The notes master a notes slide is wired to: the one the presentation's own
// `<p:notesMasterIdLst>` declares — the master its notes slides are drawn through
// — or, for a deck stating none, the first the package holds. Either way it is a
// part that is already there, so the name is the one the package really has.
function notesMasterPart(zip) {
  const presentation = zip.file(PPT_PRESENTATION);
  const rels = zip.file(PPT_RELS);
  const tag = presentation ? (presentation.asText().match(/<p:notesMasterId\b[^>]*>/) || [])[0] : undefined;
  const id = tag === undefined ? undefined : relationshipId(tag);
  const target = id === undefined || !rels ? undefined : relationshipMap(rels.asText()).get(id);
  const declared = target === undefined ? null : resolvePart(PPT_BASE, target);
  if (declared !== null && zip.file(declared)) return declared;
  return firstPartName(zip, `${PPT_NOTES_MASTERS}notesMaster`);
}

// A notes part's relationships, written relative to its own directory: back to the
// slide whose notes it holds, and to the master it draws through.
const newNotesRelsXml = (base, slidePart, master) =>
  `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
  `<Relationship Id="rId1" Type="${SLIDE_REL}" Target="${relativeTarget(base, slidePart)}"/>` +
  `<Relationship Id="rId2" Type="${NOTES_MASTER_REL}" Target="${relativeTarget(base, master)}"/></Relationships>`;

// The notes part `slidePart` is given its notes in: the one the slide declares —
// created at that very name when the package has lost it, so the relationship it
// already has resolves again — or a fresh part wired into the slide. A
// presentation with no notes master cannot hold a notes slide, so adding one to
// it is refused rather than written as a file a reader has to repair.
function createNotesPart(editor, slidePart, text) {
  const zip = editor.zip;
  const declared = slideNotesPart(zip, slidePart);
  const master = notesMasterPart(zip);
  if (!master) throw new UsageError("the presentation has no notes master, so a notes slide cannot be created for it");
  const part = declared || `${PPT_NOTES_SLIDES}notesSlide${Math.max(0, ...notesSlideNumbers(zip)) + 1}.xml`;
  editor.add(part, newNotesXml(text));
  editor.add(relsPartFor(part), newNotesRelsXml(partDirectory(part), slidePart, master));
  if (!declared) addRelationship(editor, slidePart, NOTES_REL, part);
  // A package that declares the part already says what it is — a `.pptm` declares
  // its notes slides macro-enabled — and only a part the package knows nothing
  // about is declared here. `pptxEdit` refuses a package with no
  // `[Content_Types].xml`, so the part is there to rewrite.
  editor.part("[Content_Types].xml", (xml) => (overrideFor(xml, part) === undefined ? withOverride(xml, part, NOTES_CONTENT_TYPE) : xml));
  return part;
}

// The notes body of a notes slide: the `<p:txBody>` of the shape whose `<p:ph>`
// is the body placeholder — the one place a reader draws notes from. A notes
// slide's other shapes hold the slide image and the slide number, and a paragraph
// must not land in either.
function notesTextBody(xml) {
  for (const shape of tagSpans(xml, SLIDE_SHAPE)) {
    if (shape.selfClosing || shape.depth !== 1) continue;
    const placeholder = xml.slice(shape.start, shape.end).match(/<p:ph\b[^>]*>/);
    if (!placeholder) continue;
    // Only a `<p:ph>` naming `body` is the notes one. The schema defaults
    // `p:ph/@type` to `obj`, which a reader draws no speaker notes from, so a
    // placeholder naming no type is not this body either — and the slide image
    // and slide number placeholders always name a type of their own.
    if (xmlAttribute(placeholder[0], "type") !== "body") continue;
    const body = shapeTextBody(xml, shape);
    if (body) return body;
  }
  return null;
}

// `pptx_edit`'s `add_notes`: `text` as a new paragraph of the addressed slide's
// speaker notes, the notes part being written when the slide has none — a slide
// without notes is not a failure to add them to.
function addSlideNotes(editor, edit) {
  const zip = editor.zip;
  const part = addressedSlide(zip, edit.slide);
  const text = editText(edit.text, "text");
  const declared = slideNotesPart(zip, part);
  if (declared && zip.file(declared)) {
    editor.part(declared, (xml) => {
      const body = notesTextBody(xml);
      if (!body) throw new UsageError(`the notes of slide ${edit.slide} have no body to add a paragraph to`);
      const at = body.closeAt;
      return xml.slice(0, at) + notesParagraph(text) + xml.slice(at);
    });
    return;
  }
  createNotesPart(editor, part, text);
}

// The refusal a notes edit that searched and found nothing states.
const missingNotesFind = (find, slide) => `the text ${JSON.stringify(find)} is not in the speaker notes of slide ${slide}`;

// `pptx_edit`'s `replace_notes`/`remove_notes`: the notes are plain text, so a
// fragment is matched in one notes paragraph's joined run text exactly as it is
// on a slide. A slide that holds no notes part at all has nothing to search, and
// that is said rather than reported as text that is not there.
function editSlideNotes(editor, edit, op) {
  const part = addressedSlide(editor.zip, edit.slide);
  const notes = slideNotesPart(editor.zip, part);
  if (!notes || !editor.zip.file(notes)) {
    throw new UsageError(`slide ${edit.slide} has no speaker notes to ${op === "replace_notes" ? "replace text in" : "remove text from"}`);
  }
  editPartText(editor, notes, edit, op === "replace_notes", (find) => missingNotesFind(find, edit.slide));
}

// The children of a run-properties body at its own level: `<a:ln>` holds a
// `<a:solidFill>` of its own — the outline's fill, not the run's — so the level
// a child sits at is what tells the two apart.
function topLevelChildren(body) {
  const children = [];
  const open = new RegExp(`${openTag("a:([A-Za-z0-9_]+)")}|${selfClosingTag("a:([A-Za-z0-9_]+)")}`, "g");
  let match;
  while ((match = open.exec(body)) !== null) {
    const name = match[1] ?? match[2];
    let end = match.index + match[0].length;
    if (!match[0].endsWith("/>")) {
      // The element's own close, found with the shared spelling rule (`name` is the
      // capture, so the `a:` prefix the pattern names is put back): a close tag
      // carrying whitespace before its `>` ends the child there too.
      const close = new RegExp(closeTag(`a:${name}`), "g");
      close.lastIndex = open.lastIndex;
      const found = close.exec(body);
      if (!found) continue;
      end = found.index + found[0].length;
    }
    children.push({ name, start: match.index, end });
    open.lastIndex = end;
  }
  return children;
}

// DrawingML's `CT_TextCharacterProperties` children in the sequence ECMA-376
// requires: a `<a:solidFill>` written after a later child is dropped by a reader,
// so a colour goes where the sequence puts it.
const RUN_CHILD_ORDER = ["ln", "noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill", "effectLst", "effectDag", "highlight", "uLnTx", "uLn", "uFillTx", "uFill", "latin", "ea", "cs", "sym", "hlinkClick", "hlinkMouseOver", "rtl", "extLst"];
// The members of DrawingML's `EG_FillProperties`, which is a choice group: the
// six are exclusive, so a run stating its own fill drops whichever one it had.
const RUN_FILL_CHILDREN = ["noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill"];

// The `<a:rPr>` of a run as the run wrote it, or "" when it has none. DrawingML
// states no run properties inside run properties, so the first element of the
// family in a run is the run's own.
const runPropertiesOf = (runXml) => (runXml.match(SLIDE_RUN_PROPERTIES) || [])[0] || "";

// `body` with the run's own text fill replaced by `element`. The six members of
// DrawingML's `EG_FillProperties` are a choice group, so whichever one the run
// had goes, and the replacement is written where ECMA-376's child sequence puts
// it rather than after a child it must precede. The removal and the write work on
// the body's own level (see `topLevelChildren`).
function withRunFill(body, element) {
  // A name the sequence does not list ranks last, so a fill is written before it.
  const rank = (name) => {
    const at = RUN_CHILD_ORDER.indexOf(name);
    return at < 0 ? RUN_CHILD_ORDER.length : at;
  };
  const fill = rank("solidFill");
  let out = "";
  let cursor = 0;
  let written = false;
  for (const child of topLevelChildren(body)) {
    out += body.slice(cursor, child.start);
    cursor = child.end;
    if (RUN_FILL_CHILDREN.includes(child.name)) continue;
    if (!written && rank(child.name) > fill) {
      out += element;
      written = true;
    }
    out += body.slice(child.start, cursor);
  }
  out += body.slice(cursor);
  return written ? out : out + element;
}

// The run properties a format request writes for one run: `bold`, `italic` and
// `underline` are attributes of `<a:rPr>`, `size` replaces its `sz` attribute
// (DrawingML counts hundredths of a point in it) and `color` the run's own solid
// fill. Everything the request does not name keeps the bytes it had, and a toggle
// set to `false` writes the off value so it overrides what the run inherits.
function formattedSlideRunProperties(runXml, format) {
  const current = runPropertiesOf(runXml);
  // The properties read through the shared element rule — `elementParts` opens a
  // self-closing tag and gives the body of a closed one — so a close tag carrying
  // whitespace before its `>` is not read as part of the body.
  const { open, body: own } = current === "" ? { open: "<a:rPr>", body: "" } : elementParts(current, "a:rPr");
  let body = own;
  let tag = open;
  if (format.bold !== undefined) tag = setXmlAttribute(tag, "b", format.bold ? "1" : "0");
  if (format.italic !== undefined) tag = setXmlAttribute(tag, "i", format.italic ? "1" : "0");
  if (format.underline !== undefined) tag = setXmlAttribute(tag, "u", format.underline ? "sng" : "none");
  if (format.size !== undefined) tag = setXmlAttribute(tag, "sz", format.size);
  if (format.color !== undefined) {
    body = withRunFill(body, `<a:solidFill><a:srgbClr val="${format.color}"/></a:solidFill>`);
  }
  // `elementParts`'s open tag is the opened form, so an empty body is written back
  // as the self-closing tag it was.
  return body ? `${tag}${body}</a:rPr>` : `${tag.slice(0, -1)}/>`;
}

// `xml` with the run-level properties of `format` written on the runs the named
// fragments cover. A run is the smallest unit DrawingML states run properties in,
// so a fragment covering part of a run formats that whole run — which the note
// the caller raises says.
function formatSlideRuns(xml, runs, runIndices, format) {
  const pieces = [];
  for (const index of [...runIndices].sort((a, b) => a - b)) {
    const run = runs[index];
    const runXml = xml.slice(run.start, run.end);
    const open = runXml.match(new RegExp(openTag("a:r")))[0];
    const own = runPropertiesOf(runXml);
    const at = own === "" ? -1 : runXml.indexOf(own);
    const next = formattedSlideRunProperties(runXml, format);
    pieces.push({
      start: run.start,
      end: run.end,
      xml: at >= 0 ? runXml.slice(0, at) + next + runXml.slice(at + own.length) : open + next + runXml.slice(open.length),
    });
  }
  pieces.sort((a, b) => b.start - a.start);
  let out = xml;
  for (const piece of pieces) out = out.slice(0, piece.start) + piece.xml + out.slice(piece.end);
  return out;
}

// The paragraph with its alignment set. `<a:pPr>` is a paragraph's first child,
// so one is written there when the paragraph has none; otherwise the attribute
// goes on the paragraph's own `<a:pPr>` tag, never on its close.
function alignSlideParagraph(element, align) {
  const open = element.match(new RegExp(openTag("a:p")));
  if (!open) return element;
  const ppr = element.match(SLIDE_PPR);
  if (!ppr) return `${element.slice(0, open[0].length)}<a:pPr algn="${align}"/>${element.slice(open[0].length)}`;
  const tag = ppr[0].match(new RegExp(`^(?:${selfClosingTag("a:pPr")}|${openTag("a:pPr")})`))[0];
  const at = element.indexOf(ppr[0]);
  return element.slice(0, at) + setXmlAttribute(tag, "algn", align) + ppr[0].slice(tag.length) + element.slice(at + ppr[0].length);
}

// The formatting a `format_text` states, normalized to what the writer needs:
// `size` in DrawingML's hundredths of a point, `color` as `RULES.color_digits`
// uppercase hex digits and `align` as the `algn` value its word names.
function slideFormat(edit) {
  const format = { runs: false };
  // A `null` property counts as absent — the caller's own boundary treats a
  // null-valued key that way — so only a stated one is written.
  for (const name of ["bold", "italic", "underline"]) {
    if (edit[name] == null) continue;
    if (typeof edit[name] !== "boolean") throw new UsageError(`${name} must be true or false`);
    format[name] = edit[name];
    format.runs = true;
  }
  if (edit.size != null) {
    if (!inSpan(edit.size, RULES.slide_text_size_points)) throw new UsageError(`size must be ${spanBounds(RULES.slide_text_size_points)} points, got: ${JSON.stringify(edit.size)}`);
    format.size = String(Math.round(edit.size * 100));
    format.runs = true;
  }
  if (edit.color != null) {
    format.color = hexDigits(edit.color).toUpperCase();
    format.runs = true;
  }
  if (edit.align != null) {
    // The rules' own table, so an alignment with no `algn` value — or an
    // inherited name such as "constructor" — is refused rather than written as
    // an alignment no reader knows.
    if (!SLIDE_ALIGNMENTS.has(edit.align)) throw new UsageError(`align must be ${listed([...SLIDE_ALIGNMENTS.keys()])}, got: ${JSON.stringify(edit.align)}`);
    format.align = SLIDE_ALIGNMENTS.get(edit.align);
  }
  if (!format.runs && format.align === undefined) throw new UsageError("format_text needs at least one of bold, italic, underline, size, color or align");
  return format;
}

// `pptx_edit`'s `format_text`: the run properties of `edit` on the runs the named
// fragment covers, and its alignment on the paragraph holding it. Two things the
// caller is owed a note about: run properties live on the run, so a fragment
// covering part of one formats all of it, and `algn` is a paragraph's, so it
// applies to the paragraph's whole text.
function formatSlideText(editor, edit) {
  const part = addressedSlide(editor.zip, edit.slide);
  const find = editText(edit.find, "find");
  if (!find) throw new UsageError("find must not be empty");
  const format = slideFormat(edit);
  editor.part(part, (xml) => {
    let found = false;
    const out = mapSlideParagraphs(xml, (paragraph) => {
      const { runs, slots } = slideRuns(paragraph);
      const text = slots.map((slot) => xmlUnescape(slot.raw)).join("");
      const offsets = occurrences(text, find);
      if (!offsets.length) return undefined;
      found = true;
      // The code points the fragments cover, so a run or paragraph the request
      // reaches beyond them can be reported.
      const covered = new Array([...text].length).fill(false);
      for (const span of offsets) {
        for (let at = span.start; at < span.end; at += 1) covered[at] = true;
      }
      const ranges = [];
      let walked = 0;
      for (const slot of slots) {
        const length = [...xmlUnescape(slot.raw)].length;
        ranges.push({ start: walked, end: walked + length });
        walked += length;
      }
      const runIndices = new Set();
      for (const span of offsets) {
        const first = locateSlot(slots, span.start);
        const last = locateSlot(slots, span.end, true);
        for (let index = first.index; index <= last.index; index += 1) runIndices.add(slots[index].run);
      }
      // A run the fragments only partly cover: its properties are the run's, so
      // the whole of it is formatted.
      const partial = (slot, index) => {
        if (!runIndices.has(slot.run)) return false;
        for (let point = ranges[index].start; point < ranges[index].end; point += 1) {
          if (!covered[point]) return true;
        }
        return false;
      };
      if (format.runs && slots.some(partial)) editor.note("[the named text is part of a longer run, so the formatting also applied to the rest of that run]");
      let element = paragraph;
      if (format.runs) element = formatSlideRuns(element, runs, runIndices, format);
      if (format.align !== undefined) {
        if (!covered.every(Boolean)) editor.note("[a paragraph's alignment is the whole paragraph's, so the text beside the named fragment was aligned too]");
        element = alignSlideParagraph(element, format.align);
      }
      return element;
    });
    if (!found) throw new UsageError(missingSlideFind(find, edit.slide));
    return out;
  });
}

// The `N` numbers of the `ppt/media/imageN.ext` parts the package holds, whatever
// extension each carries.
const mediaNumbers = (zip) => Object.keys(zip.files).flatMap((name) => {
  const match = name.match(new RegExp(`^${escapeRegExp(PPT_MEDIA)}image(\\d+)\\.[A-Za-z0-9]+$`));
  return match ? [Number(match[1])] : [];
});

// A free `ppt/media/imageN.ext` name: one past the largest number the package
// uses, whatever extension it carries, so the name is free by construction.
function nextMediaPart(zip, extension) {
  return `${PPT_MEDIA}image${Math.max(0, ...mediaNumbers(zip)) + 1}.${extension}`;
}

// The slide's own size in EMU, from the presentation's `<p:sldSz>`, or the
// default a presentation stating none is read at (see `SLIDE_SIZE_DEFAULT`).
function slideSize(zip) {
  const presentation = zip.file(PPT_PRESENTATION);
  const tag = presentation ? (presentation.asText().match(/<p:sldSz\b[^>]*>/) || [])[0] : undefined;
  const side = (name, fallback) => {
    const value = tag === undefined ? Number.NaN : Number(xmlAttribute(tag, name));
    return Number.isFinite(value) && value > 0 ? value : fallback;
  };
  return { width: side("cx", SLIDE_SIZE_DEFAULT.width), height: side("cy", SLIDE_SIZE_DEFAULT.height) };
}

// Where an added image goes, in EMU. `x` and `y` are fractions of the slide's
// own width and height and so are `width` and `height`, which is what lets a
// caller place an image without knowing the deck's size; an absent place is the
// slide's middle and an absent size is the largest that fits the slide with the
// image's own proportions. A size named on one side only keeps those proportions.
function imagePlacement(image, edit, slide) {
  for (const name of ["x", "y", "width", "height"]) {
    const value = edit[name];
    if (value == null) continue;
    const span = name === "width" || name === "height" ? RULES.slide_size_fraction : RULES.slide_position_fraction;
    if (!inSpan(value, span)) throw new UsageError(`${name} must be ${spanBounds(span)}, got: ${JSON.stringify(value)}`);
  }
  const natural = { width: image.width * PX_TO_EMU, height: image.height * PX_TO_EMU };
  let width;
  let height;
  if (edit.width != null && edit.height != null) {
    width = edit.width * slide.width;
    height = edit.height * slide.height;
  } else if (edit.width != null) {
    width = edit.width * slide.width;
    height = width * natural.height / natural.width;
  } else if (edit.height != null) {
    height = edit.height * slide.height;
    width = height * natural.width / natural.height;
  } else {
    const scale = Math.min(slide.width / natural.width, slide.height / natural.height);
    width = natural.width * scale;
    height = natural.height * scale;
  }
  const round = (value) => Math.max(1, Math.round(value));
  return {
    x: Math.round(edit.x != null ? edit.x * slide.width : (slide.width - width) / 2),
    y: Math.round(edit.y != null ? edit.y * slide.height : (slide.height - height) / 2),
    cx: round(width),
    cy: round(height),
  };
}

// Where a new shape goes in a slide: the end of the slide's shape tree's own
// content, before the tree's trailing `<p:extLst>`, which `CT_GroupShape` keeps
// last — a shape written after it is out of the tree's element order. The tree's
// own list is told from a nested one by its tail: a nested `<p:extLst>` sits
// inside a shape, so that shape's close follows it, while the tree's own is
// followed by nothing but whitespace to the tree's close. `null` when the xml
// holds no shape tree — there is then nowhere for a shape to go. A self-closing
// `<p:spTree/>` is not one: `CT_GroupShape` requires the group's own properties,
// so a slide holding only that element is not a slide a picture may be added to.
function shapeTreeEnd(xml) {
  const tree = tagSpans(xml, SLIDE_SHAPE_TREE).find((span) => !span.selfClosing);
  if (!tree) return null;
  // Where the tree's own close begins, read from the scan's own span rather than
  // measured off a close tag spelled out again (see `tagSpans`).
  const close = tree.closeAt;
  const trailing = (span) => tree.start < span.start && span.end <= close && xml.slice(span.end, close).trim() === "";
  const extension = tagSpans(xml, tagPattern("p:extLst")).find(trailing);
  return extension ? extension.start : close;
}

// The slide part's root declaring the relationships namespace as `r` when it does
// not already: the picture reaches its media part through `r:embed`, and an
// undeclared prefix is not XML — in a part the kit did not write itself.
function withRelationshipNamespace(xml) {
  const root = xml.match(new RegExp(openTag("p:sld")));
  if (!root || /\bxmlns:r=/.test(root[0])) return xml;
  const declared = setXmlAttribute(root[0], "xmlns:r", REL_NS);
  return xml.slice(0, root.index) + declared + xml.slice(root.index + root[0].length);
}

// `pptx_edit`'s `add_image`: the file's bytes become a media part, the slide
// gains a relationship to it and a `<p:pic>` is appended to the slide's shape
// tree — the picture is the slide's last shape and takes the next free shape id,
// so nothing else on the slide is renamed. The image is placed and sized by
// fraction of the slide (see `imagePlacement`). The part's own content type is
// declared as an Override: the deck's `<Default>` for the extension may carry
// another spelling of it (a writer's `image/jpg`, say), and an Override is the
// declaration that decides for a part.
function addSlideImage(editor, edit) {
  const zip = editor.zip;
  const part = addressedSlide(zip, edit.slide);
  const image = readImage(edit.path);
  const media = nextMediaPart(zip, image.kind);
  editor.add(media, image.bytes);
  editor.part("[Content_Types].xml", (xml) => withOverride(xml, media, image.contentType));
  const id = addRelationship(editor, part, IMAGE_REL, media);
  editor.part(part, (xml) => {
    const namespaced = withRelationshipNamespace(xml);
    const at = shapeTreeEnd(namespaced);
    if (at === null) throw new UsageError("the slide has no shape tree to place an image in");
    const shapeId = Math.max(1, ...[...namespaced.matchAll(/<p:cNvPr\b[^>]*\bid="(\d+)"/g)].map((match) => Number(match[1]))) + 1;
    const placement = imagePlacement(image, edit, slideSize(zip));
    const picture = `<p:pic><p:nvPicPr><p:cNvPr id="${shapeId}" name="Image ${shapeId}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>` +
      `<p:blipFill><a:blip r:embed="${id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>` +
      `<p:spPr><a:xfrm><a:off x="${placement.x}" y="${placement.y}"/><a:ext cx="${placement.cx}" cy="${placement.cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>`;
    return namespaced.slice(0, at) + picture + namespaced.slice(at);
  });
}

function pptxEdit(req) {
  const editor = openEdit(req, "pptx");
  // A package without content types is not a presentation this kit can edit: it
  // could not name a new slide's type. The caller can supply one that has it,
  // not the kit.
  if (!editor.zip.file("[Content_Types].xml")) throw missingContentTypes(req.input);
  const edits = editList(req);
  for (const edit of edits) {
    const op = edit && edit.op;
    if (op === "replace_text" || op === "remove_text") editSlideText(editor, edit, op);
    else if (op === "format_text") formatSlideText(editor, edit);
    else if (op === "add_paragraph") addSlideParagraph(editor, edit);
    else if (op === "remove_paragraph") removeSlideParagraphs(editor, edit);
    else if (op === "add_image") addSlideImage(editor, edit);
    else if (op === "add_slide") addSlide(editor, edit);
    else if (op === "delete_slide") deleteSlide(editor, edit);
    else if (op === "move_slide") moveSlide(editor, edit);
    else if (op === "duplicate_slide") duplicateSlide(editor, edit);
    else if (op === "replace_notes" || op === "remove_notes") editSlideNotes(editor, edit, op);
    else if (op === "add_notes") addSlideNotes(editor, edit);
    else throw new UsageError(`unknown pptx edit: ${JSON.stringify(op)}`);
  }
  return editor.finish();
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
// The digits of a colour the tool accepts: `color_digits` hex digits with an
// optional leading "#". A colour is a string, so a JSON number is refused here
// too rather than coerced into digits.
const hexDigits = (color) => {
  const digits = typeof color === "string" ? color.replace(/^#/, "") : "";
  if (!HEX_COLOR.test(digits)) throw new UsageError(`color must be ${RULES.color_digits} hex digits, with an optional leading "#", got: ${color}`);
  return digits;
};
const hexColor = (hex) => {
  const digits = hexDigits(hex);
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
    // The edit paths run over packages the create arms just wrote, and each
    // family's step runs every op it advertises, so an op that fails on the
    // smallest input really runs rather than being probed on a package that
    // skips it.
    await step("edit_docx", async () => {
      writeOut(`${scratch}/probe_edit.docx`, await createDocx({ content: [
        { type: "paragraph", text: "Проверка текста" }, { type: "paragraph", text: "Лишний абзац" },
      ] }));
      const edited = docxEdit({ input: `${scratch}/probe_edit.docx`, edits: [
        { op: "replace_text", find: "текста", replace: "правки" },
        { op: "insert_text", find: "Проверка", insert: " новая", position: "after" },
        { op: "remove_text", find: " новая" },
        { op: "format_text", find: "Проверка", bold: true, italic: true, size: 14 },
        { op: "add_paragraph", after: "Проверка", text: "Добавлено" },
        { op: "remove_paragraph", find: "Лишний" },
      ] });
      writeOut(`${scratch}/probe_edited.docx`, edited.buffer);
    });
    await step("edit_xlsx", () => {
      writeOut(`${scratch}/probe_edit.xlsx`, createXlsx({ sheets: [{ name: "Данные", rows: [["Проверка", 1], ["черновик", 2]] }] }));
      const edited = xlsxEdit({ input: `${scratch}/probe_edit.xlsx`, edits: [
        { op: "set_cell", sheet: "Данные", cell: "C3", value: "правка" },
        { op: "clear_cell", sheet: "Данные", cell: "A2" },
        { op: "format_cells", sheet: "Данные", range: "A1:B1", size: 12, bold: true, italic: false, color: "FF0000", fill: "FFF2CC", border: { top: true, bottom: false }, align: "center", vertical: "top", wrap: true, number_format: "#,##0.00" },
        { op: "format_cells", sheet: "Данные", column: "A", width: 14 },
        { op: "format_cells", sheet: "Данные", row: 1, height: 20 },
        { op: "insert_row", sheet: "Данные", row: 1 },
        { op: "delete_row", sheet: "Данные", row: 4 },
        { op: "insert_column", sheet: "Данные", column: "A" },
        { op: "delete_column", sheet: "Данные", column: "A" },
      ] });
      writeOut(`${scratch}/probe_edited.xlsx`, edited.buffer);
    });
    // A deck with two slides and notes, edited through every presentation
    // operation: a text replacement and a removal, formatting, an image, an
    // added and a removed paragraph, notes replaced, removed and added, an added
    // and a deleted slide, and a moved and a duplicated one — the last two on the
    // slide whose notes the added notes landed on.
    await step("edit_pptx", async () => {
      writeOut(`${scratch}/probe_edit.pptx`, await createPptx({ content: [
        { type: "heading", level: 1, text: "Первый" }, { type: "paragraph", text: "Текст слайда" },
        { type: "notes", text: "Заметка" }, { type: "heading", level: 1, text: "Второй" },
      ] }));
      const edited = pptxEdit({ input: `${scratch}/probe_edit.pptx`, edits: [
        { op: "replace_text", slide: 1, find: "слайда", replace: "правки" },
        { op: "format_text", slide: 1, find: "Текст", bold: true, color: "#FF0000", align: "center" },
        { op: "add_image", slide: 1, path: `${scratch}/probe.png`, width: 0.3 },
        { op: "add_paragraph", slide: 1, text: "Лишнее" },
        { op: "remove_paragraph", slide: 1, find: "Лишнее" },
        { op: "add_paragraph", slide: 1, text: "Добавлено" },
        { op: "remove_text", slide: 1, find: "правки" },
        { op: "replace_notes", slide: 1, find: "Заметка", replace: "Заметка правлена" },
        { op: "remove_notes", slide: 1, find: "правлена" },
        { op: "add_notes", slide: 2, text: "Новые заметки" },
        { op: "add_slide", after: 1, title: "Новый", bullets: ["раз", "два"] },
        { op: "delete_slide", slide: 2 },
        { op: "duplicate_slide", slide: 2 },
        { op: "move_slide", slide: 3, to: 2 },
      ] });
      writeOut(`${scratch}/probe_edited.pptx`, edited.buffer);
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
  docx_edit: async (req) => {
    const edited = docxEdit(req);
    writeOut(req.output, edited.buffer);
    return { outputs: [req.output], notes: edited.notes };
  },
  xlsx_edit: async (req) => {
    const edited = xlsxEdit(req);
    writeOut(req.output, edited.buffer);
    return { outputs: [req.output], notes: edited.notes };
  },
  pptx_edit: async (req) => {
    const edited = pptxEdit(req);
    writeOut(req.output, edited.buffer);
    return { outputs: [req.output], notes: edited.notes };
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
