const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, WidthType, BorderStyle, ShadingType, Footer, PageNumber, HeadingLevel } = require('docx');
const B = JSON.parse(fs.readFileSync('paper.json', 'utf8'));
const FONT = 'Times New Roman';
function runs(text, size = 21, base = {}) {   // **bold**, *italic*
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size, ...base }));
    const tok = m[0];
    if (tok.startsWith('**')) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, size, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size, ...base }));
  return out;
}
const W = 9360; // content width in DXA (US Letter, 1" margins)
const border = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const borders = { top: border, bottom: border, left: border, right: border };
function pngSize(p) { const b = fs.readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
const children = [];
for (const b of B) {
  if (b.t === 'title') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 34 })] }));
  else if (b.t === 'authors') b.lines.forEach((l, i) => children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: i === b.lines.length - 1 ? 240 : 40 }, children: [new TextRun({ text: l, font: FONT, size: 22, italics: i > 0 })] })));
  else if (b.t === 'abstract') children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [new TextRun({ text: 'Abstract— ', bold: true, italics: true, font: FONT, size: 20 }), ...runs(b.text, 20, { bold: false })] }));
  else if (b.t === 'keywords') children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [new TextRun({ text: 'Keywords— ', bold: true, italics: true, font: FONT, size: 20 }), new TextRun({ text: b.text, font: FONT, size: 20 })] }));
  else if (b.t === 'h1') children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: b.text.toUpperCase().replace(/^([IVX]+)\. /, '$1. '), bold: true, font: FONT, size: 22 })] }));
  else if (b.t === 'h2') children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 }, children: [new TextRun({ text: b.text, italics: true, bold: true, font: FONT, size: 21 })] }));
  else if (b.t === 'h3') children.push(new Paragraph({ spacing: { before: 120, after: 60 }, children: [new TextRun({ text: b.text, italics: true, font: FONT, size: 21 })] }));
  else if (b.t === 'p') children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 110, line: 264 }, children: runs(b.text) }));
  else if (b.t === 'eq') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: [new TextRun({ text: b.text, font: 'Cambria Math', size: 21, italics: true })] }));
  else if (b.t === 'code') children.push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ children: [new TableCell({ width: { size: W, type: WidthType.DXA }, borders, shading: { fill: 'F3F3F3', type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 140, right: 140 }, children: b.lines.map((l, i) => new Paragraph({ spacing: { after: 20 }, children: [new TextRun({ text: l, font: 'Courier New', size: 17, bold: i === 0 })] })) })] })] }), new Paragraph({ spacing: { after: 100 }, children: [] }));
  else if (b.t === 'img') {
    const [w, h] = pngSize(b.path); const wpx = Math.round(b.w * 96), hpx = Math.round(wpx * h / w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true, children: [new ImageRun({ type: 'png', data: fs.readFileSync(b.path), transformation: { width: wpx, height: hpx }, altText: { title: b.caption, description: b.caption, name: b.caption.slice(0, 20) } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: b.caption, font: FONT, size: 18, italics: true })] }));
  } else if (b.t === 'table') {
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 140, after: 60 }, children: [new TextRun({ text: b.caption, font: FONT, size: 17, bold: true })] }));
    const cw = b.widths.map(f => Math.round(f * W)); const diff = W - cw.reduce((a, c) => a + c, 0); cw[0] += diff;
    const fs_ = Math.round((b.fs || 8) * 2);
    const mk = (cells, head) => new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((c, i) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, borders, shading: head ? { fill: 'DCE6F1', type: ShadingType.CLEAR } : undefined, margins: { top: 30, bottom: 30, left: 60, right: 60 }, children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, children: [new TextRun({ text: String(c), font: FONT, size: fs_, bold: head })] })] })) });
    children.push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: cw, rows: [mk(b.header, true), ...b.rows.map(r => mk(r, false))] }));
    children.push(new Paragraph({ spacing: { after: 140 }, children: [] }));
  } else if (b.t === 'refs') {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: 'REFERENCES', bold: true, font: FONT, size: 22 })] }));
    b.items.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 50 }, indent: { left: 460, hanging: 460 }, children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 18 }), new TextRun({ text: r, font: FONT, size: 18 })], tabStops: [{ type: 'left', position: 460 }] })));
  }
}
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 22, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 21, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) }, children }]
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('/home/claude/build/paper.docx', b); console.log('written', b.length); });
