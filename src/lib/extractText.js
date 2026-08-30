import mammoth from 'mammoth';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

// A PDF page over 1000pt in either dimension reads as slide-deck-shaped
// (a standard letter/A4 page tops out around 792/842pt) — those tend to be
// image-heavy with little real text per page.
const LARGE_PAGE_PT = 1000;
// Below this many extracted characters per page, the "text" is really just
// scattered labels/captions rather than a real prose layer — a proxy for
// "would `pdffonts` report no embedded text," since shelling out to a CLI
// tool isn't available in a browser/GitHub-Pages build; this is the
// in-browser equivalent signal using what pdf.js already extracts.
const SPARSE_CHARS_PER_PAGE = 200;

// Real text extraction for the local pipeline — unlike Loupe's LLM-provider
// path (which hands Claude the raw PDF bytes and lets it read them), the
// local BM25/embedding/NLI pipeline needs actual sentence text to index and
// compare, so PDFs get parsed page-by-page here instead. Also assesses
// extraction quality while it's already walking every page, so callers can
// warn the user when a source is unlikely to be reliably searchable rather
// than silently returning UNSUPPORTED for content that's visually present
// but not text-extractable.
export async function extractPdfText(file) {
  const ab = await file.arrayBuffer();
  const pdf = await getDocument({ data: ab }).promise;
  const pages = [];
  let maxWidth = 0, maxHeight = 0, totalChars = 0;
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 1 });
    maxWidth = Math.max(maxWidth, viewport.width);
    maxHeight = Math.max(maxHeight, viewport.height);
    const content = await page.getTextContent();
    const text = content.items.map(it => it.str).join(' ');
    totalChars += text.replace(/\s/g, '').length;
    pages.push(text);
  }
  const avgCharsPerPage = pdf.numPages ? totalChars / pdf.numPages : 0;
  const quality = {
    largePage: maxWidth > LARGE_PAGE_PT || maxHeight > LARGE_PAGE_PT,
    sparseText: avgCharsPerPage < SPARSE_CHARS_PER_PAGE,
    pageDims: { width: Math.round(maxWidth), height: Math.round(maxHeight) },
  };
  return { text: pages.join('\n\n'), quality };
}

export async function extractDocxText(file) {
  const ab = await file.arrayBuffer();
  const res = await mammoth.extractRawText({ arrayBuffer: ab });
  return res.value;
}

export async function extractText(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return (await extractPdfText(file)).text;
  if (ext === 'docx') return extractDocxText(file);
  return file.text();
}
