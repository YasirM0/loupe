import mammoth from 'mammoth';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

// Real text extraction for the local pipeline — unlike Loupe's LLM-provider
// path (which hands Claude the raw PDF bytes and lets it read them), the
// local BM25/embedding/NLI pipeline needs actual sentence text to index and
// compare, so PDFs get parsed page-by-page here instead.
export async function extractPdfText(file) {
  const ab = await file.arrayBuffer();
  const pdf = await getDocument({ data: ab }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const text = content.items.map(it => it.str).join(' ');
    pages.push(text);
  }
  return pages.join('\n\n');
}

export async function extractDocxText(file) {
  const ab = await file.arrayBuffer();
  const res = await mammoth.extractRawText({ arrayBuffer: ab });
  return res.value;
}

export async function extractText(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return extractPdfText(file);
  if (ext === 'docx') return extractDocxText(file);
  return file.text();
}
