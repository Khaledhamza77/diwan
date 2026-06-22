import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { cn } from "@/lib/utils";
import remarkBreaks from "remark-breaks";
import { MermaidDiagram } from "./MermaidDiagram";

export type MessageAlignment = "left" | "right";

export interface MessageProps {
  text: string;
  align?: MessageAlignment;
  meta?: string;
  className?: string;
  bubbleClassName?: string;
  children?: React.ReactNode;
}

const remarkPlugins = [remarkGfm, remarkBreaks, remarkMath];
const rehypePlugins = [rehypeRaw, rehypeKatex];

// Strips Arabic-Indic numeral or single Arabic letter enumeration prefixes from
// markdown list item lines to prevent double-indexing when the LLM mixes both.
// e.g. "- ١. نص" → "- نص" and "1. أ. نص" → "1. نص"
function stripArabicListPrefix(text: string): string {
  return text.replace(
    /^([ \t]*(?:[-*+]|\d+[.)]) +)([٠-٩۰-۹]+|[ء-غف-ي])[.)–\-]\s*/gmu,
    "$1",
  );
}

const _ARABIC_RE = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/;

// Replace [Source: ...] tags with a styled inline span so they render as
// small, greyed-out citations instead of plain bracketed text.
const _SOURCE_RE = /\[Source:[^\]]+\]/g;
function styleSourceTags(text: string): string {
  return text.replace(
    _SOURCE_RE,
    (match) =>
      `<span style="font-size:11px;color:#6b7280;font-style:normal;white-space:nowrap;font-weight:normal;line-height:1;">${match}</span>`,
  );
}

// Convert [N] inline citations in the body text to styled superscript badges.
const _INLINE_CITE_RE = /\[(\d+)\]/g;
function styleInlineCitations(body: string): string {
  return body.replace(
    _INLINE_CITE_RE,
    (_, num) =>
      `<sup dir="ltr" style="color:#a78bfa;font-size:0.7em;font-weight:700;vertical-align:super;line-height:0;white-space:nowrap;">[${num}]</sup>`,
  );
}

// Render the footnote block (after ---) as a styled div instead of plain text.
function buildFootnotesHtml(footnotesText: string): string {
  const lines = footnotesText.trim().split("\n").filter((l) => l.trim());
  if (lines.length === 0) return "";
  const items = lines
    .map((line) => {
      const m = line.match(/^\[(\d+)\]\s*(.+)$/);
      if (!m) return "";
      const [, num, content] = m;
      return (
        `<div style="display:flex;gap:6px;align-items:baseline;padding:2px 0;">` +
        `<sup dir="ltr" style="color:#a78bfa;font-weight:700;font-size:0.75em;min-width:18px;flex-shrink:0;">[${num}]</sup>` +
        `<span style="color:#9ca3af;font-size:0.78em;">${content}</span>` +
        `</div>`
      );
    })
    .filter(Boolean);
  return (
    `<div style="margin-top:0.9em;padding-top:0.5em;border-top:1px solid #374151;">` +
    items.join("") +
    `</div>`
  );
}

// Process the full answer text: style inline [N] citations and replace the
// --- footnote block with formatted HTML.
function processAnswer(text: string): string {
  const sep = "\n---\n";
  const idx = text.indexOf(sep);
  if (idx === -1) return styleInlineCitations(text);
  const body = styleInlineCitations(text.slice(0, idx));
  const footnotesHtml = buildFootnotesHtml(text.slice(idx + sep.length));
  return body + (footnotesHtml ? "\n\n" + footnotesHtml : "");
}

// Strip Arabic letter enumeration labels that leak through from the LLM
// despite prompt instructions. Handles both "(أ)." suffix and "(أ) " prefix
// patterns on bullet-point lines.
// Covers: أ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن هـ ه و ي ة
const _AR_LABEL_SUFFIX = /\s*[\(\(][أ-يةهـ]{1,2}[\)\)][.)،,]?\s*$/gmu;
const _AR_LABEL_PREFIX = /^([ \t]*[-*]\s*)[\(\(][أ-يةهـ]{1,2}[\)\)][.)،,]?\s*/gmu;
function stripArabicLetterLabels(text: string): string {
  return text
    .replace(_AR_LABEL_SUFFIX, "")
    .replace(_AR_LABEL_PREFIX, "$1");
}

// Matches a table cell whose entire content is numeric (digits, decimal/
// thousands separators, %, currency symbols, +/-) so it can be force-LTR'd
// to avoid bidi reordering inside an RTL table row.
const _NUMERIC_CELL_RE = /^[\s0-9.,%+\-$£€٪]*$/u;
function isNumericCell(children: React.ReactNode): boolean {
  const text = React.Children.toArray(children)
    .map((c) => (typeof c === "string" || typeof c === "number" ? String(c) : ""))
    .join("");
  return text.trim().length > 0 && _NUMERIC_CELL_RE.test(text);
}

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE_URL ?? "http://localhost:8000";

export const Message: React.FC<MessageProps> = React.memo(({
  text,
  align = "left",
  meta,
  className,
  bubbleClassName,
  children,
}) => {
  const isRight = align === "right";
  // Detect direction from the first strongly-directional characters so RTL
  // is applied immediately on the first streamed token, not after accumulation.
  const textDir = _ARABIC_RE.test(text.slice(0, 120)) ? "rtl" : "ltr";

  return (
    <div
      className={cn(
        "w-full flex",
        isRight ? "justify-end" : "justify-start",
        className,
      )}
    >
      <div
        className={cn(
          "max-w-[120ch] text-[13.5px] leading-relaxed",
          bubbleClassName,
        )}
      >
        {children}
        {/* dir detected from first Arabic character in text, applied immediately
            on the first streaming token so RTL works during the heading phase. */}
        <div className="prose prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0" dir={textDir}>
          <ReactMarkdown
            remarkPlugins={remarkPlugins}
            rehypePlugins={rehypePlugins}
            components={{
              // Intercept links — prevent SPA navigation for /documents/ paths
              a: ({ href, children }) => {
                const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
                  if (!href) return;
                  if (href.startsWith("/documents/")) {
                    e.preventDefault();
                    const full = `${API_BASE}${href}`;
                    window.open(full, "_blank", "noopener,noreferrer");
                  }
                };
                const isDoc = href?.startsWith("/documents/");
                return (
                  <a
                    href={isDoc ? "#" : href}
                    onClick={handleClick}
                    target={isDoc ? undefined : "_blank"}
                    rel="noopener noreferrer"
                    className="text-purple-400 underline underline-offset-2 hover:text-purple-300 transition-colors cursor-pointer"
                  >
                    {children}
                  </a>
                );
              },
              code({ className, children, ...props }: any) {
                const language = /language-(\.\w+)/.exec(className ?? "")?.[1];
                const code = String(children).replace(/\n$/, "");
                if (language === "mermaid") {
                  return <MermaidDiagram code={code} />;
                }
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
              // Force dir=ltr on every KaTeX-generated span/div so the browser
              // bidi algorithm cannot reorder digits inside an RTL message.
              span({ className, children, ...props }: any) {
                const cls: string = className ?? "";
                if (cls.includes("katex")) {
                  return <span className={cls} dir="ltr" {...props}>{children}</span>;
                }
                return <span className={cls} {...props}>{children}</span>;
              },
              div({ className, children, ...props }: any) {
                const cls: string = className ?? "";
                if (cls.includes("katex")) {
                  return <div className={cls} dir="ltr" {...props}>{children}</div>;
                }
                return <div className={cls} {...props}>{children}</div>;
              },
              // Force dir=ltr on table cells whose content is purely numeric
              // (digits, %, currency, separators) so digits don't get
              // bidi-reordered inside an RTL table row, mirroring the KaTeX fix above.
              td: ({ children, ...props }: any) => (
                <td {...props} dir={isNumericCell(children) ? "ltr" : undefined}>
                  {children}
                </td>
              ),
              th: ({ children, ...props }: any) => (
                <th {...props} dir={isNumericCell(children) ? "ltr" : undefined}>
                  {children}
                </th>
              ),
            }}
          >
            {processAnswer(styleSourceTags(stripArabicListPrefix(stripArabicLetterLabels(text))))}
          </ReactMarkdown>
          <style>{`
            .katex-display {
              direction: ltr !important;
              text-align: center;
              unicode-bidi: isolate;
            }
            .katex {
              direction: ltr !important;
              unicode-bidi: isolate;
            }
          `}</style>
        </div>

        {meta ? (
          <div
            className={cn(
              "mt-1 text-[11px] text-green-400/80",
              isRight && "text-right",
            )}
          >
            {meta}
          </div>
        ) : null}
      </div>
    </div>
  );
});
