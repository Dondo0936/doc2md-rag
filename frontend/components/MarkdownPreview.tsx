"use client";

import { useState } from "react";
import { Check, Copy, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ParseResult } from "@/lib/types";

function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function baseName(result: ParseResult): string {
  const raw =
    result.source.filename ||
    result.source.url?.split("/").pop() ||
    result.job_id ||
    "parse";
  return raw.replace(/\.[^.]+$/, "") || "parse";
}

interface MarkdownPreviewProps {
  result: ParseResult;
}

export function MarkdownPreview({ result }: MarkdownPreviewProps) {
  const [copied, setCopied] = useState(false);

  async function copyMarkdown() {
    await navigator.clipboard.writeText(result.markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={copyMarkdown}
          className="border-[var(--ed-border)]"
        >
          {copied ? (
            <Check className="size-3.5 text-[var(--ed-green)]" />
          ) : (
            <Copy className="size-3.5" />
          )}
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() =>
            downloadBlob(`${baseName(result)}.md`, result.markdown, "text/markdown")
          }
          className="border-[var(--ed-border)]"
        >
          <Download className="size-3.5" />
          Download .md
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() =>
            downloadBlob(
              `${baseName(result)}.json`,
              JSON.stringify(result, null, 2),
              "application/json",
            )
          }
          className="border-[var(--ed-border)]"
        >
          <Download className="size-3.5" />
          Download .json
        </Button>
      </div>

      <pre className="max-h-[32rem] overflow-auto rounded-xl border border-[var(--ed-border)] bg-[var(--ed-surface)] p-4 text-sm leading-relaxed whitespace-pre-wrap text-[var(--ed-ink)]">
        {result.markdown || "(empty markdown)"}
      </pre>
    </div>
  );
}
