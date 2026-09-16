"use client";

import { useRef, useState } from "react";
import { InputTabs, type ParseSubmitPayload } from "@/components/InputTabs";
import {
  ParseProgress,
  type ProgressStage,
} from "@/components/ParseProgress";
import { ResultPanel } from "@/components/ResultPanel";
import { ApiError, parseFile, parseText, parseUrl } from "@/lib/api";
import type { ParseResult } from "@/lib/types";

export default function ParserPage() {
  const [stage, setStage] = useState<ProgressStage>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ParseResult | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const busy = stage === "upload" || stage === "extract" || stage === "normalize";

  async function handleSubmit(payload: ParseSubmitPayload) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    setResult(null);
    setStage("upload");
    setMessage("Sending to parser…");

    try {
      setStage("extract");
      setMessage("Running extractor chain…");

      const onProgress = (job: { progress?: string | null; status: string }) => {
        setStage("normalize");
        setMessage(
          job.progress ||
            (job.status === "running"
              ? "Job running…"
              : `Job status: ${job.status}`),
        );
      };

      let parsed: ParseResult;
      if (payload.mode === "upload") {
        if (!payload.file) throw new Error("Choose a file to upload.");
        parsed = await parseFile(payload.file, {
          signal: controller.signal,
          onProgress,
        });
      } else if (payload.mode === "text") {
        parsed = await parseText(payload.text || "", payload.format || "plain", {
          signal: controller.signal,
          onProgress,
        });
      } else {
        parsed = await parseUrl(payload.url || "", {
          signal: controller.signal,
          onProgress,
        });
      }

      setStage("done");
      setMessage("Done");
      setResult(parsed);
    } catch (err) {
      if (controller.signal.aborted) return;
      const msg =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Parse failed";
      setStage("error");
      setMessage(null);
      setError(msg);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-6 py-8">
      <header>
        <p className="section-label">Module</p>
        <h1 className="ed-title">Parser</h1>
        <p className="ed-sub">
          Upload a document, paste text, or fetch a URL. Structured blocks are
          the source of truth; markdown is an export.
        </p>
      </header>

      <InputTabs disabled={busy} onSubmit={handleSubmit} />

      <ParseProgress stage={stage} message={message} />

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
        >
          {error}
        </div>
      )}

      {result && <ResultPanel result={result} />}

      {!result && !error && stage === "idle" && (
        <div className="rounded-xl border border-dashed border-[var(--ed-border)] bg-white/70 px-6 py-16 text-center">
          <h2 className="text-lg font-semibold text-[var(--ed-ink)]">
            No parse yet
          </h2>
          <p className="mt-2 text-sm text-[var(--ed-slate)]">
            Results will show blocks, markdown export, and the pipeline trace.
          </p>
        </div>
      )}
    </div>
  );
}
