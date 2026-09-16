"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { PipelineStep } from "@/lib/types";

interface PipelineTraceProps {
  steps: PipelineStep[];
  warnings?: string[];
}

const STATUS_STYLE: Record<
  PipelineStep["status"],
  { label: string; className: string }
> = {
  success: {
    label: "success",
    className: "border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  failed: {
    label: "failed",
    className: "border-red-200 bg-red-50 text-red-700",
  },
  skipped: {
    label: "skipped",
    className: "border-slate-200 bg-slate-50 text-slate-600",
  },
};

export function PipelineTrace({ steps, warnings = [] }: PipelineTraceProps) {
  return (
    <div className="space-y-4">
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <p className="mb-1 font-semibold">Warnings</p>
          <ul className="list-disc space-y-1 pl-5">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {!steps.length ? (
        <div className="py-12 text-center text-sm text-[var(--ed-muted)]">
          No pipeline steps recorded.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--ed-border)]">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-[var(--ed-surface)] text-[0.7rem] font-semibold tracking-[0.06em] text-[var(--ed-muted)] uppercase">
              <tr>
                <th className="px-4 py-3">Extractor</th>
                <th className="px-4 py-3">Attempt</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Error</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((step, i) => {
                const style = STATUS_STYLE[step.status];
                return (
                  <tr
                    key={`${step.extractor}-${step.attempt}-${i}`}
                    className="border-t border-[var(--ed-border)]"
                  >
                    <td className="px-4 py-3 font-medium text-[var(--ed-ink)]">
                      {step.extractor}
                    </td>
                    <td className="px-4 py-3 font-mono text-[var(--ed-slate)]">
                      {step.attempt}
                    </td>
                    <td className="px-4 py-3">
                      <Badge
                        variant="outline"
                        className={cn("border text-[0.65rem]", style.className)}
                      >
                        {style.label}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 font-mono text-[var(--ed-slate)]">
                      {step.duration_ms} ms
                    </td>
                    <td className="max-w-xs truncate px-4 py-3 text-[var(--ed-slate)]">
                      {step.error || "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
