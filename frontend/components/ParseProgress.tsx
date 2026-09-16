"use client";

import { cn } from "@/lib/utils";

export type ProgressStage = "idle" | "upload" | "extract" | "normalize" | "done" | "error";

const STAGES = [
  { id: "upload", label: "Input" },
  { id: "extract", label: "Extract" },
  { id: "normalize", label: "Normalize" },
  { id: "done", label: "Result" },
] as const;

function stageIndex(stage: ProgressStage): number {
  if (stage === "idle") return -1;
  if (stage === "error") return -1;
  if (stage === "upload") return 0;
  if (stage === "extract") return 1;
  if (stage === "normalize") return 2;
  return 3;
}

interface ParseProgressProps {
  stage: ProgressStage;
  message?: string | null;
}

export function ParseProgress({ stage, message }: ParseProgressProps) {
  if (stage === "idle") return null;

  const active = stageIndex(stage);
  const failed = stage === "error";

  return (
    <div className="rounded-xl border border-[var(--ed-border)] bg-white px-6 py-5 shadow-[var(--ed-shadow)]">
      <div className="relative flex items-start justify-between">
        <div
          className="absolute top-5 right-10 left-10 h-0.5 bg-[var(--ed-border)]"
          aria-hidden
        />
        {STAGES.map((s, i) => {
          const done = !failed && active > i;
          const current = !failed && active === i;
          return (
            <div
              key={s.id}
              className="relative z-10 flex flex-1 flex-col items-center gap-2"
            >
              <div
                className={cn(
                  "flex size-10 items-center justify-center rounded-full border-2 text-sm font-semibold",
                  done &&
                    "border-[var(--ed-green)] bg-[var(--ed-green)] text-white",
                  current &&
                    "animate-pulse border-[var(--ed-blue)] bg-[var(--ed-blue)] text-white",
                  !done &&
                    !current &&
                    "border-[var(--ed-border-strong)] bg-[var(--ed-surface)] text-[var(--ed-muted)]",
                  failed &&
                    i === 0 &&
                    "border-red-400 bg-red-50 text-red-600",
                )}
              >
                {done ? "✓" : i + 1}
              </div>
              <span
                className={cn(
                  "text-[0.75rem] font-medium tracking-[0.05em] uppercase",
                  done && "text-[var(--ed-green)]",
                  current && "text-[var(--ed-blue)]",
                  !done && !current && "text-[var(--ed-slate)]",
                )}
              >
                {s.label}
              </span>
            </div>
          );
        })}
      </div>
      {message && (
        <p className="mt-4 text-center text-sm text-[var(--ed-slate)]">
          {message}
        </p>
      )}
    </div>
  );
}
