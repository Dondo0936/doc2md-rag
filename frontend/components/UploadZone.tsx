"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, FileUp } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPT =
  ".pdf,.docx,.pptx,.csv,.txt,.md,.png,.jpg,.jpeg,.webp,application/pdf,text/plain,text/csv,text/markdown,image/png,image/jpeg,image/webp";

interface UploadZoneProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
}

export function UploadZone({ file, onFileChange, disabled }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = useCallback(
    (list: FileList | null) => {
      if (!list?.length) return;
      onFileChange(list[0]);
    },
    [onFileChange],
  );

  return (
    <div className="space-y-3">
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) pick(e.dataTransfer.files);
        }}
        className={cn(
          "flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 transition-colors",
          "bg-white text-center",
          dragging
            ? "border-[var(--ed-blue)] bg-[var(--ed-blue-soft)]"
            : "border-[var(--ed-border)] hover:border-[var(--ed-blue)] hover:bg-[var(--ed-blue-soft)]/40",
          disabled && "pointer-events-none opacity-50",
        )}
      >
        <div className="flex size-12 items-center justify-center rounded-full bg-[var(--ed-blue-soft)] text-[var(--ed-blue)]">
          <Upload className="size-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-[var(--ed-ink)]">
            Drop a file here or click to browse
          </p>
          <p className="mt-1 text-xs text-[var(--ed-slate)]">
            PDF, DOCX, PPTX, CSV, TXT, MD, PNG, JPG, WEBP
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => pick(e.target.files)}
        />
      </button>

      {file && (
        <div className="flex items-center gap-3 rounded-xl border border-[var(--ed-border)] bg-white px-4 py-3 shadow-[var(--ed-shadow)]">
          <FileUp className="size-4 shrink-0 text-[var(--ed-blue)]" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-[var(--ed-ink)]">
              {file.name}
            </p>
            <p className="text-xs text-[var(--ed-muted)]">
              {(file.size / 1024).toFixed(1)} KB
            </p>
          </div>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onFileChange(null)}
            className="text-xs font-medium text-[var(--ed-slate)] hover:text-[var(--ed-ink)]"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
