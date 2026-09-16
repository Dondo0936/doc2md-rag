"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { UploadZone } from "@/components/UploadZone";
import type { TextFormat } from "@/lib/types";

export type InputMode = "upload" | "text" | "url";

export interface ParseSubmitPayload {
  mode: InputMode;
  file?: File | null;
  text?: string;
  format?: TextFormat;
  url?: string;
}

interface InputTabsProps {
  disabled?: boolean;
  onSubmit: (payload: ParseSubmitPayload) => void;
}

export function InputTabs({ disabled, onSubmit }: InputTabsProps) {
  const [mode, setMode] = useState<InputMode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [format, setFormat] = useState<TextFormat>("plain");
  const [url, setUrl] = useState("");

  const canSubmit =
    mode === "upload"
      ? !!file
      : mode === "text"
        ? text.trim().length > 0
        : url.trim().length > 0;

  return (
    <div className="rounded-xl border border-[var(--ed-border)] bg-white p-5 shadow-[var(--ed-shadow)]">
      <p className="section-label mb-3">Input</p>
      <Tabs
        value={mode}
        onValueChange={(v) => setMode(v as InputMode)}
        className="gap-4"
      >
        <TabsList className="h-10 w-full justify-start bg-[var(--ed-surface)] p-1">
          <TabsTrigger value="upload" disabled={disabled} className="flex-1">
            Upload
          </TabsTrigger>
          <TabsTrigger value="text" disabled={disabled} className="flex-1">
            Paste text
          </TabsTrigger>
          <TabsTrigger value="url" disabled={disabled} className="flex-1">
            URL
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="mt-0 space-y-4">
          <UploadZone file={file} onFileChange={setFile} disabled={disabled} />
        </TabsContent>

        <TabsContent value="text" className="mt-0 space-y-3">
          <div className="flex flex-wrap gap-2">
            {(["plain", "markdown", "html"] as TextFormat[]).map((f) => (
              <button
                key={f}
                type="button"
                disabled={disabled}
                onClick={() => setFormat(f)}
                className={
                  format === f
                    ? "rounded-lg border border-[var(--ed-blue)] bg-[var(--ed-blue-soft)] px-3 py-1.5 text-xs font-semibold text-[var(--ed-blue)]"
                    : "rounded-lg border border-[var(--ed-border)] bg-white px-3 py-1.5 text-xs font-medium text-[var(--ed-slate)] hover:border-[var(--ed-blue)]"
                }
              >
                {f}
              </button>
            ))}
          </div>
          <Textarea
            value={text}
            disabled={disabled}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste plain text, markdown, or HTML…"
            className="min-h-40 resize-y border-[var(--ed-border)] bg-[var(--ed-surface)] text-sm"
          />
        </TabsContent>

        <TabsContent value="url" className="mt-0 space-y-3">
          <Input
            type="url"
            value={url}
            disabled={disabled}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/document.pdf"
            className="h-11 border-2 border-[var(--ed-blue)] bg-[var(--ed-blue-soft)] text-sm focus-visible:ring-[var(--ed-blue)]/20"
          />
          <p className="text-xs text-[var(--ed-slate)]">
            HTML pages or direct links to supported file types.
          </p>
        </TabsContent>
      </Tabs>

      <div className="mt-5 flex justify-end">
        <Button
          disabled={disabled || !canSubmit}
          onClick={() =>
            onSubmit({
              mode,
              file,
              text,
              format,
              url,
            })
          }
          className="h-10 bg-[var(--ed-blue)] px-5 text-white hover:bg-[var(--ed-blue-dark)]"
        >
          Parse
        </Button>
      </div>
    </div>
  );
}
