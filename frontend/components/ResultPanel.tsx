"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BlockTree } from "@/components/BlockTree";
import { MarkdownPreview } from "@/components/MarkdownPreview";
import { PipelineTrace } from "@/components/PipelineTrace";
import type { ParseResult } from "@/lib/types";

interface ResultPanelProps {
  result: ParseResult;
}

export function ResultPanel({ result }: ResultPanelProps) {
  const blockCount = result.blocks.length;
  const { stats } = result;

  return (
    <div className="rounded-xl border border-[var(--ed-border)] bg-white p-5 shadow-[var(--ed-shadow)]">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="section-label">Results</p>
          <h2 className="text-lg font-bold tracking-[-0.02em] text-[var(--ed-ink)]">
            Parse complete
          </h2>
          <p className="mt-1 text-sm text-[var(--ed-slate)]">
            Job <span className="font-mono">{result.job_id}</span>
            {result.source.filename
              ? ` · ${result.source.filename}`
              : result.source.url
                ? ` · ${result.source.url}`
                : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-4 text-center">
          <div>
            <div className="text-xl font-bold text-[var(--ed-ink)]">{blockCount}</div>
            <div className="text-[0.7rem] font-medium text-[var(--ed-muted)]">
              Blocks
            </div>
          </div>
          <div>
            <div className="text-xl font-bold text-[var(--ed-ink)]">
              {stats.char_count.toLocaleString()}
            </div>
            <div className="text-[0.7rem] font-medium text-[var(--ed-muted)]">
              Chars
            </div>
          </div>
          <div>
            <div className="text-xl font-bold text-[var(--ed-ink)]">{stats.tables}</div>
            <div className="text-[0.7rem] font-medium text-[var(--ed-muted)]">
              Tables
            </div>
          </div>
          <div>
            <div className="text-xl font-bold text-[var(--ed-ink)]">{stats.images}</div>
            <div className="text-[0.7rem] font-medium text-[var(--ed-muted)]">
              Images
            </div>
          </div>
        </div>
      </div>

      <Tabs defaultValue="blocks" className="gap-4">
        <TabsList className="h-10 bg-[var(--ed-surface)] p-1">
          <TabsTrigger value="blocks">Blocks</TabsTrigger>
          <TabsTrigger value="markdown">Markdown</TabsTrigger>
          <TabsTrigger value="trace">Trace</TabsTrigger>
        </TabsList>

        <TabsContent value="blocks" className="mt-0">
          <BlockTree blocks={result.blocks} />
        </TabsContent>
        <TabsContent value="markdown" className="mt-0">
          <MarkdownPreview result={result} />
        </TabsContent>
        <TabsContent value="trace" className="mt-0">
          <PipelineTrace
            steps={result.pipeline_trace}
            warnings={result.warnings}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
