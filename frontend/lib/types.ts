/** Types matching the Parser Platform FastAPI contract (plan §1 / §3). */

export type BlockType =
  | "heading"
  | "paragraph"
  | "list"
  | "table"
  | "image"
  | "code"
  | "quote"
  | "page_break";

export type PipelineStepStatus = "success" | "failed" | "skipped";

export type TextFormat = "plain" | "markdown" | "html";

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface Block {
  id: string;
  type: BlockType;
  text?: string | null;
  level?: number | null;
  rows?: string[][] | null;
  items?: string[] | null;
  image_ref?: string | null;
  metadata?: Record<string, unknown>;
  children?: Block[];
}

export interface PipelineStep {
  extractor: string;
  attempt: number;
  status: PipelineStepStatus;
  duration_ms: number;
  error?: string | null;
}

export interface SourceInfo {
  input_type: "file" | "text" | "url" | "image";
  filename?: string | null;
  url?: string | null;
  mime?: string | null;
  size_bytes?: number | null;
  format?: TextFormat | null;
}

export interface ParseStats {
  char_count: number;
  tables: number;
  images: number;
  pages?: number | null;
  block_counts: Partial<Record<BlockType, number>>;
}

export interface ParseResult {
  job_id: string;
  source: SourceInfo;
  blocks: Block[];
  markdown: string;
  stats: ParseStats;
  pipeline_trace: PipelineStep[];
  warnings: string[];
}

/** Response when the API accepts a heavy job for background processing (HTTP 202). */
export interface AsyncJobAccepted {
  job_id: string;
  status: JobStatus;
  message?: string;
}

export interface JobPollResponse {
  job_id: string;
  status: JobStatus;
  result?: ParseResult | null;
  error?: string | null;
  progress?: string | null;
}

export type ParseResponse = ParseResult | AsyncJobAccepted;

export function isAsyncJob(value: ParseResponse): value is AsyncJobAccepted {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    !("blocks" in value)
  );
}

export function isParseResult(value: unknown): value is ParseResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "blocks" in value &&
    "markdown" in value &&
    "job_id" in value
  );
}
