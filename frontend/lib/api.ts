import {
  type AsyncJobAccepted,
  type JobPollResponse,
  type ParseResult,
  type ParseResponse,
  type TextFormat,
  isAsyncJob,
  isParseResult,
} from "./types";

const DEFAULT_API_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function apiBase(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim() || DEFAULT_API_URL;
  return raw.replace(/\/$/, "");
}

async function parseError(res: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = await res.text().catch(() => null);
  }
  const detail =
    typeof body === "object" &&
    body !== null &&
    "detail" in body &&
    typeof (body as { detail: unknown }).detail === "string"
      ? (body as { detail: string }).detail
      : `Request failed (${res.status})`;
  return new ApiError(detail, res.status, body);
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<{ data: T; status: number }> {
  const res = await fetch(`${apiBase()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    throw await parseError(res);
  }

  const data = (await res.json()) as T;
  return { data, status: res.status };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Poll GET /api/v1/parse/{job_id} until completed or failed. */
export async function pollJob(
  jobId: string,
  options?: {
    intervalMs?: number;
    timeoutMs?: number;
    signal?: AbortSignal;
    onProgress?: (job: JobPollResponse) => void;
  },
): Promise<ParseResult> {
  const intervalMs = options?.intervalMs ?? POLL_INTERVAL_MS;
  const timeoutMs = options?.timeoutMs ?? POLL_TIMEOUT_MS;
  const started = Date.now();

  while (true) {
    if (options?.signal?.aborted) {
      throw new ApiError("Polling aborted", 499);
    }
    if (Date.now() - started > timeoutMs) {
      throw new ApiError(`Job ${jobId} timed out after ${timeoutMs}ms`, 408);
    }

    const { data } = await requestJson<JobPollResponse>(
      `/api/v1/parse/${encodeURIComponent(jobId)}`,
      { signal: options?.signal },
    );
    options?.onProgress?.(data);

    if (data.status === "completed") {
      if (data.result && isParseResult(data.result)) {
        return data.result;
      }
      throw new ApiError("Job completed without a result payload", 500, data);
    }

    if (data.status === "failed") {
      throw new ApiError(data.error || `Job ${jobId} failed`, 500, data);
    }

    await sleep(intervalMs);
  }
}

async function resolveParseResponse(
  response: ParseResponse,
  status: number,
  options?: {
    signal?: AbortSignal;
    onProgress?: (job: JobPollResponse) => void;
  },
): Promise<ParseResult> {
  if (status === 202 || isAsyncJob(response)) {
    const job = response as AsyncJobAccepted;
    return pollJob(job.job_id, options);
  }
  if (isParseResult(response)) {
    return response;
  }
  throw new ApiError("Unexpected parse response shape", status, response);
}

/** POST /api/v1/parse/file — multipart file upload. */
export async function parseFile(
  file: File,
  options?: {
    signal?: AbortSignal;
    onProgress?: (job: JobPollResponse) => void;
  },
): Promise<ParseResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${apiBase()}/api/v1/parse/file`, {
    method: "POST",
    body: form,
    signal: options?.signal,
    headers: { Accept: "application/json" },
  });

  if (!res.ok) {
    throw await parseError(res);
  }

  const data = (await res.json()) as ParseResponse;
  return resolveParseResponse(data, res.status, options);
}

/** POST /api/v1/parse/text — JSON body. */
export async function parseText(
  text: string,
  format: TextFormat = "plain",
  options?: {
    signal?: AbortSignal;
    onProgress?: (job: JobPollResponse) => void;
  },
): Promise<ParseResult> {
  const { data, status } = await requestJson<ParseResponse>(
    "/api/v1/parse/text",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, format }),
      signal: options?.signal,
    },
  );
  return resolveParseResponse(data, status, options);
}

/** POST /api/v1/parse/url — JSON body. */
export async function parseUrl(
  url: string,
  options?: {
    signal?: AbortSignal;
    onProgress?: (job: JobPollResponse) => void;
  },
): Promise<ParseResult> {
  const { data, status } = await requestJson<ParseResponse>(
    "/api/v1/parse/url",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: options?.signal,
    },
  );
  return resolveParseResponse(data, status, options);
}

export async function getHealth(): Promise<{ status: string; [key: string]: unknown }> {
  const { data } = await requestJson<{ status: string }>("/api/v1/health");
  return data;
}

export { apiBase };
