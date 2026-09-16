"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Heading,
  Type,
  List,
  Table2,
  Image as ImageIcon,
  Code2,
  Quote,
  SeparatorHorizontal,
  type LucideIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Block, BlockType } from "@/lib/types";

const TYPE_META: Record<
  BlockType,
  { label: string; icon: LucideIcon; className: string }
> = {
  heading: {
    label: "Heading",
    icon: Heading,
    className: "bg-blue-50 text-blue-700 border-blue-200",
  },
  paragraph: {
    label: "Paragraph",
    icon: Type,
    className: "bg-slate-50 text-slate-700 border-slate-200",
  },
  list: {
    label: "List",
    icon: List,
    className: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  table: {
    label: "Table",
    icon: Table2,
    className: "bg-amber-50 text-amber-800 border-amber-200",
  },
  image: {
    label: "Image",
    icon: ImageIcon,
    className: "bg-violet-50 text-violet-700 border-violet-200",
  },
  code: {
    label: "Code",
    icon: Code2,
    className: "bg-zinc-100 text-zinc-800 border-zinc-300",
  },
  quote: {
    label: "Quote",
    icon: Quote,
    className: "bg-sky-50 text-sky-800 border-sky-200",
  },
  page_break: {
    label: "Page break",
    icon: SeparatorHorizontal,
    className: "bg-gray-50 text-gray-600 border-gray-200",
  },
};

function BlockNode({ block, depth = 0 }: { block: Block; depth?: number }) {
  const [open, setOpen] = useState(depth < 2);
  const meta = TYPE_META[block.type] ?? TYPE_META.paragraph;
  const Icon = meta.icon;
  const hasChildren = (block.children?.length ?? 0) > 0;
  const hasExpandable =
    hasChildren ||
    !!block.rows?.length ||
    !!block.items?.length ||
    !!block.text;

  const page =
    typeof block.metadata?.page === "number"
      ? block.metadata.page
      : typeof block.metadata?.page === "string"
        ? block.metadata.page
        : null;
  const extractor =
    typeof block.metadata?.extractor === "string"
      ? block.metadata.extractor
      : null;

  return (
    <div className={cn(depth > 0 && "ml-4 border-l border-[var(--ed-border)] pl-3")}>
      <button
        type="button"
        onClick={() => hasExpandable && setOpen((v) => !v)}
        className={cn(
          "flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left transition-colors hover:bg-[var(--ed-surface)]",
          !hasExpandable && "cursor-default",
        )}
      >
        <span className="mt-0.5 size-4 shrink-0 text-[var(--ed-muted)]">
          {hasExpandable ? (
            open ? (
              <ChevronDown className="size-4" />
            ) : (
              <ChevronRight className="size-4" />
            )
          ) : null}
        </span>
        <Icon className="mt-0.5 size-4 shrink-0 text-[var(--ed-slate)]" />
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              variant="outline"
              className={cn("border text-[0.65rem] font-semibold", meta.className)}
            >
              {meta.label}
              {block.type === "heading" && block.level != null
                ? ` H${block.level}`
                : ""}
            </Badge>
            {page != null && (
              <span className="font-mono text-[0.65rem] text-[var(--ed-muted)]">
                p.{page}
              </span>
            )}
            {extractor && (
              <span className="text-[0.65rem] text-[var(--ed-muted)]">
                via {extractor}
              </span>
            )}
            <span className="font-mono text-[0.65rem] text-[var(--ed-muted)]">
              {block.id}
            </span>
          </div>
          {!open && block.text && (
            <p className="truncate text-sm text-[var(--ed-slate)]">{block.text}</p>
          )}
        </div>
      </button>

      {open && (
        <div className="mb-2 ml-6 space-y-2">
          {block.text && (
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--ed-ink)]">
              {block.text}
            </p>
          )}
          {block.items && block.items.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sm text-[var(--ed-ink)]">
              {block.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          )}
          {block.rows && block.rows.length > 0 && (
            <div className="overflow-x-auto rounded-lg border border-[var(--ed-border)]">
              <table className="min-w-full text-left text-xs">
                <tbody>
                  {block.rows.map((row, ri) => (
                    <tr
                      key={ri}
                      className={
                        ri === 0
                          ? "bg-[var(--ed-surface)] font-semibold"
                          : "border-t border-[var(--ed-border)]"
                      }
                    >
                      {row.map((cell, ci) => (
                        <td key={ci} className="px-3 py-2 whitespace-pre-wrap">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {block.image_ref && (
            <p className="font-mono text-xs break-all text-[var(--ed-slate)]">
              {block.image_ref.slice(0, 120)}
              {block.image_ref.length > 120 ? "…" : ""}
            </p>
          )}
          {block.children?.map((child) => (
            <BlockNode key={child.id} block={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

interface BlockTreeProps {
  blocks: Block[];
}

export function BlockTree({ blocks }: BlockTreeProps) {
  if (!blocks.length) {
    return (
      <div className="py-16 text-center text-sm text-[var(--ed-muted)]">
        No blocks returned.
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {blocks.map((block) => (
        <BlockNode key={block.id} block={block} />
      ))}
    </div>
  );
}
