"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FileText,
  Database,
  Layers,
  Settings,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  disabled?: boolean;
  badge?: string;
}

const NAV: NavItem[] = [
  { href: "/parser", label: "Parser", icon: FileText },
  { href: "/rag", label: "RAG", icon: Database, disabled: true, badge: "Soon" },
  {
    href: "/indexer",
    label: "Indexer",
    icon: Layers,
    disabled: true,
    badge: "Soon",
  },
];

export function PlatformShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen bg-[var(--ed-bg)] text-[var(--ed-ink)]">
      <aside className="flex w-56 shrink-0 flex-col border-r border-[var(--ed-border)] bg-white">
        <div className="border-b border-[var(--ed-border)] px-5 py-5">
          <p className="text-[0.7rem] font-semibold tracking-[0.08em] text-[var(--ed-muted)] uppercase">
            Doc2MD
          </p>
          <h1 className="mt-1 text-lg font-bold tracking-[-0.02em] text-[var(--ed-ink)]">
            Platform
          </h1>
        </div>

        <nav className="flex flex-1 flex-col gap-1 p-3" aria-label="Modules">
          {NAV.map((item) => {
            const active =
              !item.disabled &&
              (pathname === item.href || pathname.startsWith(`${item.href}/`));
            const Icon = item.icon;
            const className = cn(
              "flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
              item.disabled &&
                "cursor-not-allowed text-[var(--ed-muted)] opacity-60",
              !item.disabled &&
                !active &&
                "text-[var(--ed-slate)] hover:bg-[var(--ed-surface)] hover:text-[var(--ed-ink)]",
              active && "bg-[var(--ed-blue-soft)] text-[var(--ed-blue)]",
            );

            if (item.disabled) {
              return (
                <span
                  key={item.label}
                  className={className}
                  aria-disabled="true"
                  title="Coming soon"
                >
                  <Icon className="size-4 shrink-0" />
                  <span className="flex-1">{item.label}</span>
                  {item.badge && (
                    <span className="rounded-md bg-[var(--ed-surface)] px-1.5 py-0.5 text-[0.65rem] font-semibold tracking-wide text-[var(--ed-muted)] uppercase">
                      {item.badge}
                    </span>
                  )}
                </span>
              );
            }

            return (
              <Link key={item.href} href={item.href} className={className}>
                <Icon className="size-4 shrink-0" />
                <span className="flex-1">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-[var(--ed-border)] p-3">
          <span
            className="flex cursor-not-allowed items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm font-medium text-[var(--ed-muted)] opacity-60"
            aria-disabled="true"
            title="Coming soon"
          >
            <Settings className="size-4 shrink-0" />
            Settings
          </span>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">{children}</main>
    </div>
  );
}
