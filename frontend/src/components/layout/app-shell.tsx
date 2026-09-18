"use client";

import * as React from "react";

import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-base">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main className="scrollbar-thin flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

/**
 * Page body with an optional right-hand inspector. The inspector is a detail
 * surface, not navigation: it shows what is selected in the main workspace.
 */
export function Workspace({
  children,
  inspector,
  inspectorTitle,
  className,
}: {
  children: React.ReactNode;
  inspector?: React.ReactNode;
  inspectorTitle?: string;
  className?: string;
}) {
  return (
    <div className="flex h-full min-h-0">
      <div className={cn("min-w-0 flex-1 space-y-4 p-4", className)}>{children}</div>
      {inspector ? (
        <aside className="hidden w-[320px] shrink-0 border-l border-border bg-panel/50 xl:block">
          <div className="scrollbar-thin h-full overflow-y-auto">
            {inspectorTitle ? (
              <div className="sticky top-0 z-10 border-b border-border bg-panel px-4 py-2.5 text-2xs font-semibold uppercase tracking-[0.14em] text-ink-muted">
                {inspectorTitle}
              </div>
            ) : null}
            <div className="space-y-4 p-4">{inspector}</div>
          </div>
        </aside>
      ) : null}
    </div>
  );
}
