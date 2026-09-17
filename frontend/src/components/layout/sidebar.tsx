"use client";

import {
  Activity,
  Boxes,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  LayoutDashboard,
  type LucideIcon,
  MessageSquare,
  PanelsTopLeft,
  ScrollText,
  Settings,
  Signal,
  Target,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useActiveJobs } from "@/lib/hooks/queries";
import { useWorkspace } from "@/lib/store";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  group: string;
}

const NAV: NavItem[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard, group: "Workspace" },
  { href: "/models", label: "Models", icon: Boxes, group: "Workspace" },
  { href: "/training", label: "Training", icon: Activity, group: "Workspace" },
  { href: "/datasets", label: "Datasets", icon: Database, group: "Workspace" },
  { href: "/chat", label: "Chat", icon: MessageSquare, group: "Inference" },
  { href: "/playground", label: "Playground", icon: PanelsTopLeft, group: "Inference" },
  { href: "/benchmarks", label: "Benchmarks", icon: Target, group: "Evaluation" },
  { href: "/evaluations", label: "Evaluations", icon: Gauge, group: "Evaluation" },
  { href: "/experiments", label: "Experiments", icon: FlaskConical, group: "Tracking" },
  { href: "/checkpoints", label: "Checkpoints", icon: GitBranch, group: "Tracking" },
  { href: "/hardware", label: "Hardware", icon: Cpu, group: "System" },
  { href: "/logs", label: "Logs", icon: ScrollText, group: "System" },
  { href: "/settings", label: "Settings", icon: Settings, group: "System" },
];

const GROUPS = ["Workspace", "Inference", "Evaluation", "Tracking", "System"];

export function Sidebar() {
  const pathname = usePathname();
  const collapsed = useWorkspace((state) => state.sidebarCollapsed);
  const toggle = useWorkspace((state) => state.toggleSidebar);
  const { data: activeJobs } = useActiveJobs(15_000);

  const runningCount = (activeJobs ?? []).filter((job) => job.status === "running").length;

  return (
    <aside
      className={cn(
        "flex h-full shrink-0 flex-col border-r border-border bg-panel transition-[width] duration-200",
        collapsed ? "w-[60px]" : "w-[216px]",
      )}
    >
      <div className="flex h-12 items-center gap-2.5 border-b border-border px-3">
        <button
          onClick={() => toggle()}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded border border-border bg-elevated text-accent transition-colors hover:border-border-strong"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <Signal className="h-3.5 w-3.5" />
        </button>
        {!collapsed ? (
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold tracking-tight text-ink">Foxtrot</div>
            <div className="mono truncate text-2xs text-ink-muted">llm platform</div>
          </div>
        ) : null}
      </div>

      <nav className="scrollbar-thin flex-1 overflow-y-auto px-2 py-3">
        {GROUPS.map((group) => {
          const items = NAV.filter((item) => item.group === group);
          return (
            <div key={group} className="mb-4">
              {!collapsed ? (
                <div className="px-2 pb-1.5 text-2xs font-medium uppercase tracking-[0.14em] text-ink-muted/70">
                  {group}
                </div>
              ) : null}
              <ul className="space-y-0.5">
                {items.map((item) => {
                  const active =
                    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
                  const Icon = item.icon;
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        title={collapsed ? item.label : undefined}
                        className={cn(
                          "group flex items-center gap-2.5 rounded-md px-2 py-1.5 text-xs transition-colors",
                          active
                            ? "bg-accent/10 text-ink"
                            : "text-ink-muted hover:bg-elevated hover:text-ink-secondary",
                          collapsed && "justify-center px-0",
                        )}
                      >
                        <Icon
                          className={cn("h-4 w-4 shrink-0", active ? "text-accent" : "text-current")}
                        />
                        {!collapsed ? <span className="truncate">{item.label}</span> : null}
                        {!collapsed && item.href === "/training" && runningCount > 0 ? (
                          <span className="mono ml-auto rounded bg-accent/15 px-1.5 text-2xs text-accent">
                            {runningCount}
                          </span>
                        ) : null}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
