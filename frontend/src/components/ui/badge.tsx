import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-2xs font-medium uppercase tracking-wide",
  {
    variants: {
      variant: {
        default: "border-border bg-elevated text-ink-secondary",
        accent: "border-accent/40 bg-accent/10 text-accent",
        good: "border-good/40 bg-good/10 text-good",
        warning: "border-warning/40 bg-warning/10 text-warning",
        serious: "border-serious/40 bg-serious/10 text-serious",
        critical: "border-critical/40 bg-critical/10 text-critical",
        muted: "border-border bg-transparent text-ink-muted",
        outline: "border-border-strong bg-transparent text-ink-secondary",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
