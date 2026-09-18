"use client";

import { Check, Copy } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function CopyButton({
  value,
  label,
  className,
  size = "icon",
}: {
  value: string;
  label?: string;
  className?: string;
  size?: "icon" | "xs" | "sm";
}) {
  const [copied, setCopied] = React.useState(false);

  const copy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable (insecure context) */
    }
  }, [value]);

  return (
    <Button variant="ghost" size={size} onClick={copy} className={cn(className)} title="Copy">
      {copied ? <Check className="h-3.5 w-3.5 text-good" /> : <Copy className="h-3.5 w-3.5" />}
      {label ? <span className="ml-1">{copied ? "Copied" : label}</span> : null}
    </Button>
  );
}
