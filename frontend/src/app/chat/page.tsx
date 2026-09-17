"use client";

import {
  ArrowUp,
  Check,
  MessageSquare,
  Pencil,
  Plus,
  RefreshCw,
  Square,
  Trash2,
} from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";

import { ProvenanceBadge } from "@/components/common/badges";
import { CopyButton } from "@/components/common/copy-button";
import { MetricRow } from "@/components/common/stat-tile";
import { EmptyState, ErrorState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { api } from "@/lib/api";
import { useChatMutations, useConversation, useConversations, useModels } from "@/lib/hooks/queries";
import { useWorkspace } from "@/lib/store";
import { formatFloat, formatNumber, formatRelative } from "@/lib/format";
import type { ChatMessage, GenerationUsage, Message } from "@/lib/types";
import { cn } from "@/lib/utils";

interface PendingState {
  content: string;
  usage: GenerationUsage | null;
  error: string | null;
}

function ChatPageContent() {
  const searchParams = useSearchParams();
  const presetModel = searchParams.get("model");

  const { data: models } = useModels({ limit: 200 });
  const { data: conversations, refetch: refetchConversations } = useConversations();
  const activeConversationId = useWorkspace((state) => state.activeConversationId);
  const setActiveConversation = useWorkspace((state) => state.setActiveConversation);
  const activeModelId = useWorkspace((state) => state.activeModelId);
  const setActiveModel = useWorkspace((state) => state.setActiveModel);
  const params = useWorkspace((state) => state.chatParams);
  const setParams = useWorkspace((state) => state.setChatParams);
  const systemPrompt = useWorkspace((state) => state.chatSystemPrompt);
  const setSystemPrompt = useWorkspace((state) => state.setChatSystemPrompt);

  const { data: conversation, refetch: refetchConversation } = useConversation(activeConversationId);
  const { create, remove, editMessage, deleteMessage } = useChatMutations();

  const [input, setInput] = React.useState("");
  const [pending, setPending] = React.useState<PendingState | null>(null);
  const [lastUsage, setLastUsage] = React.useState<GenerationUsage | null>(null);
  const [editing, setEditing] = React.useState<{ id: string; content: string } | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  const modelId = presetModel ?? conversation?.model_id ?? activeModelId ?? models?.items[0]?.id ?? null;
  const model = models?.items.find((item) => item.id === modelId);

  React.useEffect(() => {
    if (presetModel) setActiveModel(presetModel);
  }, [presetModel, setActiveModel]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [conversation?.messages.length, pending?.content]);

  const ensureConversation = React.useCallback(async (): Promise<string | null> => {
    if (activeConversationId) return activeConversationId;
    if (!modelId) return null;
    const created = await create.mutateAsync({
      model_id: modelId,
      system_prompt: systemPrompt,
      params,
    });
    setActiveConversation(created.id);
    return created.id;
  }, [activeConversationId, create, modelId, params, setActiveConversation, systemPrompt]);

  const send = React.useCallback(
    async (overrideMessages?: ChatMessage[], skipUserPersist = false) => {
      if (!modelId) return;
      const text = input.trim();
      if (!overrideMessages && !text) return;

      const conversationId = await ensureConversation();
      const history: ChatMessage[] =
        overrideMessages ??
        [
          ...(conversation?.messages ?? []).map((message) => ({
            role: message.role,
            content: message.content,
          })),
          { role: "user" as const, content: text },
        ];

      setInput("");
      setPending({ content: "", usage: null, error: null });

      const controller = new AbortController();
      abortRef.current = controller;

      await api.chat.stream(
        {
          model_id: modelId,
          messages: history,
          system_prompt: systemPrompt,
          params,
          conversation_id: conversationId,
          persist: !skipUserPersist,
        },
        {
          signal: controller.signal,
          onToken: (token) =>
            setPending((current) => (current ? { ...current, content: current.content + token } : current)),
          onUsage: (usage) => {
            setLastUsage(usage);
            setPending((current) => (current ? { ...current, usage } : current));
          },
          onError: (message) =>
            setPending((current) => (current ? { ...current, error: message } : current)),
        },
      );

      abortRef.current = null;
      setPending(null);
      await refetchConversation();
      await refetchConversations();
    },
    [
      conversation?.messages,
      ensureConversation,
      input,
      modelId,
      params,
      refetchConversation,
      refetchConversations,
      systemPrompt,
    ],
  );

  const stopGeneration = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(null);
  };

  const regenerate = async () => {
    const messages = conversation?.messages ?? [];
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    if (lastAssistant) {
      await deleteMessage.mutateAsync({ id: lastAssistant.id, andAfter: true });
      await refetchConversation();
    }
    const history = messages
      .filter((message) => message.id !== lastAssistant?.id)
      .map((message) => ({ role: message.role, content: message.content }));
    await send(history, true);
  };

  const messages = conversation?.messages ?? [];

  return (
    <Workspace
      inspectorTitle="Inference settings"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Model</CardTitle>
              {model?.is_demo ? <ProvenanceBadge provenance="simulated" compact /> : null}
            </CardHeader>
            <CardContent className="space-y-2 py-3">
              <Select
                value={modelId ?? undefined}
                onValueChange={(value) => {
                  setActiveModel(value);
                  if (activeConversationId) {
                    void api.chat.updateConversation(activeConversationId, { model_id: value });
                  }
                }}
              >
                <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Select a model" /></SelectTrigger>
                <SelectContent>
                  {(models?.items ?? []).map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.display_name ?? item.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {model ? (
                <div className="divide-y divide-border">
                  <MetricRow label="Context" value={formatNumber(model.context_length)} />
                  <MetricRow label="Precision" value={model.precision.toUpperCase()} />
                  <MetricRow label="Status" value={model.status} />
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>System prompt</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <Textarea
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                className="h-24 text-xs"
                placeholder="Set the assistant's behaviour…"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Sampling</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 py-3">
              <ParamSlider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.05}
                onChange={(value) => setParams({ temperature: value })}
              />
              <ParamSlider
                label="Top P"
                value={params.top_p}
                min={0}
                max={1}
                step={0.01}
                onChange={(value) => setParams({ top_p: value })}
              />
              <ParamSlider
                label="Top K"
                value={params.top_k}
                min={0}
                max={200}
                step={1}
                onChange={(value) => setParams({ top_k: value })}
              />
              <ParamSlider
                label="Max tokens"
                value={params.max_tokens}
                min={16}
                max={8192}
                step={16}
                onChange={(value) => setParams({ max_tokens: value })}
              />
              <ParamSlider
                label="Repetition penalty"
                value={params.repetition_penalty}
                min={0.5}
                max={2}
                step={0.01}
                onChange={(value) => setParams({ repetition_penalty: value })}
              />
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input
                  type="number"
                  placeholder="random"
                  value={params.seed ?? ""}
                  onChange={(event) =>
                    setParams({ seed: event.target.value === "" ? null : Number(event.target.value) })
                  }
                  className="h-8 text-xs"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Debug</CardTitle>
              {lastUsage ? <ProvenanceBadge provenance={lastUsage.provenance} compact /> : null}
            </CardHeader>
            <CardContent className="py-3">
              {lastUsage ? (
                <div className="divide-y divide-border">
                  <MetricRow label="Prompt tokens" value={formatNumber(lastUsage.prompt_tokens)} />
                  <MetricRow label="Completion tokens" value={formatNumber(lastUsage.completion_tokens)} />
                  <MetricRow label="Total tokens" value={formatNumber(lastUsage.total_tokens)} />
                  <MetricRow label="Generation speed" value={`${formatFloat(lastUsage.tokens_per_sec, 1)} tok/s`} />
                  <MetricRow label="Time to first token" value={`${formatFloat(lastUsage.time_to_first_token_ms, 0)} ms`} />
                  <MetricRow label="Total latency" value={`${formatFloat(lastUsage.latency_ms, 0)} ms`} />
                  <MetricRow label="Finish reason" value={lastUsage.finish_reason} />
                  <MetricRow label="Engine" value={lastUsage.engine} />
                </div>
              ) : (
                <p className="text-xs text-ink-muted">Send a message to populate generation stats.</p>
              )}
            </CardContent>
          </Card>
        </>
      }
    >
      <div className="flex h-[calc(100vh-7rem)] gap-4">
        <div className="hidden w-[220px] shrink-0 flex-col gap-2 lg:flex">
          <Button
            size="sm"
            onClick={async () => {
              const created = await create.mutateAsync({
                model_id: modelId ?? undefined,
                system_prompt: systemPrompt,
                params,
              });
              setActiveConversation(created.id);
            }}
          >
            <Plus className="h-3.5 w-3.5" />
            New conversation
          </Button>
          <div className="scrollbar-thin flex-1 space-y-1 overflow-y-auto">
            {(conversations?.items ?? []).map((item) => (
              <div
                key={item.id}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter") setActiveConversation(item.id);
                }}
                onClick={() => setActiveConversation(item.id)}
                className={cn(
                  "group w-full cursor-pointer rounded-md border px-2.5 py-2 text-left transition-colors",
                  item.id === activeConversationId
                    ? "border-accent/50 bg-accent/10"
                    : "border-border bg-panel hover:border-border-strong",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="line-clamp-2 text-xs text-ink">{item.title}</span>
                  <Button
                    size="xs"
                    variant="ghost"
                    className="h-5 w-5 p-0 opacity-0 transition-opacity group-hover:opacity-100"
                    onClick={(event) => {
                      event.stopPropagation();
                      remove.mutate(item.id);
                      if (item.id === activeConversationId) setActiveConversation(null);
                    }}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
                <div className="mono mt-1 text-2xs text-ink-muted">{formatRelative(item.updated_at)}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-border bg-panel">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <div className="min-w-0">
              <div className="truncate text-xs font-medium text-ink">
                {conversation?.title ?? "New conversation"}
              </div>
              <div className="mono text-2xs text-ink-muted">
                {model ? (model.display_name ?? model.name) : "no model selected"}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="muted">T {formatFloat(params.temperature, 2)}</Badge>
              <Badge variant="muted">max {params.max_tokens}</Badge>
            </div>
          </div>

          <div ref={scrollRef} className="scrollbar-thin flex-1 space-y-4 overflow-y-auto px-4 py-4">
            {messages.length === 0 && !pending ? (
              <EmptyState
                icon={<MessageSquare className="h-5 w-5" />}
                title="Start a conversation"
                description={
                  model?.is_demo
                    ? "This model is a demo entry — responses come from the simulated engine and are labelled as such."
                    : "Messages are streamed token by token from the inference engine."
                }
                className="border-0 bg-transparent"
              />
            ) : null}

            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                editing={editing?.id === message.id}
                editValue={editing?.content ?? ""}
                onEditChange={(value) => setEditing({ id: message.id, content: value })}
                onEditStart={() => setEditing({ id: message.id, content: message.content })}
                onEditCancel={() => setEditing(null)}
                onEditSave={async () => {
                  if (!editing) return;
                  await editMessage.mutateAsync({ id: editing.id, content: editing.content });
                  setEditing(null);
                  await refetchConversation();
                }}
                onDelete={async () => {
                  await deleteMessage.mutateAsync({ id: message.id });
                  await refetchConversation();
                }}
              />
            ))}

            {pending ? (
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Badge variant="accent">assistant</Badge>
                  <span className="mono text-2xs text-ink-muted">streaming…</span>
                </div>
                <div className="whitespace-pre-wrap rounded-lg border border-border bg-elevated/40 px-3 py-2.5 text-sm text-ink-secondary">
                  {pending.content}
                  <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse-dot bg-accent align-middle" />
                </div>
                {pending.error ? <ErrorState error={pending.error} /> : null}
              </div>
            ) : null}
          </div>

          <div className="border-t border-border p-3">
            <div className="flex items-end gap-2">
              <Textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void send();
                  }
                }}
                placeholder={modelId ? "Send a message… (Enter to send, Shift+Enter for newline)" : "Select a model first"}
                disabled={!modelId || Boolean(pending)}
                className="min-h-[44px] flex-1 text-sm"
              />
              {pending ? (
                <Button variant="danger" size="sm" onClick={stopGeneration}>
                  <Square className="h-3.5 w-3.5" />
                  Stop
                </Button>
              ) : (
                <>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => void regenerate()}
                    disabled={messages.length === 0}
                    title="Regenerate the last response"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </Button>
                  <Button size="sm" onClick={() => void send()} disabled={!modelId || !input.trim()}>
                    <ArrowUp className="h-3.5 w-3.5" />
                    Send
                  </Button>
                </>
              )}
            </div>
            <div className="mono mt-1.5 flex items-center gap-3 text-2xs text-ink-muted">
              <span>~{Math.max(0, Math.round(input.length / 4))} prompt tokens</span>
              {lastUsage ? <span>{formatFloat(lastUsage.tokens_per_sec, 1)} tok/s last</span> : null}
              {lastUsage ? <span>TTFT {formatFloat(lastUsage.time_to_first_token_ms, 0)} ms</span> : null}
            </div>
          </div>
        </div>
      </div>
    </Workspace>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <span className="mono text-2xs text-ink-secondary">{value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([next]) => onChange(next)} />
    </div>
  );
}

function MessageBubble({
  message,
  editing,
  editValue,
  onEditChange,
  onEditStart,
  onEditCancel,
  onEditSave,
  onDelete,
}: {
  message: Message;
  editing: boolean;
  editValue: string;
  onEditChange: (value: string) => void;
  onEditStart: () => void;
  onEditCancel: () => void;
  onEditSave: () => void;
  onDelete: () => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className="group space-y-1.5">
      <div className="flex items-center gap-2">
        <Badge variant={isUser ? "outline" : "accent"}>{message.role}</Badge>
        {!isUser ? <ProvenanceBadge provenance={message.provenance} compact /> : null}
        {!isUser && message.tokens_per_sec ? (
          <span className="mono text-2xs text-ink-muted">
            {formatFloat(message.tokens_per_sec, 1)} tok/s · {formatNumber(message.completion_tokens)} tokens ·{" "}
            {formatFloat(message.latency_ms, 0)} ms
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
          <CopyButton value={message.content} size="xs" />
          <Button size="xs" variant="ghost" onClick={onEditStart} title="Edit">
            <Pencil className="h-3 w-3" />
          </Button>
          <Button size="xs" variant="ghost" onClick={onDelete} title="Delete">
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {editing ? (
        <div className="space-y-2">
          <Textarea
            value={editValue}
            onChange={(event) => onEditChange(event.target.value)}
            className="text-sm"
          />
          <div className="flex gap-2">
            <Button size="xs" onClick={onEditSave}>
              <Check className="h-3 w-3" />
              Save
            </Button>
            <Button size="xs" variant="ghost" onClick={onEditCancel}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <div
          className={cn(
            "whitespace-pre-wrap rounded-lg border px-3 py-2.5 text-sm",
            isUser ? "border-border bg-elevated/60 text-ink" : "border-border bg-panel text-ink-secondary",
          )}
        >
          {message.content}
        </div>
      )}
    </div>
  );
}

/**
 * `useSearchParams` opts this route into client-side rendering, so the page body
 * lives behind a Suspense boundary (required by the Next.js app router).
 */
export default function ChatPage() {
  return (
    <React.Suspense
      fallback={
        <div className="p-6 text-xs text-ink-muted">Loading chat…</div>
      }
    >
      <ChatPageContent />
    </React.Suspense>
  );
}
