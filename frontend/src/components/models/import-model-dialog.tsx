"use client";

import { Download, Plus } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WarningNote } from "@/components/common/states";
import { PRECISIONS } from "@/lib/constants";
import { useImportModel } from "@/lib/hooks/queries";
import type { ModelFormat, Precision } from "@/lib/types";

const FORMATS: { value: ModelFormat; label: string }[] = [
  { value: "safetensors", label: "SafeTensors" },
  { value: "pytorch", label: "PyTorch (.bin / .pt)" },
  { value: "gguf", label: "GGUF" },
  { value: "hf_repo", label: "Hugging Face repository" },
];

export function ImportModelDialog({ trigger }: { trigger?: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const [source, setSource] = React.useState<"huggingface" | "local" | "custom_path">("huggingface");
  const [repoId, setRepoId] = React.useState("");
  const [revision, setRevision] = React.useState("");
  const [path, setPath] = React.useState("");
  const [name, setName] = React.useState("");
  const [format, setFormat] = React.useState<ModelFormat>("safetensors");
  const [precision, setPrecision] = React.useState<Precision>("bf16");
  const [contextLength, setContextLength] = React.useState("");

  const importModel = useImportModel();

  const submit = () => {
    importModel.mutate(
      {
        source,
        name: name || undefined,
        repo_id: source === "huggingface" ? repoId.trim() : undefined,
        revision: revision || undefined,
        path: source !== "huggingface" ? path.trim() : undefined,
        format,
        precision,
        context_length: contextLength ? Number(contextLength) : null,
      },
      {
        onSuccess: () => {
          setOpen(false);
          setRepoId("");
          setPath("");
          setName("");
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger ?? (
          <Button size="sm">
            <Plus className="h-3.5 w-3.5" />
            Import model
          </Button>
        )}
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Import a model</DialogTitle>
          <DialogDescription>
            Registration records metadata only — weights are fetched lazily by the inference engine,
            so this never blocks on a multi-gigabyte download.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 p-4">
          <Tabs value={source} onValueChange={(value) => setSource(value as typeof source)}>
            <TabsList>
              <TabsTrigger value="huggingface">Hugging Face</TabsTrigger>
              <TabsTrigger value="local">Local directory</TabsTrigger>
              <TabsTrigger value="custom_path">Custom path</TabsTrigger>
            </TabsList>

            <TabsContent value="huggingface" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="repo">Repository id</Label>
                <Input
                  id="repo"
                  placeholder="mistralai/Mistral-7B-Instruct-v0.3"
                  value={repoId}
                  onChange={(event) => setRepoId(event.target.value)}
                  className="mono"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="revision">Revision (optional)</Label>
                <Input
                  id="revision"
                  placeholder="main"
                  value={revision}
                  onChange={(event) => setRevision(event.target.value)}
                  className="mono"
                />
              </div>
            </TabsContent>

            <TabsContent value="local" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="path">Path inside the data root</Label>
                <Input
                  id="path"
                  placeholder="models/my-llama-7b"
                  value={path}
                  onChange={(event) => setPath(event.target.value)}
                  className="mono"
                />
              </div>
              <WarningNote>
                Paths are resolved inside <code className="mono">FOXTROT_DATA_DIR/models</code>. Anything
                that escapes that root is rejected by the API.
              </WarningNote>
            </TabsContent>

            <TabsContent value="custom_path" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="custom">Registered path</Label>
                <Input
                  id="custom"
                  placeholder="models/gguf/qwen2-1_5b-q4_k_m.gguf"
                  value={path}
                  onChange={(event) => setPath(event.target.value)}
                  className="mono"
                />
              </div>
            </TabsContent>
          </Tabs>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="name">Display name</Label>
              <Input
                id="name"
                placeholder="auto"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="ctx">Context length</Label>
              <Input
                id="ctx"
                type="number"
                placeholder="auto"
                value={contextLength}
                onChange={(event) => setContextLength(event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Format</Label>
              <Select value={format} onValueChange={(value) => setFormat(value as ModelFormat)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FORMATS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Precision</Label>
              <Select value={precision} onValueChange={(value) => setPrecision(value as Precision)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRECISIONS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {importModel.error ? (
            <p className="mono text-xs text-critical">{(importModel.error as Error).message}</p>
          ) : null}
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={submit}
            disabled={importModel.isPending || (source === "huggingface" ? !repoId.trim() : !path.trim())}
          >
            <Download className="h-3.5 w-3.5" />
            {importModel.isPending ? "Registering…" : "Register model"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
