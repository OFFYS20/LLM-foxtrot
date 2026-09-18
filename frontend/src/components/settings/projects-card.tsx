"use client";

/**
 * Project management.
 *
 * A project is the backend record that training jobs and experiments are filed
 * under. Which project is active is remembered per browser, but the projects
 * themselves live in the API — renaming one here renames it for everyone.
 */

import { FolderPlus, Trash2 } from "lucide-react";
import * as React from "react";

import { MetricRow } from "@/components/common/stat-tile";
import { WarningNote } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useProject, useProjectMutations, useProjects } from "@/lib/hooks/queries";
import { useWorkspace } from "@/lib/store";

const NONE = "__none__";

export function ProjectsCard() {
  const { data: projects, isLoading } = useProjects({ limit: 100 });
  const activeProjectId = useWorkspace((state) => state.activeProjectId);
  const setActiveProject = useWorkspace((state) => state.setActiveProject);
  const { data: detail } = useProject(activeProjectId);
  const { create, update, remove } = useProjectMutations();

  const [newName, setNewName] = React.useState("");
  const [renameTo, setRenameTo] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);

  const items = projects?.items ?? [];

  // The stored id can point at a project someone else has since deleted.
  const missing =
    Boolean(activeProjectId) && !isLoading && !items.some((p) => p.id === activeProjectId);

  React.useEffect(() => {
    setRenameTo(detail?.name ?? "");
  }, [detail?.name]);

  const act = async (action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  return (
    <Card>
      <CardContent className="space-y-4 py-3">
        {missing ? (
          <WarningNote>The selected project no longer exists. Pick another one.</WarningNote>
        ) : null}

        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[16rem] flex-1 space-y-1.5">
            <Label>Active project</Label>
            <Select
              value={activeProjectId ?? NONE}
              onValueChange={(value) =>
                setActiveProject(
                  value === NONE ? null : value,
                  items.find((p) => p.id === value)?.name,
                )
              }
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder={isLoading ? "Loading…" : "No project"} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={NONE}>No project</SelectItem>
                {items.map((project) => (
                  <SelectItem key={project.id} value={project.id}>
                    {project.name}
                    {project.is_demo ? " (demo)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-2xs text-ink-muted">
              Shown in the top bar. The selection is per browser; the projects are shared.
            </p>
          </div>

          <div className="min-w-[14rem] flex-1 space-y-1.5">
            <Label htmlFor="new-project">New project</Label>
            <div className="flex gap-2">
              <Input
                id="new-project"
                value={newName}
                placeholder="Instruction tuning"
                onChange={(event) => setNewName(event.target.value)}
                className="h-8 text-xs"
              />
              <Button
                size="sm"
                disabled={!newName.trim() || create.isPending}
                onClick={() =>
                  act(async () => {
                    const project = await create.mutateAsync({ name: newName.trim() });
                    setActiveProject(project.id, project.name);
                    setNewName("");
                  })
                }
              >
                <FolderPlus className="h-3.5 w-3.5" />
                Create
              </Button>
            </div>
          </div>
        </div>

        {detail ? (
          <div className="space-y-3 border-t border-hairline pt-3">
            <div className="flex flex-wrap items-end gap-3">
              <div className="min-w-[16rem] flex-1 space-y-1.5">
                <Label htmlFor="rename-project">Name</Label>
                <div className="flex gap-2">
                  <Input
                    id="rename-project"
                    value={renameTo}
                    onChange={(event) => setRenameTo(event.target.value)}
                    className="h-8 text-xs"
                  />
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={
                      !renameTo.trim() || renameTo.trim() === detail.name || update.isPending
                    }
                    onClick={() =>
                      act(async () => {
                        const updated = await update.mutateAsync({
                          id: detail.id,
                          payload: { name: renameTo.trim() },
                        });
                        setActiveProject(updated.id, updated.name);
                      })
                    }
                  >
                    Rename
                  </Button>
                </div>
              </div>
              <Button
                size="sm"
                variant="secondary"
                disabled={remove.isPending || detail.running_job_count > 0}
                title={
                  detail.running_job_count > 0
                    ? "Stop the running jobs in this project first"
                    : "Delete this project. Its runs are kept and detached."
                }
                onClick={() =>
                  act(async () => {
                    await remove.mutateAsync(detail.id);
                    setActiveProject(null);
                  })
                }
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </Button>
            </div>

            <div className="grid gap-x-6 sm:grid-cols-2">
              <MetricRow label="Base model" value={detail.base_model_name ?? "—"} />
              <MetricRow label="Default dataset" value={detail.default_dataset_name ?? "—"} />
              <MetricRow label="Experiments" value={detail.experiment_count} />
              <MetricRow label="Training runs" value={detail.training_job_count} />
              <MetricRow label="Running now" value={detail.running_job_count} />
              <MetricRow label="Checkpoints" value={detail.checkpoint_count} />
            </div>

            {detail.is_demo ? <Badge variant="muted">demo project</Badge> : null}
            <p className="text-2xs text-ink-muted">
              Deleting a project keeps its training runs and experiments — they are detached, not
              removed.
            </p>
          </div>
        ) : null}

        {error ? <WarningNote>{error}</WarningNote> : null}
      </CardContent>
    </Card>
  );
}
