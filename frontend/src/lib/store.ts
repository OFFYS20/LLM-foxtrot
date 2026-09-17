"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

import { DEFAULT_SAMPLING } from "@/lib/constants";
import type { SamplingParams } from "@/lib/types";

interface WorkspaceState {
  /** The model shown in the top bar and pre-selected across Chat/Training/Benchmarks. */
  activeModelId: string | null;
  activeProjectName: string;
  inspectorOpen: boolean;
  sidebarCollapsed: boolean;
  chatParams: SamplingParams;
  chatSystemPrompt: string;
  activeConversationId: string | null;
  playgroundModelIds: string[];

  setActiveModel: (id: string | null) => void;
  setActiveProject: (name: string) => void;
  toggleInspector: (open?: boolean) => void;
  toggleSidebar: (collapsed?: boolean) => void;
  setChatParams: (params: Partial<SamplingParams>) => void;
  setChatSystemPrompt: (prompt: string) => void;
  setActiveConversation: (id: string | null) => void;
  setPlaygroundModels: (ids: string[]) => void;
}

export const useWorkspace = create<WorkspaceState>()(
  persist(
    (set) => ({
      activeModelId: null,
      activeProjectName: "Foxtrot workspace",
      inspectorOpen: true,
      sidebarCollapsed: false,
      chatParams: { ...DEFAULT_SAMPLING },
      chatSystemPrompt: "You are Foxtrot, a precise assistant for ML engineers.",
      activeConversationId: null,
      playgroundModelIds: [],

      setActiveModel: (id) => set({ activeModelId: id }),
      setActiveProject: (name) => set({ activeProjectName: name }),
      toggleInspector: (open) => set((state) => ({ inspectorOpen: open ?? !state.inspectorOpen })),
      toggleSidebar: (collapsed) =>
        set((state) => ({ sidebarCollapsed: collapsed ?? !state.sidebarCollapsed })),
      setChatParams: (params) => set((state) => ({ chatParams: { ...state.chatParams, ...params } })),
      setChatSystemPrompt: (prompt) => set({ chatSystemPrompt: prompt }),
      setActiveConversation: (id) => set({ activeConversationId: id }),
      setPlaygroundModels: (ids) => set({ playgroundModelIds: ids.slice(0, 4) }),
    }),
    {
      name: "foxtrot-workspace",
      partialize: (state) => ({
        activeModelId: state.activeModelId,
        activeProjectName: state.activeProjectName,
        inspectorOpen: state.inspectorOpen,
        sidebarCollapsed: state.sidebarCollapsed,
        chatParams: state.chatParams,
        chatSystemPrompt: state.chatSystemPrompt,
        activeConversationId: state.activeConversationId,
        playgroundModelIds: state.playgroundModelIds,
      }),
    },
  ),
);
