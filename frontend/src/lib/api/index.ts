/**
 * Provider selection.
 *
 * `NEXT_PUBLIC_DATA_PROVIDER=mock` runs the whole UI on client-side fixtures;
 * anything else uses the FastAPI backend. Components import `api` only.
 */

import { httpProvider } from "@/lib/api/http";
import { mockProvider } from "@/lib/api/mock";
import type { DataProvider } from "@/lib/api/provider";

const configured = process.env.NEXT_PUBLIC_DATA_PROVIDER ?? "http";

export const api: DataProvider = configured === "mock" ? mockProvider : httpProvider;
export const isMockProvider = api.kind === "mock";

export type { DataProvider } from "@/lib/api/provider";
export { ApiError, API_BASE_URL } from "@/lib/api/http";
