import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "./types";
import { ApiError } from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const API_BASE_URL = (
  import.meta.env.PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

export class ApiUnreachableError extends Error {
  constructor(cause?: unknown) {
    super("The API could not be reached. Check that the service is running.");
    this.name = "ApiUnreachableError";
    this.cause = cause;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    throw new ApiUnreachableError(error);
  }

  if (!response.ok) {
    let detail: unknown = `Request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = (body as { detail: unknown }).detail;
      }
    } catch {
      // Preserve the status-based detail when the error body is not JSON.
    }
    throw new ApiError(
      response.status,
      detail,
      response.headers.get("X-Request-ID"),
    );
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>("/model-info");
}

export function predict(
  features: Record<string, number>,
): Promise<PredictionResponse> {
  const body: PredictionRequest = { features };
  return request<PredictionResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
