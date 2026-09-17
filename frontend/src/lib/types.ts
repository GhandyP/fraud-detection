/**
 * Hand-maintained against src/fraud_detection/api/schemas.py.
 * Keep these API contract types in sync with the FastAPI schemas.
 */

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ModelInfoResponse {
  model_name: string | null;
  model_version: string | null;
  model_identity: Record<string, string> | null;
  feature_names: string[];
  threshold: number;
  artifact_schema_version: string | null;
  split_strategy: string | null;
  dataset_sha256: string | null;
  created_at_utc: string | null;
}

export interface PredictionRequest {
  features: Record<string, number>;
}

export interface PredictionResponse {
  is_fraud: boolean;
  fraud_probability: number;
  threshold: number;
  model_version: string;
}

export interface ApiErrorBody {
  detail: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string | null;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, requestId: string | null = null) {
    super(`API request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
    this.detail = detail;
  }
}
