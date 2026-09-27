// Types mirror backend/app/models.py. Only the fields the UI uses are declared.

export interface Health {
  llm_mode: "generative" | "extractive";
  llm_model: string | null;
  retrieval: string;
  retrieval_is_semantic: boolean;
  min_evidence_score: number;
  max_query_rewrites: number;
}

export interface PolicySummary {
  doc_id: string;
  doc_name: string;
  page_count: number;
  chunk_count: number;
  empty_pages: number[];
  sections: string[];
}

export interface Citation {
  ref: string;
  chunk_id: string;
  page: number;
  section: string | null;
  quote: string;
}

export interface Evidence {
  chunk_id: string;
  page: number;
  section: string | null;
  text: string;
  score: number;
}

export interface RetrievalAttempt {
  query: string;
  strategy: "original" | "llm_rewrite";
  best_score: number | null;
  evidence_found: boolean;
  note: string | null;
}

export type AnswerStatus =
  | "answered"
  | "evidence_only"
  | "insufficient_evidence"
  | "manual_review"
  | "llm_error";

export interface AskResponse {
  status: AnswerStatus;
  mode: "generative" | "extractive";
  answer: string | null;
  citations: Citation[];
  evidence: Evidence[];
  retrieval_attempts: RetrievalAttempt[];
  verification_issues: string[];
  message: string;
}

export interface Treatment {
  treatment_code: string;
  treatment_name: string;
  category: string;
  typical_length_of_stay_days: number;
}

export interface TreatmentInput {
  treatment_code: string;
  city_tier: "tier1" | "tier2" | "tier3";
  room_type: "general_ward" | "semi_private" | "private";
  length_of_stay_days?: number;
}

export interface CostEstimate {
  treatment_name: string;
  category: string;
  length_of_stay_days: number;
  currency: string;
  low: number;
  high: number;
  estimate: number;
  room_rent_per_day: number;
  components: { name: string; low: number; high: number }[];
  is_synthetic: boolean;
  disclaimer: string;
}

export interface PolicyTerms {
  sum_insured?: number;
  deductible?: number;
  copay_pct?: number;
  room_rent_limit_per_day?: number;
  sub_limits?: Record<string, number>;
}

export interface CalculationStep {
  step: number;
  rule: string;
  description: string;
  amount_before: number;
  amount_after: number;
  patient_share_added: number;
}

export interface CoverageResult {
  status: "complete" | "incomplete_terms" | "not_covered";
  total_cost: number;
  covered_amount: number;
  out_of_pocket: number;
  steps: CalculationStep[];
  missing_terms: string[];
  assumptions: string[];
  disclaimer: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    const text = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
  return body as T;
}

const json = (data: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

export const api = {
  health: () => request<Health>("/health"),
  policies: () => request<PolicySummary[]>("/policies"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<PolicySummary>("/policies", { method: "POST", body: form });
  },
  ask: (docId: string, question: string) =>
    request<AskResponse>(`/policies/${encodeURIComponent(docId)}/ask`, json({ question })),
  treatments: () => request<Treatment[]>("/treatments"),
  estimate: (input: TreatmentInput) => request<CostEstimate>("/estimate", json(input)),
  coverage: (treatment: TreatmentInput, terms: PolicyTerms) =>
    request<CoverageResult>("/coverage", json({ treatment, terms })),
};
