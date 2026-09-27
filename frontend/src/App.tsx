import { useEffect, useState, type FormEvent } from "react";
import {
  api,
  type AskResponse,
  type CostEstimate,
  type CoverageResult,
  type Health,
  type PolicySummary,
  type PolicyTerms,
  type Treatment,
  type TreatmentInput,
} from "./api";

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const fmt = (n: number) => money.format(n);
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

const STATUS_LABEL: Record<AskResponse["status"], string> = {
  answered: "Answered from policy text",
  evidence_only: "Evidence only (no LLM configured)",
  insufficient_evidence: "Insufficient evidence",
  manual_review: "Needs manual review",
  llm_error: "LLM unavailable - evidence shown",
};

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [policies, setPolicies] = useState<PolicySummary[]>([]);
  const [docId, setDocId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(`Backend not reachable (${errorText(e)})`));
    api.policies().then((list) => {
      setPolicies(list);
      if (list.length) setDocId((current) => current || list[list.length - 1].doc_id);
    }).catch(() => undefined);
  }, []);

  const onUploaded = (summary: PolicySummary) => {
    setPolicies((list) => [...list.filter((p) => p.doc_id !== summary.doc_id), summary]);
    setDocId(summary.doc_id);
  };

  return (
    <main>
      <header>
        <h1>Policy-to-Patient</h1>
        <p className="muted">
          Prototype for HackMatrix 5.0 FIN-01. Answers come only from your policy's text; costs are{" "}
          <strong>synthetic demo figures</strong>; coverage is an estimate, not an insurance decision.
        </p>
        {health && (
          <p className="chips">
            <span className="chip">LLM: {health.llm_mode}{health.llm_model ? ` (${health.llm_model})` : ""}</span>
            <span className={`chip ${health.retrieval_is_semantic ? "" : "warn"}`}>
              Retrieval: {health.retrieval}{health.retrieval_is_semantic ? "" : " - lexical, not semantic"}
            </span>
            <span className="chip">Min relevance: {health.min_evidence_score}</span>
          </p>
        )}
        {error && <p className="error" role="alert">{error}</p>}
      </header>

      <PolicyPanel policies={policies} docId={docId} onSelect={setDocId} onUploaded={onUploaded} />
      <AskPanel docId={docId} />
      <CostAndCoverage />
    </main>
  );
}

function PolicyPanel(props: {
  policies: PolicySummary[];
  docId: string;
  onSelect: (id: string) => void;
  onUploaded: (s: PolicySummary) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const selected = props.policies.find((p) => p.doc_id === props.docId);

  async function upload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      props.onUploaded(await api.upload(file));
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="policy-h">
      <h2 id="policy-h">1. Policy</h2>
      <form onSubmit={upload} className="row">
        <label>
          Policy PDF (text-based)
          <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <button type="submit" disabled={!file || busy}>{busy ? "Extracting and indexing…" : "Upload and index"}</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {props.policies.length > 0 && (
        <label>
          Active policy
          <select value={props.docId} onChange={(e) => props.onSelect(e.target.value)}>
            {props.policies.map((p) => (
              <option key={p.doc_id} value={p.doc_id}>{p.doc_name} ({p.doc_id})</option>
            ))}
          </select>
        </label>
      )}
      {selected && (
        <p className="status ok" data-testid="policy-status">
          Indexed <strong>{selected.doc_name}</strong>: {selected.page_count} pages, {selected.chunk_count} chunks,{" "}
          {selected.sections.length} sections
          {selected.empty_pages.length > 0 && <> · pages without text (not OCR'd): {selected.empty_pages.join(", ")}</>}
        </p>
      )}
    </section>
  );
}

function AskPanel({ docId }: { docId: string }) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function ask(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      setResult(await api.ask(docId, question));
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="ask-h">
      <h2 id="ask-h">2. Ask about the policy</h2>
      <form onSubmit={ask} className="row">
        <label className="grow">
          Question
          <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. What is the deductible?" />
        </label>
        <button type="submit" disabled={!docId || question.trim().length < 3 || busy}>{busy ? "Searching…" : "Ask"}</button>
      </form>
      {!docId && <p className="muted">Upload or select a policy first.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {result && <AnswerView result={result} />}
    </section>
  );
}

function AnswerView({ result }: { result: AskResponse }) {
  const rewrites = result.retrieval_attempts.filter((a) => a.strategy === "llm_rewrite");
  return (
    <div data-testid="answer">
      <p className={`status ${result.status}`} data-testid="answer-status">{STATUS_LABEL[result.status]}</p>
      {result.answer && <p className="answer">{result.answer}</p>}
      <p className="muted">{result.message}</p>
      {rewrites.length > 0 && (
        <p className="muted">
          First search found no relevant passage; retried with:{" "}
          {rewrites.map((a) => (a.query ? `"${a.query}"` : a.note)).join(", ")}
        </p>
      )}
      {result.verification_issues.length > 0 && (
        <ul className="error">{result.verification_issues.map((i) => <li key={i}>{i}</li>)}</ul>
      )}
      {result.citations.length > 0 && (
        <>
          <h3>Policy evidence</h3>
          <ol className="citations" data-testid="citations">
            {result.citations.map((c) => (
              <li key={c.ref}>
                <strong>[{c.ref}] Page {c.page}{c.section ? ` · ${c.section}` : ""}</strong>
                <blockquote>{c.quote}</blockquote>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

function CostAndCoverage() {
  const [treatments, setTreatments] = useState<Treatment[]>([]);
  const [input, setInput] = useState<TreatmentInput>({ treatment_code: "", city_tier: "tier1", room_type: "semi_private" });
  const [estimate, setEstimate] = useState<CostEstimate | null>(null);
  const [terms, setTerms] = useState<Record<string, string>>({});
  const [coverage, setCoverage] = useState<CoverageResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.treatments().then((list) => {
      setTreatments(list);
      if (list.length) setInput((i) => ({ ...i, treatment_code: i.treatment_code || list[0].treatment_code }));
    }).catch((e) => setError(errorText(e)));
  }, []);

  const category = treatments.find((t) => t.treatment_code === input.treatment_code)?.category ?? "";
  const num = (key: string) => (terms[key]?.trim() ? Number(terms[key]) : undefined);

  async function runEstimate(e: FormEvent) {
    e.preventDefault();
    setError("");
    setCoverage(null);
    try {
      setEstimate(await api.estimate(input));
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function runCoverage(e: FormEvent) {
    e.preventDefault();
    setError("");
    const subLimit = num("sub_limit");
    const policyTerms: PolicyTerms = {
      sum_insured: num("sum_insured"),
      deductible: num("deductible"),
      copay_pct: num("copay_pct"),
      room_rent_limit_per_day: num("room_rent_limit_per_day"),
      sub_limits: subLimit !== undefined && category ? { [category]: subLimit } : {},
    };
    try {
      setCoverage(await api.coverage(input, policyTerms));
    } catch (err) {
      setError(errorText(err));
    }
  }

  const termField = (key: string, label: string) => (
    <label key={key}>
      {label}
      <input type="number" min="0" step="any" value={terms[key] ?? ""} onChange={(e) => setTerms({ ...terms, [key]: e.target.value })} />
    </label>
  );

  return (
    <>
      <section aria-labelledby="cost-h">
        <h2 id="cost-h">3. Treatment cost estimate</h2>
        <form onSubmit={runEstimate} className="row">
          <label>
            Treatment
            <select value={input.treatment_code} onChange={(e) => setInput({ ...input, treatment_code: e.target.value })}>
              {treatments.map((t) => <option key={t.treatment_code} value={t.treatment_code}>{t.treatment_name}</option>)}
            </select>
          </label>
          <label>
            City
            <select value={input.city_tier} onChange={(e) => setInput({ ...input, city_tier: e.target.value as TreatmentInput["city_tier"] })}>
              <option value="tier1">Tier 1</option>
              <option value="tier2">Tier 2</option>
              <option value="tier3">Tier 3</option>
            </select>
          </label>
          <label>
            Room
            <select value={input.room_type} onChange={(e) => setInput({ ...input, room_type: e.target.value as TreatmentInput["room_type"] })}>
              <option value="general_ward">General ward</option>
              <option value="semi_private">Semi-private</option>
              <option value="private">Private</option>
            </select>
          </label>
          <label>
            Stay (days, optional)
            <input type="number" min="0" max="90" value={input.length_of_stay_days ?? ""}
              onChange={(e) => setInput({ ...input, length_of_stay_days: e.target.value === "" ? undefined : Number(e.target.value) })} />
          </label>
          <button type="submit" disabled={!input.treatment_code}>Estimate cost</button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
        {estimate && (
          <div data-testid="estimate">
            <p className="badge">SYNTHETIC DEMO DATA</p>
            <p>
              <strong>{estimate.treatment_name}</strong>, {estimate.length_of_stay_days} day(s): estimated{" "}
              <strong>{fmt(estimate.estimate)}</strong> (range {fmt(estimate.low)} – {fmt(estimate.high)}),
              room about {fmt(estimate.room_rent_per_day)}/day.
            </p>
            <p className="muted">{estimate.disclaimer}</p>
          </div>
        )}
      </section>

      <section aria-labelledby="cov-h">
        <h2 id="cov-h">4. Coverage and out-of-pocket</h2>
        <p className="muted">
          Enter the terms as written in your policy (use the cited passages above). Blank terms are reported as missing,
          never guessed. The calculation is deterministic; no AI is involved.
        </p>
        <form onSubmit={runCoverage}>
          <div className="grid">
            {termField("sum_insured", "Sum insured (₹)")}
            {termField("deductible", "Deductible (₹ per year)")}
            {termField("copay_pct", "Co-payment (%)")}
            {termField("room_rent_limit_per_day", "Room rent limit (₹/day)")}
            {termField("sub_limit", `Sub-limit for ${category || "this treatment"} (₹)`)}
          </div>
          <button type="submit" disabled={!input.treatment_code}>Calculate coverage</button>
        </form>
        {coverage && <CoverageView result={coverage} />}
      </section>
    </>
  );
}

function CoverageView({ result }: { result: CoverageResult }) {
  return (
    <div data-testid="coverage">
      <p className={`status ${result.status === "complete" ? "ok" : "warn"}`}>
        {result.status === "complete" ? "All required terms provided" : result.status === "not_covered" ? "Not covered" : "Some terms missing"}
      </p>
      <p className="totals">
        Total cost <strong>{fmt(result.total_cost)}</strong> · Insurer pays (est.) <strong data-testid="covered">{fmt(result.covered_amount)}</strong> · You pay (est.){" "}
        <strong data-testid="oop">{fmt(result.out_of_pocket)}</strong>
      </p>
      {result.steps.length > 0 && (
        <table>
          <thead>
            <tr><th>#</th><th>Rule</th><th>Detail</th><th>Payable after</th><th>Added to your share</th></tr>
          </thead>
          <tbody>
            {result.steps.map((s) => (
              <tr key={s.step}>
                <td>{s.step}</td><td>{s.rule}</td><td>{s.description}</td>
                <td>{fmt(s.amount_after)}</td><td>{fmt(s.patient_share_added)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {result.missing_terms.length > 0 && <p className="error">Missing terms: {result.missing_terms.join(", ")}</p>}
      <ul className="muted">{result.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
      <p className="muted">{result.disclaimer}</p>
    </div>
  );
}
