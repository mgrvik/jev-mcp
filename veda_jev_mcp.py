#!/usr/bin/env python3
"""veda-jev MCP — comprehensive TypeSafe/Jev decision plane behind master-mcp singleton.

Based on ~/veda-converge-diverge (DecisionService). Extends vendor jkudish/jev-mcp
(verify / screen / find) with Veda's life, coding, agent, log, and branch surfaces.

Stdio JSON-RPC via scripts/mcp-singleton/mcp_standalone_base.StdioMcpServer.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

# Paths
def _load_typesafe_env() -> None:
    """Ensure TYPESAFE_API_KEY / JEV_API_KEY are set from typesafe-key.md or .env."""
    if os.getenv("TYPESAFE_API_KEY") and os.getenv("JEV_API_KEY"):
        return
    for path in (
        Path("/home/veda/typesafe-key.md"),
        Path.home() / "typesafe-key.md",
        Path("/home/veda/.env"),
        Path("/home/veda/veda-converge-diverge/.env"),
        Path("/home/veda/vendor/jev-mcp/.env"),
        Path("/home/veda/vendor/jev-review/.env"),
    ):
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if path.name.endswith(".md"):
                key = text.splitlines()[0].strip() if text else ""
                if key and not os.getenv("TYPESAFE_API_KEY"):
                    os.environ["TYPESAFE_API_KEY"] = key
                if key and not os.getenv("JEV_API_KEY"):
                    os.environ["JEV_API_KEY"] = key
                continue
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("TYPESAFE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        os.environ.setdefault("TYPESAFE_API_KEY", val)
                elif line.startswith("JEV_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        os.environ.setdefault("JEV_API_KEY", val)
        except OSError:
            continue


_load_typesafe_env()

REPO_ROOT = Path("/home/veda/veda-converge-diverge")
SRC = REPO_ROOT / "src"
SINGLETON = Path("/home/veda/scripts/mcp-singleton")
JEV_REVIEW_ROOT = Path("/home/veda/vendor/jev-review")
for p in (str(SINGLETON), str(SRC), "/home/veda/veda-core"):
    if p not in sys.path:
        sys.path.insert(0, p)

from mcp_standalone_base import StdioMcpServer  # noqa: E402

from veda_converge_diverge.models import (  # noqa: E402
    LIFE_SCENARIO_MAP,
    Answer,
    DecisionPack,
    Outcome,
    PolicySpec,
    ProviderResult,
    QuestionSpec,
)
from veda_converge_diverge.providers.mock import MockDecisionProvider  # noqa: E402
from veda_converge_diverge.service import DecisionService  # noqa: E402

DECISION_LOG_API = os.getenv("VEDA_DECISION_LOG_API", "http://127.0.0.1:8898/api/decision-log").rstrip("/")
MODEL = os.getenv("JEV_MCP_MODEL") or os.getenv("VEDA_TYPESAFE_MODEL") or "jev-latest"

_svc: DecisionService | None = None


def service() -> DecisionService:
    global _svc
    if _svc is None:
        _svc = DecisionService()
    return _svc


def _text(payload: Any) -> Dict[str, Any]:
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    return {"content": [{"type": "text", "text": body}], "isError": False}


def _err(msg: str, **extra: Any) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps({"ok": False, "error": msg, **extra}, indent=2, default=str)}],
        "isError": True,
    }


def _sanitize_id(raw: str, fallback: str, i: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (raw or "")).strip("_")
    return (cleaned[:64] or f"{fallback}{i}")


def _unique(items: List[dict], prefix: str) -> List[dict]:
    used: set[str] = set()
    out = []
    for i, item in enumerate(items):
        base = _sanitize_id(str(item.get("id") or ""), prefix, i)
        ident = base
        n = 1
        while ident in used:
            ident = f"{base}_{n}"
            n += 1
        used.add(ident)
        out.append({**item, "id": ident})
    return out


def _normalize_evidence(evidence: Any) -> List[dict]:
    if isinstance(evidence, str):
        return [{"id": "evidence", "text": evidence}]
    if isinstance(evidence, dict):
        return [{"id": evidence.get("id") or "evidence", "text": evidence.get("text") or ""}]
    if isinstance(evidence, list):
        return [
            {"id": e.get("id") if isinstance(e, dict) else f"evidence{i}", "text": (e.get("text") if isinstance(e, dict) else str(e))}
            for i, e in enumerate(evidence)
        ]
    return []


import hashlib
import threading

class MemoryDecisionCache:
    """Thread-safe TTL + LRU in-memory decision cache."""

    def __init__(self, maxsize: int = 1024, ttl_s: float = 600.0):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _hash_key(self, namespace: str, payload: Any) -> str:
        data = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{namespace}:{data}".encode("utf-8")).hexdigest()
        return digest

    def get(self, namespace: str, payload: Any) -> Any | None:
        key = self._hash_key(namespace, payload)
        now = time.time()
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if now - ts <= self.ttl_s:
                    self.hits += 1
                    return val
                else:
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, namespace: str, payload: Any, val: Any) -> None:
        key = self._hash_key(namespace, payload)
        now = time.time()
        with self._lock:
            if len(self._cache) >= self.maxsize:
                sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])
                for k in sorted_keys[: max(1, self.maxsize // 5)]:
                    self._cache.pop(k, None)
            self._cache[key] = (now, val)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl_s": self.ttl_s,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


_cache = MemoryDecisionCache(maxsize=1024, ttl_s=600.0)


def _provider_answers(pack: DecisionPack, state: dict, use_cache: bool = True) -> ProviderResult:
    if use_cache:
        cached = _cache.get(f"provider:{pack.id}:{pack.version}", state)
        if cached is not None:
            return cached

    svc = service()
    try:
        result = svc.provider.evaluate(pack=pack, state=state)
        if not svc.provider.available():
            # TypeSafe key missing — still produce a labeled offline judgment
            raise RuntimeError("provider_unavailable")
        if use_cache:
            _cache.set(f"provider:{pack.id}:{pack.version}", state, result)
        return result
    except Exception:
        mock = MockDecisionProvider().evaluate(pack=pack, state=state)
        return mock


def _offline_wrap(payload: dict, *, allow_offline: bool) -> dict:
    """If decide failed due to missing TypeSafe key, retry once with mock + labels."""
    if not allow_offline:
        return payload
    if not (isinstance(payload, dict) and payload.get("ok") is False):
        return payload
    err = str(payload.get("error") or "")
    if "provider_unavailable" not in err and "TYPESAFE_API_KEY" not in err:
        return payload
    return payload


def _decide_with_fallback(fn: Callable[[], Any], *, title: str | None = None) -> dict:
    """Run a DecisionService decide path; on provider_unavailable, use mock provider."""
    svc = service()
    original = svc.provider
    try:
        result = fn()
        return _text(result.model_dump() if hasattr(result, "model_dump") else result)
    except Exception as exc:
        if "provider_unavailable" not in str(exc) and "TYPESAFE_API_KEY" not in str(exc):
            return _err(str(exc))
        svc.provider = MockDecisionProvider()
        try:
            result = fn()
            data = result.model_dump() if hasattr(result, "model_dump") else result
            if isinstance(data, dict):
                data["offline_fallback"] = True
                data["warning"] = "TypeSafe API key not configured — mock System-1 judgments only; do not treat as calibrated Jev."
                data["provider_available"] = False
            return _text(data)
        except Exception as exc2:
            return _err(str(exc2))
        finally:
            svc.provider = original


def _answer_map(result: ProviderResult) -> dict:
    out = {}
    for name, ans in result.answers.items():
        out[name] = {
            "type": ans.type,
            "value": ans.value,
            "confidence": ans.confidence,
            "probabilities": ans.probabilities,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vendor-parity tools
# ─────────────────────────────────────────────────────────────────────────────

def jev_health(_args: dict) -> dict:
    svc = service()
    h = svc.health()
    h["model"] = MODEL
    h["decision_log_api"] = DECISION_LOG_API
    h["based_on"] = str(REPO_ROOT)
    h["vendor_parity"] = ["jev_verify", "jev_screen", "jev_find"]
    h["cache"] = _cache.stats()
    h["commands"] = len(TOOLS)
    return _text(h)


def jev_verify(args: dict) -> dict:
    claims = args.get("claims") or []
    if not claims:
        return _err("claims required")
    auto_accept = float(args.get("auto_accept") or 0.8)
    evidence_items = _unique(_normalize_evidence(args.get("evidence")), "evidence")
    claim_items = _unique([{"text": c if isinstance(c, str) else str(c)} for c in claims], "claim")

    questions = {}
    for c in claim_items:
        questions[f"relation_{c['id']}"] = QuestionSpec(
            type="choice",
            instructions={
                "question": f"How does the evidence relate to claim `{c['id']}` ({c['text']})?",
            },
            criteria={
                "supports": "Evidence states or implies the claim is true",
                "contradicts": "Evidence states or implies the claim is false",
                "says_nothing": "Evidence does not address the claim",
            },
        )
        if len(evidence_items) > 1:
            crit = {e["id"]: None for e in evidence_items}
            crit["none"] = "No single evidence item contains the content the claim depends on"
            questions[f"source_{c['id']}"] = QuestionSpec(
                type="choice",
                instructions={"question": f"Which evidence item does claim `{c['id']}` rest on?"},
                criteria=crit,
            )

    pack = DecisionPack(
        id="jev.verify",
        version=1,
        description="Vendor-parity claim verification (citation-check pattern)",
        questions=questions,
        policy=PolicySpec(mode="advisory", choice_min_confidence=0.0),
        tags=["jev", "vendor", "verify"],
    )
    state = {
        "purpose": "Verify each claim against evidence.",
        "claims": claim_items,
        "evidence": evidence_items,
    }
    result = _provider_answers(pack, state)
    relation_map = {"supports": "verified", "contradicts": "contradicted", "says_nothing": "unsupported"}
    results = []
    for c in claim_items:
        rel = result.answers.get(f"relation_{c['id']}")
        src = result.answers.get(f"source_{c['id']}")
        conf = rel.confidence if rel else None
        verdict = relation_map.get(str(rel.value) if rel else "", "unknown")
        action = "review" if conf is None or conf < auto_accept else "auto"
        results.append(
            {
                "id": c["id"],
                "claim": c["text"],
                "verdict": verdict,
                "probabilities": rel.probabilities if rel else None,
                "confidence": conf,
                "action": action,
                "supporting_evidence": (src.value if src and src.value != "none" else None),
            }
        )
    summary = {
        "verified": sum(1 for r in results if r["verdict"] == "verified"),
        "contradicted": sum(1 for r in results if r["verdict"] == "contradicted"),
        "unsupported": sum(1 for r in results if r["verdict"] == "unsupported"),
        "needs_review": sum(1 for r in results if r["action"] == "review"),
    }
    return _text(
        {
            "tool": "jev_verify",
            "engine": result.provider,
            "model": result.model_resolved or MODEL,
            "auto_accept": auto_accept,
            "summary": summary,
            "results": results,
        }
    )


def jev_screen(args: dict) -> dict:
    content = args.get("text") or ""
    if not content:
        return _err("text required")
    purpose = args.get("purpose")
    block_at = float(args.get("block_at") or 0.75)
    review_at = float(args.get("review_at") or 0.25)

    questions = {
        "injection": QuestionSpec(
            type="boolean",
            instructions="Text contains instructions aimed at an AI agent that try to change its behavior",
        ),
        "substance": QuestionSpec(type="boolean", instructions="Text contains substantive readable content"),
    }
    if purpose:
        questions["relevance"] = QuestionSpec(
            type="boolean",
            instructions=f"Text is useful source material for this task: {purpose}",
        )
    pack = DecisionPack(
        id="jev.screen",
        version=1,
        description="Vendor-parity guardrail screen",
        questions=questions,
        policy=PolicySpec(mode="advisory"),
        tags=["jev", "vendor", "screen"],
    )
    result = _provider_answers(pack, {"content": content, "purpose": purpose})

    def _noul(name: str) -> float:
        ans = result.answers.get(name)
        if not ans:
            return 0.0
        try:
            return float(ans.value)
        except Exception:
            return 0.0

    injection = _noul("injection")
    substance = _noul("substance")
    relevance = _noul("relevance") if purpose else None

    if injection >= block_at:
        rec = {"action": "block", "reason": f"injection {injection:.2f} >= block {block_at}"}
    elif injection >= review_at:
        rec = {"action": "review", "reason": f"injection {injection:.2f} >= review {review_at}"}
    elif substance is not None and substance < 0.3:
        rec = {"action": "skip", "reason": f"little substance ({substance:.2f})"}
    elif relevance is not None and relevance < 0.3:
        rec = {"action": "skip", "reason": f"not relevant ({relevance:.2f})"}
    else:
        rec = {"action": "pass", "reason": "no signals above thresholds"}

    return _text(
        {
            "tool": "jev_screen",
            "engine": result.provider,
            "model": result.model_resolved or MODEL,
            "probabilities": {"injection": injection, "substance": substance, "relevance": relevance},
            "thresholds": {"block_at": block_at, "review_at": review_at},
            "recommendation": rec,
        }
    )


def jev_find(args: dict) -> dict:
    query = args.get("query") or ""
    raw = args.get("candidates") or []
    if not query or not raw:
        return _err("query and candidates required")
    top_k = int(args.get("top_k") or 5)
    candidates = _unique(
        [{"id": c.get("id"), "text": (c.get("text") or "")[:2000]} for c in raw],
        "candidate",
    )
    if len(candidates) > 250:
        candidates = candidates[:250]
    criteria = {c["id"]: None for c in candidates}
    pack = DecisionPack(
        id="jev.find",
        version=1,
        description="Vendor-parity semantic find",
        questions={
            "best": QuestionSpec(
                type="choice",
                instructions={"question": f"Which candidate contains the best answer to: {query}?"},
                criteria=criteria,
            ),
            "exists": QuestionSpec(
                type="boolean",
                instructions=f"Does any candidate address or answer: {query}?",
            ),
        },
        policy=PolicySpec(mode="advisory"),
        tags=["jev", "vendor", "find"],
    )
    result = _provider_answers(pack, {"query": query, "candidates": candidates})
    best = result.answers.get("best")
    exists_ans = result.answers.get("exists")
    probs = (best.probabilities if best else {}) or {}
    ranked = sorted(
        candidates,
        key=lambda c: (-float(probs.get(c["id"], 0) or 0), candidates.index(c)),
    )[: max(1, top_k)]
    try:
        exists = float(exists_ans.value) if exists_ans else 0.0
    except Exception:
        exists = 0.0
    exists_verdict = "answered" if exists >= 0.7 else ("absent" if exists < 0.35 else "partial")
    return _text(
        {
            "tool": "jev_find",
            "engine": result.provider,
            "model": result.model_resolved or MODEL,
            "query": query,
            "exists": exists,
            "exists_verdict": exists_verdict,
            "selected": best.value if best else None,
            "top": [
                {"id": c["id"], "probability": float(probs.get(c["id"], 0) or 0), "text": c["text"]}
                for c in ranked
            ],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision plane tools (veda-converge-diverge)
# ─────────────────────────────────────────────────────────────────────────────

def jev_contract_list(_args: dict) -> dict:
    packs = [
        {
            "id": p.id,
            "version": p.version,
            "key": p.key,
            "description": p.description,
            "tags": p.tags,
            "mode": p.policy.mode,
            "questions": list(p.questions.keys()),
        }
        for p in service().registry.list()
    ]
    return _text({"count": len(packs), "contracts": packs})


def jev_contract_get(args: dict) -> dict:
    key = args.get("contract") or ""
    try:
        return _text(service().registry.get(key).model_dump())
    except KeyError as exc:
        return _err(str(exc))


def jev_context_compile(args: dict) -> dict:
    try:
        return _text(service().compile(args.get("contract") or "", args.get("context") or {}).model_dump())
    except Exception as exc:
        return _err(str(exc))


def jev_decide(args: dict) -> dict:
    contract = args.get("contract")
    if not contract:
        return _err("contract required")
    svc = service()

    def _run():
        return svc.decide(
            contract,
            args.get("context") or {},
            record=bool(args.get("record", True)),
            title=args.get("title"),
            domain=args.get("domain"),
            counterpart=args.get("counterpart"),
            parent_decision_id=args.get("parent_decision_id"),
            group_key=args.get("group_key"),
            branch_label=args.get("branch_label"),
        )

    return _decide_with_fallback(_run)


def jev_advise(args: dict) -> dict:
    args = {**args, "record": True}
    return jev_decide(args)


def jev_life_decide(args: dict) -> dict:
    scenario = args.get("scenario")
    if not scenario:
        return _err("scenario required — see jev_life_scenarios")
    svc = service()

    def _run():
        return svc.life_decide(
            scenario,
            args.get("context") or {},
            record=bool(args.get("record", True)),
            title=args.get("title"),
            counterpart=args.get("counterpart"),
            parent_decision_id=args.get("parent_decision_id"),
            branch_label=args.get("branch_label"),
        )

    return _decide_with_fallback(_run)


def jev_life_scenarios(_args: dict) -> dict:
    return _text(
        {
            "scenarios": service().life_scenarios(),
            "alias_map": LIFE_SCENARIO_MAP,
        }
    )


def jev_code_decide(args: dict) -> dict:
    svc = service()

    def _run():
        return svc.code_decide(
            task=args.get("task") or "",
            artifact=args.get("artifact"),
            candidates=args.get("candidates"),
            evidence=args.get("evidence"),
            extra_context=args.get("extra_context"),
            record=bool(args.get("record", True)),
        )

    return _decide_with_fallback(_run)


def jev_decision_history(args: dict) -> dict:
    return _text(
        service().history(
            start=args.get("start"),
            end=args.get("end"),
            domain=args.get("domain"),
            limit=int(args.get("limit") or 50),
        )
    )


def jev_decision_tree(args: dict) -> dict:
    return _text(service().tree(root_decision_id=args.get("root_decision_id"), limit=int(args.get("limit") or 200)))


def jev_decision_link(args: dict) -> dict:
    try:
        return _text(
            service().link(
                args.get("decision_id") or "",
                args.get("parent_decision_id") or "",
                args.get("branch_label"),
            )
        )
    except Exception as exc:
        return _err(str(exc))


def jev_decision_get(args: dict) -> dict:
    did = args.get("decision_id") or ""
    row = service().store.get_decision(did)
    if not row:
        return _err("not found", decision_id=did)
    outs = service().store.get_outcomes(did)
    kids = service().store.children(did)
    return _text({"decision": row, "outcomes": outs, "children": kids})


def jev_decision_recent(args: dict) -> dict:
    return _text(service().store.recent(limit=int(args.get("limit") or 50), contract=args.get("contract")))


def jev_report_outcome(args: dict) -> dict:
    try:
        service().outcome(
            Outcome(
                decision_id=args.get("decision_id") or "",
                action_taken=args.get("action_taken"),
                success=args.get("success"),
                result=args.get("result"),
                human_override=bool(args.get("human_override", False)),
                notes=args.get("notes"),
            )
        )
        return _text({"ok": True, "decision_id": args.get("decision_id")})
    except Exception as exc:
        return _err(str(exc))


def jev_bold_action(args: dict) -> dict:
    """Bold Action System 0 (80% rule) + optional Decision Log ingest + optional parent branch."""
    options = args.get("options") or []
    if len(options) < 1:
        return _err("options required")
    scored = []
    for i, opt in enumerate(options):
        o = opt if isinstance(opt, dict) else {"title": str(opt)}
        score = float(o.get("success_prob") or o.get("estimatedSuccessProb") or 50) * 0.45
        speed = o.get("speed") or o.get("speedToExecute") or "days"
        score += {"hours": 20, "days": 14, "weeks": 7, "months": 0}.get(str(speed), 0)
        cost = o.get("resource_cost") or o.get("resourceCost") or "medium"
        score += {"low": 15, "medium": 8, "high": 2}.get(str(cost), 0)
        if args.get("gut_option_id") and args.get("gut_option_id") == o.get("id"):
            score += 10
        if args.get("reversibility") in ("yes", "type-2"):
            score += 8
        scored.append({"option": {**o, "id": o.get("id") or f"opt-{i+1}"}, "score": round(score, 2)})
    scored.sort(key=lambda x: -x["score"])
    winner = scored[0]["option"]
    conf = (
        float(args.get("info_completeness") or 75) * 0.4
        + float(winner.get("success_prob") or winner.get("estimatedSuccessProb") or 70) * 0.35
        + (20 if args.get("worst_case_survivable", True) else -15)
        + (12 if args.get("reversibility") in ("yes", "type-2") else 4)
    )
    conf = int(max(52, min(98, round(conf))))
    verdict = {
        "engine": "bold_action_system0",
        "title": args.get("title") or "Untitled",
        "selected": winner.get("id"),
        "selected_title": winner.get("title"),
        "confidence": conf / 100.0,
        "confidenceScore": conf,
        "verdictType": args.get("verdictType") or ("IMMEDIATE_ACTION" if conf >= 75 else "EXPERIMENTAL_PROBE"),
        "ranking": [
            {"id": s["option"].get("id"), "title": s["option"].get("title"), "score": s["score"]} for s in scored
        ],
        "policy_action": "human_decision",
    }

    # Optionally persist into episodic store as a life/priority decision
    decision_id = None
    if args.get("persist", True):
        svc = service()
        original = svc.provider
        try:
            context = {
                "title": verdict["title"],
                "goal": args.get("goal") or verdict["title"],
                "options": [
                    {"id": s["option"].get("id"), "title": s["option"].get("title") or "opt", "description": ""}
                    for s in scored
                ],
                "constraints": [],
            }

            def _life(provider=None):
                return svc.life_decide(
                    args.get("scenario") or "priority",
                    context,
                    record=True,
                    title=verdict["title"],
                    counterpart=args.get("counterpart") or "self",
                    parent_decision_id=args.get("parent_decision_id"),
                    branch_label="bold-action" if args.get("parent_decision_id") else None,
                )

            try:
                res = _life()
            except Exception as exc:
                if "provider_unavailable" in str(exc) or "TYPESAFE_API_KEY" in str(exc):
                    svc.provider = MockDecisionProvider()
                    res = _life()
                    verdict["persist_offline_fallback"] = True
                else:
                    raise
            decision_id = res.decision_id
            verdict["decision_id"] = decision_id
            verdict["contract"] = res.contract
        except Exception as exc:
            verdict["persist_error"] = str(exc)
        finally:
            svc.provider = original

    # Optionally mirror into live-app Decision Log
    log_id = None
    if args.get("save_to_decision_log", False):
        try:
            import urllib.request

            payload = {
                "title": verdict["title"],
                "category": args.get("category") or "Strategy",
                "context": args.get("context") or "",
                "reversibility": args.get("reversibility") or "type-2",
                "confidenceScore": conf,
                "verdictType": verdict["verdictType"],
                "recommendedTitle": verdict["selected_title"],
                "options": [
                    {
                        "id": s["option"].get("id"),
                        "title": s["option"].get("title"),
                        "description": s["option"].get("description") or "",
                        "success_prob": s["option"].get("success_prob") or s["option"].get("estimatedSuccessProb"),
                        "speed": s["option"].get("speed") or s["option"].get("speedToExecute"),
                        "resource_cost": s["option"].get("resource_cost") or s["option"].get("resourceCost"),
                    }
                    for s in scored
                ],
                "rationale": args.get("rationale") or [],
                "parent_decision_id": args.get("parent_decision_id"),
                "external_id": decision_id,
            }
            req = urllib.request.Request(
                f"{DECISION_LOG_API}/bold-action",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            log_id = data.get("id") or (data.get("decision") or {}).get("id")
            verdict["decision_log_id"] = log_id
            verdict["action_plan"] = data.get("action_plan")
        except Exception as exc:
            verdict["decision_log_error"] = str(exc)

    return _text(verdict)


def jev_tool_select(args: dict) -> dict:
    candidates = args.get("candidate_tools") or args.get("candidates") or args.get("tools") or []
    intent = args.get("intent") or args.get("goal") or args.get("query") or ""
    domain_hint = args.get("domain_hint") or args.get("domain") or ""
    session_context = args.get("session_context") or args.get("context") or {}
    constraints = args.get("constraints") or []
    parameters_needed = args.get("parameters_needed") or []
    record = bool(args.get("record", False))
    use_cache = bool(args.get("cache", not record))

    svc = service()
    context_payload = {
        "intent": intent,
        "candidate_tools": candidates,
        "domain_hint": domain_hint,
        "session_context": session_context,
        "constraints": constraints,
        "parameters_needed": parameters_needed,
    }

    if use_cache and not record:
        cached = _cache.get("decide:mcp.tool-select", context_payload)
        if cached is not None:
            return _text(cached)

    def _run():
        res = svc.decide(
            "mcp.tool-select",
            context_payload,
            record=record,
            title=args.get("title") or "tool-select",
            domain="mcp",
            counterpart="agent",
        )
        data = res.model_dump() if hasattr(res, "model_dump") else res
        if use_cache and not record:
            _cache.set("decide:mcp.tool-select", context_payload, data)
        return data

    return _decide_with_fallback(_run)


def jev_agent_next_action(args: dict) -> dict:
    actions = args.get("candidate_actions") or args.get("actions") or args.get("options") or args.get("candidates") or ["continue", "retry", "ask", "delegate", "stop"]
    goal = args.get("goal") or "choose next agent action"
    phase = args.get("phase") or "execution"
    observation = args.get("observation") or args.get("state") or args.get("context") or "step completed"
    constraints = args.get("constraints") or []
    record = bool(args.get("record", False))
    use_cache = bool(args.get("cache", not record))

    svc = service()
    context_payload = {
        "goal": goal,
        "phase": phase,
        "observation": observation,
        "candidate_actions": actions if isinstance(actions, list) else [actions],
        "constraints": constraints,
        "attempts": int(args.get("attempts", 1)),
        "runtime_health": args.get("runtime_health") or {},
        "verification": args.get("verification") or {},
        "recent_failures": args.get("recent_failures") or [],
    }

    if use_cache and not record:
        cached = _cache.get("decide:agent.next-action", context_payload)
        if cached is not None:
            return _text(cached)

    def _run():
        res = svc.decide(
            "agent.next-action",
            context_payload,
            record=record,
            title=args.get("title") or "agent.next-action",
            domain="agent",
            counterpart="agent",
        )
        data = res.model_dump() if hasattr(res, "model_dump") else res
        if use_cache and not record:
            _cache.set("decide:agent.next-action", context_payload, data)
        return data

    return _decide_with_fallback(_run)


def jev_rank_options(args: dict) -> dict:
    """Generic multi-option ranking via candidate.select (or life pack when scenario given)."""
    options = args.get("options") or args.get("candidates") or []
    scenario = args.get("scenario")
    record = bool(args.get("record", False))
    use_cache = bool(args.get("cache", not record))
    svc = service()

    if scenario:
        context = {
            "title": args.get("title"),
            "goal": args.get("goal") or args.get("query") or "Rank options",
            "options": options,
            "constraints": args.get("constraints") or [],
        }

        def _run():
            return svc.life_decide(
                scenario,
                context,
                record=record,
                title=args.get("title"),
                counterpart=args.get("counterpart") or "self",
                parent_decision_id=args.get("parent_decision_id"),
            )

        return _decide_with_fallback(_run)

    context = {
        "goal": args.get("goal") or args.get("query") or "Rank options",
        "candidates": options,
        "constraints": args.get("constraints") or [],
    }
    if use_cache and not record:
        cached = _cache.get("decide:candidate.select", context)
        if cached is not None:
            return _text(cached)

    def _run():
        res = svc.decide(
            "candidate.select",
            context,
            record=record,
            title=args.get("title") or "rank-options",
            domain=args.get("domain") or "life",
            counterpart=args.get("counterpart") or "self",
        )
        data = res.model_dump() if hasattr(res, "model_dump") else res
        if use_cache and not record:
            _cache.set("decide:candidate.select", context, data)
        return data

    return _decide_with_fallback(_run)


def jev_cache_stats(_args: dict) -> dict:
    return _text(_cache.stats())


def jev_cache_clear(_args: dict) -> dict:
    _cache.clear()
    return _text({"ok": True, "message": "Jev in-memory decision cache cleared"})


def jev_decision_log_list(args: dict) -> dict:
    try:
        import urllib.parse
        import urllib.request

        q = urllib.parse.urlencode({k: v for k, v in (("q", args.get("q")), ("domain", args.get("domain"))) if v})
        url = f"{DECISION_LOG_API}/?{q}" if q else f"{DECISION_LOG_API}/"
        with urllib.request.urlopen(url, timeout=8) as resp:
            return _text(json.loads(resp.read().decode()))
    except Exception as exc:
        return _err(f"decision-log list failed: {exc}")


def jev_decision_log_history(args: dict) -> dict:
    try:
        import urllib.parse
        import urllib.request

        q = urllib.parse.urlencode(
            {k: v for k, v in (("start", args.get("start")), ("end", args.get("end")), ("domain", args.get("domain"))) if v}
        )
        url = f"{DECISION_LOG_API}/history?{q}" if q else f"{DECISION_LOG_API}/history"
        with urllib.request.urlopen(url, timeout=8) as resp:
            return _text(json.loads(resp.read().decode()))
    except Exception as exc:
        return _err(f"decision-log history failed: {exc}")


def jev_decision_log_tree(args: dict) -> dict:
    try:
        import urllib.parse
        import urllib.request

        q = urllib.parse.urlencode({k: v for k, v in (("root", args.get("root")),) if v})
        url = f"{DECISION_LOG_API}/tree?{q}" if q else f"{DECISION_LOG_API}/tree"
        with urllib.request.urlopen(url, timeout=8) as resp:
            return _text(json.loads(resp.read().decode()))
    except Exception as exc:
        return _err(f"decision-log tree failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# jev-review (NiazMorshed2007/jev-review) parity + Veda extensions
# ─────────────────────────────────────────────────────────────────────────────

JEV_REVIEW_DIMENSIONS = [
    "correctness",
    "cognitiveComplexity",
    "readability",
    "modularity",
    "coupling",
    "changeability",
    "abstractionQuality",
    "projectStructure",
    "duplication",
    "maintainability",
    "testQuality",
    "reliability",
    "security",
    "consistency",
    "documentation",
    "performance",
    "scalability",
    "compatibility",
    "observability",
]

CORE_DIMENSIONS = [
    "correctness",
    "cognitiveComplexity",
    "readability",
    "modularity",
    "changeability",
    "testQuality",
    "reliability",
    "security",
]

CONDITIONAL_DIMENSIONS = [
    "performance",
    "scalability",
    "compatibility",
    "observability",
]

SCORE_LEVELS = [
    "1 — Serious, fundamental problems; unsafe or substantially unfit.",
    "2 — Severe problems dominate; major rework is required.",
    "3 — Serious weaknesses; important behavior or design is unreliable.",
    "4 — Meaningful weaknesses materially impede quality.",
    "5 — Several consequential weaknesses remain.",
    "6 — Acceptable baseline, but notable improvement is warranted.",
    "7 — Sound overall with limited, concrete weaknesses.",
    "8 — Strong; only minor meaningful improvements are available.",
    "9 — Very strong and well fitted to its context.",
    "10 — Exceptional; little meaningful improvement is available. Use rarely.",
]

DIMENSION_GUIDANCE = {
    "correctness": "Judge requested behavior, missing behavior, edge cases, invariants, assumptions, regressions.",
    "cognitiveComplexity": "Judge nesting, branching, hidden control flow, indirection, side effects, special cases.",
    "readability": "Judge naming, control/data-flow clarity, responsibilities, useful comments.",
    "modularity": "Judge coherent grouping, separation of unrelated responsibilities, domain boundaries.",
    "coupling": "Judge unnecessary dependencies, dependency direction, leaked details, circularity, global state.",
    "changeability": "Judge shotgun surgery, scattered rules, duplicated knowledge, brittle chains.",
    "abstractionQuality": "Judge interface simplicity, information hiding, premature abstraction, leaky APIs.",
    "projectStructure": "Judge navigation, module boundaries, utility dumping grounds, god files.",
    "duplication": "Judge duplicated knowledge, repeated business rules, missed reuse (not syntactic DRY).",
    "maintainability": "Judge combined cost of understanding, modifying, debugging, testing, extending.",
    "testQuality": "Judge behavior coverage, assertions, determinism, isolation, edge cases.",
    "reliability": "Judge error propagation, invalid states, cleanup, retries, timeouts, races.",
    "security": "Judge trust boundaries, validation, injection, authn/authz, secrets, defaults.",
    "consistency": "Judge fit with repository architecture, naming, language idioms, conventions.",
    "documentation": "Judge public contracts, decisions, unusual constraints, changed APIs.",
    "performance": "Only when relevant: algorithmic cost, repeated work, N+1, I/O. Require evidence.",
    "scalability": "Only when growth/change is evidenced. Do not reward speculative architecture.",
    "compatibility": "Only when contracts/integrations are relevant: breaking changes, migrations.",
    "observability": "Only when operational behavior is relevant: logging, metrics, diagnostics.",
}

WEAKNESS_CHOICES = {
    "correctness": {
        "missing_behavior": "Requested behavior appears missing or incomplete.",
        "edge_case": "An important edge case or invalid state appears insufficiently handled.",
        "incorrect_assumption": "Implementation relies on an unsafe or incorrect assumption.",
        "regression_risk": "The change creates meaningful risk of breaking existing behavior.",
        "no_material_issue": "No material issue is evident from the supplied context.",
    },
    "security": {
        "input_validation": "Untrusted input crosses a boundary without sufficient validation.",
        "secret_handling": "Secret or sensitive-data handling creates unnecessary exposure risk.",
        "authorization": "Authentication, authorization, or privilege boundaries appear insufficient.",
        "injection": "Data may reach an interpreter or sensitive sink without appropriate separation.",
        "no_material_issue": "No material issue is evident from the supplied context.",
    },
    "testQuality": {
        "missing_coverage": "Important changed behavior lacks meaningful regression coverage.",
        "weak_assertions": "Tests execute code without proving the important outcome.",
        "brittle_tests": "Tests are coupled to implementation details or excessive mocking.",
        "nondeterminism": "Test design makes results nondeterministic or poorly isolated.",
        "no_material_issue": "No material issue is evident from the supplied context.",
    },
    "reliability": {
        "error_propagation": "Errors are swallowed, distorted, or propagated without useful boundaries.",
        "cleanup": "A failure path can leave resources or state inconsistent.",
        "timeout_retry": "Timeout or retry behavior is missing, unsafe, or disproportionate.",
        "concurrency": "A race or concurrency assumption threatens reliable behavior.",
        "no_material_issue": "No material issue is evident from the supplied context.",
    },
}


def _score_to_level(value: Any) -> float:
    """Map TypeSafe score answer (index or 1-10) to 1-10."""
    try:
        v = float(value)
    except Exception:
        return 6.0
    # Mock returns midpoint index 0..9; live score may be level index or number
    if 0 <= v <= 9.5 and v != int(v) or (0 <= v <= 9 and float(v).is_integer() and v < 10):
        # treat as 0-based level index when clearly an index
        if v <= 9:
            return round(v + 1, 1) if v <= 9 and v < 1 else float(v)
    return max(1.0, min(10.0, v))


def _normalize_score(ans_value: Any, n_levels: int = 10) -> float:
    try:
        v = float(ans_value)
    except Exception:
        return 6.0
    # If provider returns index into SCORE_LEVELS (0..n-1)
    if 0 <= v <= n_levels - 1:
        return float(int(v) + 1)
    return max(1.0, min(10.0, v))


def _build_review_questions(dimensions: list[str]) -> dict:
    questions: dict[str, QuestionSpec] = {}
    for dim in dimensions:
        guidance = DIMENSION_GUIDANCE.get(dim, dim)
        cond = dim in CONDITIONAL_DIMENSIONS
        if cond:
            app_inst = f"Is {dim} actually relevant and assessable from the supplied software-change state? {guidance}"
        else:
            app_inst = f"Does the supplied state contain enough evidence to assess {dim}? {guidance}"
        questions[f"{dim}_applicable"] = QuestionSpec(
            type="boolean",
            instructions=app_inst,
        )
        questions[f"{dim}_score"] = QuestionSpec(
            type="score",
            instructions=f"Rate {dim} for the implementation in the supplied state. {guidance}",
            criteria=list(SCORE_LEVELS),
        )
        weakness_criteria = WEAKNESS_CHOICES.get(dim) or {
            "material_issue": "There is a material weakness in this dimension.",
            "no_material_issue": "No material issue is evident from the supplied context.",
        }
        questions[f"{dim}_weakness"] = QuestionSpec(
            type="choice",
            instructions=f"Identify the single most consequential {dim} weakness. Choose no_material_issue when none is justified.",
            criteria=weakness_criteria,
        )
    return questions


def jev_review_dimensions(_args: dict) -> dict:
    return _text(
        {
            "all": JEV_REVIEW_DIMENSIONS,
            "core": CORE_DIMENSIONS,
            "conditional": CONDITIONAL_DIMENSIONS,
            "score_levels": SCORE_LEVELS,
            "source": str(JEV_REVIEW_ROOT),
        }
    )


def jev_review(args: dict) -> dict:
    """Vendor-parity software quality review (jev_review) via TypeSafe/Jev.

    Input: task, diff, files[], repositoryContext, previousEvaluation, dimensions[].
    Output: metrics{dim:{applicable,score,confidence,weakness}}, priorities, comparison.
    """
    task = args.get("task") or ""
    diff = args.get("diff") or ""
    files = args.get("files") or []
    repo_ctx = args.get("repositoryContext") or args.get("repository_context") or ""
    prev = args.get("previousEvaluation") or args.get("previous_evaluation") or {}
    if not (task or diff or files or repo_ctx):
        return _err("Provide at least one of task, diff, files, repositoryContext")

    requested = args.get("dimensions") or args.get("metrics") or CORE_DIMENSIONS
    dimensions = [d for d in requested if d in JEV_REVIEW_DIMENSIONS] or list(CORE_DIMENSIONS)

    state = {
        "purpose": "Software-quality review of a focused implementation slice.",
        "task": task or None,
        "diff": (diff[:12000] if diff else None),
        "files": files[:12] if files else None,
        "repositoryContext": (repo_ctx[:8000] if repo_ctx else None),
    }
    questions = _build_review_questions(dimensions)
    pack = DecisionPack(
        id="code.review-quality.extended",
        version=1,
        description="jev-review multi-dimension quality evaluation",
        questions=questions,
        policy=PolicySpec(mode="advisory"),
        tags=["code", "review", "jev-review"],
    )
    result = _provider_answers(pack, state)

    metrics: dict[str, Any] = {}
    priorities: list[dict] = []
    improvements: list[str] = []
    comparison: list[dict] = []

    for dim in dimensions:
        app = result.answers.get(f"{dim}_applicable")
        score_ans = result.answers.get(f"{dim}_score")
        weak = result.answers.get(f"{dim}_weakness")
        try:
            applicable = bool(float(app.value) >= 0.5) if app else True
        except Exception:
            applicable = True
        entry: dict[str, Any] = {"applicable": applicable}
        if applicable and score_ans is not None:
            score = _normalize_score(score_ans.value)
            conf = score_ans.confidence
            entry["score"] = score
            entry["confidence"] = float(conf) if conf is not None else 0.5
            weakness_key = str(weak.value) if weak else "no_material_issue"
            entry["weakness"] = weakness_key
            if weakness_key != "no_material_issue":
                entry["summary"] = WEAKNESS_CHOICES.get(dim, {}).get(weakness_key) or weakness_key
                priorities.append(
                    {
                        "metric": dim,
                        "severity": "high" if score <= 4 else ("medium" if score <= 6 else "low"),
                        "reason": entry["summary"],
                    }
                )
                if "suggestion" not in entry:
                    improvements.append(f"[{dim}] {entry['summary']}")
            # comparison vs previous
            prev_metrics = (prev or {}).get("metrics") or {}
            prev_score = (prev_metrics.get(dim) or {}).get("score")
            if isinstance(prev_score, (int, float)):
                delta = round(score - float(prev_score), 2)
                direction = "improved" if delta > 0.3 else ("regressed" if delta < -0.3 else "unchanged")
                comparison.append(
                    {
                        "metric": dim,
                        "previousScore": float(prev_score),
                        "currentScore": score,
                        "delta": delta,
                        "direction": direction,
                    }
                )
        metrics[dim] = entry

    # Prioritize by weighted severity
    weight = {"correctness": 3, "security": 3, "changeability": 3}
    priorities.sort(key=lambda p: ({"high": 0, "medium": 1, "low": 2}.get(p["severity"], 3), -weight.get(p["metric"], 1)))

    engine = result.provider
    return _text(
        {
            "tool": "jev_review",
            "engine": engine,
            "model": result.model_resolved or MODEL,
            "based_on": str(JEV_REVIEW_ROOT),
            "contract": pack.key,
            "dimensions": dimensions,
            "metrics": metrics,
            "priorities": priorities[:12],
            "improvements": improvements[:12],
            "comparison": comparison,
            "offline_fallback": engine == "mock",
            "policy_note": "Review evidence only — coding agent remains responsible for changes.",
        }
    )


def jev_review_compare(args: dict) -> dict:
    """Compare two evaluation metric maps (previous vs current) without calling Jev."""
    previous = args.get("previous") or {}
    current = args.get("current") or {}
    prev_m = previous.get("metrics") or previous
    cur_m = current.get("metrics") or current
    comparison = []
    improvements = []
    regressions = []
    for dim in JEV_REVIEW_DIMENSIONS:
        ps = (prev_m.get(dim) or {}).get("score") if isinstance(prev_m.get(dim), dict) else None
        cs = (cur_m.get(dim) or {}).get("score") if isinstance(cur_m.get(dim), dict) else None
        if not isinstance(ps, (int, float)) or not isinstance(cs, (int, float)):
            continue
        delta = round(float(cs) - float(ps), 2)
        direction = "improved" if delta > 0.3 else ("regressed" if delta < -0.3 else "unchanged")
        comparison.append(
            {
                "metric": dim,
                "previousScore": float(ps),
                "currentScore": float(cs),
                "delta": delta,
                "direction": direction,
            }
        )
        if direction == "improved":
            improvements.append(f"{dim}: {ps} → {cs}")
        elif direction == "regressed":
            regressions.append(f"{dim}: {ps} → {cs}")
    return _text({"tool": "jev_review_compare", "comparison": comparison, "improvements": improvements, "regressions": regressions})


def jev_review_via_vendor(args: dict) -> dict:
    """Call the bundled vendor Node jev-review server (dist/server.js) when available."""
    server_js = JEV_REVIEW_ROOT / "dist" / "server.js"
    if not server_js.is_file():
        return _err("vendor dist/server.js missing — run npm install/build in ~/vendor/jev-review", path=str(server_js))
    key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY") or ""
    if not key:
        return _err("JEV_API_KEY/TYPESAFE_API_KEY not set")
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "veda-jev", "version": "1"}}})
    call = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "jev_review", "arguments": args}})
    script = f"""
import json, os, subprocess, sys
env=os.environ.copy()
env['JEV_API_KEY']={key!r}
env['TYPESAFE_API_KEY']={key!r}
p=subprocess.Popen(['node', {str(server_js)!r}], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
p.stdin.write({payload!r}+'\\n'); p.stdin.flush(); p.stdout.readline()
p.stdin.write({call!r}+'\\n'); p.stdin.flush()
print(p.stdout.readline())
p.terminate()
"""
    import subprocess

    try:
        proc = subprocess.run(
            ["/home/veda/veda-core/.venv/bin/python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return _err("vendor server produced no output", stderr=proc.stderr[-500:])
        return _text({"tool": "jev_review_via_vendor", "raw": json.loads(line[-1])})
    except Exception as exc:
        return _err(str(exc))


def jev_commands(_args: dict) -> dict:
    return _text(
        {
            "server": "veda-jev",
            "based_on": str(REPO_ROOT),
            "vendor": "https://github.com/jkudish/jev-mcp + https://github.com/NiazMorshed2007/jev-review",
            "model": MODEL,
            "typesafe_key_configured": bool(os.getenv("TYPESAFE_API_KEY")),
            "commands": [
                {"name": t["name"], "summary": t["description"].split(".")[0]}
                for t in TOOLS
            ],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas + registration
# ─────────────────────────────────────────────────────────────────────────────

def _obj(props: dict, required: list | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "jev_health",
        "description": "Decision-plane + provider health (veda-converge-diverge).",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_commands",
        "description": "List every command on this comprehensive Jev/decision MCP.",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_verify",
        "description": "Vendor parity: verify claims against evidence with typed Jev verdicts.",
        "inputSchema": _obj(
            {
                "claims": {"type": "array", "items": {"type": "string"}},
                "evidence": {"description": "string | {id,text} | [{id,text}]"},
                "auto_accept": {"type": "number"},
            },
            ["claims", "evidence"],
        ),
    },
    {
        "name": "jev_screen",
        "description": "Vendor parity: screen external text for prompt injection / substance / relevance.",
        "inputSchema": _obj(
            {
                "text": {"type": "string"},
                "purpose": {"type": "string"},
                "block_at": {"type": "number"},
                "review_at": {"type": "number"},
            },
            ["text"],
        ),
    },
    {
        "name": "jev_find",
        "description": "Vendor parity: semantic rank of candidates against a query (no embeddings).",
        "inputSchema": _obj(
            {
                "query": {"type": "string"},
                "candidates": {"type": "array", "items": {"type": "object"}},
                "top_k": {"type": "integer"},
            },
            ["query", "candidates"],
        ),
    },
    {
        "name": "jev_review",
        "description": "jev-review parity: multi-dimension software quality review (correctness, security, tests, ...).",
        "inputSchema": _obj(
            {
                "task": {"type": "string"},
                "diff": {"type": "string"},
                "files": {"type": "array", "items": {"type": "object"}},
                "repositoryContext": {"type": "string"},
                "previousEvaluation": {"type": "object"},
                "dimensions": {"type": "array", "items": {"type": "string"}},
            },
        ),
    },
    {
        "name": "jev_review_dimensions",
        "description": "List jev-review quality dimensions, core vs conditional, and score levels.",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_review_compare",
        "description": "Compare two review evaluations (previous vs current) without calling Jev.",
        "inputSchema": _obj({"previous": {"type": "object"}, "current": {"type": "object"}}, ["previous", "current"]),
    },
    {
        "name": "jev_review_via_vendor",
        "description": "Optional: call bundled ~/vendor/jev-review Node MCP (dist/server.js) directly.",
        "inputSchema": _obj(
            {
                "task": {"type": "string"},
                "diff": {"type": "string"},
                "files": {"type": "array"},
                "repositoryContext": {"type": "string"},
            },
        ),
    },
    {
        "name": "jev_contract_list",
        "description": "List all decision packs (code, life, agent, conversation, system).",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_contract_get",
        "description": "Get one decision pack by id/key.",
        "inputSchema": _obj({"contract": {"type": "string"}}, ["contract"]),
    },
    {
        "name": "jev_context_compile",
        "description": "Compile/prune/redact context for a pack without calling the model.",
        "inputSchema": _obj({"contract": {"type": "string"}, "context": {"type": "object"}}, ["contract", "context"]),
    },
    {
        "name": "jev_decide",
        "description": "Run a decision pack: compile → Jev → policy → optional episodic record.",
        "inputSchema": _obj(
            {
                "contract": {"type": "string"},
                "context": {"type": "object"},
                "record": {"type": "boolean"},
                "title": {"type": "string"},
                "domain": {"type": "string"},
                "counterpart": {"type": "string"},
                "parent_decision_id": {"type": "string"},
                "branch_label": {"type": "string"},
            },
            ["contract", "context"],
        ),
    },
    {
        "name": "jev_advise",
        "description": "Advisory alias for jev_decide (always records).",
        "inputSchema": _obj(
            {"contract": {"type": "string"}, "context": {"type": "object"}, "title": {"type": "string"}},
            ["contract", "context"],
        ),
    },
    {
        "name": "jev_life_decide",
        "description": "Life decision mapper: career/money/relationship/request/negotiate/priority/agent/health/buy/meeting.",
        "inputSchema": _obj(
            {
                "scenario": {"type": "string"},
                "context": {"type": "object"},
                "title": {"type": "string"},
                "counterpart": {"type": "string", "enum": ["self", "human", "agent", "system"]},
                "parent_decision_id": {"type": "string"},
                "branch_label": {"type": "string"},
                "record": {"type": "boolean"},
            },
            ["scenario", "context"],
        ),
    },
    {
        "name": "jev_life_scenarios",
        "description": "List life scenarios and their contract packs.",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_code_decide",
        "description": "Coding decisions: refactor, bug root-cause, test priority, review triage, merge readiness, etc.",
        "inputSchema": _obj(
            {
                "task": {"type": "string"},
                "artifact": {"type": "string"},
                "candidates": {"type": "array"},
                "evidence": {"type": "object"},
                "extra_context": {"type": "object"},
                "record": {"type": "boolean"},
            },
            ["task"],
        ),
    },
    {
        "name": "jev_rank_options",
        "description": "Rank N options (candidate.select or life pack when scenario set).",
        "inputSchema": _obj(
            {
                "goal": {"type": "string"},
                "options": {"type": "array"},
                "scenario": {"type": "string"},
                "title": {"type": "string"},
                "counterpart": {"type": "string"},
                "record": {"type": "boolean"},
                "parent_decision_id": {"type": "string"},
            },
            ["options"],
        ),
    },
    {
        "name": "jev_tool_select",
        "description": "Select best tool/skill/agent for a goal via mcp.tool-select pack.",
        "inputSchema": _obj(
            {"goal": {"type": "string"}, "candidates": {"type": "array"}, "constraints": {"type": "array"}},
            ["goal", "candidates"],
        ),
    },
    {
        "name": "jev_agent_next_action",
        "description": "Agent loop control: continue/retry/ask/delegate/stop via agent.next-action pack.",
        "inputSchema": _obj(
            {"goal": {"type": "string"}, "options": {"type": "array"}, "state": {"type": "object"}, "record": {"type": "boolean"}},
            ["goal"],
        ),
    },
    {
        "name": "jev_bold_action",
        "description": "Bold Action 80% System 0 + optional episodic persist + optional Decision Log ingest.",
        "inputSchema": _obj(
            {
                "title": {"type": "string"},
                "options": {"type": "array"},
                "scenario": {"type": "string"},
                "category": {"type": "string"},
                "reversibility": {"type": "string"},
                "goal": {"type": "string"},
                "counterpart": {"type": "string"},
                "parent_decision_id": {"type": "string"},
                "persist": {"type": "boolean"},
                "save_to_decision_log": {"type": "boolean"},
                "info_completeness": {"type": "number"},
                "worst_case_survivable": {"type": "boolean"},
                "gut_option_id": {"type": "string"},
            },
            ["title", "options"],
        ),
    },
    {
        "name": "jev_decision_history",
        "description": "Date-ranged episodic decision history.",
        "inputSchema": _obj({"start": {"type": "string"}, "end": {"type": "string"}, "domain": {"type": "string"}, "limit": {"type": "integer"}}),
    },
    {
        "name": "jev_decision_tree",
        "description": "Parent-child decision tree from episodic store.",
        "inputSchema": _obj({"root_decision_id": {"type": "string"}, "limit": {"type": "integer"}}),
    },
    {
        "name": "jev_decision_link",
        "description": "Link a decision as a child branch of a parent.",
        "inputSchema": _obj(
            {"decision_id": {"type": "string"}, "parent_decision_id": {"type": "string"}, "branch_label": {"type": "string"}},
            ["decision_id", "parent_decision_id"],
        ),
    },
    {
        "name": "jev_decision_get",
        "description": "Get one decision + outcomes + children.",
        "inputSchema": _obj({"decision_id": {"type": "string"}}, ["decision_id"]),
    },
    {
        "name": "jev_decision_recent",
        "description": "Recent decisions from episodic store.",
        "inputSchema": _obj({"limit": {"type": "integer"}, "contract": {"type": "string"}}),
    },
    {
        "name": "jev_report_outcome",
        "description": "Attach ground-truth outcome for calibration.",
        "inputSchema": _obj(
            {
                "decision_id": {"type": "string"},
                "action_taken": {"type": "string"},
                "success": {"type": "boolean"},
                "result": {"type": "string"},
                "human_override": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            ["decision_id"],
        ),
    },
    {
        "name": "jev_decision_log_list",
        "description": "Live-app Decision Log list (bold-action + mapper entries).",
        "inputSchema": _obj({"q": {"type": "string"}, "domain": {"type": "string"}}),
    },
    {
        "name": "jev_decision_log_history",
        "description": "Live-app Decision Log history by date.",
        "inputSchema": _obj({"start": {"type": "string"}, "end": {"type": "string"}, "domain": {"type": "string"}}),
    },
    {
        "name": "jev_decision_log_tree",
        "description": "Live-app Decision Log parent-child tree.",
        "inputSchema": _obj({"root": {"type": "string"}}),
    },
    {
        "name": "jev_cache_stats",
        "description": "Get in-memory decision cache hit/miss statistics and size.",
        "inputSchema": _obj({}),
    },
    {
        "name": "jev_cache_clear",
        "description": "Clear the in-memory decision cache.",
        "inputSchema": _obj({}),
    },
]

HANDLERS: Dict[str, Callable[[dict], dict]] = {
    "jev_health": jev_health,
    "jev_commands": jev_commands,
    "jev_cache_stats": jev_cache_stats,
    "jev_cache_clear": jev_cache_clear,
    "jev_verify": jev_verify,
    "jev_screen": jev_screen,
    "jev_find": jev_find,
    "jev_review": jev_review,
    "jev_review_dimensions": jev_review_dimensions,
    "jev_review_compare": jev_review_compare,
    "jev_review_via_vendor": jev_review_via_vendor,
    "jev_contract_list": jev_contract_list,
    "jev_contract_get": jev_contract_get,
    "jev_context_compile": jev_context_compile,
    "jev_decide": jev_decide,
    "jev_advise": jev_advise,
    "jev_life_decide": jev_life_decide,
    "jev_life_scenarios": jev_life_scenarios,
    "jev_code_decide": jev_code_decide,
    "jev_rank_options": jev_rank_options,
    "jev_tool_select": jev_tool_select,
    "jev_agent_next_action": jev_agent_next_action,
    "jev_bold_action": jev_bold_action,
    "jev_decision_history": jev_decision_history,
    "jev_decision_tree": jev_decision_tree,
    "jev_decision_link": jev_decision_link,
    "jev_decision_get": jev_decision_get,
    "jev_decision_recent": jev_decision_recent,
    "jev_report_outcome": jev_report_outcome,
    "jev_decision_log_list": jev_decision_log_list,
    "jev_decision_log_history": jev_decision_log_history,
    "jev_decision_log_tree": jev_decision_log_tree,
}


def main() -> None:
    server = StdioMcpServer("veda-jev", version="1.0.0-comprehensive", max_workers=8)
    server.get_tools_list = lambda: TOOLS
    server.register_handlers(HANDLERS)
    server.run()


if __name__ == "__main__":
    main()
