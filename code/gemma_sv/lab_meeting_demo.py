"""One-command LongMemEval lab-meeting replay with an optional live resolver.

The resolver answers only "which committed memory record?". Confirmation ends
at that boundary. All subsequent deletion and certificate evidence is loaded
from the committed, hash-bound replay; this app never runs a deletion model.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
from typing import Any
import webbrowser

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .demo_server.resolver import (
    CatalogTurn,
    GeminiResolver,
    MemoryCatalogRecord,
    MemoryResolver,
    ProposalConsumed,
    ProposalError,
    ProposalExpired,
    ProposalInvalidated,
    ProposalNotFound,
    ProposalSelectionInvalid,
    ProposalSelectionRequired,
    ProposalStore,
    RecordedResolver,
    ResolverDecision,
    ResolverStatus,
    ResolverUnavailable,
    ResolverValidationError,
    catalog_hash,
    request_hash,
    validate_resolution_payload,
)


LAB_MODEL_ID = "gemini-3.6-flash"
LAB_CASE_RECORD_ID = "longmemeval-constrained-chat-v1-c8276e265e3db489c090cdcc"
FINAL_REPORT_SHA256 = (
    "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
)
LAB_RESOLVER_QUERY = "resolver=lab"
LAB_SESSION_COOKIE = "gemmasv_lab_session"
LAB_SESSION_MAX_AGE_SECONDS = 30 * 60
LAB_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
SITE_DIR = Path(__file__).resolve().parent / "demo_site"
CASE_ASSET = SITE_DIR / "assets" / "longmemeval_forgetting_final.json"
REPLAY_PATH = "/longmemeval_forgetting_replay.html"

_EXPECTED_PAYLOAD_SCHEMA = "gemma-sv-longmemeval-chat-forgetting-payload-v2"
_EXPECTED_FIXTURE_SCHEMA = "gemma-sv-recorded-memory-resolver-fixture-v1"
_EXPECTED_FIXTURE_LABEL = (
    "Deterministic recorded resolver fixture — no provider call"
)
_ALLOWED_EXCHANGE_REFS = {
    "/case/dialogue/target/quotes": "target",
    "/case/dialogue/retained/quotes": "retained",
}
_SESSION_PATTERN = re.compile(r"^lab_[A-Za-z0-9_-]{20,128}$")


class LabCaseError(RuntimeError):
    """The committed replay is missing or no longer satisfies the lab contract."""


class LabStartupError(RuntimeError):
    """The selected launcher mode cannot start safely."""


def load_lab_environment(path: str | Path = LAB_ENV_FILE) -> bool:
    """Load the local server-only demo environment without overriding the shell."""

    return load_dotenv(
        dotenv_path=Path(path),
        override=False,
        interpolate=False,
        encoding="utf-8",
    )


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_text: str = Field(min_length=1, max_length=1_000)

    @field_validator("request_text")
    @classmethod
    def request_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_text must not be blank")
        return value


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(
        min_length=10,
        max_length=128,
        pattern=r"^proposal_[A-Za-z0-9_-]+$",
    )
    confirmed: bool


@dataclass(frozen=True)
class LabCase:
    """Validated server-side view of the one committed LongMemEval case."""

    record_id: str
    request_text: str
    records: tuple[MemoryCatalogRecord, ...]
    public_records: Mapping[str, Mapping[str, Any]]
    server_catalog_hash: str
    committed_catalog_hash: str
    fixture_selected_record_id: str
    fixture_explanation: str
    payload_sha256: str
    method_report_sha256: str

    def public_record(self, record_id: str) -> Mapping[str, Any]:
        try:
            return self.public_records[record_id]
        except KeyError as exc:
            raise LabCaseError("selected record is absent from the lab case") from exc

    def recorded_decision(self) -> dict[str, Any]:
        return {
            "status": "resolved",
            "selected_record_id": self.fixture_selected_record_id,
            "alternative_record_ids": [],
            # This is a deterministic fixture marker, not provider confidence.
            "confidence": 0.0,
            "explanation": self.fixture_explanation,
            "requires_confirmation": True,
            "deletion_executed": False,
        }

    def deletion_evidence(self) -> dict[str, Any]:
        return {
            "kind": "recorded_replay",
            "label": "Recorded, hash-bound LongMemEval deletion evidence",
            "live_model_deletion_run": False,
            "resolver_outside_certificate": True,
            "payload_sha256": self.payload_sha256,
            "method_report_sha256": self.method_report_sha256,
        }


def load_lab_case(path: str | Path = CASE_ASSET) -> LabCase:
    """Load and validate the committed catalog and complete exchanges."""

    asset_path = Path(path)
    try:
        with asset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LabCaseError(f"cannot load committed replay: {asset_path}") from exc
    if not isinstance(payload, dict):
        raise LabCaseError("committed replay root must be an object")

    integrity = payload.get("integrity")
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    if (
        payload.get("schema") != _EXPECTED_PAYLOAD_SCHEMA
        or payload.get("schema_version") != 2
        or payload.get("status") != "final_case_study"
        or not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _canonical_sha256(unsigned)
    ):
        raise LabCaseError("committed replay integrity or final status differs")

    case = _mapping(payload.get("case"), "case")
    resolution = _mapping(case.get("memory_resolution"), "memory resolution")
    fixture_provider = _mapping(resolution.get("provider"), "fixture provider")
    model_configuration = _mapping(
        resolution.get("model_configuration"),
        "fixture model configuration",
    )
    if (
        resolution.get("schema") != _EXPECTED_FIXTURE_SCHEMA
        or resolution.get("schema_version") != 1
        or resolution.get("mode") != "deterministic_recorded_fixture"
        or resolution.get("label") != _EXPECTED_FIXTURE_LABEL
        or resolution.get("outside_certificate_boundary") is not True
        or resolution.get("network_request_performed") is not False
        or fixture_provider.get("call_performed") is not False
        or fixture_provider.get("artifact_present") is not False
        or fixture_provider.get("artifact") is not None
        or model_configuration.get("default_model_id") != LAB_MODEL_ID
        or model_configuration.get("model_produced_fixture") is not False
    ):
        raise LabCaseError("recorded resolver provenance boundary differs")

    request = _mapping(resolution.get("request"), "resolver request")
    deletion_action = _mapping(case.get("deletion_action"), "deletion action")
    request_text = request.get("text")
    if (
        not isinstance(request_text, str)
        or not request_text
        or request_text != deletion_action.get("display_text")
        or request.get("sha256") != _text_sha256(request_text)
    ):
        raise LabCaseError("committed resolver request binding differs")

    catalog_rows = resolution.get("candidate_catalog")
    if not isinstance(catalog_rows, list) or len(catalog_rows) != 2:
        raise LabCaseError("lab catalog must contain exactly two records")
    committed_catalog_hash = resolution.get("catalog_hash")
    if (
        not isinstance(committed_catalog_hash, str)
        or committed_catalog_hash != _canonical_sha256(catalog_rows)
    ):
        raise LabCaseError("committed resolver catalog hash differs")

    dialogue = _mapping(case.get("dialogue"), "dialogue")
    records: list[MemoryCatalogRecord] = []
    public_records: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in catalog_rows:
        candidate = _mapping(raw_candidate, "catalog candidate")
        exchange_ref = candidate.get("owned_exchange_ref")
        section_name = _ALLOWED_EXCHANGE_REFS.get(exchange_ref)
        if section_name is None:
            raise LabCaseError("catalog candidate has an unauthorized exchange ref")
        section = _mapping(dialogue.get(section_name), f"{section_name} dialogue")
        quotes = section.get("quotes")
        if not isinstance(quotes, list) or len(quotes) != 2:
            raise LabCaseError("catalog exchange must contain two complete turns")

        source_turn_ids = candidate.get("source_turn_ids")
        if (
            not isinstance(source_turn_ids, list)
            or len(source_turn_ids) != len(quotes)
            or any(not isinstance(value, str) for value in source_turn_ids)
        ):
            raise LabCaseError("catalog source turn IDs are invalid")

        catalog_turns: list[CatalogTurn] = []
        public_turns: list[dict[str, Any]] = []
        for turn_id, raw_quote in zip(source_turn_ids, quotes, strict=True):
            quote = _mapping(raw_quote, "dialogue quote")
            text = quote.get("display_text")
            role = quote.get("role")
            source_sha256 = quote.get("source_turn_sha256")
            source_index = quote.get("source_turn_index")
            if (
                not isinstance(text, str)
                or not text
                or quote.get("display_text_sha256") != _text_sha256(text)
                or role not in {"user", "assistant"}
                or not isinstance(source_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
                or turn_id != f"turn_{source_sha256[:20]}"
                or not isinstance(source_index, int)
            ):
                raise LabCaseError("source-bound complete exchange differs")
            catalog_turns.append(
                CatalogTurn(turn_id=turn_id, role=role, text=text)
            )
            public_turns.append(
                {
                    "turn_id": turn_id,
                    "role": role,
                    "text": text,
                    "display_text": text,
                    "source_turn_index": source_index,
                    "source_turn_sha256": source_sha256,
                }
            )

        try:
            record = MemoryCatalogRecord(
                record_id=candidate.get("record_id"),
                label=candidate.get("label"),
                summary=candidate.get("summary"),
                source_turn_ids=tuple(source_turn_ids),
                deletion_scope=candidate.get("deletion_scope"),
                owned_exchange=tuple(catalog_turns),
            )
        except Exception as exc:
            raise LabCaseError("committed catalog record is invalid") from exc
        if record.deletion_scope != "complete_exchange":
            raise LabCaseError("lab records must own complete exchanges")
        if record.record_id in public_records:
            raise LabCaseError("lab catalog record IDs must be unique")
        records.append(record)
        public_records[record.record_id] = {
            "record_id": record.record_id,
            "label": record.label,
            "summary": record.summary,
            "source_turn_ids": list(record.source_turn_ids),
            "deletion_scope": record.deletion_scope,
            "owned_exchange": public_turns,
        }

    decision = _mapping(resolution.get("decision"), "fixture decision")
    selected_record_id = decision.get("selected_record_id")
    if (
        decision.get("status") != "resolved"
        or selected_record_id not in public_records
        or decision.get("alternative_record_ids") != []
        or decision.get("requires_confirmation") is not True
        or decision.get("deletion_executed") is not False
        or not isinstance(decision.get("explanation"), str)
    ):
        raise LabCaseError("recorded resolver decision differs")

    confirmation = _mapping(resolution.get("confirmation"), "fixture confirmation")
    selected_candidate = next(
        candidate
        for candidate in catalog_rows
        if candidate.get("record_id") == selected_record_id
    )
    selected_section = _ALLOWED_EXCHANGE_REFS[
        selected_candidate["owned_exchange_ref"]
    ]
    selected_quotes = _mapping(
        dialogue.get(selected_section),
        "selected dialogue",
    ).get("quotes")
    if (
        selected_candidate.get("owned_exchange_ref")
        != "/case/dialogue/target/quotes"
        or confirmation.get("selected_record_id") != selected_record_id
        or confirmation.get("catalog_hash") != committed_catalog_hash
        or confirmation.get("requires_explicit_click") is not True
        or confirmation.get("confirmed_in_payload") is not False
        or confirmation.get("owned_exchange") != selected_quotes
    ):
        raise LabCaseError("recorded confirmation exchange binding differs")

    provenance = _mapping(payload.get("provenance"), "provenance")
    source_artifacts = _mapping(
        provenance.get("source_artifacts"),
        "source artifacts",
    )
    method_report = _mapping(
        source_artifacts.get("method_report"),
        "method report provenance",
    )
    method_report_sha256 = method_report.get("file_sha256")
    case_record_id = case.get("record_id")
    demo = _mapping(payload.get("demo"), "demo")
    scope = _mapping(payload.get("scope"), "scope")
    if (
        case_record_id != LAB_CASE_RECORD_ID
        or method_report_sha256 != FINAL_REPORT_SHA256
        or demo.get("live_compute") is not False
        or demo.get("network_api_required") is not False
        or scope.get("certificate_reference") != "fixed-C retained-key refit"
        or scope.get("behavioral_reference")
        != "fresh raw round-omitted repack"
        or scope.get("full_repack_certificate_claimed") is not False
    ):
        raise LabCaseError("recorded deletion-evidence provenance differs")

    record_tuple = tuple(records)
    return LabCase(
        record_id=case_record_id,
        request_text=request_text,
        records=record_tuple,
        public_records=public_records,
        server_catalog_hash=catalog_hash(record_tuple),
        committed_catalog_hash=committed_catalog_hash,
        fixture_selected_record_id=selected_record_id,
        fixture_explanation=decision["explanation"],
        payload_sha256=integrity["sha256"],
        method_report_sha256=method_report_sha256,
    )


def create_lab_app(
    *,
    mode: str = "recorded",
    resolver: MemoryResolver | None = None,
    proposal_store: ProposalStore | None = None,
    asset_path: str | Path = CASE_ASSET,
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    """Build the isolated same-origin lab app without running a server."""

    if mode not in {"recorded", "gemini"}:
        raise ValueError("lab mode must be 'recorded' or 'gemini'")
    lab_case = load_lab_case(asset_path)
    active_resolver = resolver or _resolver_for_mode(
        mode,
        lab_case,
        environ=os.environ if environ is None else environ,
    )
    if not active_resolver.available:
        raise LabStartupError("the selected resolver is unavailable")
    proposals = proposal_store or ProposalStore(
        ttl_seconds=90,
        max_proposals=64,
    )

    app = FastAPI(
        title="GemmaSV LongMemEval Lab Meeting Demo",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.lab_case = lab_case
    app.state.resolver = active_resolver
    app.state.proposals = proposals

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        return response

    @app.exception_handler(ProposalError)
    async def proposal_error(_request: Request, exc: ProposalError):
        if isinstance(exc, ProposalNotFound):
            status_code = 404
        elif isinstance(
            exc,
            (ProposalSelectionRequired, ProposalSelectionInvalid),
        ):
            status_code = 422
        elif isinstance(
            exc,
            (ProposalExpired, ProposalInvalidated, ProposalConsumed),
        ):
            status_code = 409
        else:
            status_code = 409
        return JSONResponse(
            {"detail": {"code": exc.code}},
            status_code=status_code,
        )

    @app.get("/healthz")
    async def health():
        return {
            "status": "ok",
            "resolver_mode": active_resolver.mode,
            "deletion_evidence": "recorded_replay",
        }

    @app.get("/api/lab/config")
    async def config():
        return {
            "schema": "gemma-sv-lab-meeting-config-v1",
            "case_record_id": lab_case.record_id,
            "request_text_sha256": request_hash(lab_case.request_text),
            "resolver": _resolver_metadata(active_resolver),
            "proposal_ttl_seconds": int(proposals.ttl_seconds),
            "server_catalog_hash": lab_case.server_catalog_hash,
            "committed_catalog_hash": lab_case.committed_catalog_hash,
            "deletion_evidence": lab_case.deletion_evidence(),
        }

    @app.post("/api/lab/resolve")
    async def resolve(body: ResolveRequest, response: Response):
        if not secrets.compare_digest(body.request_text, lab_case.request_text):
            raise HTTPException(
                status_code=422,
                detail={"code": "request_must_match_committed_case"},
            )

        session_id = _session_id_from_request_or_new(response=response)
        try:
            raw_decision = await asyncio.to_thread(
                active_resolver.resolve,
                body.request_text,
                lab_case.records,
            )
            decision = validate_resolution_payload(
                raw_decision,
                lab_case.records,
            )
        except ResolverValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "resolver_invalid_response",
                    "reason": exc.code,
                },
            ) from exc
        except ResolverUnavailable as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "resolver_provider_unavailable",
                    "reason": exc.code,
                },
            ) from exc
        except Exception as exc:
            # Never serialize provider exceptions; they can contain request details.
            raise HTTPException(
                status_code=502,
                detail={"code": "resolver_provider_failed"},
            ) from exc

        proposal = None
        if decision.status is ResolverStatus.RESOLVED:
            proposal = proposals.create(
                session_id=session_id,
                catalog_hash=lab_case.server_catalog_hash,
                selected_record_id=decision.selected_record_id,
                candidate_record_ids=(decision.selected_record_id,),
                status=decision.status,
                model_id=active_resolver.model_id,
                request_hash=request_hash(body.request_text),
            )

        return _public_resolution(
            lab_case,
            decision,
            active_resolver,
            proposal_id=proposal.proposal_id if proposal else None,
            proposal_ttl_seconds=(
                max(1, int(proposal.expires_at - proposal.created_at))
                if proposal
                else None
            ),
        )

    @app.post("/api/lab/confirm")
    async def confirm(body: ConfirmRequest, request: Request):
        session_id = _session_id_from_cookie(request)
        if session_id is None:
            raise ProposalNotFound(body.proposal_id)
        if not body.confirmed:
            proposals.cancel(body.proposal_id, session_id=session_id)
            return {
                "proposal_id": body.proposal_id,
                "confirmed": False,
                "selected_record_id": None,
                "deletion_executed": False,
                "deletion_evidence": None,
            }

        proposal, selected_record_id = proposals.claim(
            body.proposal_id,
            session_id=session_id,
            catalog_hash=lab_case.server_catalog_hash,
        )
        confirmed_record = lab_case.public_record(selected_record_id)
        return {
            "proposal_id": proposal.proposal_id,
            "confirmed": True,
            "selected_record_id": selected_record_id,
            "resolver_model_id": proposal.model_id,
            "confirmed_record": confirmed_record,
            # This app confirms routing only; it never executes a deletion model.
            "deletion_executed": False,
            "subsequent_deletion_evidence": lab_case.deletion_evidence(),
        }

    @app.get("/")
    async def replay_redirect():
        return RedirectResponse(
            url=f"{REPLAY_PATH}?{LAB_RESOLVER_QUERY}",
            status_code=307,
        )

    app.mount(
        "/",
        StaticFiles(directory=str(SITE_DIR), html=True),
        name="demo-site",
    )
    return app


def _resolver_for_mode(
    mode: str,
    lab_case: LabCase,
    *,
    environ: Mapping[str, str],
) -> MemoryResolver:
    if mode == "recorded":
        return RecordedResolver(
            lab_case.recorded_decision(),
            model_id=LAB_MODEL_ID,
        )

    api_key = next(
        (
            value.strip()
            for value in (
                environ.get("GEMINI_API_KEY"),
                environ.get("GOOGLE_API_KEY"),
            )
            if value and value.strip()
        ),
        None,
    )
    if api_key is None:
        raise LabStartupError(
            "gemini mode requires GEMINI_API_KEY or GOOGLE_API_KEY in the "
            "process environment or repository-root .env"
        )
    try:
        return GeminiResolver.from_api_key(api_key, model_id=LAB_MODEL_ID)
    except ResolverUnavailable as exc:
        if exc.code == "resolver_client_not_installed":
            message = (
                "gemini mode requires the optional google-genai dependency "
                "(install gemma_sv/requirements-demo.txt)"
            )
        else:
            message = f"cannot initialize Gemini resolver ({exc.code})"
        raise LabStartupError(message) from exc


def _public_resolution(
    lab_case: LabCase,
    decision: ResolverDecision,
    resolver: MemoryResolver,
    *,
    proposal_id: str | None,
    proposal_ttl_seconds: int | None,
) -> dict[str, Any]:
    selected = (
        lab_case.public_record(decision.selected_record_id)
        if decision.selected_record_id is not None
        else None
    )
    alternatives = [
        lab_case.public_record(record_id)
        for record_id in decision.alternative_record_ids
    ]
    payload = decision.model_dump(mode="json")
    payload.update(
        {
            "proposal_id": proposal_id,
            "proposal_expires_in_seconds": proposal_ttl_seconds,
            "selected_record": selected,
            "alternative_records": alternatives,
            "resolver": _resolver_metadata(
                resolver,
                request_performed=True,
            ),
            "deletion_evidence": lab_case.deletion_evidence(),
        }
    )
    return payload


def _resolver_metadata(
    resolver: MemoryResolver,
    *,
    request_performed: bool = False,
) -> dict[str, Any]:
    if resolver.evidence == "recorded_resolution":
        label = "Recorded resolver fixture — no provider call"
    elif resolver.mode == "gemini":
        label = "Live Gemini resolver outcome"
    else:
        label = "Injected test resolver outcome"
    return {
        "model_id": resolver.model_id,
        "mode": resolver.mode,
        "evidence": resolver.evidence,
        "label": label,
        "network_request_possible": resolver.network_requests,
        "network_request_performed": bool(
            request_performed and resolver.network_requests
        ),
        "outside_deletion_certificate": True,
    }


def _session_id_from_request_or_new(*, response: Response) -> str:
    # Resolve creates a fresh browser-bound proposal session. A prior cookie is
    # deliberately not reused, so a new resolution invalidates no other browser.
    session_id = f"lab_{secrets.token_urlsafe(24)}"
    response.set_cookie(
        LAB_SESSION_COOKIE,
        session_id,
        max_age=LAB_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )
    return session_id


def _session_id_from_cookie(request: Request) -> str | None:
    value = request.cookies.get(LAB_SESSION_COOKIE)
    if value is None or _SESSION_PATTERN.fullmatch(value) is None:
        return None
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabCaseError(f"{label} must be an object")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("recorded", "gemini"),
        default="recorded",
        help="recorded is network-free; gemini performs one live resolver call",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the replay URL in the default browser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.mode == "gemini":
        load_lab_environment(LAB_ENV_FILE)
    try:
        app = create_lab_app(mode=args.mode)
    except (LabCaseError, LabStartupError) as exc:
        raise SystemExit(f"Lab launcher refused to start: {exc}") from exc

    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = (
        f"http://{browser_host}:{args.port}{REPLAY_PATH}?"
        f"{LAB_RESOLVER_QUERY}"
    )
    print(f"GemmaSV lab meeting demo ({args.mode})", flush=True)
    print(f"Open: {url}", flush=True)
    if args.mode == "recorded":
        print(
            "Resolver: recorded fixture; no provider or model network call.",
            flush=True,
        )
    else:
        print(
            f"Resolver: live {LAB_MODEL_ID}; deletion evidence remains recorded.",
            flush=True,
        )
    if args.open:
        timer = threading.Timer(0.8, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
