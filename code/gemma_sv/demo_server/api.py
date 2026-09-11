"""FastAPI service for the hosted hero demo."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import secrets
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .engine import DemoEngine, EngineError, ReplayDemoEngine, preset_payloads
from .jobs import JobNotFound, JobQueue, JobRecord
from .rate_limit import SlidingWindowLimiter
from .resolver import (
    CatalogTurn,
    GeminiResolver,
    MemoryCatalogRecord,
    MemoryOwnershipReceipt,
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
    ResolverProposal,
    ResolverStatus,
    ResolverUnavailable,
    ResolverValidationError,
    UnavailableResolver,
    catalog_hash,
    request_hash,
    unavailable_decision,
    validate_resolution_payload,
)
from .settings import DemoSettings
from .span import (
    CharacterRange,
    SelectedSpan,
    default_audit_probe,
    validate_deletion_ranges,
)
from .state import SessionNotFound, SessionRecord, SessionStore


class CreateSessionRequest(BaseModel):
    domain: Literal["cybersecurity", "medicine", "tofu", "custom"] = "cybersecurity"


class DeletionRangeRequest(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class IngestRequest(BaseModel):
    memory_text: str = Field(min_length=1, max_length=2_000)
    secret_start: int | None = Field(default=None, ge=0)
    secret_end: int | None = Field(default=None, gt=0)
    deletion_ranges: list[DeletionRangeRequest] | None = Field(
        default=None,
        max_length=64,
    )
    audit_probe: str | None = Field(default=None, max_length=1_000)
    audit_target: str | None = Field(default=None, min_length=1, max_length=240)
    delete_scope: Literal["span", "field", "record"] = "span"
    record_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    memory_label: str | None = Field(default=None, min_length=1, max_length=160)
    memory_summary: str | None = Field(
        default=None,
        min_length=1,
        max_length=1_600,
    )
    source_turn_ids: list[str] | None = Field(default=None, max_length=64)
    owned_exchange: list[CatalogTurn] | None = Field(default=None, max_length=16)


class ResolveMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_text: str = Field(min_length=1, max_length=1_000)

    @field_validator("request_text")
    @classmethod
    def request_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_text must not be blank")
        return value


class ConfirmMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(
        min_length=10,
        max_length=128,
        pattern=r"^proposal_[A-Za-z0-9_-]+$",
    )
    confirmed: bool
    selected_record_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )


class AttackRequest(BaseModel):
    method: Literal["exact", "icul", "decay", "never"]
    kind: Literal["extraction", "elicitation", "freeform"] = "extraction"
    budget: Literal[1, 2, 4, 8] | None = None
    prompt: str | None = Field(default=None, max_length=1_000)


class TwinGuessRequest(BaseModel):
    pane: Literal["A", "B"]


class AppContext:
    def __init__(
        self,
        settings: DemoSettings,
        engine: DemoEngine,
        resolver: MemoryResolver,
        *,
        proposal_store: ProposalStore | None = None,
    ):
        self.settings = settings
        self.engine = engine
        self.resolver = resolver
        self.jobs = JobQueue(
            max_workers=settings.certificate_workers,
            max_pending=settings.max_certificate_jobs,
        )
        self.proposals = proposal_store or ProposalStore(
            ttl_seconds=settings.resolver_proposal_ttl_seconds,
            max_proposals=settings.max_resolver_proposals,
        )
        self.session_locks: dict[str, asyncio.Lock] = {}
        self.memory_catalogs: dict[
            str,
            dict[str, MemoryCatalogRecord],
        ] = {}
        self.ownership_receipts: dict[
            str,
            dict[str, MemoryOwnershipReceipt],
        ] = {}
        self.active_memory_record_ids: dict[str, str] = {}
        self.sessions = SessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
            on_remove=self._remove_session_state,
        )
        self.inference_slots = asyncio.Semaphore(settings.max_concurrent_inference)
        self.resolver_slots = asyncio.Semaphore(
            settings.max_concurrent_resolver_requests
        )
        self.limiter = SlidingWindowLimiter(limit=settings.requests_per_minute)

    async def run_inference(self, fn, /, *args, **kwargs):
        async with self.inference_slots:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def run_resolver(self, request_text, candidates):
        async with self.resolver_slots:
            return await asyncio.to_thread(
                self.resolver.resolve,
                request_text,
                candidates,
            )

    def session_lock(self, session_id: str) -> asyncio.Lock:
        return self.session_locks.setdefault(session_id, asyncio.Lock())

    def replace_memory_record(
        self,
        session_id: str,
        record: MemoryCatalogRecord,
        receipt: MemoryOwnershipReceipt,
    ) -> None:
        if not receipt.authorizes(session_id, record.record_id):
            raise ValueError("ownership receipt does not bind this session record")
        self.memory_catalogs[session_id] = {record.record_id: record}
        self.ownership_receipts[session_id] = {record.record_id: receipt}
        self.active_memory_record_ids[session_id] = record.record_id

    def memory_catalog(
        self,
        session_id: str,
    ) -> dict[str, MemoryCatalogRecord]:
        return self.memory_catalogs.get(session_id, {})

    def ownership_receipt(
        self,
        session_id: str,
        record_id: str,
    ) -> MemoryOwnershipReceipt | None:
        return self.ownership_receipts.get(session_id, {}).get(record_id)

    def active_memory_record_id(self, session_id: str) -> str | None:
        return self.active_memory_record_ids.get(session_id)

    def _remove_session_state(self, record: SessionRecord) -> None:
        self.jobs.delete_for_session(record.session_id)
        self.proposals.delete_for_session(record.session_id)
        self.session_locks.pop(record.session_id, None)
        self.memory_catalogs.pop(record.session_id, None)
        self.ownership_receipts.pop(record.session_id, None)
        self.active_memory_record_ids.pop(record.session_id, None)


def create_app(
    *,
    settings: DemoSettings | None = None,
    engine: DemoEngine | None = None,
    resolver: MemoryResolver | None = None,
    proposal_store: ProposalStore | None = None,
) -> FastAPI:
    settings = settings or DemoSettings.from_environment()
    if engine is None:
        if settings.engine == "gemma":
            from .gemma_engine import GemmaDemoEngine

            engine = GemmaDemoEngine.from_environment()
        else:
            engine = ReplayDemoEngine()
    if resolver is None:
        resolver = _resolver_from_settings(settings)
    context = AppContext(
        settings,
        engine,
        resolver,
        proposal_store=proposal_store,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.demo = context
        yield
        context.jobs.shutdown(wait=False)

    app = FastAPI(
        title="Hosted Hero Demo API",
        version="1.0.0",
        docs_url=None if settings.review_mode else "/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.demo = context
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Demo-Session-ID"],
        max_age=600,
    )

    @app.middleware("http")
    async def security_and_rate_limit(request: Request, call_next):
        # Job-status polling is exempt: a ~60 s certificate polled every second
        # would otherwise exhaust the per-minute budget mid-job.
        is_job_poll = request.method == "GET" and request.url.path.startswith(
            "/api/v1/jobs/"
        )
        if (
            request.method != "OPTIONS"
            and not is_job_poll
            and request.url.path.startswith("/api/")
        ):
            client = request.client.host if request.client else "unknown"
            allowed, retry_after = context.limiter.allow(client)
            if not allowed:
                return JSONResponse(
                    {"detail": {"code": "rate_limited"}},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        return response

    @app.exception_handler(SessionNotFound)
    async def session_not_found(_request: Request, _exc: SessionNotFound):
        return JSONResponse(
            {"detail": {"code": "session_not_found"}},
            status_code=404,
        )

    @app.exception_handler(JobNotFound)
    async def job_not_found(_request: Request, _exc: JobNotFound):
        return JSONResponse(
            {"detail": {"code": "job_not_found"}},
            status_code=404,
        )

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

    @app.exception_handler(EngineError)
    async def engine_error(_request: Request, exc: EngineError):
        return JSONResponse(
            {"detail": {"code": str(exc)}},
            status_code=409,
        )

    @app.get("/healthz")
    async def health():
        return {
            "status": "ok",
            "engine": context.engine.name,
            "live_compute": context.engine.live_compute,
        }

    @app.get("/api/v1/config")
    async def public_config():
        return {
            "engine": context.engine.name,
            "model_id": getattr(context.engine, "model_id", None),
            "live_compute": context.engine.live_compute,
            "review_mode": settings.review_mode,
            "certificate_scope": "registered_audit_probe_next_token_distribution",
            "behavioral_score": "teacher_forced_answer_span",
            "resolver": {
                "mode": context.resolver.mode,
                "model_id": context.resolver.model_id,
                "configured": context.resolver.available,
                "evidence": context.resolver.evidence,
                "network_requests_possible": context.resolver.network_requests,
                "requires_confirmation": True,
                "proposal_ttl_seconds": int(context.proposals.ttl_seconds),
            },
            "presets": preset_payloads(),
            "limits": {
                "session_ttl_seconds": settings.session_ttl_seconds,
                "max_fact_characters": 2_000,
                "max_selected_characters": 240,
                "max_record_characters": 1_600,
                "max_deletion_ranges": 64,
            },
            "privacy": {
                "ephemeral": True,
                "request_body_logging": False,
                "real_secrets_allowed": False,
            },
        }

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(body: CreateSessionRequest):
        try:
            session = context.sessions.create(domain=body.domain)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail={"code": "session_capacity_reached"},
                headers={"Retry-After": "30"},
            )
        return {
            "session_id": session.session_id,
            "domain": session.domain,
            "expires_in_seconds": settings.session_ttl_seconds,
        }

    @app.post("/api/v1/sessions/{session_id}/ingest")
    async def ingest(session_id: str, body: IngestRequest):
        session = context.sessions.get(session_id)
        deletion_scope = "field" if body.delete_scope == "span" else body.delete_scope
        ranges = body.deletion_ranges
        if ranges is None:
            if body.secret_start is None or body.secret_end is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_deletion_ranges",
                        "message": (
                            "provide deletion_ranges or both secret_start and secret_end"
                        ),
                    },
                )
            raw_ranges = (CharacterRange(body.secret_start, body.secret_end),)
        else:
            raw_ranges = tuple(CharacterRange(item.start, item.end) for item in ranges)
        try:
            selection = validate_deletion_ranges(
                body.memory_text,
                raw_ranges,
                max_selected_chars=(
                    1_600 if deletion_scope == "record" else 240
                ),
                deletion_scope=deletion_scope,
                record_id=body.record_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_deletion_ranges", "message": str(exc)},
            )
        probe = body.audit_probe
        if probe is None or not probe.strip():
            probe = default_audit_probe(session.domain, selection.value)
        target = (body.audit_target or selection.value).strip()
        if not selection.contains_value(target):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "audit_target_outside_deletion",
                    "message": "audit_target must occur inside the selected deletion scope",
                },
            )
        try:
            catalog_record, ownership_receipt = _build_catalog_record(
                session,
                body,
                selection,
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_memory_catalog"},
            )
        async with context.session_lock(session_id):
            result = await context.run_inference(
                context.engine.ingest, session, selection, probe, target
            )
            context.replace_memory_record(
                session_id,
                catalog_record,
                ownership_receipt,
            )
        return {
            **result,
            "resolver_record_id": catalog_record.record_id,
        }

    @app.post("/api/v1/sessions/{session_id}/recall")
    async def recall(session_id: str):
        session = context.sessions.get(session_id)
        async with context.session_lock(session_id):
            return await context.run_inference(context.engine.recall, session)

    @app.post("/api/v1/sessions/{session_id}/memory/resolve")
    async def resolve_memory(session_id: str, body: ResolveMemoryRequest):
        session = context.sessions.get(session_id)
        async with context.session_lock(session_id):
            candidates = tuple(context.memory_catalog(session_id).values())
            resolved_catalog_hash = catalog_hash(candidates)

        resolver_failed = False
        try:
            raw_decision = await context.run_resolver(
                body.request_text,
                candidates,
            )
            decision = validate_resolution_payload(raw_decision, candidates)
        except ResolverValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "resolver_invalid_response",
                    "reason": exc.code,
                },
            )
        except ResolverUnavailable:
            resolver_failed = True
            decision = unavailable_decision()
        except Exception:
            # Provider exception text can contain request or credential details.
            # Return a fixed safe outcome and never log or serialize the exception.
            resolver_failed = True
            decision = unavailable_decision()

        async with context.session_lock(session_id):
            session = context.sessions.get(session_id)
            current_candidates = tuple(
                context.memory_catalog(session_id).values()
            )
            if catalog_hash(current_candidates) != resolved_catalog_hash:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "resolver_catalog_changed"},
                )
            proposal = None
            if decision.status is ResolverStatus.RESOLVED:
                proposal_ids = (decision.selected_record_id,)
            elif decision.status is ResolverStatus.AMBIGUOUS:
                proposal_ids = decision.alternative_record_ids
            else:
                proposal_ids = ()
            if proposal_ids:
                by_id = {
                    candidate.record_id: candidate
                    for candidate in current_candidates
                }
                if any(
                    not by_id[record_id].owned_exchange
                    for record_id in proposal_ids
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "confirmation_context_unavailable"},
                    )
                try:
                    proposal = context.proposals.create(
                        session_id=session_id,
                        catalog_hash=resolved_catalog_hash,
                        selected_record_id=decision.selected_record_id,
                        candidate_record_ids=proposal_ids,
                        status=decision.status,
                        model_id=context.resolver.model_id,
                        request_hash=request_hash(body.request_text),
                    )
                except RuntimeError:
                    raise HTTPException(
                        status_code=503,
                        detail={"code": "resolver_proposal_capacity_reached"},
                        headers={"Retry-After": "30"},
                    )

        return _public_resolution(
            decision,
            candidates,
            proposal=proposal,
            resolver=context.resolver,
            resolver_failed=resolver_failed,
        )

    @app.post("/api/v1/sessions/{session_id}/memory/confirm")
    async def confirm_memory(session_id: str, body: ConfirmMemoryRequest):
        session = context.sessions.get(session_id)
        if not body.confirmed:
            context.proposals.cancel(
                body.proposal_id,
                session_id=session_id,
            )
            return {
                "proposal_id": body.proposal_id,
                "confirmed": False,
                "selected_record_id": None,
                "deletion_executed": False,
            }

        async with context.session_lock(session_id):
            candidates = tuple(context.memory_catalog(session_id).values())
            proposal, selected_record_id = context.proposals.claim(
                body.proposal_id,
                session_id=session_id,
                catalog_hash=catalog_hash(candidates),
                selected_record_id=body.selected_record_id,
            )
            record = context.memory_catalog(session_id).get(selected_record_id)
            receipt = context.ownership_receipt(
                session_id,
                selected_record_id,
            )
            if (
                record is None
                or receipt is None
                or not receipt.authorizes(session_id, selected_record_id)
                or context.active_memory_record_id(session_id)
                != selected_record_id
            ):
                raise HTTPException(
                    status_code=403,
                    detail={"code": "unauthorized_memory_record"},
                )
            deletion = await _execute_id_based_forget(
                context,
                session,
                selected_record_id,
            )
        return {
            "proposal_id": proposal.proposal_id,
            "confirmed": True,
            "selected_record_id": selected_record_id,
            "resolver_model_id": proposal.model_id,
            "confirmed_record": record.public_dict(),
            "deletion_executed": True,
            **deletion,
        }

    @app.post("/api/v1/sessions/{session_id}/forget")
    async def forget(session_id: str):
        session = context.sessions.get(session_id)
        async with context.session_lock(session_id):
            return await _execute_id_based_forget(
                context,
                session,
                context.active_memory_record_id(session_id),
            )

    @app.get("/api/v1/jobs/{job_id}")
    async def job_status(
        job_id: str,
        session_id: Annotated[str, Header(alias="X-Demo-Session-ID")],
    ):
        context.sessions.get(session_id)
        return _public_job(context.jobs.get(job_id, session_id=session_id))

    @app.post("/api/v1/sessions/{session_id}/attacks")
    async def attack(session_id: str, body: AttackRequest):
        session = context.sessions.get(session_id)
        # Do not hold the per-session lock during a certificate job: engine state used
        # by attacks is immutable after ingest and the fast/certificate models differ.
        return await context.run_inference(
            context.engine.attack,
            session,
            method=body.method,
            kind=body.kind,
            budget=body.budget,
            prompt=body.prompt,
        )

    @app.post("/api/v1/sessions/{session_id}/twin")
    async def twin_start(session_id: str):
        session = context.sessions.get(session_id)
        async with context.session_lock(session_id):
            return await context.run_inference(context.engine.twin_start, session)

    @app.post("/api/v1/sessions/{session_id}/twin/guess")
    async def twin_guess(session_id: str, body: TwinGuessRequest):
        session = context.sessions.get(session_id)
        async with context.session_lock(session_id):
            return context.engine.twin_guess(session, body.pane)

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        context.jobs.delete_for_session(session_id)
        if not context.sessions.delete(session_id):
            raise SessionNotFound(session_id)
        context.session_locks.pop(session_id, None)
        return Response(status_code=204)

    return app


def _resolver_from_settings(settings: DemoSettings) -> MemoryResolver:
    if settings.resolver_mode == "disabled":
        return UnavailableResolver(
            model_id=settings.resolver_model,
            reason="resolver_disabled",
            mode="disabled",
        )
    if settings.resolver_mode == "recorded":
        if not settings.resolver_recorded_json:
            return UnavailableResolver(
                model_id=settings.resolver_model,
                reason="recorded_resolution_missing",
                mode="recorded",
            )
        return RecordedResolver(
            settings.resolver_recorded_json,
            model_id=settings.resolver_model,
        )
    if not settings.resolver_api_key:
        return UnavailableResolver(
            model_id=settings.resolver_model,
            reason="resolver_api_key_missing",
            mode="gemini",
        )
    try:
        return GeminiResolver.from_api_key(
            settings.resolver_api_key,
            model_id=settings.resolver_model,
        )
    except ResolverUnavailable as exc:
        return UnavailableResolver(
            model_id=settings.resolver_model,
            reason=exc.code,
            mode="gemini",
        )


def _build_catalog_record(
    session: SessionRecord,
    body: IngestRequest,
    selection: SelectedSpan,
) -> tuple[MemoryCatalogRecord, MemoryOwnershipReceipt]:
    record_id = body.record_id or f"memory_{secrets.token_urlsafe(12)}"
    owned_exchange = tuple(body.owned_exchange or ())
    source_turn_ids = (
        tuple(body.source_turn_ids)
        if body.source_turn_ids is not None
        else tuple(turn.turn_id for turn in owned_exchange)
    )
    turn_cursor = 0
    for turn in owned_exchange:
        turn_start = selection.text.find(turn.text, turn_cursor)
        if turn_start < 0:
            raise ValueError("owned exchange is not contained in memory text")
        turn_cursor = turn_start + len(turn.text)
    selected_text = " / ".join(value.strip() for value in selection.values)
    label = body.memory_label or f"Saved memory: {selected_text}"
    default_summary = (
        selection.text
        if selection.deletion_scope == "record"
        else selected_text
    )
    summary = body.memory_summary or default_summary
    record = MemoryCatalogRecord(
        record_id=record_id,
        label=_truncate_visible(label, 160),
        summary=_truncate_visible(summary, 1_600),
        source_turn_ids=source_turn_ids,
        deletion_scope=selection.deletion_scope,
        owned_exchange=owned_exchange,
    )
    source_texts = (
        tuple(turn.text for turn in owned_exchange)
        if owned_exchange
        else (selection.text,)
    )
    receipt = MemoryOwnershipReceipt(
        session_id=session.session_id,
        record_id=record.record_id,
        owned_ranges=tuple((item.start, item.end) for item in selection.ranges),
        source_message_hashes=tuple(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in source_texts
        ),
    )
    return record, receipt


def _truncate_visible(value: str, max_length: int) -> str:
    value = value.strip()
    if not value:
        raise ValueError("catalog text must not be blank")
    return value[:max_length]


def _public_resolution(
    decision: ResolverDecision,
    candidates: tuple[MemoryCatalogRecord, ...],
    *,
    proposal: ResolverProposal | None,
    resolver: MemoryResolver,
    resolver_failed: bool,
) -> dict[str, Any]:
    by_id = {candidate.record_id: candidate for candidate in candidates}
    selected = (
        by_id[decision.selected_record_id].public_dict()
        if decision.selected_record_id is not None
        else None
    )
    alternatives = [
        by_id[record_id].public_dict()
        for record_id in decision.alternative_record_ids
    ]
    if resolver.evidence == "recorded_resolution":
        source_label = "recorded Gemini resolver outcome"
    elif decision.status is ResolverStatus.UNAVAILABLE:
        source_label = "resolver unavailable"
    elif resolver.mode == "gemini":
        source_label = "live Gemini resolver outcome"
    else:
        source_label = "server-side resolver outcome"
    payload = decision.model_dump(mode="json")
    payload.update(
        {
            "proposal_id": proposal.proposal_id if proposal else None,
            "proposal_expires_in_seconds": (
                max(1, int(proposal.expires_at - proposal.created_at))
                if proposal
                else None
            ),
            "selected_record": selected,
            "alternative_records": alternatives,
            "resolver": {
                "model_id": resolver.model_id,
                "mode": resolver.mode,
                "evidence": resolver.evidence,
                "label": source_label,
                "network_request_performed": bool(
                    resolver.network_requests and candidates
                ),
                "service_failed": resolver_failed,
            },
        }
    )
    return payload


async def _execute_id_based_forget(
    context: AppContext,
    session: SessionRecord,
    record_id: str | None,
) -> dict[str, Any]:
    """Route a confirmed catalog ID through the pre-existing engine operation."""

    if record_id is not None:
        receipt = context.ownership_receipt(session.session_id, record_id)
        if (
            receipt is None
            or not receipt.authorizes(session.session_id, record_id)
            or context.active_memory_record_id(session.session_id) != record_id
        ):
            raise HTTPException(
                status_code=403,
                detail={"code": "unauthorized_memory_record"},
            )
    result = await context.run_inference(context.engine.forget, session)
    try:
        job = context.jobs.submit_with_progress(
            session.session_id,
            _certificate_job,
            context.engine,
            session,
        )
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail={"code": "certificate_queue_full"},
            headers={"Retry-After": "30"},
        )
    return {"forget": result, "certificate_job": _public_job(job)}


def _certificate_job(progress, engine: DemoEngine, session: SessionRecord):
    return engine.certify(session, progress=progress)


def _public_job(job: JobRecord) -> dict:
    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "error_code": job.error_code,
    }
    if job.result is not None:
        payload["result"] = job.result
    return payload


app = create_app()
