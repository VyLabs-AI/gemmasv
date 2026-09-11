"""Environment-only configuration; no credentials are shipped to GitHub Pages."""

from __future__ import annotations

from dataclasses import dataclass, field
import os


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DemoSettings:
    engine: str = "replay"
    review_mode: bool = True
    allowed_origins: tuple[str, ...] = (
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )
    session_ttl_seconds: int = 30 * 60
    max_sessions: int = 128
    certificate_workers: int = 1
    max_certificate_jobs: int = 16
    requests_per_minute: int = 30
    max_concurrent_inference: int = 1
    resolver_mode: str = "disabled"
    resolver_model: str = "gemini-3.6-flash"
    resolver_api_key: str | None = field(default=None, repr=False, compare=False)
    resolver_recorded_json: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    resolver_proposal_ttl_seconds: int = 2 * 60
    max_resolver_proposals: int = 512
    max_concurrent_resolver_requests: int = 4

    def __post_init__(self) -> None:
        if self.engine not in {"replay", "gemma"}:
            raise ValueError("engine must be 'replay' or 'gemma'")
        if self.resolver_mode not in {"disabled", "gemini", "recorded"}:
            raise ValueError(
                "resolver_mode must be 'disabled', 'gemini', or 'recorded'"
            )
        if (
            not self.resolver_model
            or len(self.resolver_model) > 200
            or any(character.isspace() for character in self.resolver_model)
        ):
            raise ValueError("resolver_model must be a non-empty model identifier")
        positive_values = {
            "session_ttl_seconds": self.session_ttl_seconds,
            "max_sessions": self.max_sessions,
            "certificate_workers": self.certificate_workers,
            "max_certificate_jobs": self.max_certificate_jobs,
            "requests_per_minute": self.requests_per_minute,
            "max_concurrent_inference": self.max_concurrent_inference,
            "resolver_proposal_ttl_seconds": self.resolver_proposal_ttl_seconds,
            "max_resolver_proposals": self.max_resolver_proposals,
            "max_concurrent_resolver_requests": (
                self.max_concurrent_resolver_requests
            ),
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"{invalid[0]} must be positive")

    @classmethod
    def from_environment(cls) -> "DemoSettings":
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv(
                "HERO_ALLOWED_ORIGINS",
                "http://localhost:8000,http://127.0.0.1:8000",
            ).split(",")
            if origin.strip()
        )
        engine = os.getenv("HERO_ENGINE", "replay").strip().lower()
        if engine not in {"replay", "gemma"}:
            raise ValueError("HERO_ENGINE must be 'replay' or 'gemma'")
        resolver_mode = os.getenv("HERO_RESOLVER_MODE", "disabled").strip().lower()
        if resolver_mode not in {"disabled", "gemini", "recorded"}:
            raise ValueError(
                "HERO_RESOLVER_MODE must be 'disabled', 'gemini', or 'recorded'"
            )
        if not origins:
            raise ValueError("HERO_ALLOWED_ORIGINS must contain at least one origin")
        resolver_api_key = next(
            (
                value.strip()
                for value in (
                    os.getenv("GEMINI_API_KEY"),
                    os.getenv("GOOGLE_API_KEY"),
                )
                if value and value.strip()
            ),
            None,
        )
        return cls(
            engine=engine,
            review_mode=_bool("HERO_REVIEW_MODE", True),
            allowed_origins=origins,
            session_ttl_seconds=int(os.getenv("HERO_SESSION_TTL_SECONDS", "1800")),
            max_sessions=int(os.getenv("HERO_MAX_SESSIONS", "128")),
            certificate_workers=int(os.getenv("HERO_CERTIFICATE_WORKERS", "1")),
            max_certificate_jobs=int(os.getenv("HERO_MAX_CERTIFICATE_JOBS", "16")),
            requests_per_minute=int(os.getenv("HERO_REQUESTS_PER_MINUTE", "30")),
            max_concurrent_inference=int(
                os.getenv("HERO_MAX_CONCURRENT_INFERENCE", "1")
            ),
            resolver_mode=resolver_mode,
            resolver_model=os.getenv(
                "HERO_RESOLVER_MODEL",
                "gemini-3.6-flash",
            ).strip(),
            resolver_api_key=resolver_api_key,
            resolver_recorded_json=os.getenv("HERO_RESOLVER_RECORDED_JSON"),
            resolver_proposal_ttl_seconds=int(
                os.getenv("HERO_RESOLVER_PROPOSAL_TTL_SECONDS", "120")
            ),
            max_resolver_proposals=int(
                os.getenv("HERO_MAX_RESOLVER_PROPOSALS", "512")
            ),
            max_concurrent_resolver_requests=int(
                os.getenv("HERO_MAX_CONCURRENT_RESOLVER_REQUESTS", "4")
            ),
        )
