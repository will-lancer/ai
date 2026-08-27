"""Fail-closed compute and paid-dispatch policy.

Local execution is limited to small CPU work.  GPU, open-weight, and training
intents require Colab.  Hosted model calls require a config-bound approval,
an explicit live flag, provider credentials, and a non-generative preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time
import tomllib
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_bytes, hash_config, strict_loads


LOCAL_MAX_CPU_CORES = 2
LOCAL_MAX_MEMORY_GIB = 4
LOCAL_MAX_WALL_SECONDS = 300
PAID_APPROVAL_SCHEMA = "proof-interface/paid-approval-v1"
SUPPORTED_CONFIG_SCHEMAS = {
    "proof-interface/config-v1",
    "lean-eval/config-v1",
}
PROVIDER_CREDENTIALS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}
PINNED_MODELS = {
    "openai": frozenset({"gpt-5.6-sol", "gpt-5.6-luna"}),
    "anthropic": frozenset({"claude-opus-5"}),
    "google": frozenset({"gemini-3.7-flash"}),
    "mock": frozenset({"mock"}),
}


class ComputePolicyError(RuntimeError):
    """Base policy failure."""


class ComputeGuardError(ComputePolicyError):
    """A compute intent violates the local/Colab boundary."""


class ApprovalError(ComputePolicyError):
    """A paid approval artifact is absent, invalid, stale, or exhausted."""


class DispatchDenied(ComputePolicyError):
    """A hosted model dispatch was denied before transport construction."""


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    floor = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ComputePolicyError(f"{name} must be an integer >= {floor}")
    return value


@dataclass(frozen=True, slots=True)
class ComputeIntent:
    operation: str
    environment: str = "local"
    cpu_cores: int = 1
    memory_gib: int = 1
    wall_seconds: int = 60
    gpu: bool = False
    open_weight: bool = False
    training: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ComputePolicyError("operation must be non-empty text")
        if self.environment not in {"local", "colab"}:
            raise ComputePolicyError("environment must be local or colab")
        _positive_int(self.cpu_cores, "cpu_cores")
        _positive_int(self.memory_gib, "memory_gib")
        _positive_int(self.wall_seconds, "wall_seconds")
        for name in ("gpu", "open_weight", "training"):
            if not isinstance(getattr(self, name), bool):
                raise ComputePolicyError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed: bool
    reason_code: str
    required_environment: str


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    local_cpu: bool = True
    local_gpu: bool = False
    local_open_weight: bool = False
    local_training: bool = False
    colab_required_for_accelerators: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "local_cpu": self.local_cpu,
            "local_gpu": self.local_gpu,
            "local_open_weight": self.local_open_weight,
            "local_training": self.local_training,
            "colab_required_for_accelerators": self.colab_required_for_accelerators,
        }


class ComputeGuard:
    """Validate every process intent before local or Colab execution."""

    def decide(self, intent: ComputeIntent) -> GuardDecision:
        if not isinstance(intent, ComputeIntent):
            raise TypeError("intent must be ComputeIntent")
        accelerator_work = intent.gpu or intent.open_weight or intent.training
        if accelerator_work and intent.environment != "colab":
            return GuardDecision(False, "colab_required", "colab")
        if intent.environment == "local":
            if intent.cpu_cores > LOCAL_MAX_CPU_CORES:
                return GuardDecision(False, "local_cpu_limit", "local")
            if intent.memory_gib > LOCAL_MAX_MEMORY_GIB:
                return GuardDecision(False, "local_memory_limit", "local")
            if intent.wall_seconds > LOCAL_MAX_WALL_SECONDS:
                return GuardDecision(False, "local_wall_limit", "local")
        return GuardDecision(True, "allowed", intent.environment)

    def require(self, intent: ComputeIntent) -> GuardDecision:
        decision = self.decide(intent)
        if not decision.allowed:
            raise ComputeGuardError(decision.reason_code)
        return decision

    @staticmethod
    def capabilities() -> CapabilityStatus:
        return CapabilityStatus()


validate_intent = ComputeGuard().decide


def require_intent(intent: ComputeIntent) -> GuardDecision:
    return ComputeGuard().require(intent)


def capability_status() -> CapabilityStatus:
    return CapabilityStatus()


def _load_mapping(value: Mapping[str, Any] | str | bytes | os.PathLike[str] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, os.PathLike):
        raw = Path(value).read_bytes()
        suffix = Path(value).suffix.lower()
    elif isinstance(value, str):
        candidate = Path(value)
        if "\n" not in value and candidate.is_file():
            raw = candidate.read_bytes()
            suffix = candidate.suffix.lower()
        else:
            raw = value.encode("utf-8", "strict")
            suffix = ""
    elif isinstance(value, bytes):
        raw = value
        suffix = ""
    else:
        raise ComputePolicyError("policy artifact must be a mapping, path, or bytes")
    try:
        decoded = tomllib.loads(raw.decode("utf-8", "strict")) if suffix == ".toml" else strict_loads(raw)
    except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ComputePolicyError("policy artifact is malformed") from exc
    if not isinstance(decoded, dict):
        raise ComputePolicyError("policy artifact must decode to an object")
    return decoded


def config_digest(config: Mapping[str, Any] | str | bytes | os.PathLike[str]) -> str:
    return hash_config(_load_mapping(config))


def _as_string_set(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ApprovalError(f"{name} must be an array of strings")
    result = frozenset(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ApprovalError(f"{name} must contain non-empty strings")
    return result


def _parse_expiry(value: Any) -> float:
    if isinstance(value, bool):
        raise ApprovalError("approval expiry is invalid")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        raise ApprovalError("approval expiry is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApprovalError("approval expiry is invalid") from exc
    if parsed.tzinfo is None:
        raise ApprovalError("approval expiry must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


@dataclass(frozen=True, slots=True)
class Approval:
    approval_id: str
    nonce: str
    config_hash: str
    providers: frozenset[str]
    models: frozenset[str]
    max_total_microdollars: int
    max_request_microdollars: int
    expires_at: float
    schema_version: str = PAID_APPROVAL_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Approval":
        if not isinstance(value, Mapping):
            raise ApprovalError("approval must be an object")
        if value.get("schema_version") != PAID_APPROVAL_SCHEMA:
            raise ApprovalError("approval schema is not supported")
        approval_id = value.get("approval_id")
        nonce = value.get("nonce")
        config_hash = value.get("config_hash")
        for name, item in (("approval_id", approval_id), ("nonce", nonce), ("config_hash", config_hash)):
            if not isinstance(item, str) or not item:
                raise ApprovalError(f"approval {name} is required")
        if len(config_hash) != 64 or any(char not in "0123456789abcdef" for char in config_hash):
            raise ApprovalError("approval config_hash must be lowercase SHA-256")
        total = _positive_int(value.get("max_total_microdollars"), "max_total_microdollars", allow_zero=True)
        request = _positive_int(value.get("max_request_microdollars"), "max_request_microdollars", allow_zero=True)
        if request > total:
            raise ApprovalError("per-request approval exceeds total approval")
        return cls(
            approval_id=approval_id,
            nonce=nonce,
            config_hash=config_hash,
            providers=_as_string_set(value.get("providers"), "providers"),
            models=_as_string_set(value.get("models"), "models"),
            max_total_microdollars=total,
            max_request_microdollars=request,
            expires_at=_parse_expiry(value.get("expires_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "approval_id": self.approval_id,
            "nonce": self.nonce,
            "config_hash": self.config_hash,
            "providers": sorted(self.providers),
            "models": sorted(self.models),
            "max_total_microdollars": self.max_total_microdollars,
            "max_request_microdollars": self.max_request_microdollars,
            "expires_at": self.expires_at,
        }


def load_approval(value: Approval | Mapping[str, Any] | str | bytes | os.PathLike[str] | None) -> Approval | None:
    if value is None:
        return None
    if isinstance(value, Approval):
        return value
    return Approval.from_mapping(_load_mapping(value))


class NonceRegistry:
    """Process-local replay guard shared by paid gates."""

    def __init__(self) -> None:
        self._claimed: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, nonce: str) -> None:
        with self._lock:
            if nonce in self._claimed:
                raise ApprovalError("approval nonce was already claimed")
            self._claimed.add(nonce)


_GLOBAL_NONCES = NonceRegistry()


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    allowed: bool
    reason_code: str
    provider: str
    model_id: str
    estimated_microdollars: int
    remaining_microdollars: int


class PaidDispatchGate:
    """Authorize one hosted generation before a transport can be constructed."""

    def __init__(
        self,
        config: Mapping[str, Any] | str | bytes | os.PathLike[str] | None,
        *,
        live: bool = False,
        cli_live: bool = False,
        approval: Approval | Mapping[str, Any] | str | bytes | os.PathLike[str] | None = None,
        credentials: Mapping[str, Any] | Sequence[str] | bool | None = None,
        preflight_ok: bool = False,
        preflight: Callable[[str, str], bool] | None = None,
        clock: Callable[[], float] = time.time,
        nonce_registry: NonceRegistry | None = None,
    ) -> None:
        self.config = _load_mapping(config)
        self.live = live is True
        self.cli_live = cli_live is True
        self.approval = load_approval(approval)
        self.credentials = credentials
        self.preflight_ok = preflight_ok is True
        self.preflight = preflight
        self.clock = clock
        self.nonce_registry = nonce_registry or _GLOBAL_NONCES
        self._spent = 0
        self._nonce_claimed = False
        self._lock = threading.Lock()

    @property
    def spent_microdollars(self) -> int:
        return self._spent

    def _provider_models(self, provider: str) -> frozenset[str]:
        configured = self.config.get("models", ())
        models: set[str] = set()
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes, bytearray)):
            for item in configured:
                if isinstance(item, str):
                    models.add(item)
                elif isinstance(item, Mapping):
                    item_provider = item.get("provider")
                    item_model = item.get("model_id", item.get("model"))
                    if item_provider == provider and isinstance(item_model, str):
                        models.add(item_model)
        return frozenset(models)

    def _has_credential(self, provider: str) -> bool:
        expected = PROVIDER_CREDENTIALS.get(provider)
        if expected is None:
            return False
        value = self.credentials
        if isinstance(value, Mapping):
            candidate = value.get(expected)
            return isinstance(candidate, str) and bool(candidate)
        return False

    def _deny(self, reason: str, provider: str, model_id: str, cost: int) -> DispatchDenied:
        return DispatchDenied(f"{reason}: {provider}/{model_id} ({cost} microdollars)")

    def static_authorize(
        self,
        provider: str,
        *,
        model_id: str,
        estimated_microdollars: int,
    ) -> DispatchDecision:
        cost = _positive_int(estimated_microdollars, "estimated_microdollars", allow_zero=True)
        if provider == "mock":
            return DispatchDecision(True, "mock_offline", provider, model_id, cost, 0)
        if provider not in PROVIDER_CREDENTIALS or model_id not in PINNED_MODELS.get(provider, ()):
            raise self._deny("provider_or_model_not_pinned", provider, model_id, cost)
        if not self.live or not self.cli_live or self.config.get("live") is not True:
            raise self._deny("live_flags_required", provider, model_id, cost)
        if self.config.get("schema_version") not in SUPPORTED_CONFIG_SCHEMAS:
            raise self._deny("config_schema_invalid", provider, model_id, cost)
        configured = self._provider_models(provider)
        if model_id not in configured:
            raise self._deny("model_not_in_config", provider, model_id, cost)
        approval = self.approval
        if approval is None:
            raise self._deny("approval_required", provider, model_id, cost)
        if approval.config_hash != hash_config(self.config):
            raise self._deny("approval_config_mismatch", provider, model_id, cost)
        if self.clock() >= approval.expires_at:
            raise self._deny("approval_expired", provider, model_id, cost)
        if provider not in approval.providers or model_id not in approval.models:
            raise self._deny("approval_scope_mismatch", provider, model_id, cost)
        if cost > approval.max_request_microdollars:
            raise self._deny("request_budget_exceeded", provider, model_id, cost)
        if self._spent + cost > approval.max_total_microdollars:
            raise self._deny("total_budget_exceeded", provider, model_id, cost)
        if not self._has_credential(provider):
            raise self._deny("provider_credential_missing", provider, model_id, cost)
        remaining = approval.max_total_microdollars - self._spent - cost
        return DispatchDecision(True, "authorized", provider, model_id, cost, remaining)

    def reserve(
        self,
        provider: str,
        *,
        model_id: str,
        estimated_microdollars: int,
    ) -> DispatchDecision:
        with self._lock:
            decision = self.static_authorize(
                provider,
                model_id=model_id,
                estimated_microdollars=estimated_microdollars,
            )
            if provider == "mock":
                return decision
            approval = self.approval
            assert approval is not None
            if not self._nonce_claimed:
                self.nonce_registry.claim(approval.nonce)
                self._nonce_claimed = True
            self._spent += estimated_microdollars
            return DispatchDecision(
                True,
                "reserved",
                provider,
                model_id,
                estimated_microdollars,
                approval.max_total_microdollars - self._spent,
            )

    def dispatch_factory(
        self,
        provider: str,
        callback_factory: Callable[[], Callable[[], Any]],
        *,
        model_id: str,
        estimated_microdollars: int,
    ) -> Any:
        if not callable(callback_factory):
            raise TypeError("callback_factory must be callable")
        self.static_authorize(
            provider,
            model_id=model_id,
            estimated_microdollars=estimated_microdollars,
        )
        if provider != "mock":
            if self.preflight is not None:
                try:
                    ok = self.preflight(provider, model_id)
                except Exception as exc:
                    raise self._deny("preflight_failed", provider, model_id, estimated_microdollars) from exc
                if ok is not True:
                    raise self._deny("preflight_failed", provider, model_id, estimated_microdollars)
            elif not self.preflight_ok:
                raise self._deny("preflight_required", provider, model_id, estimated_microdollars)
        # Recheck expiry and budget after preflight, then consume the budget.
        self.reserve(
            provider,
            model_id=model_id,
            estimated_microdollars=estimated_microdollars,
        )
        callback = callback_factory()
        if not callable(callback):
            raise DispatchDenied("callback factory did not return a callable")
        return callback()


def estimate_spend(requests: int, microdollars_per_request: int) -> int:
    return _positive_int(requests, "requests", allow_zero=True) * _positive_int(
        microdollars_per_request,
        "microdollars_per_request",
        allow_zero=True,
    )


__all__ = [
    "Approval",
    "ApprovalError",
    "CapabilityStatus",
    "ComputeGuard",
    "ComputeGuardError",
    "ComputeIntent",
    "ComputePolicyError",
    "DispatchDecision",
    "DispatchDenied",
    "GuardDecision",
    "LOCAL_MAX_CPU_CORES",
    "LOCAL_MAX_MEMORY_GIB",
    "LOCAL_MAX_WALL_SECONDS",
    "NonceRegistry",
    "PINNED_MODELS",
    "PROVIDER_CREDENTIALS",
    "PaidDispatchGate",
    "capability_status",
    "config_digest",
    "estimate_spend",
    "load_approval",
    "require_intent",
    "validate_intent",
]
