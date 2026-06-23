"""Reusable PII/PHI de-identification engine: one shared model, thin wrappers.

This is the core of the app, independent of any web framework. It holds a single
:class:`openmed.ModelLoader` and reuses it across every call (the documented best
practice) so the ~44M-parameter PII model loads at most once per process.

The OpenMed import is deferred to first use, so importing this module — and the
service's ``/health`` endpoint — never pulls in Torch/Transformers or downloads a
model.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

# pysbd (a transitive openmed dependency) raises harmless SyntaxWarnings from its
# regex literals on Python >=3.12. Silence them before openmed imports pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning)

if TYPE_CHECKING:
    from openmed import ModelLoader

DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

# Inference backends openmed exposes. ``None`` (the engine default) lets openmed
# auto-detect — it prefers MLX on Apple Silicon when the `mlx` extra is installed,
# else HuggingFace/PyTorch. Forcing ``"mlx"`` raises if MLX is unavailable (e.g.
# non-Apple hosts); ``"hf"`` pins the portable backend.
Backend = Literal["hf", "mlx"]

# The de-identification strategies openmed's deidentify() accepts. Mirrors
# openmed.core.pii.DeidentificationMethod; a test enforces they stay in sync.
DeidMethod = Literal["mask", "remove", "replace", "hash", "shift_dates"]


def _entities(result: Any) -> list[Any]:
    """``extract_pii`` may return a list or an object exposing the entities."""
    for attr in ("entities", "pii_entities"):
        if hasattr(result, attr):
            return list(getattr(result, attr))
    return list(result)


class PIIEngine:
    """Detect and de-identify PII/PHI in clinical text.

    The OpenMed model is loaded lazily on first use and then reused, so
    constructing the engine is cheap and the model downloads only when the first
    detection/redaction call is made.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        model_name: str | None = None,
        backend: Backend | None = None,
        loader: ModelLoader | None = None,
    ) -> None:
        self.lang = lang
        self.model_name = model_name
        self.backend = backend
        self._loader: ModelLoader | None = loader

    @property
    def loader(self) -> ModelLoader:
        """The shared ModelLoader, created on first access.

        With ``backend=None`` openmed auto-detects (MLX on Apple Silicon when the
        `mlx` extra is installed, else HuggingFace). A non-None ``backend`` is
        pinned via ``OpenMedConfig`` — ``"mlx"`` raises at first model load on a
        host without MLX, so prefer ``None`` for portable auto-fallback.
        """
        loader = self._loader
        if loader is None:
            from openmed import ModelLoader

            if self.backend is None:
                loader = ModelLoader()
            else:
                from openmed import OpenMedConfig

                loader = ModelLoader(OpenMedConfig(backend=self.backend))
            self._loader = loader
        return loader

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying ModelLoader has been instantiated yet."""
        return self._loader is not None

    def _model_kwargs(
        self, *, lang: str | None = None, model_name: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"loader": self.loader, "lang": lang or self.lang}
        resolved = model_name or self.model_name
        if resolved:
            kwargs["model_name"] = resolved
        return kwargs

    def extract(
        self,
        text: str,
        *,
        confidence_threshold: float = 0.5,
        use_smart_merging: bool = True,
        lang: str | None = None,
        model_name: str | None = None,
    ) -> list[Any]:
        """Detect PII entities; each has ``.label``/``.text``/``.start``/``.end``/``.confidence``."""
        from openmed import extract_pii

        result = extract_pii(
            text,
            confidence_threshold=confidence_threshold,
            use_smart_merging=use_smart_merging,
            **self._model_kwargs(lang=lang, model_name=model_name),
        )
        return _entities(result)

    def deidentify(
        self,
        text: str,
        *,
        method: DeidMethod = "mask",
        confidence_threshold: float = 0.7,
        keep_mapping: bool = False,
        consistent: bool = False,
        seed: int | None = None,
        lang: str | None = None,
        model_name: str | None = None,
        date_shift_days: int | None = None,
        keep_year: bool = True,
        use_safety_sweep: bool = True,
    ) -> Any:
        """Rewrite ``text`` with PII redacted via ``method``.

        Returns OpenMed's ``DeidentificationResult`` (``.deidentified_text``,
        ``.pii_entities``, and ``.mapping`` when ``keep_mapping=True``). Every
        method — including ``"shift_dates"`` — is delegated straight to openmed;
        ``date_shift_days``/``keep_year`` apply only to ``shift_dates``.

        ``use_safety_sweep`` (default on, openmed 1.6.0's default) runs a
        deterministic structured-identifier sweep after model detection — it can
        redact identifiers the model misses, so de-identification may catch a few
        entities the ``Detect`` tab's ``extract_pii`` (which has no sweep) does not.
        It is passed explicitly rather than inherited so the behavior is controlled.

        openmed >=1.6.0 shifts dates correctly on the default model: it matches
        date entities by canonical label (normalizing the model's lowercase
        ``"date"``) rather than the literal ``"DATE"``, so ``shift_dates`` no
        longer falls back to masking. (Earlier versions masked dates instead;
        ``tests/test_pii_model.py`` verifies the shift now happens.)
        """
        from openmed import deidentify

        return deidentify(
            text,
            method=method,
            confidence_threshold=confidence_threshold,
            keep_mapping=keep_mapping,
            consistent=consistent,
            seed=seed,
            date_shift_days=date_shift_days,
            keep_year=keep_year,
            use_safety_sweep=use_safety_sweep,
            **self._model_kwargs(lang=lang, model_name=model_name),
        )

    @staticmethod
    def reidentify(deidentified_text: str, mapping: dict[str, str]) -> str:
        """Restore originals from a kept mapping.

        openmed.reidentify applies one ``str.replace`` per entry in mapping order, so a
        key that is a substring/prefix of another (``ALIAS_1`` vs ``ALIAS_10``, or
        unbracketed ``hash``/``replace`` surrogates) would corrupt the longer one. We
        reorder the mapping longest-key-first before delegating — openmed preserves dict
        insertion order — so each longer key is restored before its prefix. (openmed's
        raw function keeps the limitation, pinned by the xfail in ``tests/test_pii_pure.py``.)
        """
        from openmed import reidentify

        ordered = dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))
        return reidentify(deidentified_text, ordered)
