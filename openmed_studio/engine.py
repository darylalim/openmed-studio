"""Reusable PII/PHI de-identification engine: one shared model, thin wrappers.

This is the core of the app, independent of any web framework. It holds a single
:class:`openmed.ModelLoader` and reuses it across every call (the documented best
practice) so the ~44M-parameter PII model loads at most once per process.

The OpenMed import is deferred to first use, so importing this module — and the
service's ``/health`` endpoint — never pulls in Torch/Transformers or downloads a
model.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
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

# Date-entity labels across OpenMed's model families, compared after stripping
# every non-letter so lowercase ``date``/``date_of_birth`` (SuperClinical),
# UPPERCASE ``DATE``/``DATEOFBIRTH`` (Portuguese, privacy-filter) and ``dob`` all
# match. This is why our shift works where openmed's ``== "DATE"`` gate does not.
_DATE_LABELS = frozenset({"date", "dateofbirth", "dob", "datetime", "birthdate"})

# Date string layouts we shift, tried in order; the matched layout is reused for
# output so the surrogate keeps the original format (US slash/dash/dot, ISO).
_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m.%d.%Y", "%Y-%m-%d")


def _is_date_label(label: str) -> bool:
    """True if ``label`` denotes a date entity in any OpenMed model's taxonomy."""
    return "".join(ch for ch in label.lower() if ch.isalpha()) in _DATE_LABELS


def shift_date_text(date_str: str, days: int, *, keep_year: bool = True) -> str | None:
    """Shift a date string by ``days``, preserving its original format.

    Returns the shifted date in the same layout as the input, or ``None`` if the
    text doesn't parse as one of :data:`_DATE_FORMATS` (callers then fall back to
    masking). With ``keep_year`` the year is pinned to the original — only the
    month/day move — matching openmed's ``shift_dates`` ``keep_year`` semantics.

    Only the US/ISO **numeric** layouts in :data:`_DATE_FORMATS` are recognized,
    and parsing is month-first and language-blind: day-first locale dates and
    month-name dates return ``None`` (and are masked by the caller, not shifted).
    """
    text = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        shifted = parsed + timedelta(days=days)
        if keep_year:
            try:
                shifted = shifted.replace(year=parsed.year)
            except ValueError:  # shifted is Feb 29 but the pinned year isn't leap
                shifted = shifted.replace(year=parsed.year, day=28)
        return shifted.strftime(fmt)
    return None


@dataclass(frozen=True)
class _ShiftDatesResult:
    """Duck-types openmed's DeidentificationResult for the engine-side shift path.

    ``main._deidentify_one`` only reads ``deidentified_text``, ``pii_entities``
    and ``mapping``, so this carries exactly those.
    """

    deidentified_text: str
    pii_entities: list[Any]
    mapping: dict[str, str] | None = None


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
    ) -> Any:
        """Rewrite ``text`` with PII redacted via ``method``.

        Returns OpenMed's ``DeidentificationResult`` (``.deidentified_text``,
        ``.pii_entities``, and ``.mapping`` when ``keep_mapping=True``).
        ``date_shift_days``/``keep_year`` apply only to ``method="shift_dates"``.

        ``method="shift_dates"`` is handled by :meth:`_shift_dates` rather than
        openmed, whose shift path is a documented no-op with this model family
        (see the method docstring); the return value still duck-types
        ``DeidentificationResult`` for the HTTP layer.
        """
        if method == "shift_dates":
            return self._shift_dates(
                text,
                confidence_threshold=confidence_threshold,
                lang=lang,
                model_name=model_name,
                date_shift_days=date_shift_days,
                keep_year=keep_year,
                keep_mapping=keep_mapping,
            )

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
            **self._model_kwargs(lang=lang, model_name=model_name),
        )

    def _shift_dates(
        self,
        text: str,
        *,
        confidence_threshold: float,
        lang: str | None,
        model_name: str | None,
        date_shift_days: int | None,
        keep_year: bool,
        keep_mapping: bool,
    ) -> _ShiftDatesResult:
        """Engine-side ``shift_dates``: shift date entities, mask the rest.

        OpenMed's own ``shift_dates`` only fires when an entity's label is the
        exact string ``"DATE"`` (openmed/core/pii.py:905), but its models and
        smart-merging emit lowercase ``"date"`` — so dates get masked, not
        shifted (a documented no-op no parameter fixes). We reproduce the method
        off our own extraction, normalizing the label ourselves (:func:`
        _is_date_label`) so any date-typed entity actually shifts, while every
        non-date entity is masked just as openmed's ``shift_dates`` does. A
        single offset is applied to all dates (random when ``date_shift_days`` is
        ``None``, like openmed), preserving intervals within the document.

        Caveats: date parsing is US/ISO numeric and language-blind (see
        :func:`shift_date_text`), so localized/day-first dates are masked rather
        than shifted. A ``keep_mapping`` mapping is keyed by replacement text, so
        repeated mask placeholders (e.g. two ``[ssn]``) or dates that collide
        after shifting overwrite earlier entries — it is not round-trip-safe for
        duplicated values (the same caveat applies to openmed's own ``mask``).
        """
        entities = self.extract(
            text,
            confidence_threshold=confidence_threshold,
            use_smart_merging=True,
            lang=lang,
            model_name=model_name,
        )
        shift = (
            date_shift_days
            if date_shift_days is not None
            else random.randint(-365, 365)
        )
        mapping: dict[str, str] | None = {} if keep_mapping else None

        # Keep non-overlapping spans (smart-merging can emit overlaps, and
        # splicing those would corrupt offsets): walk left-to-right, drop any
        # span that starts before the previous one ended.
        chosen: list[Any] = []
        last_end = -1
        for entity in sorted(entities, key=lambda e: (int(e.start), -int(e.end))):
            start, end = int(entity.start), int(entity.end)
            if end <= start or start < last_end:
                continue
            chosen.append(entity)
            last_end = end

        # Apply replacements right-to-left so earlier offsets stay valid.
        out = text
        for entity in sorted(chosen, key=lambda e: int(e.start), reverse=True):
            start, end = int(entity.start), int(entity.end)
            original = text[start:end]
            label = (
                getattr(entity, "label", "") or getattr(entity, "entity_type", "") or ""
            )
            replacement = (
                shift_date_text(original, shift, keep_year=keep_year)
                if _is_date_label(label)
                else None
            )
            if replacement is None:  # non-date, or an unparseable date -> mask
                replacement = f"[{label}]"
            out = out[:start] + replacement + out[end:]
            if mapping is not None and replacement != original:
                mapping[replacement] = original

        return _ShiftDatesResult(
            deidentified_text=out, pii_entities=list(entities), mapping=mapping
        )

    @staticmethod
    def reidentify(deidentified_text: str, mapping: dict[str, str]) -> str:
        """Restore originals from a kept mapping (see README for the prefix caveat)."""
        from openmed import reidentify

        return reidentify(deidentified_text, mapping)
