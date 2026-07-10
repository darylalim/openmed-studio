"""Reusable engine for OpenMed clinical NLP: one shared model loader, thin wrappers.

This is the core of the app, independent of any web framework. :class:`PIIEngine`
holds a single :class:`openmed.ModelLoader` and reuses it across every call (the
documented best practice). It wraps both PII/PHI **de-identification**
(``extract``/``deidentify``/``reidentify`` over the ~44M-parameter PII model) and
clinical **NER** (``analyze`` over a per-domain token-classification model — see
:data:`NER_MODELS`); the one shared loader serves every model, keyed by name.

The OpenMed import is deferred to first use, so importing this module never pulls in
Torch/Transformers or downloads a model.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

# pysbd (a transitive openmed dependency) raises harmless SyntaxWarnings from its
# regex literals on Python >=3.12. Silence them before openmed imports pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning)

if TYPE_CHECKING:
    from openmed import ModelLoader

DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


class NerModel(NamedTuple):
    """A curated clinical-NER domain model plus its openmed registry metadata.

    The metadata fields are baked in at authoring time so the UI can show them with **no
    runtime openmed import** (every openmed import here is deferred to first model use).
    The drift guard ``tests/test_validation.py::test_validation_ner_models_resolve_in_openmed``
    pins ``alias``/``recommended_confidence``/``entity_types`` against the live registry,
    so a registry change fails CI rather than silently leaving this table stale.
    """

    # registry alias passed to analyze_text(model_name=...)
    alias: str
    # friendly name for the UI (vs the raw alias)
    display_name: str
    # the model's own suggested threshold (the UI slider default)
    recommended_confidence: float
    # labels it emits (an empty tuple = "not declared", e.g. Medical)
    entity_types: tuple[str, ...]
    # human model size, e.g. "141M"
    params: str


# Clinical NER (token-classification) is one model PER domain, not one universal model, so
# the app curates one representative ~141M "superclinical" model per clinical category
# (Medical uses the broader 434M ClinicalNER model — its only non-MLX option). Keys are
# openmed's own category names (openmed.list_model_categories() minus the "Privacy" PII bucket).
NER_MODELS: dict[str, NerModel] = {
    "Disease": NerModel(
        "disease_detection_superclinical_141m",
        "DiseaseDetect SuperClinical 141M",
        0.6,
        ("DISEASE", "CONDITION", "PATHOLOGY"),
        "141M",
    ),
    "Pharmaceutical": NerModel(
        "pharma_detection_superclinical_141m",
        "PharmaDetect SuperClinical 141M",
        0.65,
        ("SIMPLE_CHEMICAL", "CHEM", "DRUG", "MEDICATION"),
        "141M",
    ),
    "Chemical": NerModel(
        "chemical_detection_superclinical_141m",
        "ChemicalDetect SuperClinical 141M",
        0.6,
        ("SIMPLE_CHEMICAL", "CHEM", "DRUG", "MEDICATION"),
        "141M",
    ),
    "Anatomy": NerModel(
        "anatomy_detection_superclinical_141m",
        "AnatomyDetect SuperClinical 141M",
        0.6,
        ("ORGAN", "TISSUE", "ANATOMY"),
        "141M",
    ),
    "Genomics": NerModel(
        "dna_detection_superclinical_141m",
        "DNADetect SuperClinical 141M",
        0.65,
        (
            "GENE_OR_GENE_PRODUCT",
            "DNA",
            "RNA",
            "GENE",
            "PROTEIN",
            "CELL_LINE",
            "CELL_TYPE",
        ),
        "141M",
    ),
    "Protein": NerModel(
        "protein_detection_superclinical_141m",
        "ProteinDetect SuperClinical 141M",
        0.6,
        (
            "GENE_OR_GENE_PRODUCT",
            "PROTEIN",
            "PROTEIN_COMPLEX",
            "PROTEIN_ENUM",
            "PROTEIN_FAMILIY_OR_GROUP",
            "PROTEIN_VARIANT",
        ),
        "141M",
    ),
    "Oncology": NerModel(
        "oncology_detection_superclinical_141m",
        "OncologyDetect SuperClinical 141M",
        0.65,
        (
            "SIMPLE_CHEMICAL",
            "CHEM",
            "CANCER",
            "CELL",
            "GENE_OR_GENE_PRODUCT",
            "ORGANISM",
            "SPECIES",
            "AMINO_ACID",
            "ANATOMICAL_SYSTEM",
            "CELLULAR_COMPONENT",
            "DEVELOPING_ANATOMICAL_STRUCTURE",
            "IMMATERIAL_ANATOMICAL_ENTITY",
            "MULTI_TISSUE_STRUCTURE",
            "ORGAN",
            "ORGANISM_SUBDIVISION",
            "ORGANISM_SUBSTANCE",
            "TISSUE",
            "PATHOLOGICAL_FORMATION",
        ),
        "141M",
    ),
    "Species": NerModel(
        "species_detection_superclinical_141m",
        "SpeciesDetect SuperClinical 141M",
        0.6,
        ("ORGANISM", "SPECIES"),
        "141M",
    ),
    "Pathology": NerModel(
        "pathology_detection_superclinical_141m",
        "PathologyDetect SuperClinical 141M",
        0.6,
        ("DISEASE", "CONDITION", "PATHOLOGY"),
        "141M",
    ),
    "Hematology": NerModel(
        "bloodcancer_detection_superclinical_141m",
        "BloodCancerDetect SuperClinical 141M",
        0.65,
        ("CANCER", "DISEASE", "CL"),
        "141M",
    ),
    "Medical": NerModel(
        "clinicalner_superclinical_large_434m",
        "ClinicalNER SuperClinical Large 434M",
        0.6,
        (),
        "434M",
    ),
}

# The default NER model: the Disease detector's alias (smallest superclinical family, known
# entity types), mirroring openmed.analyze_text's own disease-domain default.
DEFAULT_NER_MODEL = NER_MODELS["Disease"].alias


class ZeroShotModel(NamedTuple):
    """A curated GLiNER zero-shot model plus the metadata the Zero-shot tab needs.

    Zero-shot extraction lets the user name **arbitrary** entity labels, so unlike
    :class:`NerModel` the ``entity_types`` here are *not* the output vocabulary — they
    are the checkpoint's training focus, shown as a "tuned for" hint. The actual labels
    come from the user (seeded from :func:`PIIEngine.default_labels` for ``label_domain``).

    ``alias``/``recommended_confidence``/``entity_types`` are pinned against openmed's live
    registry, and ``label_domain`` against ``openmed.ner.available_domains()``, by
    ``tests/test_validation.py::test_zero_shot_models_resolve_in_openmed`` — so a registry
    rename or a dropped label vocabulary fails CI rather than leaving this table stale.
    """

    # registry alias -> resolved to the HF repo id passed to openmed.ner.infer()
    alias: str
    # friendly name for the UI (vs the raw alias)
    display_name: str
    # the checkpoint's suggested threshold (the UI slider default)
    recommended_confidence: float
    # the labels the checkpoint was tuned on — a "tuned for" hint, not the output vocab
    entity_types: tuple[str, ...]
    # human model size, e.g. "166M"
    params: str
    # openmed.ner label-vocabulary domain used to SEED the label picker (a suggestion the
    # user edits freely); a key of openmed.ner.available_domains(), distinct from the
    # model-category names above (openmed's label vocab is its own 24-domain taxonomy).
    label_domain: str


# Curated zero-shot (GLiNER) models: one representative Small/166M checkpoint per clinical
# domain, mirroring NER_MODELS' domain vocabulary. Every OpenMed zero-shot checkpoint is
# domain-tuned (there is no universal one), so the domain picks the backbone while the labels
# stay free-text. Keys are the same display domains the Clinical NER tab uses (minus Medical,
# whose broad ClinicalNER model has no zero-shot sibling — Disease covers the general case).
ZERO_SHOT_MODELS: dict[str, ZeroShotModel] = {
    "Disease": ZeroShotModel(
        "zeroshot_disease_small_166m",
        "ZeroShot Disease 166M",
        0.6,
        ("DISEASE", "CONDITION", "PATHOLOGY"),
        "166M",
        "clinical",
    ),
    "Pharmaceutical": ZeroShotModel(
        "zeroshot_pharma_small_166m",
        "ZeroShot Pharma 166M",
        0.6,
        ("SIMPLE_CHEMICAL", "CHEM", "DRUG", "MEDICATION"),
        "166M",
        "biomedical",
    ),
    "Chemical": ZeroShotModel(
        "zeroshot_chemical_small_166m",
        "ZeroShot Chemical 166M",
        0.6,
        ("SIMPLE_CHEMICAL", "CHEM", "DRUG", "MEDICATION"),
        "166M",
        "chemistry",
    ),
    "Anatomy": ZeroShotModel(
        "zeroshot_anatomy_small_166m",
        "ZeroShot Anatomy 166M",
        0.6,
        ("ORGAN", "TISSUE", "ANATOMY"),
        "166M",
        "clinical",
    ),
    "Genomics": ZeroShotModel(
        "zeroshot_dna_small_166m",
        "ZeroShot DNA 166M",
        0.6,
        ("GENE_OR_GENE_PRODUCT", "DNA", "RNA", "GENE", "PROTEIN"),
        "166M",
        "genomic",
    ),
    "Protein": ZeroShotModel(
        "zeroshot_protein_small_166m",
        "ZeroShot Protein 166M",
        0.6,
        ("GENE_OR_GENE_PRODUCT", "PROTEIN"),
        "166M",
        "biomedical",
    ),
    "Oncology": ZeroShotModel(
        "zeroshot_oncology_small_166m",
        "ZeroShot Oncology 166M",
        0.6,
        (
            "SIMPLE_CHEMICAL",
            "CHEM",
            "CANCER",
            "CELL",
            "GENE_OR_GENE_PRODUCT",
            "ORGANISM",
            "SPECIES",
        ),
        "166M",
        "biomedical",
    ),
    "Species": ZeroShotModel(
        "zeroshot_species_small_166m",
        "ZeroShot Species 166M",
        0.6,
        ("ORGANISM", "SPECIES"),
        "166M",
        "organism",
    ),
    "Pathology": ZeroShotModel(
        "zeroshot_pathology_small_166m",
        "ZeroShot Pathology 166M",
        0.6,
        ("DISEASE", "CONDITION", "PATHOLOGY"),
        "166M",
        "clinical",
    ),
    "Hematology": ZeroShotModel(
        "zeroshot_bloodcancer_small_166m",
        "ZeroShot BloodCancer 166M",
        0.65,
        (
            "CANCER",
            "DISEASE",
            "CONDITION",
            "PATHOLOGY",
            "GENE_OR_GENE_PRODUCT",
            "PROTEIN",
        ),
        "166M",
        "biomedical",
    ),
}

# The default zero-shot domain/model: Disease (the general clinical case), mirroring
# DEFAULT_NER_MODEL's choice.
DEFAULT_ZERO_SHOT_MODEL = ZERO_SHOT_MODELS["Disease"].alias

# A fixed timestamp for the in-memory ModelIndex the zero-shot path fabricates (see
# PIIEngine.extract_zero_shot). openmed.ner.infer only echoes it into result metadata the
# app discards, so its value is irrelevant — a constant keeps the call deterministic.
_ZERO_SHOT_INDEX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Inference backends openmed exposes. ``None`` (the engine default) lets openmed
# auto-detect — it prefers MLX on Apple Silicon when the `mlx` extra is installed,
# else HuggingFace/PyTorch. Forcing ``"mlx"`` raises if MLX is unavailable (e.g.
# non-Apple hosts); ``"hf"`` pins the portable backend.
Backend = Literal["hf", "mlx"]

# The de-identification strategies openmed's deidentify() accepts. Mirrors
# openmed.core.pii.DeidentificationMethod; a test enforces they stay in sync.
# ``format_preserve`` (added in openmed 1.7.0) is a ``replace`` sibling: it swaps
# structured identifiers for synthetic values of the same shape (a phone stays
# phone-shaped), falling back to masking for entities it can't format-preserve.
DeidMethod = Literal[
    "mask", "remove", "replace", "hash", "shift_dates", "format_preserve"
]


def _entities(result: Any) -> list[Any]:
    """``extract_pii`` may return a list or an object exposing the entities."""
    for attr in ("entities", "pii_entities"):
        if hasattr(result, attr):
            return list(getattr(result, attr))
    return list(result)


class PIIEngine:
    """Detect and de-identify PII/PHI, and detect clinical entities (NER), in text.

    Despite the name, this engine spans both capabilities: ``extract``/``deidentify``/
    ``reidentify`` for PII/PHI and ``analyze`` for clinical NER (a rename to a more
    general name is deferred). Models load lazily on first use and are then reused, so
    constructing the engine is cheap; each model downloads only on the first call that
    needs it.
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

        Built with ``OpenMedConfig(backend=self.backend,
        torch_attention_backend="eager")``. ``backend`` stays ``None`` unless pinned,
        so openmed still auto-detects it (MLX on Apple Silicon when the `mlx` extra is
        installed, else HuggingFace); a pinned ``"mlx"`` raises at first model load on a
        host without MLX, so prefer ``None`` for portable auto-fallback.

        ``torch_attention_backend="eager"`` is pinned deliberately: the OpenMed models
        are DeBERTa-v2, which has no SDPA/flash-attention kernel. openmed's default
        ``"auto"`` requests SDPA, which transformers <=5.12 silently downgraded to eager
        but 5.13+ rejects outright (``DebertaV2ForTokenClassification does not support
        ... scaled_dot_product_attention``). eager is the implementation these models run
        under either way, so pinning it keeps model loading working on the latest
        transformers. (The ``OPENMED_TORCH_ATTENTION_BACKEND`` env var still overrides it.)
        """
        loader = self._loader
        if loader is None:
            from openmed import ModelLoader, OpenMedConfig

            loader = ModelLoader(
                OpenMedConfig(backend=self.backend, torch_attention_backend="eager")
            )
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

    def analyze(
        self,
        text: str,
        *,
        model_name: str,
        confidence_threshold: float = 0.0,
        aggregation_strategy: str = "simple",
        group_entities: bool = False,
    ) -> list[Any]:
        """Detect clinical entities with a token-classification (NER) model.

        Wraps openmed's ``analyze_text`` the way :meth:`extract` wraps ``extract_pii``:
        the entities returned each expose ``.label``/``.text``/``.start``/``.end``/
        ``.confidence`` (labels are UPPERCASE, e.g. ``"DISEASE"``). Unlike PII detection,
        clinical NER is one model PER domain, so ``model_name`` is **required** — pass a
        registry alias from :data:`NER_MODELS` (a missing model would silently fall back
        to openmed's disease-only default). The shared :class:`ModelLoader` is reused as
        ``loader=`` (``analyze_text`` dispatches/caches by ``model_name``), so switching
        domains loads another model into the same loader rather than rebuilding it.

        ``analyze_text`` returns a ``PredictionResult`` *object* whose ``.entities`` holds
        the spans (``output_format="dict"`` is a misnomer — it is not a plain dict), so
        ``_entities`` unwraps it via its ``.entities`` attribute, not by iterating it.
        ``lang`` is intentionally not threaded: ``analyze_text`` has no ``lang`` parameter
        (it uses ``sentence_language``, left at its ``"en"`` default).
        """
        from openmed import analyze_text

        result = analyze_text(
            text,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            aggregation_strategy=aggregation_strategy,
            group_entities=group_entities,
            output_format="dict",
            loader=self.loader,
        )
        return _entities(result)

    @staticmethod
    def zero_shot_available() -> bool:
        """Whether the optional GLiNER backend (the ``gliner`` extra) is importable.

        The Zero-shot tab is gated on this so it can show install instructions instead of
        failing at call time. Delegates to openmed's own probe, which checks ``gliner`` plus
        its ancillary deps without importing Torch or loading a model.
        """
        from openmed.ner import is_gliner_available

        return is_gliner_available()

    @staticmethod
    def default_labels(label_domain: str) -> list[str]:
        """Suggested entity labels for a domain, from openmed's label vocabulary.

        Seeds the Zero-shot tab's label picker (the user edits them freely). Read live from
        ``openmed.ner.get_default_labels`` so the suggestions track openmed rather than a
        baked copy — and so the Streamlit layer needs no openmed import of its own. These
        are natural-language prompts (``"Problem"``, ``"Treatment"``), which GLiNER reads
        better than UPPERCASE tag names.
        """
        from openmed.ner import get_default_labels

        return list(get_default_labels(label_domain))

    def extract_zero_shot(
        self,
        text: str,
        *,
        model_name: str,
        labels: list[str],
        confidence_threshold: float = 0.6,
    ) -> list[Any]:
        """Extract user-named ``labels`` from ``text`` with a GLiNER zero-shot model.

        Unlike :meth:`analyze` (a fixed per-domain label set), zero-shot takes **arbitrary**
        ``labels`` and a domain-tuned backbone selected by ``model_name`` (a
        :data:`ZERO_SHOT_MODELS` alias). The entities returned each expose ``.label``/
        ``.text``/``.start``/``.end``/``.score`` — note ``.score``, not ``.confidence``
        (``openmed.ner.Entity``); the service adapter maps it.

        This path deliberately does **not** use the shared :attr:`loader`: openmed's GLiNER
        inference (``openmed.ner.infer``) bypasses ``ModelLoader`` entirely, caching its own
        model instances, and needs no DeBERTa-v2 eager pin (the ``gliner`` fork runs on an
        older transformers where the SDPA request degrades to eager on its own). So
        :attr:`is_loaded` does not reflect a loaded zero-shot model; the UI tracks that
        separately. ``infer`` also defaults to a on-disk model index that isn't shipped, so
        a one-entry :class:`~openmed.ner.ModelIndex` is fabricated in memory to point it at
        the resolved HF repo id.
        """
        from openmed import get_all_models
        from openmed.ner import ModelIndex, ModelRecord, NerRequest, infer

        info = get_all_models().get(model_name)
        if info is None:
            # model_name passed the format check but isn't a registry alias. The UI only ever
            # sends a curated ZERO_SHOT_MODELS alias (pinned by the drift guard), so this is
            # unreachable from the app — but a direct service caller (or an openmed rename)
            # gets a clear message instead of an opaque "failed unexpectedly" (ValueError maps
            # to a pass-through ServiceError in the seam).
            raise ValueError(f"unknown zero-shot model: {model_name!r}")
        model_id = info.model_id
        index = ModelIndex(
            models=(ModelRecord(id=model_id, family="gliner"),),
            generated_at=_ZERO_SHOT_INDEX_EPOCH,
            source_dir=Path(),
        )
        result = infer(
            NerRequest(
                model_id=model_id,
                text=text,
                labels=labels,
                threshold=confidence_threshold,
            ),
            index=index,
        )
        return _entities(result)

    def deidentify(
        self,
        text: str,
        *,
        method: DeidMethod = "mask",
        confidence_threshold: float = 0.7,
        use_smart_merging: bool = True,
        keep_mapping: bool = False,
        consistent: bool = False,
        seed: int | None = None,
        locale: str | None = None,
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
        ``date_shift_days``/``keep_year`` apply only to ``shift_dates``, and the
        surrogate knobs ``consistent``/``seed``/``locale`` (``locale`` a Faker
        locale, e.g. ``"pt_BR"``, picking the surrogate locale instead of the default
        openmed derives from ``lang``) apply to the surrogate methods ``"replace"``
        and ``"format_preserve"``. They are forwarded unconditionally; openmed ignores
        the ones a method doesn't consume, and the UI only surfaces each where it applies.

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
            use_smart_merging=use_smart_merging,
            keep_mapping=keep_mapping,
            consistent=consistent,
            seed=seed,
            locale=locale,
            date_shift_days=date_shift_days,
            keep_year=keep_year,
            use_safety_sweep=use_safety_sweep,
            **self._model_kwargs(lang=lang, model_name=model_name),
        )

    @staticmethod
    def reidentify(deidentified_text: str, mapping: dict[str, str]) -> str:
        """Restore originals from a kept mapping, in a single correct pass.

        openmed.reidentify applies one ``str.replace`` per entry, which corrupts output
        two ways: a key that is a substring of another (``ALIAS_1`` vs ``ALIAS_10``, or
        unbracketed ``hash``/``replace`` surrogates) clobbers the longer one, and a
        replacement value that contains another key gets re-substituted. We instead match
        every key in one regex pass (longest key first, so the longest match wins at each
        position) and substitute via the mapping, so a replacement is never re-scanned —
        eliminating both failure modes. (openmed's raw function keeps the limitation,
        pinned by the xfail in ``tests/test_pii_pure.py``.)
        """
        if not mapping:
            return deidentified_text
        pattern = re.compile(
            "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
        )
        return pattern.sub(lambda m: mapping[m.group(0)], deidentified_text)
