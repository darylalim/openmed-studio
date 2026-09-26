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
import threading
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
    # labels it emits (an empty tuple = the registry declares none)
    entity_types: tuple[str, ...]
    # human model size, e.g. "141M"
    params: str


# Clinical NER (token-classification) is one model PER domain, not one universal model, so
# the app curates one representative ~141M "superclinical" model per clinical category. Keys are
# openmed's own category names (openmed.list_model_categories() minus the "Privacy" PII bucket
# and "Medical" — openmed 2.0 dropped the broad 434M ClinicalNER alias the Medical domain used,
# leaving that category with no plain-PyTorch token-classification model). The 10 domains left
# mirror ZERO_SHOT_MODELS exactly.
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
        ("CHEM", "CHEMICAL", "DRUG", "MEDICATION"),
        "141M",
    ),
    "Chemical": NerModel(
        "chemical_detection_superclinical_141m",
        "ChemicalDetect SuperClinical 141M",
        0.6,
        ("SIMPLE_CHEMICAL", "CHEM", "CHEMICAL", "DRUG", "MEDICATION"),
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
            "CELL",
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
            "CHEMICAL",
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
            "ANATOMY",
            "PATHOLOGICAL_FORMATION",
            "PATHOLOGY",
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
        ("CANCER", "CELL", "CL", "DISEASE"),
        "141M",
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
    # model-category names above (openmed's label vocab is its own, larger taxonomy).
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
# ``aadhaar_mask`` (added in openmed 2.0) is India-specific: a value that passes openmed's
# Aadhaar (Verhoeff) checksum renders as ``XXXX XXXX NNNN`` — the UIDAI display form, which
# KEEPS the last four digits — and every other entity falls through to the ordinary mask
# placeholder, so it is byte-identical to ``mask`` on a note with no Aadhaar number. Retaining
# four digits makes it strictly *weaker* than ``mask``, so it is listed LAST: the guard
# compares sets, not order, and ``METHODS`` in streamlit_app.py feeds this order straight into
# the picker — an India-specific niche method should not outrank ``replace``/``hash`` there.
DeidMethod = Literal[
    "mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
    "aadhaar_mask",
]

# The regulatory anonymization policies the app OFFERS: openmed.core.policy.PolicyName minus
# HIDDEN_POLICIES below, in PolicyName order. test_validation_policy_matches_openmed pins that
# the two partition PolicyName exactly. A policy is a per-entity-label ACTION rulebook (a
# compliance profile) that OVERRIDES ``method`` when set — it assigns an action per label per
# that legal standard, and can force reversibility (a kept mapping) and the safety sweep. The
# Literal itself is narrowed, not just the picker, so AnonymizePolicyRequest rejects a hidden
# name (PHI-safely, before the engine) on the UI and the HTTP API alike. See POLICY_MODELS.
Policy = Literal[
    "hipaa_safe_harbor",
    "gdpr_art9_health",
    "strict_no_leak",
    "china_pipl",
    "africa_malabo_baseline",
    "za_popia",
    "ng_ndpa",
    "ke_dpa",
    "eg_pdpl",
    "ma_law_09_08",
]

# The openmed profiles the app deliberately does NOT offer. Each maps openmed's catch-all
# OTHER label to `keep`, and openmed's `normalize_label` (core/labels.py) files 26 of the PII
# model's 54 entity types under OTHER — license, tax and employee IDs, fax and health-plan
# numbers, company names, religion, political views, sexuality — so these profiles pass those
# identifiers through verbatim (bar the few formats the safety sweep's patterns catch, such
# as IP addresses), and because keep spans are dropped from the result (core/pipeline.py),
# the entity table doesn't list them either. Probe (openmed 2.5, the tab's defaults) on
# "Mr. Tom Hale, driver's license D4417-2290, tax ID 98-7654321, works at Brightline
# Logistics and is a devout Catholic.": GDPR Pseudonymization surrogated the name and
# returned the license, tax ID, employer and religion untouched, listing only the two name
# spans; HIPAA Safe Harbor masked all four. No profile left in Policy keeps any label.
# To re-expose one: once openmed stops keeping OTHER in it,
# test_hidden_policies_are_exactly_those_that_keep_other fails naming it (and its case in
# test_engine_hidden_policies_keep_other_identifiers_verbatim, --run-model, fails too). Move
# it back into Policy + POLICY_MODELS; the descriptions these ten last had, and the
# --run-model tests that pinned them, are in commit 8e1e50f — re-check both with a real run.
HIDDEN_POLICIES: frozenset[str] = frozenset(
    {
        "hipaa_expert_review_assist",
        "gdpr_pseudonymization",
        "research_limited_dataset",
        "clinical_minimal_redaction",
        "clinical_preserve",
        "canada_pipeda",
        "uk_ico_anonymisation",
        "australia_privacy_act",
        "india_dpdp_act",
        "india_health_id",
    }
)


class PolicyModel(NamedTuple):
    """A curated regulatory anonymization policy plus its openmed ``PolicyProfile`` metadata.

    Unlike :class:`NerModel`/:class:`ZeroShotModel` a policy loads **no model of its own** — it
    reuses the shared PII model and only changes the per-label ACTION assignment (a compliance
    profile). The behavioral flags below are baked at authoring time so the UI can preview a
    policy with **no runtime openmed import**; the drift guard
    ``tests/test_validation.py::test_policy_models_resolve_in_openmed`` pins them against
    openmed's live ``PolicyProfile`` (``load_policy``), so a policy-schema change fails CI rather
    than leaving this table stale. ``description`` is the one hand-authored field — openmed ships
    no per-policy description (only a ``name``/``posture`` slug).
    """

    # canonical policy name passed to deidentify(policy=...); a Policy Literal value
    name: str
    # short, hand-authored summary of what the policy does (openmed ships none). This — NOT
    # `default_action` below — is the honest account of a profile's behavior, so write it from
    # `load_policy(name).actions` / `.policy_label_actions`, never from `default_action`. Never
    # claim a profile masks clinical terms: the shared PII model has no clinical labels, so
    # those `mask` actions never fire. No offered profile keeps a label; if a hidden one
    # returns, confirm every "keeps X" claim with a real run — the safety sweep runs after keep
    # spans are dropped, so it can still mask what the rules keep (commit 8e1e50f has details).
    description: str
    # The profile's DECLARED fallback action (mask/redact/replace/keep), pinned against
    # load_policy. Caveat worth knowing before you surface it as a headline: openmed never
    # reaches it. `PolicyProfile.action_for` resolves through `actions[normalize_label(label)]`,
    # `_canonical_actions` requires `actions` to cover all 139 canonical labels exactly, and
    # `normalize_label` funnels anything unrecognized to OTHER — so the fallback is unreachable
    # and a profile can declare "replace" while masking 123 of 139 labels (za_popia does).
    default_action: str
    # whether the policy keeps a surrogate->original mapping (i.e. is reversible) — pinned
    keep_mapping: bool
    # whether the policy forces the deterministic structured-identifier safety sweep — pinned
    safety_sweep_mandatory: bool


# The openmed built-in compliance profiles the app offers (HIDDEN_POLICIES has the rest, and
# why), keyed by a friendly display name (the picker shows the key). Each maps to a canonical
# policy that, passed as deidentify(policy=...), OVERRIDES the flat method and assigns a
# per-label action encoding that legal standard. Reversibility is a SEPARATE flag, not implied
# by the action: masking profiles never keep a mapping, and of the four `replace` profiles only
# GDPR Art. 9 and China PIPL do — ZA POPIA and NG NDPA surrogate irreversibly. Ordered from the
# most widely used (HIPAA Safe Harbor) outward.
POLICY_MODELS: dict[str, PolicyModel] = {
    "HIPAA Safe Harbor": PolicyModel(
        "hipaa_safe_harbor",
        "Mask all 18 HIPAA identifiers; irreversible (US HIPAA §164.514(b)).",
        "mask",
        False,
        True,
    ),
    "GDPR Art. 9 Health": PolicyModel(
        "gdpr_art9_health",
        "Surrogate direct identifiers, mask every other detected span; high-recall, "
        "reversible with a key — identical to China PIPL (EU GDPR Art. 9).",
        "replace",
        True,
        True,
    ),
    "Strict No-Leak": PolicyModel(
        "strict_no_leak",
        "Maximum-recall detection, mask every detected span — the most aggressive profile.",
        "mask",
        False,
        True,
    ),
    # openmed 2.x additions (APAC and African regimes). Three things to keep straight when
    # editing these descriptions — the first two verified against `load_policy(...).actions`
    # (139 canonical labels each as of openmed 2.5), not inferred from `default_action`, which
    # openmed never actually applies (see PolicyModel's docstring); the third by real runs:
    #   * Reversibility: only China PIPL keeps a mapping. The other `replace` profiles here
    #     produce IRREVERSIBLE surrogates, so their descriptions must not promise a key.
    #   * "replace" rarely means "surrogate everything". za_popia replaces 16 of 139 labels and
    #     masks 123; ng_ndpa replaces 9 and masks 130. Four profiles (Malabo, Kenya DPA, Egypt
    #     PDPL, Morocco 09-08) are `mask`-everything and behaviorally identical to Strict
    #     No-Leak — say so rather than implying four distinct regimes.
    #   * "Mask everything" never reaches clinical text. All seven map DISEASE, DRUG and the
    #     other clinical labels to `mask`, but the shared PII model has none of them (its 54
    #     entity types are identifiers and personal attributes such as religion or blood type)
    #     and the sweep has no clinical patterns, so diagnoses pass through every profile.
    #     Hence "every detected span", never "clinical terms included".
    # All seven also carry threshold_profile="strict_no_leak" (higher recall than the slider's
    # nominal setting); that is what "high-recall" means below.
    "China PIPL": PolicyModel(
        "china_pipl",
        "Surrogate direct identifiers, mask every other detected span; high-recall, "
        "reversible with a key — identical to GDPR Art. 9 Health (China PIPL).",
        "replace",
        True,
        True,
    ),
    "African Union (Malabo)": PolicyModel(
        "africa_malabo_baseline",
        "Mask every detected span with high-recall detection — identical to Strict "
        "No-Leak (AU Malabo baseline).",
        "mask",
        False,
        True,
    ),
    "South Africa POPIA": PolicyModel(
        "za_popia",
        "Surrogate names, contacts and places; mask dates, ages, jobs, ID numbers and "
        "sensitive traits; irreversible (South Africa POPIA).",
        "replace",
        False,
        True,
    ),
    "Nigeria NDPA": PolicyModel(
        "ng_ndpa",
        "Surrogate names and contacts, mask every other detected span; high-recall, "
        "irreversible (Nigeria NDPA 2023).",
        "replace",
        False,
        True,
    ),
    "Kenya DPA": PolicyModel(
        "ke_dpa",
        "Mask every detected span with high-recall detection — identical to Strict "
        "No-Leak (Kenya DPA 2019).",
        "mask",
        False,
        True,
    ),
    "Egypt PDPL": PolicyModel(
        "eg_pdpl",
        "Mask every detected span with high-recall detection — identical to Strict "
        "No-Leak (Egypt Law 151/2020).",
        "mask",
        False,
        True,
    ),
    "Morocco Law 09-08": PolicyModel(
        "ma_law_09_08",
        "Mask every detected span with high-recall detection — identical to Strict "
        "No-Leak (Morocco Law 09-08).",
        "mask",
        False,
        True,
    ),
}

# The default policy: HIPAA Safe Harbor (the most widely used de-identification standard).
DEFAULT_POLICY_MODEL = POLICY_MODELS["HIPAA Safe Harbor"].name


# openmed 2.x's occurrence-mapping protocol. Mirrors
# ``openmed.core.pii._OCCURRENCE_MAPPING_PREFIX``, which is private, so it is baked here
# rather than imported — ``PIIEngine.reidentify`` is a pure, lock-free staticmethod with no
# openmed import at all. ``tests/test_pii_pure.py::test_occurrence_prefix_matches_openmed``
# pins it against the real constant, so an upstream rename fails CI.
_OCCURRENCE_MAPPING_PREFIX = "__openmed_occurrence_v1__:"


def _parse_occurrence_key(key: str) -> tuple[int, str] | None:
    """Split ``__openmed_occurrence_v1__:00000001:[last_name]`` into ``(1, "[last_name]")``.

    Returns ``None`` for a plain mapping key (and for a malformed occurrence key, which is
    then treated as literal text — the conservative reading for caller-supplied mappings).
    """
    if not key.startswith(_OCCURRENCE_MAPPING_PREFIX):
        return None
    ordinal_text, separator, surface = key[len(_OCCURRENCE_MAPPING_PREFIX) :].partition(
        ":"
    )
    if not separator or not ordinal_text.isdigit() or not surface:
        return None
    return int(ordinal_text), surface


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

    The engine is process-wide and shared across threads (FastAPI requests, cached
    Streamlit sessions). An internal lock serializes the model-calling methods
    (``extract``/``analyze``/``extract_zero_shot``/``deidentify``) so concurrent
    inference runs one call at a time; ``reidentify`` is lock-free (pure regex).
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
        # Serializes model inference across threads. The engine is shared — one instance
        # per process, handed to concurrent FastAPI requests and (via st.cache_resource)
        # concurrent Streamlit sessions — but the underlying transformers/GLiNER pipelines
        # are not guaranteed thread-safe. This lock guards every model-calling method so
        # inference runs one call at a time (and the first-call model download can't race);
        # `reidentify` is exempt (pure regex, no model). It trades concurrent throughput
        # for correctness, which suits this local/small-scale tool.
        self._lock = threading.Lock()

    @property
    def loader(self) -> ModelLoader:
        """The shared ModelLoader, created on first access.

        Built with ``OpenMedConfig(backend=self.backend,
        torch_attention_backend="eager")``. ``backend`` stays ``None`` unless pinned,
        so openmed still auto-detects it (MLX on Apple Silicon when the `mlx` extra is
        installed, else HuggingFace); a pinned ``"mlx"`` raises at first model load on a
        host without MLX, so prefer ``None`` for portable auto-fallback.

        ``torch_attention_backend="eager"`` is pinned deliberately: the OpenMed models
        are DeBERTa-v2, which has no SDPA kernel, and transformers raises
        ``DebertaV2ForTokenClassification does not support ... scaled_dot_product_attention``
        for any caller that requests SDPA *explicitly* (a caller that requests nothing still
        degrades to eager silently). openmed used to request it on ``"auto"``; as of 2.x it
        does not — ``openmed/torch/attention.py::select_attn_implementation("auto")`` returns
        ``None`` precisely so transformers can pick a kernel the architecture supports — so
        this pin is now belt-and-braces rather than load-bearing. Keep it: eager is the
        implementation these models run under either way, and pinning it means an openmed
        regression here cannot silently break every model load. (The
        ``OPENMED_TORCH_ATTENTION_BACKEND`` env var still overrides it.)
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

        with self._lock:
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

        ``analyze_text`` returns an ``AnalyzeResult`` *object* whose ``.entities`` holds
        the spans (``output_format="dict"`` is a misnomer — it is not a plain dict; the class
        was ``PredictionResult`` before openmed 2.0 and exposes the same ``.entities``), so
        ``_entities`` unwraps it via its ``.entities`` attribute, not by iterating it.
        ``lang`` is intentionally not threaded: ``analyze_text`` has no ``lang`` parameter
        (it uses ``sentence_language``, left at its ``"en"`` default).
        """
        from openmed import analyze_text

        with self._lock:
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
        with self._lock:
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
        policy: str | None = None,
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

        ``policy`` (a compliance-profile name, e.g. ``"hipaa_safe_harbor"`` — a value
        of :data:`Policy`) is forwarded to openmed unchanged. When set it **overrides**
        ``method``: openmed assigns a per-label action (mask/redact/replace/keep) from
        that profile and may force ``keep_mapping``/the safety sweep. The method-driven
        tabs leave it ``None`` (no policy); the ``Policy de-ID`` tab sets it via
        :data:`POLICY_MODELS`. ``None`` is forwarded too — openmed treats it as "no
        policy override", so the default de-identification path is unaffected.
        """
        from openmed import deidentify

        with self._lock:
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
                policy=policy,
                **self._model_kwargs(lang=lang, model_name=model_name),
            )

    @staticmethod
    def reidentify(deidentified_text: str, mapping: dict[str, str]) -> str:
        """Restore originals from a kept mapping, in a single pass.

        openmed.reidentify applies one ``str.replace`` per *plain* entry, which corrupts
        output two ways: a key that is a substring of another (``ALIAS_1`` vs ``ALIAS_10``,
        or unbracketed ``hash``/``replace`` surrogates) clobbers the longer one, and a
        replacement value that contains another key gets re-substituted. We instead match
        every key in one regex pass (longest key first, so the longest match wins at each
        position), so a replacement is never re-scanned — eliminating both failure modes.
        (openmed's raw function keeps the limitation, pinned by the xfail in
        ``tests/test_pii_pure.py``.)

        openmed 2.x additionally emits **occurrence-keyed** entries
        (``__openmed_occurrence_v1__:00000001:<surface>``) whenever one redacted surface
        stands for several distinct originals — e.g. ``method="aadhaar_mask"``, which is not
        in openmed's unique-placeholder set, collapses every last name onto ``[last_name]``.
        Those keys are protocol, not literal text: matching them verbatim finds nothing and
        silently leaves the placeholders in place. We therefore group them by surface and
        hand out their originals in ordinal order as the single pass walks the document
        (ordinals are assigned in entity order upstream, so ordinal order *is* document
        order), falling back to leaving the surface untouched once a group is exhausted —
        the same contract openmed's own reader has, minus its substring bug for plain keys.

        One limit no mapping-only restore can avoid: the mapping has no span offsets, so text
        that merely equals a surrogate (an age surrogated to ``6`` vs. "6 weeks") is restored
        too — pinned by the strict xfail ``test_reidentify_restores_only_the_surrogate_spans``.
        """
        if not mapping:
            return deidentified_text

        regular: dict[str, str] = {}
        occurrences: dict[str, list[tuple[int, str]]] = {}
        for key, original in mapping.items():
            parsed = _parse_occurrence_key(key)
            if parsed is None:
                regular[key] = original
            else:
                ordinal, surface = parsed
                occurrences.setdefault(surface, []).append((ordinal, original))

        pending = {
            surface: iter([original for _, original in sorted(items)])
            for surface, items in occurrences.items()
        }
        surfaces = set(regular) | set(pending)
        if not surfaces:
            # Every entry was a malformed occurrence key; nothing is safely restorable.
            return deidentified_text

        def restore(match: re.Match[str]) -> str:
            surface = match.group(0)
            remaining = pending.get(surface)
            if remaining is not None:
                # Exhausted (more matches than mapped originals) => leave it alone.
                return next(remaining, surface)
            return regular[surface]

        pattern = re.compile(
            "|".join(re.escape(key) for key in sorted(surfaces, key=len, reverse=True))
        )
        return pattern.sub(restore, deidentified_text)
