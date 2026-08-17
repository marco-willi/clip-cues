"""Guards for the revision figure set: canonical cue space, shared taxonomy, saving contract.

The figure work sits directly downstream of the 2026-07-17 double-projection bug. Two artifacts
that *look* usable are in the retracted W-squared space — `outputs/e8/paired/paired_cue_shifts_*`
and `outputs/e8/stable_interp/*_cue_profile.csv` — so a figure that quietly reads one would put a
retracted number in the paper. These tests make that a build failure.

Beyond the W-squared guard they pin four things a layout change could silently break:

- Fig 3's poles are labelled the way the montage claims (all synthetic on one side, all real on the other);
- Fig 4's single example and Fig 5's population panel are the **same quantity**, not two analyses;
- Fig 6's left/right annotations sit on the halves they describe (this inverted once already);
- Fig 7's concept names come from the text space the checkpoint actually uses.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from clip_cues_research.figures import style  # noqa: E402

# Artifacts in the retracted W-squared text space, plus the pre-fix vocabulary pool.
RETRACTED = re.compile(
    r"antonyms_diff_embeddings\.pt|antonyms_embeddings\.pt|textspan_embeddings\.pt|vocab_pool/"
)
# Derived outputs of the retracted artifacts — equally unusable, and easy to reach for by name.
RETRACTED_OUTPUTS = re.compile(r"outputs/e8/paired/paired_cue_shifts|outputs/e8/stable_interp/")

# Figure modules behind the current figure set; each must be canonical-only.
FIGURE_MODULES = [
    "src/clip_cues_research/figures/style.py",
    "src/clip_cues_research/figures/extreme_scores.py",
    "src/clip_cues_research/figures/paired_example.py",
    "src/clip_cues_research/figures/cue_population.py",
    "src/clip_cues_research/figures/boundary_mechanism.py",
    "src/clip_cues_research/figures/concept_model_profile.py",
    "src/clip_cues_research/figures/dataset_examples.py",
    "src/clip_cues_research/figures/corpus_examples.py",
    "src/clip_cues_research/figures/paired_cue_delta.py",
]


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every string constant in a module **except** docstrings.

    Prose may — and should — name the hazard: several of these modules exist precisely to explain
    why the W-squared artifacts must not be used. Only a string that could become a *path* counts,
    so docstrings are excluded structurally rather than by a comment-stripping heuristic.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def _existing(paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths if Path(p).exists()]


def _offenders(paths: list[Path]) -> list[str]:
    out = []
    for path in paths:
        for lineno, text in _string_literals(path):
            if RETRACTED.search(text) or RETRACTED_OUTPUTS.search(text):
                out.append(f"{path}:{lineno}: {text!r}")
    return out


def test_no_figure_module_reads_a_retracted_w2_artifact():
    offenders = _offenders(_existing(FIGURE_MODULES))
    assert not offenders, "figure code reads retracted W-squared cue data:\n" + "\n".join(offenders)


def test_no_figure_driver_reads_a_retracted_w2_artifact():
    offenders = _offenders(sorted(Path("scripts/plot").glob("plot_fig*.py")))
    assert not offenders, "figure driver reads retracted W-squared cue data:\n" + "\n".join(
        offenders
    )


def test_the_guard_would_catch_a_real_path(tmp_path):
    """The guard is worthless if excluding docstrings also excludes real code — check it still bites."""
    p = tmp_path / "bad.py"
    p.write_text(
        '"""Mentions antonyms_diff_embeddings.pt safely."""\nV = "data/embeddings/antonyms_diff_embeddings.pt"\n'
    )
    assert _offenders([p]), "docstring exclusion swallowed a genuine retracted path"


# ── cue-family taxonomy ──────────────────────────────────────────────────────────────────────
def test_cue_families_cover_the_expected_share_per_vocabulary():
    """Pins the coverage that drove the per-figure vocabulary choice."""
    ant = Path("data/vocabularies/antonyms.csv")
    if not ant.exists():
        pytest.skip("vocabulary CSV not present")

    fam_ant = style.cue_families(ant)
    assert len(fam_ant) == 168
    # antonyms groups poorly -> used only for per-cue LABELS (Fig 4/5), never for family
    # aggregation. The optimized vocabulary that grouped well is not published (E9 has no
    # manuscript section), so only the antonyms half of this pin remains.
    assert (fam_ant == "other").mean() > 0.3
    assert set(fam_ant.unique()) <= set(style.FAMILY_ORDER)


def test_every_family_has_a_label_and_a_colour():
    for fam in style.FAMILY_ORDER:
        assert fam in style.FAMILY_LABEL
        assert fam in style.FAMILY_PALETTE


# ── Fig 3: the poles are what the montage says they are ──────────────────────────────────────
def _snapshot_or_skip():
    from clip_cues_research.finalexp import data as D

    if not D.MANIFEST.exists():
        pytest.skip("input snapshot not built (make finalexp-data)")
    return D


def test_cnnspot_real_source_map_is_present_and_correct():
    """Pin the B.1 provenance so a config edit cannot silently relabel a real image's corpus.

    CNNSpot files real photographs under the generator group they are paired into, so the *only*
    thing standing between the figures and "this LSUN photo was made by ProGAN" is this map.
    """
    from clip_cues_research.figures.extreme_scores import real_source_map

    m = real_source_map()
    # Wang et al. 2020 appendix B.1, "Dataset Collection"
    expected = {
        "progan": "LSUN",
        "stylegan": "LSUN",
        "stylegan2": "LSUN",
        "biggan": "ImageNet",
        "stargan": "CelebA",
        "gaugan": "COCO",
        "crn": "GTA",
        "imle": "GTA",
        "deepfake": "FaceForensics++",
    }
    for group, corpus in expected.items():
        assert m.get(group) == corpus, f"{group} should map to {corpus}, got {m.get(group)!r}"

    # Deliberately absent — B.1 names no corpus for cyclegan, and the diffusion-era groups post-date
    # it. If someone fills these, it must be from a source, not from the group name.
    for group in ("cyclegan", "whichfaceisreal", "ldm_200", "glide_100_10", "guided", "dalle"):
        assert group not in m, (
            f"{group!r} gained a corpus — confirm the provenance and update this test, "
            "or remove the entry"
        )


def test_no_cnnspot_real_panel_is_labelled_with_its_generator():
    """The bug this guards: a real LSUN photo annotated 'real / StyleGAN2' because of its folder.

    A bare generator name on a `label == 0` panel is always wrong. The `"<Group> subset"` fallback is
    allowed precisely because it says *subset* — it describes where the file sits, not what made it.
    """
    _snapshot_or_skip()
    from clip_cues_research.figures.dataset_examples import DEFAULT_SOURCES, median_examples
    from clip_cues_research.figures.extreme_scores import (
        PRETTY_SOURCE,
        annotate,
        cnnspot_real_origin,
    )

    generators = set(PRETTY_SOURCE.values())

    from clip_cues_research.figures.extreme_scores import poles

    for _, row in poles("cnnspot", k=5).iterrows():
        cls, origin = annotate("cnnspot", row)
        if row["label"] == 0:
            assert origin not in generators, (
                f"real image {row['image_id']} annotated {origin!r}, which is a generator name"
            )

    for r in median_examples(DEFAULT_SOURCES).itertuples():
        if r.label == 0:
            origin = cnnspot_real_origin(r.image_id, r.source)
            assert origin not in generators, (
                f"real image {r.image_id} annotated {origin!r}, which is a generator name"
            )


def test_fig3_poles_are_cleanly_labelled():
    _snapshot_or_skip()
    from clip_cues_research.figures.extreme_scores import poles

    for dataset in ("synthclic", "cnnspot"):
        p = poles(dataset, k=5)
        real = p[p["pole"] == "real_like"]
        synth = p[p["pole"] == "synthetic_like"]
        assert (real["label"] == 0).all(), f"{dataset}: a synthetic image is in the real-like pole"
        assert (synth["label"] == 1).all(), f"{dataset}: a real image is in the synthetic-like pole"
        assert real["logit"].max() < synth["logit"].min()


# ── Fig 4 and Fig 5 measure the same thing ───────────────────────────────────────────────────
def test_fig4_example_deltas_equal_the_fig5_population_quantity():
    """The single example must be an instance of the aggregate, not a parallel computation.

    Both read `cue_scores/synthclic__antonyms`, so equality should be exact — any drift means one
    of them changed construction (normalization, vocabulary, split) and the figures no longer agree.
    """
    _snapshot_or_skip()
    from clip_cues_research.figures.cue_population import paired_deltas
    from clip_cues_research.figures.paired_example import GENERATORS, pair_deltas

    ex_deltas, ids, names = pair_deltas(GENERATORS, "test")
    pop, pop_ids, pop_names, pop_gens = paired_deltas("synthclic", "test")
    assert names == pop_names

    image_id = ids[0]
    for gen in GENERATORS:
        m = (pop_ids == image_id) & (pop_gens == gen)
        assert m.sum() == 1, f"expected exactly one ({image_id}, {gen}) pair, got {m.sum()}"
        np.testing.assert_allclose(
            ex_deltas[gen][ids.index(image_id)], pop[m][0], rtol=0, atol=1e-9
        )


# ── Fig 6: the annotations sit on the halves they describe ───────────────────────────────────
# The cues the two annotations name explicitly. Asserting on *these* rather than on whichever cue
# happens to top each half is deliberate: the ranking within a half is not the claim, and an
# over-strict version of this test failed on `posterization` topping the SynthCLIC half — a tone cue
# that is genuinely there and does not invalidate the labelling.
CNNSPOT_PROMISED = {"compression_artifacts", "upscaler_artifacts"}
SYNTHCLIC_PROMISED = {"film_grain", "chromatic_aberration"}


def test_fig6_annotation_sides_match_the_data():
    """Delta = w(CNNSpot) - w(SynthCLIC): positive => CNNSpot-associated, i.e. processing artifacts.

    The labelled halves inverted once during a layout change, which reverses the figure's claim.
    This asserts the *data* still supports the label placement hard-coded in the module: every cue
    each annotation names by hand must actually fall on the side that annotation labels.
    """
    from clip_cues_research.figures.boundary_mechanism import DELTA_AXES, delta_axes

    if not DELTA_AXES.exists():
        pytest.skip("F6 artifacts not present (run scripts/finalexp/run_f6_cross_dataset.py)")
    df = delta_axes(top_n=14).set_index("cue")["alpha_coef"]

    for cue in CNNSPOT_PROMISED:
        assert cue in df.index, f"{cue!r} left the top axes; the CNNSpot annotation names it"
        assert df[cue] > 0, (
            f"{cue!r} is on the SynthCLIC half ({df[cue]:+.3f}) but the CNNSpot annotation "
            "names it — the labelled halves are inverted"
        )
    for cue in SYNTHCLIC_PROMISED:
        assert cue in df.index, f"{cue!r} left the top axes; the SynthCLIC annotation names it"
        assert df[cue] < 0, (
            f"{cue!r} is on the CNNSpot half ({df[cue]:+.3f}) but the SynthCLIC annotation "
            "names it — the labelled halves are inverted"
        )


# ── Fig 7: concept names come from the checkpoint's own text space ───────────────────────────
def test_fig7_concept_names_are_pinned_to_the_checkpoint_text_space(tmp_path):
    from clip_cues_research.figures.concept_model_profile import CKPT, assert_canonical_vocabulary

    ckpt = Path(CKPT["synthclic"])
    canon = Path("data/embeddings/vocab_canon/antonyms.pt")
    if not (ckpt.exists() and canon.exists()):
        pytest.skip("concept checkpoint or canonical vocabulary not present")

    names = assert_canonical_vocabulary(str(ckpt), str(canon))
    assert len(names) == 168

    # ... and a wrong-but-shape-identical vocabulary must be REJECTED, not silently accepted. The
    # 2026-07-17 double-projection bug produced exactly this: a (168, 768) float32 file that loads
    # cleanly and passes every shape check, so only the diagonal cosine separates it from the
    # canonical basis. Rather than ship a real retracted artifact as a fixture — a wrong file in a
    # public repository is the very hazard this guard exists to prevent — the near-miss is built
    # here: the checkpoint's own text space, perturbed just past `tol`. That probes the tolerance
    # boundary, which a merely-unrelated tensor would not.
    import torch

    v = torch.load(canon, weights_only=False)
    E = np.asarray(v["embeddings"], dtype=np.float64)
    E = E / np.clip(np.linalg.norm(E, axis=-1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(0)
    perturbed = E + 0.005 * rng.standard_normal(E.shape)  # diagonal cosine ~0.99, just past tol
    near_miss = tmp_path / "near_miss_vocab.pt"
    torch.save({"embeddings": perturbed, "vocabulary": v["vocabulary"]}, near_miss)

    with pytest.raises(ValueError, match="not the text space"):
        assert_canonical_vocabulary(str(ckpt), str(near_miss))


# ── saving contract ──────────────────────────────────────────────────────────────────────────
def test_save_figure_writes_png_pdf_and_source_table(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "FIGURES_ROOT", tmp_path)
    style.apply_style()
    fig, ax = plt.subplots()
    _ = ax.plot([0, 1], [0, 1])
    table = pd.DataFrame({"cue": ["a"], "value": [1.0]})
    paths = style.save_figure(fig, "fig-test", "my-figure", table=table)
    plt.close(fig)

    assert paths["png"].exists() and paths["pdf"].exists() and paths["csv"].exists()
    assert paths["png"].name == "my-figure.png"
    assert pd.read_csv(paths["csv"]).equals(table)


# ── shared presentation: one palette, one title convention ───────────────────────────────────
#: Colour literals that predate `reproduction/config/figures.yaml`. A hex code in figure code means a figure has
#: gone its own way, which is exactly the drift the config exists to stop.
HEX_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b")
#: `style.py` holds the fallback defaults, and `concept_model_profile` documents the original
#: figure's own encoding in prose; both are checked by other tests instead.
PALETTE_EXEMPT = {"style.py"}


def test_no_figure_module_hardcodes_a_colour():
    """Every colour comes from `reproduction/config/figures.yaml`, so the palette can be changed in one place."""
    offenders = []
    for path in _existing(FIGURE_MODULES):
        if path.name in PALETTE_EXEMPT:
            continue
        for lineno, text in _string_literals(path):
            if HEX_LITERAL.fullmatch(text.strip()):
                offenders.append(f"{path}:{lineno}: {text!r}")
    assert not offenders, (
        "figure code hardcodes a colour instead of reading reproduction/config/figures.yaml:\n"
        + "\n".join(offenders)
    )


def test_palette_matches_the_original_paper():
    """The published palette (archived reproduction/config/plotting.yaml `color_palette_real_synthetic`)."""
    assert style.color("real") == "#1f77b4"  # RGB (0.122, 0.467, 0.706)
    assert style.color("synthetic") == "#ff7f0e"  # RGB (1.000, 0.498, 0.055)
    # Signed quantities reuse the same two hues so "orange = toward synthetic" reads everywhere.
    assert style.color("positive") == style.color("synthetic")
    assert style.color("negative") == style.color("real")


def test_title_case_preserves_acronyms_and_technical_terms():
    """`str.title()` would wreck every one of these, which is why the helper is hand-written."""
    cases = {
        "class separation": "Class Separation",
        "most real-like": "Most Real-like",
        "A  Cues associated with detector score": "A  Cues Associated with Detector Score",
        "SynthCLIC": "SynthCLIC",
        "CNNSpot": "CNNSpot",
        "predictive power": "Predictive Power",
    }
    for raw, want in cases.items():
        assert style.title_case(raw) == want, f"{raw!r} -> {style.title_case(raw)!r}, want {want!r}"
