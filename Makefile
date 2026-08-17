SHELL := /bin/bash
.DEFAULT_GOAL := help

# Load .env if present (optional; e.g. WANDB_* for the beta sweep)
-include .env
export

CNNSPOT_DIR := data/checkpoints/cnnspot
AUTO_TERMINATE ?= false

# ── Environment ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Create .venv from uv.lock with every extra (CPU torch; GPU=1 for the cu128 wheels)
	uv sync --extra all $(if $(GPU),--no-default-groups --group gpu,)

.PHONY: test
test: ## Run the test suite that pins the science
	uv run pytest tests/

.PHONY: lint
lint: ## Ruff check + format over the tracked Python sources
	uv run ruff check --fix src scripts tests
	uv run ruff format src scripts tests

.PHONY: requirements
requirements: ## Re-export the pinned requirements*.txt from uv.lock (run after any dependency change)
	uv export --no-hashes --no-emit-project --extra all --no-default-groups -o requirements.txt
	uv export --no-hashes --no-emit-project --extra all --no-default-groups --group gpu \
		-o requirements-gpu.txt
	@# uv export does not emit the index line the cu128 wheels need; re-add it in place.
	@grep -q '^--extra-index-url' requirements-gpu.txt \
		|| sed -i '2a --extra-index-url https://download.pytorch.org/whl/cu128' requirements-gpu.txt
	@echo "requirements.txt + requirements-gpu.txt re-exported from uv.lock"

.PHONY: check-publication
check-publication: ## Publication gates: retracted artifacts, checkpoint identity, redistributable weights, figure caveats, lockfile drift, blob size
	uv run python scripts/utils/check_publication.py --strict

# ── CNNSpot weights ───────────────────────────────────────────────────────────

.PHONY: download-cnnspot
download-cnnspot: ## Download pre-trained CNNSpot checkpoints (Wang et al., CVPR 2020)
	mkdir -p $(CNNSPOT_DIR)
	@echo "Downloading blur_jpg_prob0.5.pth ..."
	wget -q --show-progress \
		"https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1" \
		-O $(CNNSPOT_DIR)/blur_jpg_prob0.5.pth
	@echo "Downloading blur_jpg_prob0.1.pth ..."
	wget -q --show-progress \
		"https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1" \
		-O $(CNNSPOT_DIR)/blur_jpg_prob0.1.pth
	@echo "SHA-256 checksums:"
	sha256sum $(CNNSPOT_DIR)/*.pth


.PHONY: package-revision
package-revision: ## Bundle all E1-E7 tables/figures into reproduction/revision_export/ (+ tarball) for the write-up repo
	uv run python scripts/export/package_revision_export.py

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' Makefile \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ── F1-F7 final consolidation (see reproduction/experiments/final_consolidation/README.md) ────

.PHONY: finalexp-fetch
finalexp-fetch: ## F1-F7: download the released input snapshot and verify it (no GPU, no rebuild)
	uv run python scripts/finalexp/fetch_snapshot.py $(if $(FROM_DIR),--from-dir $(FROM_DIR),) $(if $(REPO),--repo $(REPO),)

.PHONY: finalexp-release
finalexp-release: ## Convert the built snapshot into its distributable (object-free .npz) form
	uv run python scripts/finalexp/export_snapshot_release.py --out $(or $(OUT),dist/snapshot-release)

.PHONY: finalexp-data
finalexp-data: ## F1-F7: build the frozen reproduction/experiments/data snapshot + manifest (WITH_CFEVAL=1 for Table A, WITH_APPENDIX=1 for Table B)
	uv run python scripts/finalexp/build_data_snapshot.py $(if $(WITH_CFEVAL),--with-cfeval,) $(if $(WITH_APPENDIX),--with-appendix,)
	uv run python scripts/finalexp/prepare_features.py

.PHONY: finalexp-verify
finalexp-verify: ## F1-F7: verify every snapshot artifact against the manifest (run before any experiment)
	uv run python scripts/finalexp/verify_data_snapshot.py

.PHONY: finalexp-f1
finalexp-f1: finalexp-verify ## F1: canonical 1024-d detector stability (5 seeds) + regression anchor
	uv run python scripts/finalexp/run_f1_canonical_stability.py

.PHONY: finalexp-f2
finalexp-f2: ## F2: matched k=1 vs k=8 (effective direction vs individual axes)
	uv run python scripts/finalexp/run_f2_matched_k8.py

.PHONY: finalexp-f3
finalexp-f3: ## F3: projected 768-d analysis head + projection cost
	uv run python scripts/finalexp/run_f3_projected_head.py

.PHONY: finalexp-f4
finalexp-f4: ## F4: 168-cue restricted-information probe vs D_e
	uv run python scripts/finalexp/run_f4_cue_capacity.py

.PHONY: finalexp-f5
finalexp-f5: ## F5: extreme-image rankings from the matched canonical probe
	uv run python scripts/finalexp/run_f1_canonical_stability.py --dataset cnnspot
	uv run python scripts/finalexp/export_f5_rankings.py

.PHONY: finalexp-f6
finalexp-f6: ## F6: cross-dataset projected heads + boundary decomposition
	uv run python scripts/finalexp/run_f6_cross_dataset.py

.PHONY: finalexp-f7
finalexp-f7: ## F7: bridge from the matched heads to the deployed checkpoints and the proxies
	uv run python scripts/finalexp/run_f7_bridge.py

.PHONY: finalexp-tableb
finalexp-tableb: ## TableB: regenerate the appendix per-generator table + export it (needs WITH_APPENDIX=1 snapshot)
	uv run python scripts/finalexp/run_appendix_per_generator.py
	uv run python scripts/export/export_per_generator_table.py

.PHONY: finalexp-all
finalexp-all: finalexp-verify finalexp-f1 finalexp-f2 finalexp-f3 finalexp-f4 finalexp-f6 finalexp-f5 finalexp-f7 ## F1-F7: verify the snapshot, then run every experiment (CPU)
	@echo "All F-experiments complete -> reproduction/experiments/final_consolidation/"

# ── Figures ───────────────────────────────────────────────────────────────────

HF := HF_HOME=data/hf_cache

.PHONY: fig2 fig3 fig4 fig5 fig6 fig7 figures-appendix figures-all tables-compact
fig3: ## Fig 3: extreme canonical detector-score montages (needs the HF image cache)
	$(HF) uv run python scripts/plot/plot_fig3_extreme_scores.py

fig4: ## Fig 4: content-controlled pairs + their named cue changes, N variants (needs the HF image cache)
	$(HF) uv run python scripts/plot/plot_fig4_paired_example.py

fig5: ## Fig 5: population cue interpretation, both layouts + the full 168-cue table
	uv run python scripts/plot/plot_fig5_cue_population.py

fig6: ## Fig 6: cross-dataset boundary difference (single signed panel)
	uv run python scripts/plot/plot_fig6_boundary.py

fig7: ## Fig 7: sparse concept-model profile per dataset
	uv run python scripts/plot/plot_fig7_concept_model.py

figures-appendix: ## Appendix Fig 8 (CNNSpot examples) + Fig 9a/9b (CLIP-IQA distributions + axes)
	$(HF) uv run python scripts/plot/plot_appendix_figures.py

fig2: ## Figs 2+3: SynthBuster+ / SynthCLIC corpus example collages, PDF (needs the HF image cache)
	$(HF) uv run python scripts/plot/plot_fig2_corpus_examples.py

tables-compact: ## The two F-experiment LaTeX tables (cascade + stability) -> reproduction/experiments/figures/tables/
	uv run python scripts/plot/plot_compact_panels.py --tex-only

figures-all: fig2 fig3 fig4 fig5 fig6 fig7 figures-appendix ## Rebuild the whole figure set
	@echo "Figure set rebuilt -> reproduction/experiments/figures/ (see reproduction/experiments/figures/README.md)"
