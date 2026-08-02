# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

`yoloe_data_engine` holds the working scripts used to build and refine YOLOE grounding and detection training data: load labels from Ultralytics `.cache` or grounding JSON, run YOLOE predictions over them, merge the new boxes back into the labels, and visualize the result. It is a script collection, not a package — there is no `pyproject.toml` and no tests, and dataset/model/buffer paths are hard-coded in each script. The repo is AGPL-3.0 licensed.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# The do_*.sh scripts call `python3 yoloe_data_engine/<script>.py`, so run them from the checkout's PARENT directory
bash yoloe_data_engine/do_flickr.sh # Flickr refinement + visualization flow
bash yoloe_data_engine/do_mixed.sh  # mixed-grounding refinement flow

# Individual scripts run from the repo root
python3 refine_text.py              # refine grounding text prompts, write updated cache
python3 data_visual_flickr.py       # render generated labels/predictions
ruff format . && ruff check --fix . # formatting is applied on PRs by Ultralytics Actions, not a repo config
```

- Every entry point hard-codes dataset, model, cache, and buffer paths (many under `/root/ultra_louis_work/...`) and both `do_*.sh` scripts activate a `clipenv` conda env. Edit those paths before running anything.
- There are no tests and no CI beyond `format.yml` (Ultralytics Actions) and `cla.yml`; nothing validates a change automatically, so run the affected script against a small slice of data before pushing.
- Requires a Python environment with `ultralytics`, PyTorch, NumPy, Pillow, and Matplotlib. There is no `requirements.txt`.

## Architecture

- `data_engine.py` — `DataEngine` is the single-process core: `load_cached_label()` reads Ultralytics `.cache` files in `grounding` or `detection` style, `load_yoloe()`/`set_classes()` configure the model (class names from a YAML or a precomputed `text_embed_pt`), `*_predict_and_update_labels*()` run predictions and merge them into the in-memory labels, and `save_cached_label()` writes the result back.
- `data_engine_agent.py` — `DataEngineAgent` wraps the same work in multiprocessing: `multi_process_load_grounding_data()` explodes grounding JSON into per-image sample JSON under a buffer dir, `multi_process_batch_model_predict()` writes per-image prediction JSON, and `multi_process_merge_prediction()` folds predictions back into the samples. It communicates through JSON files in the buffer dir, not shared memory.
- Merge rule: a prediction whose IoU against an existing box exceeds the threshold is dropped; the rest are appended to the label.
- `refine_text.py` — refines grounding text prompts using YOLOE visual prompt embeddings and writes updated cache files.
- `visual_json.py`, `data_visual*.py`, `grounding_dataset_visualizer.py` — rendering helpers, one per dataset flavor (Flickr, mixed, Object365).
- **Known duplication:** `YoloBox` (box conversion + IoU) exists twice and the two copies have diverged — `data_engine.py` holds the original that `refine_text.py` imports, and `data_engine_agent.py` defines its own larger variant. Fix both together, or consolidate into `data_engine.py` and delete the other rather than adding a third.

## Conventions

- Ultralytics Actions adds the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header and runs Ruff, docformatter, prettier, and codespell on PRs — don't add or revert headers manually, and expect bot commits on the branch.
- The default branch is `master`, not `main`.
- `log.md` is a scratch record of past runs, not documentation; generated buffer directories and cache files stay local and are not committed.
