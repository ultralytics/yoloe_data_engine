# YOLOE Data Engine Pipeline

This repository contains working scripts for preparing and refining YOLOE grounding and detection training data. The
pipeline is currently script-driven and expects dataset paths, text embeddings, model weights, and buffer directories to
be configured in the relevant Python or shell script before running.

## Pipeline

1. Load grounding or detection labels from JSON/cache files.
   - `DataEngine.load_cached_label(...)` loads Ultralytics `.cache` files for `grounding` or `detection` data.
   - `DataEngineAgent.multi_process_load_grounding_data(...)` converts grounding JSON annotations into per-image sample
     JSON files.
2. Run YOLOE predictions.
   - `DataEngine.load_yoloe()` loads the configured YOLOE model.
   - `DataEngineAgent.multi_process_batch_model_predict(...)` writes per-image model prediction JSON files under the
     configured buffer directory.
3. Merge model predictions into labels.
   - Predictions with high overlap against existing boxes are skipped.
   - Remaining predictions are added to sample labels and saved for cache generation or inspection.
4. Inspect and refine outputs.
   - Visualization helpers such as `visual_json.py`, `data_visual_flickr.py`, `data_visual_mixed.py`, and
     `grounding_dataset_visualizer.py` can render generated labels and predictions.
   - `refine_text.py` refines grounding text prompts and writes updated cache files.

## Example Scripts

- `do_flickr.sh` runs the Flickr refinement and visualization flow.
- `do_mixed.sh` runs the mixed-grounding refinement flow.
- Edit the hard-coded dataset, model, cache, and environment paths in these scripts before running them.

## Notes

- Several scripts assume an Ultralytics checkout and local dataset layout under `/root/ultra_louis_work/...`.
- Use a Python environment with `ultralytics`, PyTorch, NumPy, Pillow, Matplotlib, and the other imports required by the
  selected script.
- Generated buffer directories and cache files are local artifacts and are not committed here.
