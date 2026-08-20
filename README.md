<a href="https://www.ultralytics.com"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 YOLOE Data Engine

The YOLOE Data Engine builds and refines training data for [YOLOE](https://docs.ultralytics.com/models/yoloe), the real-time open-vocabulary model in the [Ultralytics](https://www.ultralytics.com) family. It closes the loop between a dataset and a model: existing labels are loaded, a trained YOLOE model predicts over the same images, and any prediction not already covered by a ground-truth box is merged back into the labels. The result is a denser, better-covered dataset for the next training run.

The pipeline handles both **grounding** data (free-form text phrases tied to image regions, such as Flickr30k Entities or mixed-grounding GQA) and **detection** data (a fixed class vocabulary, such as Objects365), and it includes tooling to refine the text prompts attached to grounding labels and to visually inspect everything it produces.

This is a working research repository: a collection of scripts rather than an installable package, with dataset, model, and buffer paths configured inside each script rather than passed on the command line.

[![Ultralytics Actions](https://github.com/ultralytics/yoloe_data_engine/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yoloe_data_engine/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🔧 Requirements

A [Python](https://www.python.org/) environment with the [`ultralytics`](https://github.com/ultralytics/ultralytics) package and its dependencies, plus [PyTorch](https://pytorch.org/), NumPy, [Pillow](https://python-pillow.org/), OpenCV, and Matplotlib for the visualizers:

```bash
pip install ultralytics matplotlib
```

There is no `requirements.txt` — the scripts import from `ultralytics` and are meant to run alongside a local Ultralytics checkout. A CUDA GPU is required for prediction; `DataEngineAgent` takes a list of devices and defaults to a single `cuda:0`, while the example at the bottom of `data_engine_agent.py` spreads the work across four.

## ⚙️ How It Works

The engine has two layers over the same logic:

- **`DataEngine`** (`data_engine.py`) — the single-process core. It loads an Ultralytics `.cache` label file, loads a YOLOE model, sets the class vocabulary, predicts, merges the predictions into the in-memory labels, and writes a new cache file.
- **`DataEngineAgent`** (`data_engine_agent.py`) — the multi-GPU wrapper. It runs the same stages as parallel worker processes that communicate through JSON files in a buffer directory, one file per image, so a large dataset can be processed in stages and inspected in between.

A typical run moves through four stages:

1. **Load labels.** `DataEngine.load_cached_label()` reads an Ultralytics `.cache` file in either `grounding` or `detection` style. For raw grounding annotations, `DataEngineAgent.multi_process_load_grounding_data()` first explodes the source JSON into one sample JSON per image under the buffer directory.
2. **Predict.** `DataEngine.load_yoloe()` loads the YOLOE segmentation checkpoint, and `set_classes()` sets the vocabulary — from a dataset YAML, an explicit list of names, or the keys of a MobileCLIP text-embedding `.pt` file (that file supplies the class names only; the prompt embeddings are recomputed by the model). `DataEngineAgent.multi_process_batch_model_predict()` then writes one prediction JSON per image.
3. **Merge.** Predictions are matched against the existing boxes, and any that overlap a ground-truth box above the IoU threshold are dropped; the rest are appended to the label. `multi_process_merge_prediction()` folds the results back into the sample files, and `save_cached_label()` writes the updated cache.
4. **Refine and inspect.** `refine_text.py` re-scores the text phrases on grounding labels using YOLOE visual prompt embeddings and writes an updated cache. The `data_visual_*.py` and `grounding_dataset_visualizer.py` scripts render labels and predictions so the output can be checked by eye.

## 🖥️ Usage

Both end-to-end flows are shell scripts. They invoke `python3 yoloe_data_engine/<script>.py`, so run them from the **parent** directory of this checkout:

```bash
# Flickr grounding: refine text prompts, then visualize
bash yoloe_data_engine/do_flickr.sh

# Mixed-grounding (GQA): refine text prompts across two GPUs
bash yoloe_data_engine/do_mixed.sh
```

Individual scripts run from the repository root. `refine_text.py` is the only one that takes command-line arguments:

```bash
# Refine the text prompts on a grounding dataset and write an updated .cache
python3 refine_text.py \
  --img_path ../datasets/flickr/full_images/ \
  --json_file ../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json

# Render generated labels and predictions
python3 data_visual_flickr.py
```

Every other entry point is configured by editing the paths at the top of the file — dataset roots, the YOLOE checkpoint, text-embedding `.pt` files, and buffer directories are hard-coded, and the shell scripts also activate a `clipenv` conda environment. Set these for your own machine before running anything.

## 📁 Repository Structure

| Path                                                | Purpose                                                                      |
| --------------------------------------------------- | ---------------------------------------------------------------------------- |
| `data_engine.py`                                    | `DataEngine` — cache loading, YOLOE prediction, label merging, cache saving  |
| `data_engine_agent.py`                              | `DataEngineAgent` — multi-GPU multiprocess versions of the same stages       |
| `refine_text.py`                                    | Refines grounding text prompts with YOLOE visual prompt embeddings           |
| `remove_segment.py`                                 | Strips masks and segments from a cached dataset                              |
| `visual_json.py`, `grounding_dataset_visualizer.py` | Render labels and predictions for inspection                                 |
| `data_visual*.py`                                   | Per-dataset visualization entry points (Flickr, mixed-grounding, Objects365) |
| `utils.py`                                          | Small dataset counting helpers                                               |
| `do_flickr.sh`, `do_mixed.sh`                       | End-to-end example flows                                                     |
| `log.md`                                            | Scratch record of past runs — label and box counts, not documentation        |

Buffer directories and generated `.cache` files are local artifacts and are not committed.

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs, suggesting features, or submitting code changes, your involvement is crucial.

- **Reporting Issues**: Encounter a bug? Please report it on [GitHub Issues](https://github.com/ultralytics/yoloe_data_engine/issues).
- **Feature Requests**: Have an idea for improvement? Share it via [GitHub Issues](https://github.com/ultralytics/yoloe_data_engine/issues).
- **Pull Requests**: Want to contribute code? Please read our [Contributing Guide](https://docs.ultralytics.com/help/contributing) first, then submit a Pull Request.
- **Feedback**: Share your thoughts and experiences by participating in our official [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey).

A heartfelt thank you 🙏 goes out to all our contributors! Your efforts help make Ultralytics tools better for everyone.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open collaboration and knowledge sharing. This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license promotes transparency and community involvement. See the [LICENSE](LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the copyleft requirements of AGPL-3.0. For commercial use cases, please inquire about an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

## 📮 Contact

For bug reports or feature suggestions, please use [GitHub Issues](https://github.com/ultralytics/yoloe_data_engine/issues). For general questions, discussions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
