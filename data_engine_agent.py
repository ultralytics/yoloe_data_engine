"""Multiprocess helpers for generating and merging YOLOE data-engine labels."""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import ultralytics
from tqdm import tqdm

from data_engine import DataEngine

workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

IMAGES_CACHE = None
IMNAME_ANNS_CACHE = None


def to_serializable(obj):
    """Convert numpy and path-like objects into JSON-serializable values."""
    if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


######################## Grounding Data Loading Worker ########################


def _load_grounding_data(buffer_dir, im_dir, imid, anns, folder_name):
    """Worker invoked in subprocesses to build per-image grounding labels."""
    global IMAGES_CACHE, IMNAME_ANNS_CACHE
    dst_dir = os.path.join(buffer_dir, folder_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst_file = os.path.join(dst_dir, str(imid) + ".json")
    if os.path.exists(dst_file):
        return
    from ultralytics.data.converter import merge_multi_segment
    from ultralytics.data.dataset import segments2boxes

    img = IMAGES_CACHE[f"{imid:d}"]
    h, w, f = img["height"], img["width"], img["file_name"]
    im_file = Path(im_dir) / f  # Use the passed im_dir
    bboxes_xyxy = []
    bboxes = []
    segments = []
    cat2id = {}
    texts = []
    if IMNAME_ANNS_CACHE is not None:
        anns_for_img = IMNAME_ANNS_CACHE.get(f, [])
    else:
        anns_for_img = []
    for ann in anns + anns_for_img:
        if (
            len(bboxes_xyxy) > 0
            and YoloBox([int(h), int(w)])
            .load_from_xyxy(bboxes_xyxy)
            .iou(ann["bbox"])
            .max()
            > 0.98
        ):
            continue
        if ann["iscrowd"]:
            continue
        box = np.array(ann["bbox"], dtype=np.float32)
        box[:2] += box[2:] / 2
        box[[0, 2]] /= float(w)
        box[[1, 3]] /= float(h)
        if box[2] <= 0 or box[3] <= 0:
            continue
        caption = ann["caption"]
        cat_name = (
            " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]])
            .lower()
            .strip()
        )
        if not cat_name:
            continue
        if cat_name not in cat2id:
            cat2id[cat_name] = len(cat2id)
            texts.append([cat_name])
        cls = cat2id[cat_name]
        box = [cls, *box.tolist()]
        if box not in bboxes:
            bboxes.append(box)
            if ann.get("segmentation") is not None:
                if len(ann["segmentation"]) == 0:
                    segments.append(box)
                    continue
                elif len(ann["segmentation"]) > 1:
                    s = merge_multi_segment(ann["segmentation"])
                    s = (
                        (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32))
                        .reshape(-1)
                        .tolist()
                    )
                else:
                    s = [j for i in ann["segmentation"] for j in i]
                    s = (
                        (
                            np.array(s, dtype=np.float32).reshape(-1, 2)
                            / np.array([w, h], dtype=np.float32)
                        )
                        .reshape(-1)
                        .tolist()
                    )
                s = [cls, *s]
                segments.append(s)
        bboxes_xyxy.append(ann["bbox"])
    lb = (
        np.array(bboxes, dtype=np.float32)
        if len(bboxes)
        else np.zeros((0, 5), dtype=np.float32)
    )
    if segments:
        classes = np.array([x[0] for x in segments], dtype=np.float32)
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
    lb = np.array(lb, dtype=np.float32)
    label = {
        "im_file": str(im_file),
        "shape": (h, w),
        "cls": lb[:, 0:1] if lb.size > 0 else [],
        "bboxes": lb[:, 1:] if lb.size > 0 else [],
        "segments": [],
        "normalized": True,
        "bbox_format": "xywh",
        "texts": texts,
    }

    def serializeLabel(label):
        lc = copy.deepcopy(label)
        lc["im_file"] = str(lc.get("im_file", ""))
        lc["shape"] = list(lc.get("shape", []))
        lc["cls"] = to_serializable(lc.get("cls", []))
        try:
            cls_arr = np.array(lc["cls"]).reshape(-1)
            lc["cls"] = [int(x) for x in cls_arr.tolist()]
        except Exception:
            pass
        lc["bboxes"] = to_serializable(lc.get("bboxes", []))
        lc["segments"] = to_serializable(lc.get("segments", []))
        lc["texts"] = to_serializable(lc.get("texts", []))
        lc["normalized"] = bool(lc.get("normalized", True))
        lc["bbox_format"] = str(lc.get("bbox_format", "xywh"))
        return lc

    label_serialized = serializeLabel(label)
    # tmp_file = str(dst_file) + ".tmp"
    import json

    with open(dst_file, "w") as file:
        json.dump(label_serialized, file, indent=4, ensure_ascii=False)
    # os.replace(tmp_file, str(dst_file))


def worker_wrapper(args):
    """Unpack worker arguments and load grounding data."""
    return _load_grounding_data(*args)


def init_worker(images_data, imname_anns_data):
    """Initialize worker processes with shared annotation state."""
    global IMAGES_CACHE, IMNAME_ANNS_CACHE
    IMAGES_CACHE = images_data
    IMNAME_ANNS_CACHE = imname_anns_data


# Multi-processing model prediction.


def _batch_model_predict_single_process(self, buffer_dir, im_files, **kwargs):
    """Batch model predict in a single process. This can be a method of DataEngine.

    Args:
        self: DataEngine instance
        buffer_dir: str, buffer directory to save results
        im_files: list of str, image file paths
        kwargs: other keyword arguments for model.predict.

    """
    assert isinstance(self, DataEngine)
    engine = self
    dst_dir = os.path.join(buffer_dir, "model_predict")
    os.makedirs(dst_dir, exist_ok=True)
    conf = kwargs.get("conf", 0.5)
    iou = kwargs.get("iou", 0.4)
    im_names_wo_ext = [
        os.path.splitext(os.path.basename(im_file))[0] for im_file in im_files
    ]
    dst_files = [os.path.join(dst_dir, f"{name}.json") for name in im_names_wo_ext]
    indices = [i for i in range(len(im_files)) if not os.path.exists(dst_files[i])]
    if len(indices) == 0:
        print("All images have been processed, skip.")
        return
    process_img_files = [im_files[i] for i in indices]
    results = list(
        engine.model.predict(
            process_img_files,
            conf=conf,
            iou=iou,
            batch=len(process_img_files),
            stream=True,
        )
    )
    print(f"Processed {len(process_img_files)} images.")
    for i, sample_index in enumerate(indices):
        sample = Sample()
        result = results[i]
        sample.load_from_yoloe_result(result)
        sample.save_to_json(dst_files[sample_index])
    return


def _device_predict_worker(args):
    """Run multi-process model prediction on a specific device.

    args: tuple containing (device, buffer_dir, batches, kwargs).
    """
    device, buffer_dir, batches, kwargs = args

    worker_kwargs = dict(kwargs or {})
    texts = worker_kwargs.pop("texts", None)

    engine = DataEngine(device=device)
    engine.load_yoloe()
    engine.set_classes(name_list=texts)

    for im_files in tqdm(batches, desc=f"Device {device} processing batches"):
        _batch_model_predict_single_process(
            engine, buffer_dir, im_files, **worker_kwargs
        )
    return True


##############################################################################


def _merge_prediction_to_sample_label(buffer_dir, sample_json, model_predict_json):
    """Merge model prediction results into one grounding sample label.

    step 1: first check the filename match, if false, raise error.
    step 2: check the dst file exist, if true, skip.
    step 3: merge model predictions into the sample label, ignoring IoU > 0.5.
    step 4: save the merged label to buffer_dir/merge_prediction/.

    Args:
        buffer_dir: str, buffer directory to save results
        sample_json: str, path to sample grounding label json file
        model_predict_json: str, path to model prediction json file.

    """
    dst_dir = os.path.join(buffer_dir, "merge_prediction")
    os.makedirs(dst_dir, exist_ok=True)
    sample_basename = os.path.basename(sample_json)
    dst_file = os.path.join(dst_dir, sample_basename)
    if os.path.exists(dst_file):
        print(f"[merge] Skip existing: {dst_file}")
        return True

    ground_sample = Sample()
    ground_sample.load_from_grounding_label(sample_json)

    predict_sample = Sample()
    predict_sample.load_from_json(model_predict_json)

    for model_inst in predict_sample.instances:
        # Defensive: skip invalid model instances
        if getattr(model_inst, "bbox", None) is None:
            print(
                "[merge][WARN] skipping model instance with empty bbox in "
                f"'{sample_json}'"
            )
            continue
        try:
            model_bbox = YoloBox(ground_sample.shape).load_from_xyxy(model_inst.bbox)
        except Exception as e:
            print(f"[merge][WARN] failed to parse model bbox for '{sample_json}': {e}")
            continue
        ignore_flag = False
        for sample_inst in ground_sample.instances:
            sample_bbox = YoloBox(ground_sample.shape).load_from_xyxy(sample_inst.bbox)
            iou = sample_bbox.iou(model_bbox.xyxy[0])
            if iou > 0.5:
                ignore_flag = True
                break
        if not ignore_flag:
            ground_sample.instances.append(model_inst)

    ground_sample.save_to_json(dst_file)
    # print(f"[merge] Saved: {dst_file}")
    return True
    # print(f"Merged label saved to {dst_file}")


def merge_prediction_worker(args):
    """Merge one sample and prediction pair inside a worker process."""
    # Support indexed and non-indexed worker argument tuples.
    try:
        if len(args) == 4:
            idx, buffer_dir, sample_json, model_predict_json = args
        else:
            buffer_dir, sample_json, model_predict_json = args
            idx = -1
    except Exception:
        # Fallback if args isn't a tuple/list
        buffer_dir, sample_json, model_predict_json = args
        idx = -1

    if idx < 5 or (idx >= 0 and idx % 5000 == 0):
        print(f"[worker] idx={idx} merging sample='{os.path.basename(sample_json)}'")
    try:
        return _merge_prediction_to_sample_label(
            buffer_dir, sample_json, model_predict_json
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[worker][ERROR] idx={idx} file='{sample_json}': {e!r}\n{tb}")
        return False


##############################################################################


class YoloBox:
    """Convert boxes between xyxy and normalized xywh formats."""

    def __init__(self, img_shape: list):
        """Initialize the converter with image height and width."""
        assert len(img_shape) == 2, "img_sz should be (height,width)"
        # Coerce to numeric floats to handle numpy scalars or strings
        try:
            self.img_h = float(img_shape[0])
            self.img_w = float(img_shape[1])
        except Exception:
            # Try a second attempt via numpy
            try:
                arr = np.array(img_shape).astype(np.float32).reshape(-1)
                self.img_h = float(arr[0])
                self.img_w = float(arr[1])
            except Exception as e:
                raise ValueError(
                    f"Invalid image shape provided to YoloBox: {img_shape}"
                ) from e
        if self.img_h <= 0 or self.img_w <= 0:
            raise ValueError(
                "Image width/height must be positive, "
                f"got img_h={self.img_h}, img_w={self.img_w}"
            )
        self.xyxy = None
        self.xywhn = None  # normalized xywh

    def load_from_xywhn_normalized(self, bboxes_xywhn):
        """Load normalized xywh boxes and compute xyxy boxes."""
        bboxes_xyxy = np.zeros_like(bboxes_xywhn)
        if bboxes_xywhn.shape[0] > 0:
            bboxes_xyxy[:, 0] = (
                bboxes_xywhn[:, 0] - bboxes_xywhn[:, 2] / 2
            ) * self.img_w
            bboxes_xyxy[:, 1] = (
                bboxes_xywhn[:, 1] - bboxes_xywhn[:, 3] / 2
            ) * self.img_h
            bboxes_xyxy[:, 2] = (
                bboxes_xywhn[:, 0] + bboxes_xywhn[:, 2] / 2
            ) * self.img_w
            bboxes_xyxy[:, 3] = (
                bboxes_xywhn[:, 1] + bboxes_xywhn[:, 3] / 2
            ) * self.img_h
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def load_from_xyxy(self, bboxes_xyxy):
        """Load xyxy boxes and compute normalized xywh boxes."""

        # Robustly coerce input to a numeric numpy array of shape (N,4)
        def _to_float_array(x):
            # Accept lists, tuples, numpy arrays, nested shapes
            if isinstance(x, (list, tuple)):
                try:
                    arr = np.array(x, dtype=np.float32)
                except Exception:
                    # Try element-wise conversion
                    flat = []
                    for el in x:
                        if isinstance(el, (list, tuple, np.ndarray)):
                            flat.append([float(v) for v in el])
                        else:
                            flat.append(float(el))
                    arr = np.array(flat, dtype=np.float32)
            elif isinstance(x, np.ndarray):
                arr = x.astype(np.float32, copy=False)
            else:
                # attempt generic conversion
                arr = np.array(x, dtype=np.float32)

            # Normalize shape: if 1D length==4 -> (1,4)
            if arr.ndim == 1 and arr.size == 4:
                arr = arr.reshape(1, 4)
            # If the last dimension is >4, prefer the last 4 entries.
            if arr.ndim == 2 and arr.shape[1] > 4:
                # prefer last 4 entries (common in some formats)
                arr = arr[:, -4:]
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(
                    f"Invalid bbox shape after conversion: {arr.shape}, "
                    f"original={type(x)}"
                )
            return arr

        try:
            bboxes_xyxy = _to_float_array(bboxes_xyxy)
        except Exception as e:
            # Re-raise with more context for upstream logging
            raise TypeError(f"Failed to convert bboxes to float array: {e}") from e

        bboxes_xywhn = np.zeros_like(bboxes_xyxy, dtype=np.float32)
        if bboxes_xyxy.shape[0] > 0:
            bboxes_xywhn[:, 0] = (
                (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2.0
            ) / float(self.img_w)
            bboxes_xywhn[:, 1] = (
                (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2.0
            ) / float(self.img_h)
            bboxes_xywhn[:, 2] = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / float(
                self.img_w
            )
            bboxes_xywhn[:, 3] = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / float(
                self.img_h
            )
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def iou(self, bbox_xyxy):
        """Calculate IoU between loaded boxes and one xyxy box."""
        assert self.xyxy is not None, "self.xyxy is None, please load the box first"
        ious = []
        for i in range(self.xyxy.shape[0]):
            box = self.xyxy[i]
            xi1 = max(box[0], bbox_xyxy[0])
            yi1 = max(box[1], bbox_xyxy[1])
            xi2 = min(box[2], bbox_xyxy[2])
            yi2 = min(box[3], bbox_xyxy[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (box[2] - box[0]) * (box[3] - box[1])
            box2_area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            ious.append(iou)
        return np.array(ious)


class Instance:
    """Store one predicted or labeled object instance."""

    def __init__(self, bbox=None, **kwargs):
        """Initialize an instance with optional bbox and extra metadata."""
        self.bbox = bbox
        self.text = None
        self.conf = None
        self.embed = None
        self.vpe = None
        self.segment = None
        self.other_data = {**kwargs}

    def set_segment(self, segment: np.ndarray):
        """Attach a polygon segment to the instance."""
        self.segment = segment
        assert len(segment.shape) == 2 and segment.shape[1] == 2

    def set_embed(self, embed):
        """Attach a text or visual embedding to the instance."""
        self.embed = embed

    def set_vpe(self, vpe: np.ndarray):
        """Attach a visual prompt embedding to the instance."""
        self.vpe = vpe.squeeze()

    def set_text(self, texts: list, conf: list | None = None):
        """Attach text labels and confidence scores to the instance."""
        self.text = texts
        self.conf = conf
        assert len(texts) == len(conf)

    def get_top_text_conf(self):
        """Return the highest-confidence text label and score."""
        assert self.text is not None and self.conf is not None
        max_conf_index = np.argmax(self.conf)
        return self.text[max_conf_index], self.conf[max_conf_index]

    def to_dict(self):
        """Convert the instance to a JSON-serializable dictionary."""
        return {
            "bbox": to_serializable(self.bbox),
            "text": to_serializable(self.text),
            "conf": to_serializable(self.conf),
            "embed": to_serializable(self.embed),
            "vp": to_serializable(self.vpe),
            "other_data": to_serializable(self.other_data),
        }

    def from_dict(self, data: dict):
        """Load instance fields from a dictionary."""
        self.bbox = data.get("bbox")
        self.text = data.get("text")
        self.conf = data.get("conf")
        self.embed = data.get("embed")
        self.vpe = data.get("vpe")
        self.other_data = data.get("other_data", {})


class Sample:
    """Store image-level sample metadata and object instances."""

    def __init__(self):
        """Initialize an empty sample."""
        self.im_file = None
        self.shape = None
        self.instances = []
        self.texts = []
        self.other_data = {}

    def load_from_grounding_label(self, grounding_data):
        """Load sample instances from a grounding label dictionary or JSON path."""
        if isinstance(grounding_data, str):
            assert grounding_data.endswith(".json"), (
                "If grounding_data is str, it should be a json file path."
            )
            import json

            with open(grounding_data) as f:
                grounding_data = json.load(f)

        assert isinstance(grounding_data, dict), "grounding_data should be a dict"
        self.im_file = grounding_data.get("im_file")
        self.shape = grounding_data.get("shape")
        for text in grounding_data.get("texts"):
            if isinstance(text, list):
                assert len(text) == 1
                self.texts.append(text[0])
            elif isinstance(text, str):
                self.texts.append(text)
            else:
                raise ValueError("text should be str or list of str")
        normalized = grounding_data.get("normalized")
        bbox_format = grounding_data.get("bbox_format")
        self.other_data["bbox_format"] = bbox_format
        self.other_data["normalized"] = normalized
        assert normalized is True
        # assert bbox_format == "xywhn"
        for cls, box, segment in zip(
            grounding_data.get("cls", []),
            grounding_data.get("bboxes", []),
            grounding_data.get("segments", []),
        ):
            # Convert normalized xywh to xyxy for internal consistency
            bbox_xyxy = (
                YoloBox(self.shape)
                .load_from_xywhn_normalized(np.array([box], dtype=np.float32))
                .xyxy[0]
            )
            # Create instance with xyxy bbox
            inst = Instance(bbox=bbox_xyxy.tolist())
            # Attach segment only if well-formed (N,2)
            try:
                seg_arr = np.array(segment, dtype=np.float32)
                if seg_arr.ndim == 2 and seg_arr.shape[1] == 2:
                    inst.set_segment(seg_arr)
            except Exception:
                pass
            cls = int(cls)
            assert cls < len(self.texts)
            text = self.texts[cls]
            assert isinstance(text, str)
            inst.set_text([self.texts[cls]], [-1])
            self.instances.append(inst)

    # def to_grounding_label(self) -> dict:
    #     grounding_data = {}
    #     grounding_data['im_file'] = self.im_file
    #     grounding_data['shape'] = self.shape
    #     grounding_data['texts'] = [[text] for text in self.texts]
    #     bboxes = []
    #     segments = []
    #     cls_list = []
    #     for inst in self.instances:
    #         bboxes.append(inst.bbox)
    #         segments.append(inst.segment)
    #         text, _ = inst.get_top_text_conf()
    #         cls_index = self.texts.index(text)
    #         cls_list.append(cls_index)
    #     grounding_data['bboxes'] = bboxes
    #     grounding_data['segments'] = segments
    #     grounding_data['cls'] = cls_list
    #     grounding_data['normalized'] = True
    #     grounding_data['bbox_format'] = 'xywhn'
    #     return grounding_data

    def load_from_yoloe_result(self, yoloe_result):
        """Load sample instances from a YOLOE result object or JSON path."""
        if isinstance(yoloe_result, str):
            assert yoloe_result.endswith(".json"), (
                "If yoloe_result is str, it should be a json file path."
            )
            import json

            with open(yoloe_result) as f:
                yoloe_result = json.load(f)
            assert isinstance(yoloe_result, dict), "yoloe_result should be a dict"

            self.instances = []
            self.im_file = yoloe_result.get("im_file")
            self.shape = (
                yoloe_result.get("orig_shape", [0, 0])[0],
                yoloe_result.get("orig_shape", [0, 0])[1],
            )
            boxes = yoloe_result.get("boxes", [])
            names = yoloe_result.get("names", [])
        else:
            self.instances = []
            self.im_file = yoloe_result.path
            self.shape = (yoloe_result.orig_shape[0], yoloe_result.orig_shape[1])
            boxes = yoloe_result.boxes
            names = yoloe_result.names
        for box in boxes:
            bbox_xyxy = box.xyxy.cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())
            inst = Instance(bbox=bbox_xyxy.tolist())
            inst.set_text([names[cls]], [float(conf)])
            self.instances.append(inst)

    def to_dict(self):
        """Convert the sample to a JSON-serializable dictionary."""
        return {
            "im_file": to_serializable(self.im_file),
            "instances": [inst.to_dict() for inst in self.instances],
            "other_data": to_serializable(self.other_data),
        }

    def save_to_json(self, json_path):
        """Save the sample to a JSON file."""
        import json

        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        # print(f"Saved sample to {json_path}")

    def load_from_json(self, json_path):
        """Load the sample from a JSON file."""
        import json

        with open(json_path) as f:
            data = json.load(f)
        self.im_file = data.get("im_file")
        self.instances = []
        for inst_data in data.get("instances", []):
            inst = Instance()
            inst.from_dict(inst_data)
            self.instances.append(inst)
        self.other_data = data.get("other_data", {})


class DataEngineAgent:
    """Coordinate data-engine model loading and multiprocess processing."""

    def __init__(
        self, devices=["cuda:0"], buffer_dir="/root/ultra_louis_work/engine_buffer"
    ):
        """Initialize the agent with devices and an output buffer directory."""
        self.buffer_dir = buffer_dir
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.devices = devices

    def load_model_engine(self):
        """Load one DataEngine model per configured device."""
        # self.model_path = model_path
        self.models = []
        for device in self.devices:
            de = DataEngine(device=device)
            de.load_yoloe()
            self.models.append(de)

    def set_classes(self, texts: list | None):
        """Set class text prompts on all loaded models."""
        if not texts:
            self.texts = texts
            return
        for model in self.models:
            model.set_classes(name_list=texts)
        self.texts = texts

    def multi_process_batch_model_predict(
        self, im_dir, texts=None, conf=0.5, iou=0.4, batch_size=3, max_workers=None
    ):
        """Run batch model prediction across multiple device workers."""
        im_files = []
        for file_name in os.listdir(im_dir):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                im_files.append(os.path.join(im_dir, file_name))

        # im_files=im_files[:128]
        print(f"Total images to process: {len(im_files)}")
        batches = [
            im_files[i : i + batch_size] for i in range(0, len(im_files), batch_size)
        ]
        print(f"Total batches: {len(batches)}, batch size: {batch_size}")
        if not batches:
            return []

        if not self.devices:
            raise ValueError("No devices available for multi-process prediction.")

        if max_workers is None:
            max_workers = len(self.devices)
        else:
            max_workers = min(max_workers, len(self.devices))

        worker_devices = self.devices[:max_workers]
        device_count = len(worker_devices)

        process_args = []
        for idx, device in enumerate(worker_devices):
            assigned_batches = batches[idx::device_count]
            if not assigned_batches:
                continue
            kwargs = {"conf": conf, "iou": iou, "texts": texts}
            process_args.append((device, self.buffer_dir, assigned_batches, kwargs))

        if not process_args:
            print("No batches assigned to workers.")
            return []

        results = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(process_args), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_device_predict_worker, args) for args in process_args
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Model predict ..."
            ):
                future.result()
        return results

        # print(f"Saved sample to {dst_file}")

    def multi_process_load_grounding_data(
        self, im_dir, json_file, merge_within_one_image, max_workers=8
    ):
        """Load grounding data from COCO-style annotations with worker processes."""
        print("Start multi-process loading of grounding data...")
        self.im_dir = im_dir
        with open(json_file) as f:
            annotations = json.load(f)
        images_data = {f"{x['id']:d}": x for x in annotations["images"]}
        imid_imname = {f"{im['id']:d}": im["file_name"] for im in annotations["images"]}
        if merge_within_one_image:
            imname_anns_data = defaultdict(list)
            for ann in annotations["annotations"]:
                imid = ann["image_id"]
                imname = imid_imname[f"{imid:d}"]
                ann["caption"] = images_data[f"{ann['image_id']:d}"]["caption"]
                imname_anns_data[imname].append(ann)
            folder_name = "grounding_data_merged"
        else:
            imname_anns_data = None
            folder_name = "grounding_data"
        imid_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            ann["caption"] = images_data[f"{ann['image_id']:d}"]["caption"]
            imid_anns[ann["image_id"]].append(ann)
        self.img_path = annotations.get("img_path", "")
        imids = list(imid_anns.keys())

        print(f"Total images to process: {len(imids)}")

        init_args = (images_data, imname_anns_data)
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)

        # Use an initializer so each worker receives heavy state only once.
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=init_worker, initargs=init_args
        ) as executor:
            tasks = [
                (self.buffer_dir, self.im_dir, imid, imid_anns[imid], folder_name)
                for imid in imids
            ]

            # Chunksize controls how many tasks are sent to a worker at once.
            chunk_size = max(
                1, min(500, len(imids) // (worker_count * 4) if worker_count > 0 else 1)
            )
            print(f"Using {worker_count} workers and chunksize: {chunk_size}")

            list(
                tqdm(
                    executor.map(worker_wrapper, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc="Loading grounding data",
                )
            )

        print("Finished loading grounding data.")

    def multi_process_merge_prediction(self, json_dir, predict_json_dir, max_workers=8):
        """Merge model prediction JSON files into grounding sample JSON files."""
        json_files = []
        predict_json_files = []
        for sample_file_name in os.listdir(json_dir):
            if sample_file_name.endswith(".json"):
                json_path = os.path.join(json_dir, sample_file_name)
                json_files.append(json_path)

                # read json_path and get im_file name
                with open(json_path) as f:
                    sample_data = json.load(f)
                im_file = sample_data.get("im_file")
                im_name = os.path.splitext(os.path.basename(im_file))[0]
                predict_json_path = os.path.join(predict_json_dir, f"{im_name}.json")
                if os.path.exists(predict_json_path):
                    predict_json_files.append(predict_json_path)
                else:
                    predict_json_files.append(None)
        print(f"Total samples to merge: {len(json_files)}")
        # check number of json_files with none predict_json_files
        valid_json_files = []
        valid_predict_json_files = []
        for json_file, predict_json_file in zip(json_files, predict_json_files):
            if predict_json_file is not None:
                valid_json_files.append(json_file)
                valid_predict_json_files.append(predict_json_file)
        json_files = valid_json_files
        predict_json_files = valid_predict_json_files

        print(f"Total samples with predictions: {len(json_files)}")
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)
        print(f"[merge] Using worker_count={worker_count}")

        process_args = []
        for i in range(len(json_files)):
            # include index for debug prints inside workers
            process_args.append(
                (i, self.buffer_dir, json_files[i], predict_json_files[i])
            )

        # Show a few samples for debugging
        preview_n = min(3, len(process_args))
        for k in range(preview_n):
            _, _, s, p = process_args[k]
            print(f"[merge] Task preview[{k}]: sample='{s}', predict='{p}'")

        # Use 'spawn' to avoid fork-related issues and set a chunksize for throughput
        ctx = mp.get_context("spawn")
        chunksize = max(
            1,
            min(
                500, len(process_args) // (worker_count * 4) if worker_count > 0 else 1
            ),
        )
        print(
            f"[merge] Submitting {len(process_args)} tasks with chunksize={chunksize}"
        )

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            iterable = executor.map(
                merge_prediction_worker, process_args, chunksize=chunksize
            )
            ok = 0
            total = 0
            for result in tqdm(
                iterable, total=len(process_args), desc="Merging predictions"
            ):
                total += 1
                if result:
                    ok += 1
                if total % 10000 == 0:
                    print(f"[merge] Progress: {ok}/{total} succeeded")
        print(f"[merge] Done: {ok}/{total} succeeded")

    def _merge_predict(self):
        pass


def read_numpy_and_print(path=None):
    """Load a cache file and print its contents for inspection."""

    def load_dataset_cache_file(path: Path) -> dict:
        import gc

        gc.disable()
        cache = np.load(str(path), allow_pickle=True).item()
        gc.enable()
        return cache

    path = "/root/ultra_louis_work/engine_buffer/grounding_data/5.cache"
    data = load_dataset_cache_file(path)
    print(data)


if __name__ == "__main__":
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    DATA = "mixed_grounding"  # "mixed_grounding"

    if DATA == "flickr":
        agent = DataEngineAgent(
            devices=devices,
            buffer_dir="/root/ultra_louis_work/runs/flickr_engine_buffer",
        )
        json_file = (
            "/root/ultra_louis_work/datasets/flickr/annotations/"
            "final_flickr_separateGT_train_segm.json"
        )
        im_dir = "../datasets/flickr/full_images/"
        mobileclip_text_embed_pt = (
            "/root/ultra_louis_work/datasets/mixed_grounding/gqa/"
            "text_embeddings_mobileclip_blt.pt"
        )

        import torch

        txt_map = torch.load(mobileclip_text_embed_pt, map_location="cuda:0")
        name_list = list(txt_map.keys())[:50000]
        agent.multi_process_merge_prediction(
            json_dir="/root/ultra_louis_work/runs/flickr_engine_buffer/grounding_data_merged",
            predict_json_dir="/root/ultra_louis_work/runs/flickr_engine_buffer/model_predict",
            max_workers=8,
        )

    elif DATA == "mixed_grounding":
        agent = DataEngineAgent(
            devices=devices,
            buffer_dir="/root/ultra_louis_work/runs/mixed_engine_buffer",
        )
        json_file = (
            "../datasets/mixed_grounding/annotations/"
            "final_mixed_train_no_coco_segm.json"
        )
        im_dir = "../datasets/mixed_grounding/gqa/images"
        mobileclip_text_embed_pt = (
            "/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"
        )

        # import torch
        # txt_map= torch.load(mobileclip_text_embed_pt, map_location="cuda:0")
        # name_list=list(txt_map.keys())[:50000]
        agent.multi_process_merge_prediction(
            json_dir="/root/ultra_louis_work/runs/mixed_engine_buffer/grounding_data_merged",
            predict_json_dir="/root/ultra_louis_work/runs/mixed_engine_buffer/model_predict",
            max_workers=8,
        )
