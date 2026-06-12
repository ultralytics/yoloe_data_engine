"""Refine grounding text labels with YOLOE visual prompt embeddings."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
from ultralytics.data.converter import merge_multi_segment
from ultralytics.data.dataset import (
    DATASET_CACHE_VERSION,
    GroundingDataset,
    get_hash,
    save_dataset_cache_file,
    segments2boxes,
)
from ultralytics.models import yolo
from ultralytics.utils import LOGGER, TQDM

from data_engine import DataEngine, YoloBox

workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


class RefineGroundingDataset(GroundingDataset, DataEngine):
    """Grounding dataset that augments and refines text labels with YOLOE."""

    def vpe_text(self, source, visual_prompts, texts):
        """Cal the visual prompt embedding for the current image and visual prompts.

        Args:
            source: image source
            visual_prompts: dict, containing "bboxes" and "cls" lists
            texts: list of str, the texts to be matched
        Returns:
            matched texts for each box: tensor, (N,).

        """
        yoloe_model = self.model
        predictor = yolo.yoloe.YOLOEVPDetectPredictor
        if type(yoloe_model.predictor) is not predictor:
            yoloe_model.predictor = predictor(
                overrides={
                    "task": yoloe_model.model.task,
                    "mode": "predict",
                    "save": False,
                    "batch": 1,
                },
                _callbacks=yoloe_model.callbacks,
            )
        # get the vpe from current image and visual prompts
        prompts = {
            "bboxes": visual_prompts["bboxes"],
            "cls": list(range(len(visual_prompts["cls"]))),
        }
        num_cls = len(set(prompts["cls"]))
        yoloe_model.model.model[-1].nc = num_cls
        yoloe_model.model.model[-1].no = (
            num_cls + yoloe_model.model.model[-1].reg_max * 4
        )
        yoloe_model.model.names = [f"object{i}" for i in range(num_cls)]
        yoloe_model.predictor.set_prompts(prompts.copy())
        yoloe_model.predictor.setup_model(model=yoloe_model.model)
        vpe = yoloe_model.predictor.get_vpe(source).squeeze(0)

        tpe = yoloe_model.get_text_pe(texts).squeeze(0)

        # normalize
        vpe = torch.nn.functional.normalize(vpe, dim=-1, p=2)
        tpe = torch.nn.functional.normalize(tpe, dim=-1, p=2)
        # cal the similarity and return the text for each box
        similarities = (vpe @ tpe.T).softmax(dim=-1)  # (N, M)
        matched_indices = similarities.argmax(dim=-1)  # (N,)
        matched_texts = [texts[i] for i in matched_indices.tolist()]
        return matched_texts

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """Load annotations and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Cached labels and related information.

        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)

        # images = {f"{im['id']:d}": im for im in annotations["images"]}

        # Map image IDs to file names
        imid_imname = {f"{im['id']:d}": im["file_name"] for im in annotations["images"]}

        # map image names to annotations
        imname_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            imid = ann["image_id"]
            imname = imid_imname[f"{imid:d}"]
            imname_anns[imname].append(ann)

        # # map sample id to the annotations
        # img_ids= [im["id"] for im in annotations["images"]]
        # imid_anns= defaultdict(list)
        # for id in img_ids:
        #     imid_anns[id]=imname_anns[imid_imname[f"{id:d}"]]

        images = {f"{x['id']:d}": x for x in annotations["images"]}
        imid_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            imid_anns[ann["image_id"]].append(ann)

        if not hasattr(self, "model") or self.model is None:
            self.load_yoloe()

        for img_id, anns in TQDM(
            imid_anns.items(), desc=f"Reading annotations {self.json_file}"
        ):
            # if img_id > 16*10: break  # for testing
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes_xyxy = []
            bboxes = []
            segments = []
            cat2id = {}
            texts = []

            anns_for_img = imname_anns[f]

            for ann in anns + anns_for_img:
                if (
                    len(bboxes_xyxy) > 0
                    and YoloBox([int(h), int(w)])
                    .load_from_xyxy(bboxes_xyxy)
                    .iou(ann["bbox"])
                    .max()
                    > 0.98
                ):
                    # print("skip duplicate box")
                    continue
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
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
                cls = cat2id[cat_name]  # class
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
                                (
                                    np.concatenate(s, axis=0)
                                    / np.array([w, h], dtype=np.float32)
                                )
                                .reshape(-1)
                                .tolist()
                            )
                        else:
                            s = [
                                j for i in ann["segmentation"] for j in i
                            ]  # all segments concatenated
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
                bboxes_xyxy.append(ann["bbox"])  # add xyxy box for iou calculation

            lb = (
                np.array(bboxes, dtype=np.float32)
                if len(bboxes)
                else np.zeros((0, 5), dtype=np.float32)
            )

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [
                    np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments
                ]  # (cls, xy1...)
                lb = np.concatenate(
                    (classes.reshape(-1, 1), segments2boxes(segments)), 1
                )  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            label = {
                "im_file": im_file,
                "shape": (h, w),
                "cls": lb[:, 0:1],  # n, 1
                "bboxes": lb[:, 1:],  # n, 4
                "segments": segments,
                "normalized": True,
                "bbox_format": "xywh",
                "texts": texts,
            }

            #

            x["labels"].append(label)

        #######  append boxes

        batch_size = 64

        self.data_style = "grounding"
        for start in TQDM(range(0, len(x["labels"]), batch_size)):
            batch_indices = list(
                range(start, min(start + batch_size, len(x["labels"])))
            )
            batch_texts = []
            for indice in batch_indices:
                label_texts = x["labels"][indice].get("texts", [])
                if isinstance(label_texts, list):
                    for entry in label_texts:
                        if isinstance(entry, (list, tuple)):
                            batch_texts.extend(str(t) for t in entry)
                        else:
                            batch_texts.append(str(entry))

            if batch_texts:
                unique_texts = list(dict.fromkeys(batch_texts))
                self.set_classes(name_list=unique_texts)
            else:
                self.set_classes(name_list=None)

            results = self.yoloe_predict_batch(
                [x["labels"][i] for i in batch_indices], conf=0.1, iou=0.4
            )
            assert len(results) == len(batch_indices), (
                "Mismatch between results and batch_indices length"
            )
            for indice, res in zip(batch_indices, results):
                iou = 0.1  # append new boxes when iou < 0.1
                replace = False  # do not replace existing boxes
                x["labels"][indice] = self._update_grounding_label(
                    x["labels"][indice], res, iou=iou, replace=replace
                )

        self.load_yoloe()  # reload to reset class number

        #####  refine the bbox texts
        imname_image = {im["file_name"]: im for im in annotations["images"]}
        for indice, label in TQDM(
            enumerate(x["labels"]), desc="Refining texts for grounding data"
        ):
            bboxes_xyxy = (
                YoloBox((int(label["shape"][0]), int(label["shape"][1])))
                .load_from_xywhn_normalized(label["bboxes"])
                .xyxy
            )
            visual = {"bboxes": bboxes_xyxy, "cls": list(range(bboxes_xyxy.shape[0]))}
            texts = []
            for text_list in label["texts"]:
                texts.extend(text_list)
            print("original texts for image ", ":", texts)
            caption = imname_image[Path(label["im_file"]).name]["caption"].replace(
                ".", ""
            )
            caption_texts = caption.split()
            texts.extend(caption_texts)
            print("caption_texts for image ", ":", caption_texts)
            texts = list(set(texts))
            matched_texts = self.vpe_text(
                source=label["im_file"], visual_prompts=visual, texts=texts
            )
            matches_texts_set = list(set(matched_texts))
            label["texts"] = [[text] for text in matches_texts_set]
            # take cls as the index in the matched texts set
            label["cls"] = [matches_texts_set.index(text) for text in matched_texts]

            print(label["cls"])
            print(matched_texts)
            print(label["texts"])
            x["labels"][indice] = label

        x["hash"] = get_hash(self.json_file)

        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x


DATA_DIR = "../datasets/"
Objects365v1 = "../datasets/Objects365v1.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default=DATA_DIR + "flickr/full_images/")
parser.add_argument(
    "--json_file",
    type=str,
    default=DATA_DIR + "flickr/annotations/final_flickr_separateGT_train_segm.json",
)
args = parser.parse_args()


data = RefineGroundingDataset(
    img_path=args.img_path,
    json_file=args.json_file,
)
