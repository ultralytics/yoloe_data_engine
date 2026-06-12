"""Utilities for loading, updating, and visualizing YOLOE data caches."""

import os
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image
from tqdm import tqdm
from ultralytics.data.utils import load_dataset_cache_file
from ultralytics.engine.results import Results

workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_names_from_yaml_config(yaml_config):
    """Read class names from a dataset YAML config."""
    import yaml

    if not os.path.exists(yaml_config):
        raise FileNotFoundError(f"YAML config file not found: {yaml_config}")
    with open(yaml_config) as f:
        data_dict = yaml.safe_load(f)
        names = data_dict["names"]
    return names


class YoloBox:
    """Convert bounding boxes between YOLO normalized and xyxy formats."""

    def __init__(self, img_shape: list):
        """Initialize the converter with image height and width."""
        assert len(img_shape) == 2, "img_sz should be (height,width)"
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.xyxy = None
        self.xywhn = None  # normalized xywh

    def load_from_xywhn_normalized(self, bboxes_xywhn):
        """Load normalized xywh boxes and compute xyxy boxes."""
        # xywhn: [N,4]  x_center,y_center,w,h (normalized)
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
        if isinstance(bboxes_xyxy, list):
            bboxes_xyxy = np.array(bboxes_xyxy)

        # Ensure the array is of a numeric type
        bboxes_xyxy = np.array(bboxes_xyxy, dtype=np.float32)

        # xyxy: [N,4]  x0,y0,x1,y1
        bboxes_xywhn = np.zeros_like(bboxes_xyxy)
        if bboxes_xyxy.shape[0] > 0:
            bboxes_xywhn[:, 0] = (
                (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2
            ) / self.img_w
            bboxes_xywhn[:, 1] = (
                (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2
            ) / self.img_h
            bboxes_xywhn[:, 2] = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / self.img_w
            bboxes_xywhn[:, 3] = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / self.img_h

        self.xyxy = bboxes_xyxy

        self.xywhn = bboxes_xywhn
        return self

    def iou(self, bbox_xyxy):
        """Calculate IoU between loaded boxes and one xyxy box."""
        # bbox_xyxy: [4,]  x0,y0,x1,y1
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


class DataEngine:
    """Manage YOLOE model inference and cached dataset labels."""

    def __init__(self, device="cuda"):
        """Initialize the engine for a target device."""
        self.device = device

    def load_yoloe(self):
        """Load the configured YOLOE segmentation model."""
        from ultralytics import YOLOE

        model_path = "/root/ultra_louis_work/ultralytics/yoloe-11l-seg.pt"
        yaml_file = "yoloe-11l-seg.yaml"
        if hasattr(self, "model"):
            # clear the existing model
            del self.model

            torch.cuda.empty_cache()

        self.model = YOLOE(yaml_file).load(model_path).to(self.device)
        print("load model from:", model_path)

    def set_classes(self, yaml_config=None, name_list=None, text_embed_pt=None):
        """Set model class names from a YAML file, list, or text embedding."""
        # only one of yaml_config and name_list should be provided
        assert (yaml_config is None) or (name_list is None), (
            "Only one of yaml_config and name_list should be provided"
        )
        if yaml_config is not None:
            assert name_list is None, (
                "If yaml_config is provided, name_list should be None"
            )
            name_list = get_names_from_yaml_config(yaml_config)
            name_list = list(name_list.values())
            print("Load names from yaml:", yaml_config)

        if text_embed_pt is not None:
            assert os.path.exists(text_embed_pt), (
                f"Text embed pt file not found: {text_embed_pt}"
            )
            txt_map = torch.load(text_embed_pt, map_location=self.device)
            name_list = list(txt_map.keys())
            print("Load text embed from:", text_embed_pt)

        assert name_list is None or isinstance(name_list, list), (
            "name_list should be a list of strings or None"
        )

        if name_list is not None:
            print(f"Set {len(name_list)} classes")
            self.model.set_classes(name_list, self.model.get_text_pe(name_list))
            self.names = name_list

        else:
            print("No classes set")

    def yoloe_predict(self, indice, conf=0.05, save_path=None):
        """Run YOLOE prediction for one cached label index."""
        img_file = self.labels[indice]["im_file"]
        if hasattr(self, "img_source"):
            img_file = os.path.join(self.img_source, img_file)
        result = self.model.predict(img_file, conf=conf)

        if save_path is not None:
            result[0].save(save_path)
            print("save to:", save_path)
        return result

    def yoloe_predict_batch(self, labels, conf=0.05, iou=0.4):
        """Run YOLOE prediction for a batch of cached labels."""
        img_files = []
        for label in labels:
            img_file = label["im_file"]
            if hasattr(self, "img_source"):
                img_file = os.path.join(self.img_source, img_file)
            img_files.append(img_file)
        if not img_files:
            return []

        return list(
            self.model.predict(
                img_files, conf=conf, iou=iou, batch=len(img_files), stream=True
            )
        )

    def __len__(self):
        """Return the number of loaded labels."""
        return len(self.labels)

    def set_img_folder(self, img_source):
        """Set the base image folder used to resolve relative image paths."""
        self.img_source = img_source

    def load_cached_label(
        self, cache_path, data_style="grounding", yaml_config=None, text_embed_pt=None
    ):
        """Load cached labels and metadata for detection or grounding data."""
        self.cache_path = cache_path

        cache = load_dataset_cache_file(Path(cache_path))
        self.cache = cache
        self.labels = cache["labels"]
        print(len(self.labels))

        assert data_style in ["grounding", "detection"]
        self.data_style = data_style

        if data_style == "detection":
            assert yaml_config is not None, (
                "yaml_config must be provided for detection data_style"
            )
            if not os.path.exists(yaml_config):
                raise FileNotFoundError(f"YAML config file not found: {yaml_config}")
            self.yaml_config = yaml_config

            # read names from the  yaml file
            import yaml

            with open(yaml_config) as f:
                data_dict = yaml.safe_load(f)
                self.names = data_dict["names"]

        elif data_style == "grounding":
            assert text_embed_pt is not None, (
                "text_embed_pt must be provided for grounding data_style"
            )
            self.text_embed_pt = text_embed_pt
            if not os.path.exists(text_embed_pt):
                raise FileNotFoundError(
                    f"Text embed pt file not found: {text_embed_pt}"
                )
            else:
                print("Load text embed from:", text_embed_pt)
            txt_map = torch.load(
                text_embed_pt, map_location=self.device, weights_only=False
            )
            self.names = list(txt_map.keys())

    def print_data_info(self):
        """Print information about the dataset labels.

        - data_style
        - Total number of labels
        - Total number of boxes (for detection and grounding).
        """
        print(f"Data style: {self.data_style}")
        print(
            "Keys: {}".format(
                self.labels[0].keys() if len(self.labels) > 0 else "No labels"
            )
        )
        print(f"Total number of labels: {len(self.labels)}")
        if self.data_style in ["detection", "grounding"]:
            total_boxes = sum(len(label.get("bboxes", [])) for label in self.labels)
            print(f"Total number of boxes: {total_boxes}")

    def remove_masks_and_segments(self):
        """Remove mask and segment data from all loaded labels."""
        for label in tqdm(self.labels):
            label["segments"] = []

    def save_cached_label(self, save_path=None):
        """Save the current labels back to a cache file."""
        if save_path is None:
            save_path = self.cache_path
        from copy import deepcopy

        copy_cache = deepcopy(self.cache)
        copy_cache["labels"] = self.labels
        with open(save_path, "wb") as f:
            np.save(f, copy_cache)

    def print_one_label(self, indice):
        """Print key and type information for one loaded label."""
        label = self.labels[indice]
        # print(self.labels[indice])
        print(label.keys())
        print(label["im_file"])
        for key, val in label.items():
            print(f"{key}: {type(val)}")
            if isinstance(val, list):
                print(f"  Length: {len(val)}")
                # if len(val)>0:
                #     print(f"  First 3 elements: {val[:3]}")

            elif isinstance(val, np.ndarray):
                print(f"  Shape: {val.shape}")
                print(f"  Dtype: {val.dtype}")
                print(f"  First 5 elements: {val.flatten()[:5]}")

            elif isinstance(val, dict):
                print(f"  Dict with keys: {list(val.keys())}")

            else:
                print(f"  Value: {val}")

    def detection_predict_and_update_labels(
        self, indice, iou=0.3, replace=True, conf=0.1
    ):
        """Predict and merge detection boxes for one label."""
        result = self.yoloe_predict(indice=indice, conf=conf)
        if not result:
            return
        self._update_detection_label(indice, result[0], iou=iou, replace=replace)

    def detection_predict_and_update_labels_batch(
        self, indices, iou=0.3, replace=False, conf=0.1
    ):
        """Predict and merge detection boxes for a batch of labels."""
        results = self.yoloe_predict_batch([self.labels[i] for i in indices], conf=conf)
        assert len(results) == len(indices), (
            "Mismatch between results and indices length"
        )
        for indice, res in zip(indices, results):
            self._update_detection_label(indice, res, iou=iou, replace=replace)

    def _update_detection_label(self, indice, result_obj, iou=0.3, replace=True):
        assert self.data_style == "detection", (
            "_update_detection_label requires detection data_style"
        )
        boxes = result_obj.boxes
        bboxes_xyxy = boxes.xyxy.cpu().numpy()
        yolo_box = YoloBox(img_shape=result_obj.orig_img.shape[:2]).load_from_xyxy(
            bboxes_xyxy
        )
        bboxes_xywhn = yolo_box.xywhn
        cls = boxes.cls.cpu().numpy()
        assert bboxes_xywhn.shape[0] == cls.shape[0], (
            "Mismatch between number of boxes and classes"
        )
        if replace:
            self.labels[indice]["bboxes"] = bboxes_xywhn
            self.labels[indice]["cls"] = cls
            print(f"Replace with {bboxes_xywhn.shape[0]} boxes")
            return
        keep_indices = []
        for i in range(bboxes_xywhn.shape[0]):
            bbox = bboxes_xywhn[i]
            max_iou = 0
            for j in range(self.labels[indice]["bboxes"].shape[0]):
                exist_bbox = self.labels[indice]["bboxes"][j]
                box1 = (
                    YoloBox(img_shape=result_obj.orig_img.shape[:2])
                    .load_from_xywhn_normalized(bbox[np.newaxis, :])
                    .xyxy[0]
                )
                box2 = (
                    YoloBox(img_shape=result_obj.orig_img.shape[:2])
                    .load_from_xywhn_normalized(exist_bbox[np.newaxis, :])
                    .xyxy[0]
                )
                xi1 = max(box1[0], box2[0])
                yi1 = max(box1[1], box2[1])
                xi2 = min(box1[2], box2[2])
                yi2 = min(box1[3], box2[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = box1_area + box2_area - inter_area
                current_iou = inter_area / union_area if union_area > 0 else 0
                if current_iou > max_iou:
                    max_iou = current_iou
            if max_iou < iou:
                keep_indices.append(i)
        print(f"Append {len(keep_indices)} new boxes out of {bboxes_xywhn.shape[0]}")
        for i in keep_indices:
            bbox = bboxes_xywhn[i]
            c = cls[i]
            self.labels[indice]["bboxes"] = np.vstack(
                [self.labels[indice]["bboxes"], bbox]
            )
            self.labels[indice]["cls"] = np.vstack([self.labels[indice]["cls"], c])

    def grounding_predict_and_update_labels_batch(
        self, indices, iou=0.05, replace=False, conf=0.1
    ):
        """Predict and merge grounding boxes for a batch of labels."""
        results = self.yoloe_predict_batch(indices, conf=conf)
        assert len(results) == len(indices), (
            "Mismatch between results and indices length"
        )
        for indice, res in zip(indices, results):
            self.labels[indice] = self._update_grounding_label(
                self.labels[indice], res, iou=iou, replace=replace
            )

    def _update_grounding_label(self, label, result_obj, iou=0.1, replace=True):
        assert self.data_style == "grounding", (
            "_update_grounding_label requires grounding data_style"
        )
        boxes = result_obj.boxes
        bboxes_xyxy = boxes.xyxy.cpu().numpy()
        yolo_box = YoloBox(img_shape=result_obj.orig_img.shape[:2]).load_from_xyxy(
            bboxes_xyxy
        )
        bboxes_xywhn = yolo_box.xywhn
        cls = boxes.cls.cpu().numpy()

        assert bboxes_xywhn.shape[0] == cls.shape[0], (
            "Mismatch between number of boxes and classes"
        )

        if replace:
            label["bboxes"] = bboxes_xywhn
            label["cls"] = cls
            print(f"Replace with {bboxes_xywhn.shape[0]} boxes")
            return
        keep_indices = []
        for i in range(bboxes_xywhn.shape[0]):
            bbox = bboxes_xywhn[i]
            max_iou = 0
            for j in range(label["bboxes"].shape[0]):
                exist_bbox = label["bboxes"][j]
                box1 = (
                    YoloBox(img_shape=result_obj.orig_img.shape[:2])
                    .load_from_xywhn_normalized(bbox[np.newaxis, :])
                    .xyxy[0]
                )
                box2 = (
                    YoloBox(img_shape=result_obj.orig_img.shape[:2])
                    .load_from_xywhn_normalized(exist_bbox[np.newaxis, :])
                    .xyxy[0]
                )
                xi1 = max(box1[0], box2[0])
                yi1 = max(box1[1], box2[1])
                xi2 = min(box1[2], box2[2])
                yi2 = min(box1[3], box2[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = box1_area + box2_area - inter_area
                current_iou = inter_area / union_area if union_area > 0 else 0
                if current_iou > max_iou:
                    max_iou = current_iou
            if max_iou < iou:
                keep_indices.append(i)

        # get  current texts
        current_texts = label["texts"]
        current_texts = [
            text[0] for text in current_texts
        ]  # remote the list structure inside

        # Get boxes to append from result_obj with class ids and text labels.
        append_bboxes, append_cls, append_text = [], [], []
        for i in keep_indices:
            append_bboxes.append(bboxes_xywhn[i])
            append_cls.append(cls[i])
            append_text.append(self.names[int(cls[i])])

        # update the current texts
        for text in append_text:
            if text not in current_texts:
                current_texts.append(text)
        label["texts"] = [[text] for text in current_texts]  # keep the list structure

        # update the append_cls to match the updated texts
        updated_append_cls = []
        for text in append_text:
            updated_cls = current_texts.index(text)  #
            updated_append_cls.append(updated_cls)
        append_cls = updated_append_cls

        # format
        append_bboxes = np.array(append_bboxes).reshape(-1, 4)
        append_cls = np.array(append_cls).reshape(-1, 1)

        # append the boxes and cls
        for i in range(append_bboxes.shape[0]):
            bbox = append_bboxes[i]
            c = append_cls[i]
            label["bboxes"] = np.vstack([label["bboxes"], bbox])
            label["cls"] = np.vstack([label["cls"], c])
        # print how many boxes are appended
        print(
            f"Append {append_bboxes.shape[0]} new boxes out of {bboxes_xywhn.shape[0]}"
        )
        return label

    def label_append_instance(self, indice, bboxes, cls, texts=None):
        """Validate instance data before appending it to one label."""
        assert len(bboxes) == len(cls), "Length of bboxes and cls must be the same"

    def visual_and_save2(
        self, indice=None, filename=None, save_path="./visualize2.jpg"
    ):
        """Visualizes a label using ultralytics.engine.results.Results and saves it."""
        assert self.data_style in ["grounding", "detection"]
        print("Visualizing index:", indice)

        if indice is None:
            assert filename is not None, "Either indice or filename must be provided"

            for idx, label in enumerate(self.labels):
                im_file = label["im_file"]
                # if hasattr(self, 'img_source'):
                #     im_file = os.path.join(self.img_source, im_file)
                print(im_file)
                if os.path.basename(im_file) == filename:
                    indice = idx
                    print(f"Found filename {filename} at index {indice}")
                    break
            assert indice is not None, f"Filename {filename} not found in labels"

        label = self.labels[indice]
        print("label keys:", label.keys())
        im_file = label["im_file"]

        if hasattr(self, "img_source"):
            im_file = os.path.join(self.img_source, im_file)

        orig_img = np.array(Image.open(im_file))
        img_h, img_w = orig_img.shape[:2]

        bboxes_xywhn = label["bboxes"]
        cls = label["cls"]
        if self.data_style == "detection":
            assert self.yaml_config is not None, (
                "yaml_config must be provided for detection data_style"
            )
            names = self.names

        elif self.data_style == "grounding":
            names = label.get("texts", None)
            names = [text[0] for text in names]  # remote the list structure inside

        if isinstance(names, (list, tuple)):
            names = {int(i): str(n) for i, n in enumerate(names)}
        elif isinstance(names, dict):
            names = {int(k): str(v) for k, v in names.items()}
        else:
            raise TypeError(f"Unsupported type for names: {type(names)}")

        yolo_box = YoloBox(img_shape=(img_h, img_w)).load_from_xywhn_normalized(
            bboxes_xywhn
        )
        bboxes_xyxy = yolo_box.xyxy.astype(np.float32)

        print("Number of boxes to visualize:", bboxes_xywhn.shape[0])
        num_boxes = bboxes_xyxy.shape[0]

        cls_array = np.asarray(cls)
        if cls_array.ndim == 0:
            cls_array = cls_array.reshape(1)
        cls_array = cls_array.reshape(-1)

        if cls_array.dtype == object:
            processed_cls = []
            for item in cls_array:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat_item = np.asarray(item).flatten()
                    processed_cls.append(float(flat_item[0]) if flat_item.size else 0.0)
                else:
                    processed_cls.append(float(item))
            cls_array = np.array(processed_cls, dtype=np.float32)
        else:
            cls_array = cls_array.astype(np.float32)

        if cls_array.shape[0] != num_boxes:
            raise ValueError(
                f"Mismatch between boxes ({num_boxes}) and class labels "
                f"({cls_array.shape[0]})"
            )

        conf = np.ones((num_boxes, 1), dtype=np.float32)
        plot_cls = cls_array.reshape(num_boxes, 1)

        if num_boxes == 0:
            boxes_tensor = torch.empty((0, 6), dtype=torch.float32)
        else:
            boxes_array = np.hstack([bboxes_xyxy, conf, plot_cls]).astype(np.float32)
            boxes_tensor = torch.from_numpy(boxes_array)

        print("Boxes data shape:", boxes_tensor.shape)

        # Create Results object
        result = Results(
            orig_img=np.array(orig_img), path=im_file, names=names, boxes=boxes_tensor
        )
        # print each bbox witth cls and name from the result object
        for i in range(result.boxes.shape[0]):
            box = result.boxes[i]
            cls_id = int(box.cls.item())
            cls_name = result.names.get(cls_id, "unknown")
            print(
                f"Box {i}: Class ID = {cls_id}, Class Name = {cls_name}, "
                f"Box Coordinates = {box.xyxy.tolist()}"
            )

        print(
            "Number of boxes in Results object:",
            len(result.boxes) if result.boxes is not None else 0,
        )
        if result.boxes:
            result.boxes.is_track = False  # Set to false to avoid printing track_ids

        result.save(save_path)

        # # Plot the results

        # im_array = result.plot(conf=False) # conf=False to not show confidence scores

        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    DATA_NAME = "flickr"  #

    if DATA_NAME == "Objects365v1":
        de = DataEngine(device="cuda")
        yaml_config = "/root/ultra_louis_work/datasets/Objects365v1.yaml"
        cache_path = "/root/ultra_louis_work/datasets/Objects365v1/labels/train.cache"
        de.load_cached_label(
            cache_path=cache_path, data_style="detection", yaml_config=yaml_config
        )
        de.load_yoloe()
        de.set_classes(yaml_config=yaml_config)  # set classes for the dataset

        batch_size = 64
        for start in tqdm(range(0, len(de), batch_size)):
            batch_indices = list(range(start, min(start + batch_size, len(de))))
            de.detection_predict_and_update_labels_batch(
                batch_indices, iou=0.1, conf=0.1
            )
        de.save_cached_label(save_path=cache_path.replace(".cache", "_updated.cache"))

    elif DATA_NAME == "mixed_grounding":
        # set gpu 3
        device = "cuda:1"
        de = DataEngine(device=device)
        cache_path = (
            "/root/ultra_louis_work/datasets/mixed_grounding/annotations/"
            "final_mixed_train_no_coco_segm.merged.cache"
        )
        text_embed_pt = (
            "/root/ultra_louis_work/datasets/mixed_grounding/gqa/"
            "text_embeddings_mobileclip_blt.pt"
        )
        de.load_cached_label(
            cache_path=cache_path, data_style="grounding", text_embed_pt=text_embed_pt
        )
        de.load_yoloe()

        batch_size = 32
        for start in tqdm(range(1000, len(de), batch_size)):
            batch_indices = list(range(start, min(start + batch_size, len(de))))
            batch_texts = []
            for indice in batch_indices:
                label_texts = de.labels[indice].get("texts", [])
                if isinstance(label_texts, list):
                    for entry in label_texts:
                        if isinstance(entry, (list, tuple)):
                            batch_texts.extend(str(t) for t in entry)
                        else:
                            batch_texts.append(str(entry))

            if batch_texts:
                unique_texts = list(dict.fromkeys(batch_texts))
                de.set_classes(name_list=unique_texts)
            else:
                de.set_classes(name_list=None)

            # debug_indice = batch_indices[10]
            try:
                de.grounding_predict_and_update_labels_batch(
                    batch_indices, iou=0.1, conf=0.1
                )
            except Exception as e:
                print(f"Error processing batch starting at index {start}: {e}")

        de.save_cached_label(save_path=cache_path.replace(".cache", ".updated.cache"))

    elif DATA_NAME == "flickr":
        # set gpu 2
        device = "cuda:2"
        de = DataEngine(device=device)
        cache_path = (
            "/root/ultra_louis_work/datasets/flickr/annotations/"
            "final_flickr_separateGT_train_segm.merged.cache"
        )
        text_embed_pt = (
            "/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"
        )
        de.load_cached_label(
            cache_path=cache_path, data_style="grounding", text_embed_pt=text_embed_pt
        )
        de.load_yoloe()

        batch_size = 128
        for start in tqdm(range(1000, len(de), batch_size)):
            batch_indices = list(range(start, min(start + batch_size, len(de))))
            batch_texts = []
            for indice in batch_indices:
                label_texts = de.labels[indice].get("texts", [])
                if isinstance(label_texts, list):
                    for entry in label_texts:
                        if isinstance(entry, (list, tuple)):
                            batch_texts.extend(str(t) for t in entry)
                        else:
                            batch_texts.append(str(entry))

            if batch_texts:
                unique_texts = list(dict.fromkeys(batch_texts))
                de.set_classes(name_list=unique_texts)
            else:
                de.set_classes(name_list=None)

            # debug_indice = batch_indices[10]
            de.grounding_predict_and_update_labels_batch(
                batch_indices, iou=0.1, conf=0.1
            )

        de.save_cached_label(save_path=cache_path.replace(".cache", ".updated.cache"))
