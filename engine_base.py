
from matplotlib.pylab import sample
import ultralytics,os   
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



def CHECK_SEGMENT(segment):
    # Allow None segments (e.g., when loading data before mask generation)
    if segment is None:
        pass
    
    try:
        seg_arr = np.array(segment, dtype=np.float32)
        if seg_arr.ndim != 2 or seg_arr.shape[1] != 2:
            raise ValueError("Segment must be a 2D array with shape (N, 2).")
    except Exception as e:
        print("Segment conversion error:", e)
        raise ValueError(f"Invalid segment format: {e}")



import os
from pathlib import Path
import numpy as np
import cv2

from pathlib import Path as _Path
def to_serializable(obj):
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



class YoloBox:
    def __init__(self, img_shape: list):
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
                raise ValueError(f"Invalid image shape provided to YoloBox: {img_shape}") from e
        if self.img_h <= 0 or self.img_w <= 0:
            raise ValueError(f"Image width/height must be positive, got img_h={self.img_h}, img_w={self.img_w}")
        self.xyxy = None
        self.xywhn = None  # normalized xywh

    def load_from_xywhn_normalized(self, bboxes_xywhn):
        bboxes_xyxy = np.zeros_like(bboxes_xywhn)
        if bboxes_xywhn.shape[0] > 0:
            bboxes_xyxy[:, 0] = (bboxes_xywhn[:, 0] - bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 1] = (bboxes_xywhn[:, 1] - bboxes_xywhn[:, 3] / 2) * self.img_h
            bboxes_xyxy[:, 2] = (bboxes_xywhn[:, 0] + bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 3] = (bboxes_xywhn[:, 1] + bboxes_xywhn[:, 3] / 2) * self.img_h
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def load_from_xyxy(self, bboxes_xyxy):
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

            # Squeeze excessive dims e.g., (1,1,4) -> (1,4)
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            # Normalize shape: if 1D length==4 -> (1,4)
            if arr.ndim == 1 and arr.size == 4:
                arr = arr.reshape(1, 4)
            # If last dimension is >4 (e.g., [cls,x,y,w,h]), try to take first 4 or last 4
            if arr.ndim == 2 and arr.shape[1] > 4:
                # prefer last 4 entries (common in some formats)
                arr = arr[:, -4:]
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"Invalid bbox shape after conversion: {arr.shape}, original={type(x)}")
            return arr

        try:
            bboxes_xyxy = _to_float_array(bboxes_xyxy)
        except Exception as e:
            # Re-raise with more context for upstream logging
            raise TypeError(f"Failed to convert bboxes to float array: {e}") from e

        bboxes_xywhn = np.zeros_like(bboxes_xyxy, dtype=np.float32)
        if bboxes_xyxy.shape[0] > 0:
            bboxes_xywhn[:, 0] = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2.0) / float(self.img_w)
            bboxes_xywhn[:, 1] = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2.0) / float(self.img_h)
            bboxes_xywhn[:, 2] = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / float(self.img_w)
            bboxes_xywhn[:, 3] = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / float(self.img_h)
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def iou(self, bbox_xyxy):
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


num_inst=0
num_none_segment=0

class Instance:
    def __init__(self, bbox=None, **kwargs):
        self.bbox = bbox
        self.text = None
        self.conf = None
        self.embed = None
        self.vpe = None
        self.segment = None
        self.other_data = {**kwargs}

    def set_segment(self, segment: np.ndarray):
        self.segment = segment
        CHECK_SEGMENT(segment)

    def set_embed(self, embed):
        self.embed = embed

    def set_vpe(self, vpe: np.ndarray):
        self.vpe = vpe.squeeze()

    def set_text(self, texts: list, conf: list = None):
        self.text = texts
        self.conf = conf
        assert len(texts) == len(conf)

    def get_top_text_conf(self):
        assert self.text is not None and self.conf is not None
        max_conf_index = np.argmax(self.conf)
        return self.text[max_conf_index], self.conf[max_conf_index]

    def to_dict(self):
        CHECK_SEGMENT(self.segment)

        return {
            'bbox': to_serializable(self.bbox),
            'text': to_serializable(self.text),
            'conf': to_serializable(self.conf),
            'embed': to_serializable(self.embed),
            'vpe': to_serializable(self.vpe),
            "segment": to_serializable(self.segment),
            'other_data': to_serializable(self.other_data)
        }
    def from_dict(self, data: dict):

        
        # Normalize bbox to 1D length-4 if possible
        bbox = data.get('bbox')
        if bbox is not None:
            try:
                arr = np.array(bbox, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.reshape(-1, 4)[0]
                elif arr.ndim == 1 and arr.size == 4:
                    pass
                else:
                    # leave as-is; upper layer may skip if malformed
                    pass
                bbox = arr.tolist()
            except Exception:
                pass
        self.bbox = bbox
        self.text = data.get('text')
        self.conf = data.get('conf')
        self.embed = data.get('embed')
        self.segment = data.get('segment')
        # backward compatibility: some files may use 'vp' key
        self.vpe = data.get('vpe', data.get('vp'))
        self.other_data = data.get('other_data', {})



class Sample:
    def __init__(self):
        self.im_file = None
        self.shape = None
        self.instances = []
        self.texts = []
        self.other_data = {}

    @property
    def all_texts(self):
        unique_texts = set()
        for inst in self.instances:
            if inst.text:
                unique_texts.update(inst.text)
        return list(unique_texts)
    

    def load_from_grounding_label(self, grounding_data: dict | str | Path):

        if isinstance(grounding_data, Path):
            grounding_data = str(grounding_data)

        if isinstance(grounding_data, str):
            assert grounding_data.endswith(".json"), "If grounding_data is str, it should be a json file path."
            import json
            with open(grounding_data, 'r') as f:
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
        segments=grounding_data.get("segments", [])
        if len(segments)==0:
            grounding_data["segments"]=[None for _ in range(len(grounding_data.get("bboxes", [])))]
        for cls, box, segment in zip(grounding_data.get("cls", []), grounding_data.get("bboxes", []), grounding_data.get("segments", [])):
            # Convert normalized xywh to xyxy for internal consistency
            bbox_xyxy = YoloBox(self.shape).load_from_xywhn_normalized(np.array([box], dtype=np.float32)).xyxy[0]
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
        return self
  
    
    def get_shape(self,im_file):
        from PIL import Image
        with Image.open(im_file) as img:
            return img.height, img.width

    def to_grounding_label(self) -> dict:



        global num_inst, num_none_segment


        grounding_data = {}
        grounding_data['im_file'] = self.im_file        
        if self.shape:
        
            grounding_data['shape'] = self.shape
        else:
            grounding_data['shape'] = self.get_shape(self.im_file)


        texts= self.all_texts
        # store as [[text]] for compatibility
        grounding_data['texts'] = [[t] for t in texts]
        bboxes = []
        segments = []
        cls_list = []
        for inst in self.instances:
            # Normalize bbox input shape to 1D length-4
            try:
                bb = np.array(inst.bbox, dtype=np.float32).reshape(-1, 4)[0]
            except Exception:
                # skip invalid bbox
                continue
            bbox_n = YoloBox(grounding_data['shape']).load_from_xyxy(bb).xywhn[0]
            bboxes.append(bbox_n)
            
            

            num_inst = num_inst + 1
            segment=inst.segment

            if  segment is None or segment==[]:
                num_none_segment = num_none_segment + 1
                segment=np.array([inst.bbox[0], inst.bbox[1], inst.bbox[2], inst.bbox[1],
                                    inst.bbox[2], inst.bbox[3], inst.bbox[0], inst.bbox[3]], dtype=np.float32).reshape(-1,2)
                
                # print("bbox area:", (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
                print(f"num_inst: {num_inst}, num_none_segment: {num_none_segment}")



            if isinstance(segment, list):
                segment = np.array(segment, dtype=np.float32)


            if isinstance(segment, np.ndarray) and segment.size > 0:
                
            
                CHECK_SEGMENT(segment)
                # Ensure h and w are Python int/float, not numpy types
                h, w = int(h), int(w)
                seg_normalized = segment.astype(np.float32)
                seg_normalized[:, 0] = seg_normalized[:, 0] / w
                seg_normalized[:, 1] = seg_normalized[:, 1] / h
                segment=seg_normalized
            elif isinstance(segment, np.ndarray) and segment.size == 0:
                # empty segment
                segment = np.zeros((0, 2), dtype=np.float32)
                assert False, "segment should not be empty here."
            else:
                # Use empty numpy array instead of list for consistency
                segment = np.zeros((0, 2), dtype=np.float32)


            segments.append(segment)

            text, _ = inst.get_top_text_conf()
            if text not in texts:
                texts.append(text)
                grounding_data['texts'].append([text])
            cls_index = texts.index(text)
            cls_list.append(cls_index)
        grounding_data['bboxes'] = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        grounding_data['cls'] = np.array(cls_list, dtype=np.float32).reshape(-1, 1)
        grounding_data['normalized'] = True
        grounding_data['bbox_format'] = 'xywh'
        # grounding_data['segments'] = segments # disable segments in cache


        return grounding_data

    def load_from_yoloe_result(self, yoloe_result):
        
        if isinstance(yoloe_result, str):
            assert yoloe_result.endswith(".json"), "If yoloe_result is str, it should be a json file path."
            import json
            with open(yoloe_result, 'r') as f:
                yoloe_result = json.load(f)
            assert isinstance(yoloe_result, dict), "yoloe_result should be a dict"
            
            self.instances = []
            self.im_file = yoloe_result.get("im_file")
            self.shape = (yoloe_result.get("orig_shape", [0, 0])[0], yoloe_result.get("orig_shape", [0, 0])[1])
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
            bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32).reshape(-1, 4)[0]
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())
            inst = Instance(bbox=bbox_xyxy.tolist())
            inst.set_text([names[cls]], [float(conf)])
            self.instances.append(inst)

    def load_from_yolo_txt(self, im_file, txt_path: str, yaml_file:str):
        """
        Load YOLO-style annotation .txt supporting both bbox and segmentation polygon formats.

        Args:
            txt_path (str): Path to the YOLO .txt annotation file.
            yaml_file (str): Path to the YAML file containing class names.
        """
        self.im_file = im_file

        self.shape=self.get_shape(im_file)

        def read_names_from_yaml(yaml_path):
            """
            
            Read class names from a YAML file.
            Args:
                yaml_path (str or Path): Path to the YAML file.
            Returns:
                list: List of class names.
            
            """
            # 不修改任何文件，直接读取 YAML
            from ultralytics.utils import YAML

            data = YAML.load(yaml_path)
            names = data["names"]  # dict: {0: 'person', 1: 'car', ...}
            if isinstance(names, dict):
                names = [names[i] for i in range(len(names))]
            return names
        
        class_names = read_names_from_yaml(yaml_file)
        self.texts=class_names

        if self.shape is None:
            raise ValueError("Sample.shape must be set (height,width) before calling load_from_yolo_txt.")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"YOLO txt file not found: {txt_path}")
        h, w = self.shape
        self.im_file = self.im_file or txt_path  # keep existing if already set
        with open(txt_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 5:  # need at least cls + 4 numbers for bbox
                print(f"[yolo_txt][WARN] Skip line (too few tokens): {line}")
                continue
            try:
                cls_id = int(parts[0])
            except Exception:
                print(f"[yolo_txt][WARN] Invalid class id in line: {line}")
                continue
            coords = parts[1:]
            # Attempt float conversion
            try:
                nums = [float(x) for x in coords]
            except Exception:
                print(f"[yolo_txt][WARN] Non-float coordinate encountered, skip line: {line}")
                continue
            # Distinguish bbox vs polygon
            if len(nums) == 4:
                # bbox in normalized xywh -> convert to xyxy pixel space for internal consistency
                cx, cy, bw, bh = nums
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                bbox_xyxy = [x1, y1, x2, y2]
                inst = Instance(bbox=bbox_xyxy)
                label_type = "bbox"
            elif len(nums) >= 6 and len(nums) % 2 == 0:
                # polygon segmentation (normalized)
                arr = np.array(nums, dtype=np.float32).reshape(-1, 2)
                # derive bbox from polygon
                x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
                x_max, y_max = arr[:, 0].max(), arr[:, 1].max()
                bbox_xyxy = [x_min * w, y_min * h, x_max * w, y_max * h]
                inst = Instance(bbox=bbox_xyxy)
                inst.set_segment(arr)  # store normalized polygon (will be recognized downstream)
                label_type = "segment"
            else:
                print(f"[yolo_txt][WARN] Unrecognized line (coords count {len(nums)}): {line}")
                continue
            # Attach text label
            if cls_id < 0 or cls_id >= len(class_names):
                print(f"[yolo_txt][WARN] cls_id {cls_id} out of range for class_names, skip line: {line}")
                continue
            inst.set_text([class_names[cls_id]], [-1])
            inst.other_data["label_type"] = label_type
            self.instances.append(inst)
            # if class_names[cls_id] not in self.texts:
            #     self.texts.append(class_names[cls_id])
        return self

    def save_to_yolo_txt(self, txt_path:str,inst_format:str="bbox",):
        """

        Save annotations to a YOLO-style .txt file supporting both bbox and segmentation polygon formats.
        Args:
            txt_path (str): Path to save the YOLO .txt annotation file.
            inst_format (str): "bbox" to save bounding boxes, "segment" to save segmentation polygons.
        """
        h, w = self.shape
        lines = []
        assert inst_format in ["bbox","segment"], "inst_format should be 'bbox' or 'segment'"

        for inst in self.instances:
            bbox_xywh= YoloBox(self.shape).load_from_xyxy(np.array(inst.bbox, dtype=np.float32).reshape(-1,4)).xywhn[0]
            cls= self.texts.index(inst.text[0]) 
            if inst_format=="bbox":
                line = f"{cls} {bbox_xywh[0]:.6f} {bbox_xywh[1]:.6f} {bbox_xywh[2]:.6f} {bbox_xywh[3]:.6f}"
                lines.append(line)
        
        # Write to file
        if not os.path.exists(os.path.dirname(txt_path)):
            os.makedirs(os.path.dirname(txt_path))

        with open(txt_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')


    def to_dict(self):
        return {
            'im_file': to_serializable(self.im_file),
            "shape": to_serializable(self.shape),
            'instances': [inst.to_dict() for inst in self.instances],
            'other_data': to_serializable(self.other_data)
        }
    


    def save_to_json(self, json_path):
        import json
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        # print(f"Saved sample to {json_path}")


    def gauss_json_style(self,json_data: dict):
        if all (key in json_data.keys() for key in ["im_file", "instances"]):
            return "default"
        elif all (key in json_data.keys() for key in ["texts", "bboxes", "cls"]):
            return "grounding"
        else:
            return "unknown"

    def load_from_json(self,json_path,sytle="default"):
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            style= self.gauss_json_style(data)

            if style=="grounding":
                # raise Exception("Please use load_from_grounding_label to load grounding style json.")
                return self.load_from_grounding_label(data)
            elif style=="default":
                return self._load_from_dict(data)
            else:
                raise Exception("Unknown json style.")

    def _load_from_dict(self, data: dict):
        self.im_file = data.get('im_file')
        self.shape= data.get('shape',None)
        self.instances = []
        for inst_data in data.get('instances', []):
            inst = Instance()
            inst.from_dict(inst_data)
            self.instances.append(inst)
        self.other_data = data.get('other_data', {})
        return self




def _resolve_image_path(sample: Sample, image_root: Path | str | None = None) -> Path:
    """Resolve the image path for a sample, considering optional root hints."""

    if sample.im_file is None:
        raise ValueError("Sample does not specify an image file")

    candidates = []
    raw_path = Path(sample.im_file)

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        source_json = sample.other_data.get("_source_json")
        if source_json is not None:
            candidates.append(Path(source_json).parent / raw_path)
        if image_root is not None:
            candidates.append(Path(image_root) / raw_path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Unable to locate image file for sample: {sample.im_file}")


import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
import ultralytics
from ultralytics.engine.results import Results

def sample_to_results(sample: Sample, image_root: Path | str | None = None) -> List[Results]:
    """Convert a `Sample` into a list containing a single Ultralytics `Results` object."""

    img_path = _resolve_image_path(sample, image_root=image_root)
    orig_img = np.array(Image.open(img_path).convert("RGB"))
    img_h, img_w = orig_img.shape[:2]

    text_instances={}
    all_instances=[]
    for inst in sample.instances:
        if inst.text[0] not in text_instances.keys():
            text_instances[inst.text[0]]=[]
        text_instances[inst.text[0]].append(inst)
        all_instances.append(inst)
    text_instances["all_instances"]=all_instances


    text_result = {}

    for text, instances in text_instances.items():

        boxes_data: List[List[float]] = []
        masks_list: List[np.ndarray] = []
        names: List[str] = []
        name_to_idx: Dict[str, int] = {}

        for inst in instances:
            bbox_array = np.array(inst.bbox, dtype=np.float32).reshape(-1, 4)
            if bbox_array.size == 0:
                continue

            label = inst.text[0] if inst.text else "unknown"
            if label not in name_to_idx:
                name_to_idx[label] = len(names)
                names.append(label)
            cls_idx = float(name_to_idx[label])

            conf_value = float(inst.conf[0]) if inst.conf else 0.0

            for bbox in bbox_array:
                boxes_data.append([
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    conf_value,
                    cls_idx,
                ])
            
            # Handle segmentation masks - convert polygon to binary mask
            if inst.segment is not None:
                seg_array = np.array(inst.segment, dtype=np.float32)
                # Create binary mask from polygon
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                if seg_array.ndim == 2 and seg_array.shape[1] == 2 and len(seg_array) >= 3:
                    # Convert to integer coordinates for cv2.drawContours
                    poly_points = seg_array.astype(np.int32)
                    cv2.drawContours(mask, [poly_points], 0, 1, -1)
                    masks_list.append(mask.astype(bool))
                else:
                    # Invalid segment, append empty mask
                    masks_list.append(np.zeros((img_h, img_w), dtype=bool))
            else:
                # No segment for this instance
                masks_list.append(np.zeros((img_h, img_w), dtype=bool))
        
        print(names)
        boxes_tensor = torch.from_numpy(np.array(boxes_data, dtype=np.float32)) if boxes_data else torch.zeros((0, 6), dtype=torch.float32)
        names_dict = {idx: name for idx, name in enumerate(names)}

        # Convert masks to tensor if any exist
        masks_tensor = None
        if masks_list and any(m.any() for m in masks_list):
            # Stack binary masks into a single tensor (N, H, W)
            masks_array = np.stack(masks_list, axis=0)
            masks_tensor = torch.from_numpy(masks_array)

        result = Results(
            orig_img=orig_img,
            path=str(img_path),
            names=names_dict,
            boxes=boxes_tensor,
            masks=masks_tensor,
        )
        text_result[text]=result
    return text_result




def visualize_sample(sample: Sample, dst_vis_img: Path | str, image_root: Path | str | None = None) -> Path:
    """Render sample predictions to an image and save it to ``dst_vis_img``."""

    text_result = sample_to_results(sample, image_root=image_root)
    dst_vis_path = Path(dst_vis_img)
    dst_vis_path.parent.mkdir(parents=True, exist_ok=True)

    for text, results in text_result.items():
        # split the dst_vis_path into name and ext
        name = dst_vis_path.stem
        ext = dst_vis_path.suffix
        parent = dst_vis_path.parent
        new_dst_vis_path = parent / f"{name}_{text}{ext}"

        results.save(str(new_dst_vis_path))

    return dst_vis_path



if __name__ == "__main__":
    # sam= Sample().load_from_yolo_txt(im_file="/root/ultra_louis_work/datasets/Objects365v1_5000/images/train/obj365_train_000000000003.jpg",
    #                                  txt_path="/root/ultra_louis_work/datasets/Objects365v1_5000/labels/train/obj365_train_000000000003.txt", yaml_file="/root/ultra_louis_work/datasets/Objects365v1.yaml")


    # sam.save_to_yolo_txt(txt_path="../runs/visual_yolo_txt/obj365_train_000000000003_saved.txt",inst_format="bbox")


    # sam2=Sample().load_from_yolo_txt(im_file="/root/ultra_louis_work/datasets/Objects365v1_5000/images/train/obj365_train_000000000003.jpg", txt_path="../runs/visual_yolo_txt/obj365_train_000000000003_saved.txt", yaml_file="/root/ultra_louis_work/datasets/Objects365v1.yaml")

    # output_path = Path(f"../runs/visual_yolo_txt1/visual_img.jpg")
    # saved_path = visualize_sample(sam2, output_path)


    # print(f"Saved visualization to {saved_path}")


    json_dir="../buffer/mixed_engine_buffer/4merge_prediction_with_masks"

    if not os.path.exists(json_dir):
        print(f"{json_dir} not exists")
        exit(0)

    index=7
    json_name=os.listdir(json_dir)[index]
    
    json_path=os.path.join(json_dir, json_name)
    

    sam=Sample().load_from_json(json_path)
    
    output_path = Path(f"../runs/visual_json_detection_{index}/visual_img.jpg")
    saved_path = visualize_sample(sam, output_path)