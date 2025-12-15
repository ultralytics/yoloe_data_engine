
from matplotlib.pylab import sample
import ultralytics,os   
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from yoloe_data_engine.data_engine import DataEngine

import copy
from pathlib import Path as _Path

IMAGES_CACHE = None
IMNAME_ANNS_CACHE = None



from engine_base import *



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
        if len(bboxes_xyxy) > 0 and YoloBox([int(h), int(w)]).load_from_xyxy(bboxes_xyxy).iou(ann["bbox"]).max() > 0.98:
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
        cat_name = " ".join([caption[t[0]:t[1]] for t in ann["tokens_positive"]]).lower().strip()
        if not cat_name:
            continue
        if cat_name not in cat2id:
            cat2id[cat_name] = len(cat2id)
            texts.append([cat_name])
        cls = cat2id[cat_name]
        box = [cls] + box.tolist()
        if box not in bboxes:
            bboxes.append(box)
            if ann.get("segmentation") is not None:
                if len(ann["segmentation"]) == 0:
                    segments.append(box)
                    continue
                elif len(ann["segmentation"]) > 1:
                    s = merge_multi_segment(ann["segmentation"])
                    s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                else:
                    s = [j for i in ann["segmentation"] for j in i]
                    s = (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                s = [cls] + s
                segments.append(s)
        bboxes_xyxy.append(ann["bbox"])
    lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
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
    return _load_grounding_data(*args)


def init_worker(images_data, imname_anns_data):
    """Initializer for worker processes to avoid repeatedly pickling large state."""
    global IMAGES_CACHE, IMNAME_ANNS_CACHE
    IMAGES_CACHE = images_data
    IMNAME_ANNS_CACHE = imname_anns_data


########################################################################################################
def _load_detection_data(buffer_dir,im_dir,txt_file, folder_name,yaml_file):
    dst_dir = os.path.join(buffer_dir, folder_name)
    os.makedirs(dst_dir, exist_ok=True)
    file_name= os.path.basename(txt_file)
    file_name_wo_ext= os.path.splitext(file_name)[0]
    dst_file = os.path.join(dst_dir, file_name_wo_ext + ".json")
    if os.path.exists(dst_file):
        return True
    from yoloe_data_engine.engine_base import Sample, YoloBox

    try:
        sample = Sample()
        im_name = file_name_wo_ext + ".jpg"
        im_file = os.path.join(im_dir, im_name)
        sample.im_file = im_file

        sample.load_from_yolo_txt(im_file=im_file, txt_path=txt_file, yaml_file=yaml_file)

        sample.save_to_json(dst_file)

    except Exception as e:
        print(f"[detection][WARN] could not open image '{im_file}': {e}")
        return False
    

def txt2json_worker(args):
    return _load_detection_data(*args)






################################## multi-processing model prediction ############################################




def _batch_model_predict_single_process(self,buffer_dir, im_files, **kwargs):
    """
    Batch model predict in a single process. This can be a method of DataEngine.
    Args:
        self: DataEngine instance
        buffer_dir: str, buffer directory to save results
        im_files: list of str, image file paths
        kwargs: other keyword arguments for model.predict
    """
    assert isinstance(self, DataEngine)
    engine=self
    dst_dir = os.path.join(buffer_dir, "2model_predict")
    os.makedirs(dst_dir, exist_ok=True)
    conf = kwargs.get("conf", 0.5)
    iou = kwargs.get("iou", 0.4)
    im_names_wo_ext = [os.path.splitext(os.path.basename(im_file))[0] for im_file in im_files]
    dst_files = [os.path.join(dst_dir, f"{name}.json") for name in im_names_wo_ext]
    indices = [i for i in range(len(im_files)) if not os.path.exists(dst_files[i])]
    if len(indices) == 0:
        print("All images have been processed, skip.")
        return
    process_img_files = [im_files[i] for i in indices]
    results = list(engine.model.predict(process_img_files, conf=conf, iou=iou, batch=len(process_img_files), stream=True))
    print(f"Processed {len(process_img_files)} images.")
    for i, sample_index in enumerate(indices):
        sample = Sample()
        result = results[i]
        sample.load_from_yoloe_result(result)
        sample.save_to_json(dst_files[sample_index])
    return



def _device_yoloe_predict_worker(args):
    """
    Worker function for multi-process model prediction on a specific device.
    args: tuple containing (device, buffer_dir, batches, kwargs)
    """
    device, buffer_dir, batches, kwargs = args

    worker_kwargs = dict(kwargs or {})
    texts = worker_kwargs.pop("texts", None)

    engine = DataEngine(device=device)
    engine.load_yoloe()
    engine.set_classes(name_list=texts)

    for im_files in tqdm(batches, desc=f"Device {device} processing batches"):
        _batch_model_predict_single_process(engine, buffer_dir, im_files, **worker_kwargs)
    return True

def _device_yolo26_predict_worker(args):
    """
    Worker function for multi-process model prediction on a specific device.
    args: tuple containing (device, buffer_dir, batches, kwargs)
    """
    device, buffer_dir, batches, kwargs = args

    worker_kwargs = dict(kwargs or {})
    texts = worker_kwargs.pop("texts", None)

    engine = DataEngine(device=device)
    engine.load_yolo26_objv1()

    for im_files in tqdm(batches, desc=f"Device {device} processing batches"):
        _batch_model_predict_single_process(engine, buffer_dir, im_files, **worker_kwargs)
    return True


##############################################################################



def _merge_prediction_to_sample_label(buffer_dir,sample_json, model_predict_json):
    """
        Each sample have a file_name, we merge the model prediction results (model_predict_json) into the sample grounding label.
        step 1: first check the filename match, if false, raise error.
        step 2: check the dst file exist, if true, skip.
        step 3: merge model prediction results into sample grounding label, iou score > 0.5 will be ignored.
        step 4: save the merged label to buffer_dir/merge_prediction/
    Args:
        buffer_dir: str, buffer directory to save results
        sample_json: str, path to sample grounding label json file
        model_predict_json: str, path to model prediction json file
    """
    dst_dir = os.path.join(buffer_dir, "3merge_prediction")
    os.makedirs(dst_dir, exist_ok=True)
    sample_basename = os.path.basename(sample_json)
    dst_file = os.path.join(dst_dir, sample_basename)
    if os.path.exists(dst_file):
        print(f"[merge] Skip existing: {dst_file}")
        return True

    ground_sample = Sample()
    ground_sample.load_from_json(sample_json)
    origin_num=len(ground_sample.instances)
    # print(f"{ground_sample.im_file} ground_sample instances:", len(ground_sample.instances))


    predict_sample = Sample()
    predict_sample.load_from_json(model_predict_json)
    # print(f"{predict_sample.im_file} predict_sample instances:", len(predict_sample.instances))

    for model_inst in predict_sample.instances:

        if model_inst.conf[0] <0.5:
            continue
        # Defensive: skip invalid model instances
        if getattr(model_inst, 'bbox', None) is None:
            print(f"[merge][WARN] skipping model instance with empty bbox in '{sample_json}'")
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
    # print(f"{ground_sample.im_file} merged instances:", len(ground_sample.instances), " (added ", len(ground_sample.instances)-origin_num,")")
    # print(f"[merge] Saved: {dst_file}")
    return True
    # print(f"Merged label saved to {dst_file}")

def merge_prediction_worker(args):
    # Support both (idx, buffer_dir, sample_json, model_predict_json) and (buffer_dir, sample_json, model_predict_json)
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
        return _merge_prediction_to_sample_label(buffer_dir, sample_json, model_predict_json)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[worker][ERROR] idx={idx} file='{sample_json}': {repr(e)}\n{tb}")
        return False
    

##############################################################################



class DataEngineAgent:
    def __init__(self, devices=["cuda:0"], buffer_dir="../engine_buffer"):
        self.buffer_dir = buffer_dir
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.devices = devices

    def load_model_engine(self):
        # self.model_path = model_path
        self.models = []
        for device in self.devices:
            de = DataEngine(device=device)
            de.load_yoloe()
            self.models.append(de)

    def set_classes(self, texts: list | None):
        if not texts:
            self.texts = texts
            return
        for model in self.models:
            model.set_classes(name_list=texts)
        self.texts = texts




    
    def multi_process_batch_model_predict(self, im_dir, texts=None, conf=0.5, iou=0.4, batch_size=3, max_workers=None,
                                          type="grounding"):
        
        """
        
        """
        assert type in ["grounding", "detection"], "type must be grounding or detection"
        if type=="grounding":
            assert texts is not None, "texts must be provided for grounding"
            _predict_worker=_device_yoloe_predict_worker
        if type=="detection":
            texts = None  # ignore texts for detection
            _predict_worker=_device_yolo26_predict_worker
        print("Start multi-process batch model prediction...")


        im_files = []
        for file_name in os.listdir(im_dir):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                im_files.append(os.path.join(im_dir, file_name))

        # im_files=im_files[:128]
        print(f"Total images to process: {len(im_files)}")
        batches = [im_files[i:i+batch_size] for i in range(0, len(im_files), batch_size)]
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
            kwargs = {'conf': conf, 'iou': iou, 'texts': texts}
            process_args.append((device, self.buffer_dir, assigned_batches, kwargs))

        if not process_args:
            print("No batches assigned to workers.")
            return []

        results = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(process_args), mp_context=ctx) as executor:
            futures = [executor.submit(_predict_worker, args) for args in process_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Model predict ..."):
                future.result()
        return results
    


        # print(f"Saved sample to {dst_file}")


    def multi_process_load_detection_data(self, im_dir, txt_dir,yaml_file, max_workers=8):
        """
        Multi-process load detection data from txt files.
        Args:
            im_dir: str, image directory
            txt_dir: str, txt annotation directory
            yaml_file: str, yaml file for class names
            max_workers: int, maximum number of worker processes

        """


        print("Start multi-process loading of detection data...")
        
        self.im_dir = im_dir

        txt_files = []
        for file_name in os.listdir(txt_dir):
            if file_name.endswith(".txt"):
                txt_files.append(os.path.join(txt_dir, file_name))
        
        # txt_files=txt_files[:128]
        print(f"Total txt files to process: {len(txt_files)}")
        folder_name = "1detection_data"
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)
        process_args = []
        for txt_file in txt_files:
            process_args.append((self.buffer_dir, self.im_dir, txt_file, folder_name,yaml_file))
        print(f"Using worker_count={worker_count}")
        # Use 'spawn' to avoid fork-related issues and set a chunksize for throughput
        ctx = mp.get_context("spawn")
        chunksize = max(1, min(500, len(process_args) // (worker_count * 4) if worker_count > 0 else 1))
        print(f"Submitting {len(process_args)} tasks with chunksize={chunksize}")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            iterable = executor.map(txt2json_worker, process_args, chunksize=chunksize)
            ok = 0
            total = 0
            for result in tqdm(iterable, total=len(process_args), desc="Loading detection data"):
                total += 1
                if result:
                    ok += 1
                if total % 10000 == 0:
                    print(f"Progress: {ok}/{total} succeeded")
        print(f"Done: {ok}/{total} succeeded")





    def multi_process_load_grounding_data(self, im_dir, json_file, merge_within_one_image, max_workers=8):

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
            folder_name = "1grounding_data_merged"
        else:
            imname_anns_data = None
            folder_name = "1grounding_data_merged"
        imid_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            ann["caption"] = images_data[f"{ann['image_id']:d}"]["caption"]
            imid_anns[ann["image_id"]].append(ann)
        self.img_path = annotations.get("img_path", "")
        imids = list(imid_anns.keys())
        
        print(f"Total images to process: {len(imids)}")

        init_args = (images_data, imname_anns_data)
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)

        # Use executor.map with initializer so each worker receives heavy state only once.
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=init_args) as executor:
            tasks = [(self.buffer_dir, self.im_dir, imid, imid_anns[imid], folder_name) for imid in imids]

            # The chunksize is critical for performance. It determines how many tasks are sent to a worker at once.
            chunk_size = max(1, min(500, len(imids) // (worker_count * 4) if worker_count > 0 else 1))
            print(f"Using {worker_count} workers and chunksize: {chunk_size}")

            list(tqdm(executor.map(worker_wrapper, tasks, chunksize=chunk_size), total=len(tasks), desc="Loading grounding data"))

        print("Finished loading grounding data.")


    def multi_process_merge_prediction(self,json_dir,predict_json_dir,max_workers=8):
        
        json_files= []
        predict_json_files = []
        for index, sample_file_name in enumerate(os.listdir(json_dir)):

            if sample_file_name.endswith(".json"):
                json_path= os.path.join(json_dir, sample_file_name)
                json_files.append(json_path)

                # read json_path and get im_file name
                with open(json_path, 'r') as f:
                    sample_data = json.load(f)
                im_file = sample_data.get("im_file")
                im_name = os.path.splitext(os.path.basename(im_file))[0]
                predict_json_path= os.path.join(predict_json_dir, f"{im_name}.json")
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
            process_args.append((i, self.buffer_dir, json_files[i], predict_json_files[i]))

        # Show a few samples for debugging
        preview_n = min(3, len(process_args))
        for k in range(preview_n):
            _, _, s, p = process_args[k]
            print(f"[merge] Task preview[{k}]: sample='{s}', predict='{p}'")

        # Use 'spawn' to avoid fork-related issues and set a chunksize for throughput
        ctx = mp.get_context("spawn")
        chunksize = max(1, min(500, len(process_args) // (worker_count * 4) if worker_count > 0 else 1))
        print(f"[merge] Submitting {len(process_args)} tasks with chunksize={chunksize}")

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            iterable = executor.map(merge_prediction_worker, process_args, chunksize=chunksize)
            ok = 0
            total = 0
            for result in tqdm(iterable, total=len(process_args), desc="Merging predictions"):
                total += 1
                if result:
                    ok += 1
                if total % 10000 == 0:
                    print(f"[merge] Progress: {ok}/{total} succeeded")
        print(f"[merge] Done: {ok}/{total} succeeded")
                

    def _merge_predict(self):
        pass




def read_numpy_and_print(path=None):
    def load_dataset_cache_file(path: Path) -> dict:
        import gc
        gc.disable()
        cache = np.load(str(path), allow_pickle=True).item()
        gc.enable()
        return cache
    path = "../engine_buffer/grounding_data/5.cache"
    data = load_dataset_cache_file(path)
    print(data)

def read_ram_tag_list():
    txt_path="../buffer/ram_tag_list.txt"
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(lines[:10])
    return lines
def read_flickr_texts(num=50000):
    mobileclip_text_embed_pt = "../datasets/flickr/text_embeddings_mobileclip_blt.pt"
    import torch
    txt_map= torch.load(mobileclip_text_embed_pt, map_location="cuda:0")
    if num < len(txt_map):
        return list(txt_map.keys())[:num]
    else:
        return list(txt_map.keys())






