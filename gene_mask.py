

import os
from multiprocessing import Pool
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings("ignore")

import json
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from engine_base import Sample

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str, required=False, help="Path to the image folder")
    parser.add_argument("--json-path", type=str, required=False, help="Path to the annotation file")
    parser.add_argument("--sample-json-dir", type=str, default=None, help="Path to the directory containing Sample JSON files")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Devices")
    parser.add_argument("--batch", action="store_true", default=False)
    return parser.parse_args()


def mask_to_coco(annotation: dict, mask: np.ndarray) -> dict:
    box_area = annotation['area']
    mask_area = np.sum(mask)
    
    if box_area < 10 or mask_area / box_area < 0.05:
        # delete the annotation with small mask or box
        return {}
    
    if mask_area < 64 * 64:
        kernel_size = (3, 3)  # use smaller kernel for small mask
    else:
        kernel_size = (7, 7)  # use larger kernel for large mask

    blurred = cv2.GaussianBlur(mask * 255, kernel_size, 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    segmentation = []
    simplified_contours = []
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # hierarchical simplification (smaller, simpler)
        area = cv2.contourArea(contour)
        r = 0.01 if area < 32 * 32 else 0.005 if area < 64 * 64 else 0.003 if area < 96 * 96 else 0.002 if area < 128 * 128 else 0.001
        epsilon = r * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # adjust epsilon if less than 3 points
        while contour.size >= 6 and len(approx) < 3:
            r *= 0.5  # decrease ratio
            epsilon = r * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(approx)
        approx_area = cv2.contourArea(approx)
        if approx.size >= 6 and approx_area / mask_area >= 0.05:
            segmentation.append(approx.astype(float).flatten().tolist())

    if len(segmentation) == 0:
        # delete empty mask annotations after simplification
        return {}

    annotation['segmentation'] = segmentation
    return annotation


def generate_mask_annos(model: SAM2ImagePredictor, rank, total, image_path: Path, annotation_path: Path, batch: bool):

    # mask annotation path
    mask_stem = annotation_path.stem + f'_segm_{rank}'
    annotation_mask_path = annotation_path.with_stem(mask_stem)
    print("Generate segmentation annotation file: ", annotation_mask_path)
    
    data_api = COCO(annotation_path)
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    data['annotations'] = [] # reset annotations
    del_anno_ids = [] # deleted annotation ids
    
    image_ids = sorted(data_api.getImgIds())
    image_ids = np.array_split(image_ids, total)[rank].tolist()
    
    pbar = tqdm(image_ids, desc='Images Segmentation')
    for image_id in pbar:
        pbar.set_postfix({'id': image_id})
        anno_ids = data_api.getAnnIds(image_id)

        if len(anno_ids) == 0:
            pbar.update(1)
            continue

        image_info = data_api.loadImgs(image_id)[0]
        image_pil = Image.open(image_path / image_info['file_name']).convert('RGB')
        image = np.array(image_pil)
        instances = data_api.loadAnns(anno_ids)
        boxes = []
        for instance in instances:
            x1, y1, w, h = instance['bbox']
            boxes.append([x1, y1, x1 + w, y1 + h])
            
        # split by chunks, avoiding OOM
        input_boxes = np.array(boxes)
        if batch:
            chunk_size = 256  # Optimized for 96GB GPUs
            chunks = [input_boxes[i:i + chunk_size] for i in range(0, len(input_boxes), chunk_size)]
        else:
            chunks = [input_boxes]

        try:
            model.set_image(image)
            mask_list = []
            for chunk in chunks:
                mask, _, _ = model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=chunk,
                    multimask_output=False,
                )
                if len(mask.shape) == 3:
                    mask = mask[None]
                mask_list.append(mask)
            masks = np.concatenate(mask_list, axis=0)
        except Exception as e:
            print("Error:", image_id, len(anno_ids))
            pbar.update(1)
            continue
        
        for anno, mask in zip(instances, masks):
            mask = mask.squeeze()
            mask = (mask > 0.5).astype(np.uint8)
            # save mask to coco format
            segm_anno = mask_to_coco(anno, mask)
            if segm_anno:
                data['annotations'].append(segm_anno)
            else:
                del_anno_ids.append(anno['id'])

        pbar.update(1)

    print(f"Filtered {len(del_anno_ids)} annotations,{len(del_anno_ids) / len(data_api.getAnnIds()) * 100:.2f}%")
    with open(annotation_mask_path, 'w') as f:
        json.dump(data, f)

def generate_mask_from_samples(model: SAM2ImagePredictor, rank, total, sample_json_dir: Path, batch: bool):
    """Generate segmentation masks from Sample JSON files."""
    
    # Output directory for results
    output_dir = Path(sample_json_dir).parent / "4merge_prediction_with_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files in the directory
    json_files = sorted(Path(sample_json_dir).glob("*.json")) # [:128]
    
    # Split by rank for multi-GPU processing
    json_files = np.array_split(json_files, total)[rank].tolist()
    
    pbar = tqdm(json_files, desc=f'Rank {rank} - Processing Samples')
    for json_file in pbar:
        pbar.set_postfix({'file': json_file.name})
        
        try:
            # Load sample from JSON
            sample = Sample()
            sample.load_from_json(str(json_file))
            
            # Resolve image path
            im_file = sample.im_file
            if not Path(im_file).exists():
                # Try to resolve relative path
                if Path(sample_json_dir).parent.parent / im_file:
                    im_file = Path(sample_json_dir).parent.parent / im_file
            
            if not Path(im_file).exists():
                print(f"Warning: Image file not found for {json_file.name}, skipping")
                pbar.update(1)
                continue
            
            # Load image
            image_pil = Image.open(im_file).convert('RGB')
            image = np.array(image_pil)
            
            # Extract bounding boxes from instances
            boxes = []
            for inst in sample.instances:
                if inst.bbox is not None:
                    bbox = np.array(inst.bbox, dtype=np.float32).reshape(-1, 4)
                    if bbox.size > 0:
                        x1, y1, x2, y2 = bbox[0]
                        boxes.append([x1, y1, x2, y2])
            
            if len(boxes) == 0:
                pbar.update(1)
                continue
            
            # Generate masks using SAM2
            input_boxes = np.array(boxes)
            if batch:
                # With 96GB VRAM per GPU, can process much larger chunks
                # Batch size of 256 is safe for SAM2 on 96GB GPUs
                chunk_size = 512
                chunks = [input_boxes[i:i + chunk_size] for i in range(0, len(input_boxes), chunk_size)]
            else:
                # Process all boxes in one batch
                chunks = [input_boxes]
            
            try:
                model.set_image(image)
                mask_list = []
                for chunk in chunks:
                    mask, _, _ = model.predict(
                        point_coords=None,
                        point_labels=None,
                        box=chunk,
                        multimask_output=False,
                    )
                    if len(mask.shape) == 3:
                        mask = mask[None]
                    mask_list.append(mask)
                masks = np.concatenate(mask_list, axis=0)
                print(f"  Generated {len(masks)} masks for {len(sample.instances)} instances, mask shape={masks.shape}")
            except Exception as e:
                print(f"Error generating masks for {json_file.name}: {e}")
                pbar.update(1)
                continue    
            
            # Add masks to sample instances
            for i, (inst, mask) in enumerate(zip(sample.instances, masks)):
                mask = mask.squeeze()
                mask = (mask > 0.5).astype(np.uint8)
                
                # Convert mask to polygon segmentation
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    print(f"  Instance {i}: No contours found in mask (mask shape={mask.shape}, unique values={np.unique(mask)})")
                    continue
                
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                area = cv2.contourArea(largest_contour)
                r = 0.01 if area < 32 * 32 else 0.005
                epsilon = r * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if approx.size >= 6:
                    # Keep in pixel space (not normalized)
                    segment = approx.astype(np.float32).reshape(-1, 2)
                    inst.set_segment(segment)
                else:
                    print(f"  Instance {i}: Contour too small (approx.size={approx.size}, area={area:.1f}, contour_len={len(largest_contour)})")
            
            # Check which instances got segments
            instances_with_segments = sum(1 for inst in sample.instances if inst.segment is not None)
            instances_without_segments = len(sample.instances) - instances_with_segments
            if instances_without_segments > 0:
                print(f"  {json_file.name}: {instances_with_segments}/{len(sample.instances)} instances have segments")
            
            # Save updated sample
            output_file = output_dir / json_file.name
            sample.save_to_json(str(output_file))
            
        except Exception as e:
            import traceback
            print(f"Error processing {json_file.name}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
        
        pbar.update(1)
    
    print(f"Rank {rank} finished processing samples")
    
def worker(args):
    device, rank, total, image_path, annotation_path, batch = args
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    # check the sam2_checkpoint
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found at {sam2_checkpoint}")

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    generate_mask_annos(predictor, rank, total, image_path, annotation_path, batch)

def worker_sample(args):
    """Worker for processing Sample JSON files."""
    device, rank, total, sample_json_dir, batch = args

    sam2_checkpoint = "../yoloe_data_engine/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    generate_mask_from_samples(predictor, rank, total, sample_json_dir, batch)

def main(args):
    # model config
    gpu_ids = [int(idx) for idx in args.gpus.split(",")]
    processes_per_gpu = 2  # Run 2 processes per GPU
    
    # Create device and rank assignments
    devices = []
    ranks = []
    for gpu_id in gpu_ids:
        for _ in range(processes_per_gpu):
            devices.append(gpu_id)
            ranks.append(len(ranks))
    
    total = len(ranks)  # Total number of processes (8 GPUs * 2 = 16 processes)
    
    print(f"Running {total} processes across {len(gpu_ids)} GPUs ({processes_per_gpu} processes per GPU)")
    print(f"Devices: {devices}")
    print(f"Ranks: {ranks}")
    
    # Check if using Sample JSON files
    if args.sample_json_dir:
        sample_json_dir = Path(args.sample_json_dir)
        with Pool(total) as pool:
            _ = pool.map(worker_sample, zip(devices, ranks, [total] * total, [sample_json_dir] * total, [args.batch] * total))
    else:
        # Original COCO annotation mode
        if not args.img_path or not args.json_path:
            raise ValueError("Either --sample-json-dir or both --img-path and --json-path must be provided")
        
        image_path = Path(args.img_path)
        annotation_path = Path(args.json_path)
        
        with Pool(total) as pool:
            _ = pool.map(worker, zip(devices, ranks, [total] * total, [image_path] * total, [annotation_path] * total, [args.batch] * total))

        annotations = []
        for rank in ranks:
            mask_path = annotation_path.with_stem(annotation_path.stem + f'_segm_{rank}')
            with open(mask_path, 'r') as f:
                annotations.extend(json.load(f)["annotations"])
            os.remove(mask_path)
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        data["annotations"] = annotations
        
        mask_path = annotation_path.with_stem(annotation_path.stem + f'_segm')
        with open(mask_path, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    # assert(torch.__version__ == "2.5.1+cu124"),f"current torch version is {torch.__version__}, please use torch 2.5.1+cu124"
    
    # Allow specifying parameters directly in script or via command line
    USE_SCRIPT_PARAMS = True  # Set to False to uce command line args
    
    if USE_SCRIPT_PARAMS:
        # Script parameters - modify these directly
        SAMPLE_JSON_DIR = "/home/louis/ultra_louis_work/buffer/mixed_engine_buffer/3merge_prediction"
        GPUS = "0,1,2,3,4,5,6,7" # 
        BATCH = False
        
        # Create args object manually
        class ScriptArgs:
            def __init__(self):
                self.sample_json_dir = SAMPLE_JSON_DIR
                self.img_path = None
                self.json_path = None
                self.gpus = GPUS
                self.batch = BATCH
        
        args = ScriptArgs()
    else:
        # Use command line arguments
        args = parse_args()
    
    print(vars(args))
    main(args)