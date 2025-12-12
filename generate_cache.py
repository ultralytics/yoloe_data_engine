import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from ultralytics.data import GroundingDataset
from ultralytics.data.dataset import DATASET_CACHE_VERSION

from pathlib import Path

import json
from typing import Any
from ultralytics.utils.torch_utils import LOGGER
from ultralytics.data.utils import load_dataset_cache_file, save_dataset_cache_file,get_hash
from ultralytics.utils import TQDM
from ultralytics.utils import LOCAL_RANK


CACHE_SUFFIX=".engine.segment.cache"

phase_folder="4merge_prediction_with_masks"
json_folders = {}
json_folders["final_flickr_separateGT_train_segm.json"] = f"../buffer/flickr_engine_buffer/{phase_folder}"
json_folders["final_mixed_train_no_coco_segm.json"] = f"../buffer/mixed_engine_buffer/{phase_folder}"
json_folders["objects365_train_segm.json"] = f"../buffer/objv1_engine_buffer/{phase_folder}"
from data_engine_agent import Sample

class GroundingDatasetJsonFolder(GroundingDataset):
    """
    Dataset class for object detection tasks using annotations from multiple JSON files in a folder.
    """

    def get_labels(self) -> list[dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(CACHE_SUFFIX)
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError, EOFError, Exception):
            # Regenerate cache if file not found, corrupted, or version mismatch
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]

        if CACHE_SUFFIX == ".cache":
            self.verify_labels(labels)

        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels
    

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")

        json_folder = json_folders[os.path.basename(self.json_file)]

        json_files = list(Path(json_folder).glob("*.json"))#[:1000]
        print(f"Found {len(json_files)} json files in folder {json_folder}")   
        for json_file in TQDM(json_files, desc=f"Reading annotations from folder {json_folder}"):
            sam = Sample().load_from_json(json_file) 

            label=sam.to_grounding_label()
            x["labels"].append(label)

            
        x["hash"] = get_hash(self.json_file)


        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x



if __name__ == "__main__":
    # dataset = GroundingDatasetJsonFolder(task="detect",
    #                                      json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
    #                                      img_path="../datasets/flickr/full_images/",
    # )

    # dataset = GroundingDatasetJsonFolder(task="detect",
    #                                      json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
    #                                      img_path="../datasets/mixed_grounding/gqa/images",
    # )

    dataset = GroundingDatasetJsonFolder(task="detect",
                                         json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
                                         img_path="../datasets/Objects365v1/images/train",
    )