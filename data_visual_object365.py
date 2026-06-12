"""Visualize Object365 detection labels from cached data."""

import os

import ultralytics

from data_engine import DataEngine

workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

if __name__ == "__main__":
    im_index = 0

    de = DataEngine(device="cuda")
    de = DataEngine(device="cuda")
    yaml_config = "/root/ultra_louis_work/datasets/Objects365v1.yaml"
    cache_path = "/root/ultra_louis_work/datasets/Objects365v1/labels/train.cache"
    de.load_cached_label(
        cache_path=cache_path, data_style="detection", yaml_config=yaml_config
    )
    de.print_data_info()

    # de.visual_and_save2(im_index, save_path="./visualized_grounding_example.jpg")
    de = DataEngine(device="cuda")
    yaml_config = "/root/ultra_louis_work/datasets/Objects365v1.yaml"
    cache_path = (
        "/root/ultra_louis_work/datasets/Objects365v1/labels/train.updated.cache"
    )
    de.load_cached_label(
        cache_path=cache_path, data_style="detection", yaml_config=yaml_config
    )

    de.print_data_info()
