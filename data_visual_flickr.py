"""Visualize Flickr grounding labels from cached data."""

import os

import ultralytics

from data_engine import DataEngine

workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

if __name__ == "__main__":
    im_index = 0

    de = DataEngine()
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
    de.print_data_info()

    # de.visual_and_save2(im_index, save_path="./visualized_grounding_example.jpg")

    de = DataEngine()
    cache_path = (
        "/root/ultra_louis_work/datasets/flickr/annotations/"
        "final_flickr_separateGT_train_segm.updated.cache"
    )
    text_embed_pt = (
        "/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"
    )
    de.load_cached_label(
        cache_path=cache_path, data_style="grounding", text_embed_pt=text_embed_pt
    )
    de.print_data_info()

    # de.visual_and_save2(im_index, save_path="./visualized_grounding_example1.jpg")

    de.visual_and_save2(
        filename="353913.jpg", save_path="./visualized_grounding_example_v2.jpg"
    )
