"""Remove mask and segment data from generated dataset caches."""

from data_engine import DataEngine

if __name__ == "__main__":
    de = DataEngine(device="cuda")
    yaml_config = "/root/ultra_louis_work/datasets/Objects365v1.yaml"
    cache_path = (
        "/root/ultra_louis_work/datasets/Objects365v1/labels/train.updated.cache"
    )
    de.load_cached_label(
        cache_path=cache_path, data_style="detection", yaml_config=yaml_config
    )
    de.remove_masks_and_segments()
    de.save_cached_label(save_path=cache_path)

    de = DataEngine()
    cache_path = (
        "/root/ultra_louis_work/datasets/mixed_grounding/annotations/"
        "final_mixed_train_no_coco_segm.updated.cache"
    )
    text_embed_pt = (
        "/root/ultra_louis_work/datasets/mixed_grounding/gqa/"
        "text_embeddings_mobileclip_blt.pt"
    )
    de.load_cached_label(
        cache_path=cache_path, data_style="grounding", text_embed_pt=text_embed_pt
    )

    de.remove_masks_and_segments()
    de.save_cached_label(save_path=cache_path)

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
    de.remove_masks_and_segments()
    de.save_cached_label(save_path=cache_path)
