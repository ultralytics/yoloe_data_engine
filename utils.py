"""Small dataset counting utilities."""

from pathlib import Path


def get_img_num(folder):
    """Calculate the number of images in a folder."""
    img_suffix = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
    img_num = 0
    for suffix in img_suffix:
        img_num += len(list(Path(folder).rglob(f"*{suffix}")))
    return img_num


def get_json_num(folder):
    """Calculate the number of json files in a folder."""
    json_num = len(list(Path(folder).rglob("*.json")))
    return json_num


flickr_res_json_dir = "/root/ultra_louis_work/runs/flickr_engine_buffer/model_predict"

print("number of json files:", get_json_num(flickr_res_json_dir))

flickr_img_dir = "/root/ultra_louis_work/datasets/flickr/full_images"
print("number of flickr images:", get_img_num(flickr_img_dir))


mixed_img_dir = "/root/ultra_louis_work/datasets/mixed_grounding/gqa/images"

print("number of images:", get_img_num(mixed_img_dir))


mixed_res_json_dir = "/root/ultra_louis_work/runs/mixed_engine_buffer/model_predict"

print("number of json files:", get_json_num(mixed_res_json_dir))
