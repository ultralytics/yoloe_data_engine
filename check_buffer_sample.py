from yoloe_data_engine.engine_base import Sample,visualize_sample
import os
from pathlib import Path

def viusal_flickr_buffer_3merge_prediction_sample(index=None):
    folder = "../buffer/flickr_engine_buffer/3merge_prediction"

    json_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    if index is None:
        import random
        index= random.randint(0, len(json_files)-1)
    json_file = json_files[index]
    print(f"Using json file: {json_file}")
    json_path = os.path.join(folder, json_file)
    sam = Sample().load_from_json(json_path)
 
    output_path = Path(f"../runs/visual_flickr_json_detection_{index}/visual_img.jpg")
    saved_path = visualize_sample(sam, output_path)
    print(f"Saved visualization to {saved_path}")


def visulize_flickr_buffer_1grounding_data_merged_sample(index=None):
    folder = "../buffer/flickr_engine_buffer/1grounding_data_merged"

    json_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    if index is None:
        import random
        index= random.randint(0, len(json_files)-1)
    json_file = json_files[index]
    print(f"Using json file: {json_file}")
    json_path = os.path.join(folder, json_file)
    sam = Sample().load_from_json(json_path)
 
    output_path = Path(f"../runs/visual_flickr_json_grounding_{index}/visual_img.jpg")
    saved_path = visualize_sample(sam, output_path)
    print(f"Saved visualization to {saved_path}")

if __name__ == "__main__":

    for i in range(10):
        import random
        index=random.randint(0, 10000)
        viusal_flickr_buffer_3merge_prediction_sample(index=index)
        visulize_flickr_buffer_1grounding_data_merged_sample(index=index)