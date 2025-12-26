import os,sys
sys.path.append("/home/louis/ultra_louis_work")

from lyutils import set_ultra_dir
set_ultra_dir()




from yoloe_data_engine.data_engine import DataEngine
from yoloe_data_engine.data_engine_agent import DataEngineAgent
from yoloe_data_engine.data_engine_agent import read_ram_tag_list



def generate_data(DATA="flickr" ):
    num_divide = 8
    devices=[ "cuda:{}".format(i) for i in range(num_divide)]
    # devices = ["cuda:0","cuda:1","cuda:2","cuda:3"]

    # agent = DataEngineAgent(devices=devices, buffer_dir="../buffer/flickr_engine_buffer")
    # json_file = "../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json"pip 
    # im_dir = "../datasets/flickr/full_images/"
    # mobileclip_text_embed_pt="../datasets/flickr/text_embeddings_mobileclip_blt.pt"



 # "mixed_grounding" #objv1

    if DATA=="flickr":


        agent = DataEngineAgent(devices=devices, buffer_dir="../buffer/flickr_engine_buffer")
        json_file = "../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json"
        im_dir = "../datasets/flickr/full_images/"
        # mobileclip_text_embed_pt = "../datasets/flickr/text_embeddings_mobileclip_blt.pt"

        import torch
        texts = read_ram_tag_list()
        # agent.multi_process_batch_model_predict(im_dir=im_dir, texts=texts, conf=0.5, iou=0.4, batch_size=2)
        # agent.multi_process_load_grounding_data(json_file=json_file, im_dir=im_dir, merge_within_one_image=False, max_workers=8)
        agent.multi_process_merge_prediction(json_dir="../buffer/flickr_engine_buffer/1grounding_data_merged",
                                            predict_json_dir="../buffer/flickr_engine_buffer/2model_predict",
                                            max_workers=32)

    elif DATA=="mixed_grounding":
 

        agent = DataEngineAgent(devices=devices, buffer_dir="../buffer/mixed_engine_buffer")
        json_file= "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json"
        im_dir="../datasets/mixed_grounding/gqa/images"
        mobileclip_text_embed_pt = "../datasets/flickr/text_embeddings_mobileclip_blt.pt"
    
        # import torch
        # txt_map= torch.load(mobileclip_text_embed_pt, map_location="cuda:0")
        # name_list=list(txt_map.keys())[:50000] 
        texts = read_ram_tag_list()
        # agent.multi_process_batch_model_predict(im_dir=im_dir, texts=texts, conf=0.5, iou=0.4,batch_size=8)


        # agent.multi_process_load_grounding_data(json_file=json_file, im_dir=im_dir, merge_within_one_image=True, max_workers=8)
        # agent.multi_process_merge_prediction(json_dir="../buffer/mixed_engine_buffer/1grounding_data_merged",
        #                                     predict_json_dir="../buffer/mixed_engine_buffer/2model_predict",
        #                                     max_workers=32)

        # agent.multi_process_load_grounding_data(json_file=json_file, im_dir=im_dir, merge_within_one_image=False, max_workers=8)

    elif DATA=="objv1":
        agent = DataEngineAgent(devices=devices, buffer_dir="../buffer/objv1_engine_buffer")
        im_dir="../datasets/Objects365v1/images/train"
        txt_dir="../datasets/Objects365v1/labels/train"
        yaml_file="../datasets/Objects365v1.yaml"


        # agent.multi_process_load_detection_data(im_dir=im_dir, txt_dir=txt_dir, yaml_file=yaml_file, max_workers=8)
        texts = read_ram_tag_list()
        # texts = read_flickr_texts(num=50000)

        # agent.multi_process_batch_model_predict(im_dir=im_dir, texts=texts, conf=0.1, iou=0.4,batch_size=64,type="grounding")

        agent.multi_process_merge_prediction(json_dir="../buffer/objv1_engine_buffer/1detection_data",
                                            predict_json_dir="../buffer/objv1_engine_buffer/2model_predict",
                                            max_workers=8)

from generate_cache import GroundingDatasetJsonFolder


def generate_cache_process_flickr():


    # generate cache 
    GroundingDatasetJsonFolder.CACHE_SUFFIX=".engine1.cache"
    # phase_folder="4merge_prediction_with_masks"
    phase_folder="3merge_prediction"
    json_folders = {}
    json_folders["final_flickr_separateGT_train_segm.json"] = f"../buffer/flickr_engine_buffer/{phase_folder}"
    json_folders["final_mixed_train_no_coco_segm.json"] = f"../buffer/mixed_engine_buffer/{phase_folder}"
    json_folders["objects365_train_segm.json"] = f"../buffer/objv1_engine_buffer/{phase_folder}"
    GroundingDatasetJsonFolder.json_folders=json_folders


    GroundingDatasetJsonFolder(task="detect",
                                         json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
                                         img_path="../datasets/flickr/full_images/",
    )

def generate_cache_process_mixedgrounding():
    

    # generate cache 
    GroundingDatasetJsonFolder.CACHE_SUFFIX=".engine1.cache"
    phase_folder="5final"
    json_folders = {}
    json_folders["final_mixed_train_no_coco_segm.json"] = f"../buffer/mixed_engine_buffer/{phase_folder}"
    GroundingDatasetJsonFolder.json_folders=json_folders


    GroundingDatasetJsonFolder(task="detect",
                                         json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                                         img_path="../datasets/mixed_grounding/gqa/images",
    )

def generate_cache_process_objv1():


    # generate cache 
    # GroundingDatasetJsonFolder.CACHE_SUFFIX=".engine.cache"

    # phase_folder="3merge_prediction"
    # json_folders = {}
    # json_folders["final_flickr_separateGT_train_segm.json"] = f"../buffer/flickr_engine_buffer/{phase_folder}"
    # json_folders["final_mixed_train_no_coco_segm.json"] = f"../buffer/mixed_engine_buffer/{phase_folder}"
    # json_folders["objects365_train_segm.json"] = f"../buffer/objv1_engine_buffer/{phase_folder}"
    # GroundingDatasetJsonFolder.json_folders=json_folders

    # GroundingDatasetJsonFolder(task="detect",
    #                                      json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
    #                                      img_path="../datasets/Objects365v1/images/train")

    # generate cache 
    GroundingDatasetJsonFolder.CACHE_SUFFIX=".engine1.cache"
    phase_folder="5final"
    json_folders = {}
    json_folders["objects365_train_segm.json"] = f"../buffer/objv1_engine_buffer/{phase_folder}"
    GroundingDatasetJsonFolder.json_folders=json_folders

    GroundingDatasetJsonFolder(task="detect",
                                         json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
                                         img_path="../datasets/Objects365v1/images/train")




# dataset = GroundingDatasetJsonFolder(task="detect",
#                                      json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
#                                      img_path="../datasets/mixed_grounding/gqa/images",
# )

# dataset = GroundingDatasetJsonFolder(task="detect",
#                                      json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
#                                      img_path="../datasets/Objects365v1/images/train",
# )






if __name__ == "__main__":


    # generate_data()
    # generate_cache_process_flickr()
    generate_cache_process_mixedgrounding()
    generate_cache_process_objv1()








