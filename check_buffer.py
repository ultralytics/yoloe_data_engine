import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)




from yoloe_data_engine.engine_base import Sample

# check how many files under the json_dir

def count_json_files(json_dir):
    import os
    count=0
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            count+=1
    return count


def count_inst_num_in_json(json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    sam = Sample().load_from_json(json_path)
    num_instances = len(sam.instances)
    return num_instances

def count_inst_num_in_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    num_instances = len(lines)
    return num_instances

def count_total_instances_in_dir(json_dir):
    import os
    total_instances=0
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path=os.path.join(json_dir,filename)
            num_instances=count_inst_num_in_json(json_path)
            total_instances+=num_instances
    return total_instances

from tqdm import tqdm

def check_all_txts_in_dir(txt_dir):
    # check if there is a crashed txt file
    # check if there is a txt file with zero instances

    import os
    total_file_num=0
    total_txt_files=0
    empty_txt_files=[]
    crashed_txt_files=[]

    for filename in tqdm(os.listdir(txt_dir)):
        total_file_num+=1
        if filename.endswith(".txt"):
            total_txt_files+=1
            txt_path=os.path.join(txt_dir,filename)
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                if len(lines)==0:
                    empty_txt_files.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                crashed_txt_files.append(filename)

    print(f"Total files in {txt_dir}: {total_file_num}")
    print(f"Total txt files in {txt_dir}: {total_txt_files}")
    print(f"Number of empty txt files: {len(empty_txt_files)}")
    # if len(empty_txt_files)>0:
    #     print("Empty txt files:", empty_txt_files)
    print(f"Number of crashed txt files: {len(crashed_txt_files)}")
    # if len(crashed_txt_files)>0:
    #     print("Crashed txt files:", crashed_txt_files)


def check_all_jsons_in_dir(json_dir):
    # check if there is a crashed json file
    # check if there is a json file with zero instances

    import os
    total_file_num=0
    total_json_files=0
    empty_json_files=[]
    crashed_json_files=[]

    for filename in tqdm(os.listdir(json_dir)):
        total_file_num+=1
        if filename.endswith(".json"):
            total_json_files+=1
            json_path=os.path.join(json_dir,filename)
            try:
                num_instances=count_inst_num_in_json(json_path)
                if num_instances==0:
                    empty_json_files.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                crashed_json_files.append(filename)

    print(f"Total files in {json_dir}: {total_file_num}")
    print(f"Total json files in {json_dir}: {total_json_files}")
    print(f"Number of empty json files: {len(empty_json_files)}")
    # if len(empty_json_files)>0:
    #     print("Empty json files:", empty_json_files)
    print(f"Number of crashed json files: {len(crashed_json_files)}")
    # if len(crashed_json_files)>0:
    #     print("Crashed json files:", crashed_json_files)


# folder="objv1_engine_buffer"
# folder="flickr_engine_buffer"
# json_dir=f"../buffer/{folder}/1grounding_data_merged"
# predict_json_dir=f"../buffer/{folder}/2model_predict"
# merge_json_dir=f"../buffer/{folder}/3merge_prediction" 
# num_files=count_json_files(json_dir)
# print(f"Number of json files in {json_dir}: {num_files}")


# num_files=count_json_files(predict_json_dir)
# print(f"Number of json files in {predict_json_dir}: {num_files}")



# num_files=count_json_files(merge_json_dir)
# print(f"Number of json files in {merge_json_dir}: {num_files}")



def check_ojbect356_json_files():



    buffer_dir="../buffer/objv1_engine_buffer"
    detlabel_dir="../datasets/Objects365v1/labels/train"

    json_dir=f"{buffer_dir}/1detection_data"
    predict_json_dir=f"{buffer_dir}/2model_predict"
    merge_json_dir=f"{buffer_dir}/3merge_prediction" 
    merge_json_withmask_dir=f"{buffer_dir}/4merge_prediction_with_masks"

    # check_all_txts_in_dir(detlabel_dir) # Number of empty txt files: 7830

    # check_all_jsons_in_dir(json_dir) # Number of empty json files: 7830

    check_all_jsons_in_dir(merge_json_withmask_dir) # Number of empty json files: 7830


    # num_files=count_json_files(json_dir)
    # print(f"Number of json files in {json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(json_dir)
    # print(f"Total instances in {json_dir}: {num_inst}")


    # num_files=count_json_files(predict_json_dir)
    # print(f"Number of json files in {predict_json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(predict_json_dir)
    # print(f"Total instances in {predict_json_dir}: {num_inst}")


    # num_files=count_json_files(merge_json_dir)
    # print(f"Number of json files in {merge_json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(merge_json_dir)
    # print(f"Total instances in {merge_json_dir}: {num_inst}")



    # num_files=count_json_files(merge_json_withmask_dir)
    # print(f"Number of json files in {merge_json_withmask_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(merge_json_withmask_dir)
    # print(f"Total instances in {merge_json_withmask_dir}: {num_inst}")


def check_mixedgrounding_json_files():



    buffer_dir="../buffer/mixed_engine_buffer"


    json_dir=f"{buffer_dir}/1grounding_data_merged"
    predict_json_dir=f"{buffer_dir}/2model_predict"
    merge_json_dir=f"{buffer_dir}/3merge_prediction" 
    merge_json_withmask_dir=f"{buffer_dir}/4merge_prediction_with_masks"

    check_all_jsons_in_dir(json_dir) 

    num_files=count_json_files(json_dir)
    print(f"Number of json files in {json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(json_dir)
    # print(f"Total instances in {json_dir}: {num_inst}")


    num_files=count_json_files(predict_json_dir)
    print(f"Number of json files in {predict_json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(predict_json_dir)
    # print(f"Total instances in {predict_json_dir}: {num_inst}")


    num_files=count_json_files(merge_json_dir)
    print(f"Number of json files in {merge_json_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(merge_json_dir)
    # print(f"Total instances in {merge_json_dir}: {num_inst}")



    num_files=count_json_files(merge_json_withmask_dir)
    print(f"Number of json files in {merge_json_withmask_dir}: {num_files}")
    # num_inst=count_total_instances_in_dir(merge_json_withmask_dir)
    # print(f"Total instances in {merge_json_withmask_dir}: {num_inst}")



# check_ojbect356_json_files()
check_mixedgrounding_json_files()

