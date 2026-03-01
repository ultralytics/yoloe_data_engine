



import json 


json_data_path="/data/shared-datasets/yoloe26_data/yolo-enterprise/pipeline_outputs/train/v5/merged.json"


# read the json  and get the annotations 

def read_json(json_data_path):

    with open(json_data_path, 'r') as f:
        jsondata = json.load(f)
    return jsondata


jsondata=read_json(json_data_path)
annos=jsondata['annotations']


# print the first one 
print(annos[0])


tail -n 200 /data/shared-datasets/yoloe26_data/yolo-enterprise/pipeline_outputs/train/v5/merged.json