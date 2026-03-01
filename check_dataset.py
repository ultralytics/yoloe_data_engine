import lyutils
from lyutils import set_ultra_dir

set_ultra_dir()

import ultralytics

from ultralytics.data import GroundingDataset
from ultralytics.data.dataset import DATASET_CACHE_VERSION




data_param=dict(
    img_path="../datasets/mixed_grounding/gqa/images",
    json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
)


data=GroundingDataset(**data_param)


for i in range(1):
    sam=data[i]
    # for k,v in sam.items():
    #     print(k,type(v))
    #     if k=="texts":
    #         print(len(v))
    #         # print("----sample texts----")
    #         # for index, text in enumerate(v):
    #         #     print(index, text)

        
    #     if k=="cls":
    #         print(v.shape, v.dtype)
            
    #         print(set(v.numpy().flatten().tolist()))