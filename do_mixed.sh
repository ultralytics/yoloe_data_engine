# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# activate clipenv conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clipenv

# set gpu id to 2,3
export CUDA_VISIBLE_DEVICES=2,3

# run the refine_text.py script to generate refined labels and cache for Flickr dataset
python3 yoloe_data_engine/refine_text.py --img_path /root/ultra_louis_work/datasets/mixed_grounding/gqa/images \
  --json_file /root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json
