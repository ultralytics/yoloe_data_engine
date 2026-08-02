# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# activate clipenv conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clipenv

# remove /root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache.A if it exists
if [ -f /root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache.A ]; then
  rm /root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache.A
fi

# run the refine_text.py script to generate refined labels and cache for Flickr dataset
python3 yoloe_data_engine/refine_text.py

# run data visualization script
python3 yoloe_data_engine/data_visual_flickr.py
