


from lyutils import set_ultra_dir
set_ultra_dir()

from lyutils.utils import random_choice_a_json,list_all_jsonpath,compare_files_in_two_dirs
import os 
from yoloe_data_engine.engine_base import Instance,Sample


def update_segments_and_save(path1,path2,save_path):
    os.makedirs(save_path, exist_ok=True)

    from tqdm import tqdm 
    for json_file1 in tqdm(list_all_jsonpath(path1)):
        json_file2= os.path.join( path2, os.path.basename(json_file1) )

        sample1=Sample().load_from_json(json_file1)
        sample1_insts={ inst.get_inst_id():inst  for inst in sample1.instances }
        sample2=Sample().load_from_json(json_file2)
        sample2_insts={ inst.get_inst_id():inst  for inst in sample2.instances }


        # check
        assert len(sample1_insts)==len(sample2_insts), f"num instances not match for {json_file1}"
        for inst_id in sample1_insts:
            assert inst_id in sample2_insts, f"inst_id {inst_id} not in sample2"    

        #update segments
        for inst in sample1.instances:
            inst.segment=sample2_insts[ inst.get_inst_id()].segment
            
        # save
        save_file=os.path.join(save_path, os.path.basename(json_file1))
        sample1.save_to_json(save_file)


def check_the_inst_sequence(path1,path2):
    from tqdm import tqdm 
    for json_file1 in tqdm(list_all_jsonpath(path1)):
        json_file2= os.path.join( path2, os.path.basename(json_file1) )

        sample1=Sample().load_from_json(json_file1)
        sample1_insts_ids=[ inst.get_inst_id() for inst in sample1.instances ]
        sample2=Sample().load_from_json(json_file2)
        sample2_insts_ids=[ inst.get_inst_id() for inst in sample2.instances ]
        # check
        assert len(sample1_insts_ids)==len(sample2_insts_ids), f"num instances not match for {json_file1}"
        for idx,inst_id in enumerate(sample1_insts_ids):
            assert inst_id == sample2_insts_ids[idx], f"inst_id {inst_id} not in sample2 at idx {idx}"



mixed_path1="/home/louis/ultra_louis_work/buffer/mixed_engine_buffer/3merge_prediction"
mixed_path2="/home/louis/ultra_louis_work/buffer/mixed_engine_buffer/4merge_prediction_with_masks"
save_path=os.path.join(os.path.dirname(mixed_path1),"5final")
if not os.path.exists(save_path):
    os.makedirs(save_path)
    assert compare_files_in_two_dirs(mixed_path1,mixed_path2,suffix=".json"), "files in two dirs not match"
    print("update_segments_and_save now...")
    update_segments_and_save(mixed_path1,mixed_path2,save_path)

    print("check_the_inst_sequence now...")
    check_the_inst_sequence(mixed_path1,save_path)


obj365_path1="/home/louis/ultra_louis_work/buffer/objv1_engine_buffer/3merge_prediction"
obj365_path2="/home/louis/ultra_louis_work/buffer/objv1_engine_buffer/4merge_prediction_with_masks"
save_path=os.path.join(os.path.dirname(obj365_path1),"5final")
if not os.path.exists(save_path):
    os.makedirs(save_path)
    assert compare_files_in_two_dirs(obj365_path1,obj365_path2,suffix=".json"), "files in two dirs not match"

    print("update_segments_and_save now...")
    update_segments_and_save(obj365_path1,obj365_path2,save_path)
    print("check_the_inst_sequence now...")
    check_the_inst_sequence(obj365_path1,save_path)
