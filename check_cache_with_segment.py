import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)




from ultralytics.data.utils import load_dataset_cache_file




def check_cache_func(cache_path):
    cache_data=load_dataset_cache_file(cache_path)
    print("="*100)
    print("cache_path:", cache_path)
    print("cache_data keys:", cache_data.keys())
 
    labels=cache_data.get('labels', None)
    if labels is None:
        print("ERROR: No labels found in cache!")
        return
    
    print(f"Total labels: {len(labels)}")

    # Statistics
    total_boxes = 0
    total_segments = 0
    labels_with_segments = 0
    labels_without_segments = 0
    mismatch_count = 0
    none_segments_count = 0
    invalid_shape_count = 0
    empty_segments_count = 0
    type_error_count = 0
    
    segment_shapes = {}
    
    print("\nScanning all labels for segment validation...")
    
    for idx, label in enumerate(labels):
        boxes = label.get('cls', None)
        segments = label.get('segments', None)
        
        if boxes is not None:
            num_boxes = len(boxes)
            total_boxes += num_boxes
        else:
            num_boxes = 0
        
        if segments is not None:
            labels_with_segments += 1
            num_segments = len(segments)
            total_segments += num_segments
            
            # Check if number of segments matches number of boxes
            if num_boxes != num_segments:
                mismatch_count += 1
                print(f"\n[MISMATCH] Label {idx}: {num_boxes} boxes but {num_segments} segments")
                print(f"  im_file: {label.get('im_file', 'unknown')}")
            
            # Check each segment
            for seg_idx, segment in enumerate(segments):
                # Check if segment is None
                if segment is None:
                    none_segments_count += 1
                    print(f"\n[NONE] Label {idx}, segment {seg_idx}: segment is None")
                    print(f"  im_file: {label.get('im_file', 'unknown')}")
                    continue
                
                # Check if segment is empty array
                if hasattr(segment, 'shape'):
                    if segment.shape[0] == 0:
                        empty_segments_count += 1
                        print(f"\n[EMPTY] Label {idx}, segment {seg_idx}: empty segment {segment.shape}")
                        print(f"  im_file: {label.get('im_file', 'unknown')}")
                        continue
                    
                    # Check segment shape
                    shape_key = str(segment.shape)
                    segment_shapes[shape_key] = segment_shapes.get(shape_key, 0) + 1
                    
                    # Validate shape (should be (N, 2) where N >= 3)
                    if len(segment.shape) != 2 or segment.shape[1] != 2:
                        invalid_shape_count += 1
                        print(f"\n[INVALID SHAPE] Label {idx}, segment {seg_idx}: invalid shape {segment.shape}")
                        print(f"  im_file: {label.get('im_file', 'unknown')}")
                    elif segment.shape[0] < 3:
                        invalid_shape_count += 1
                        print(f"\n[INVALID SHAPE] Label {idx}, segment {seg_idx}: too few points {segment.shape}")
                        print(f"  im_file: {label.get('im_file', 'unknown')}")
                else:
                    print(segment)
                    assert False
                    type_error_count += 1
                    print(f"\n[TYPE ERROR] Label {idx}, segment {seg_idx}: not a numpy array, type={type(segment)}")
                    print(f"  im_file: {label.get('im_file', 'unknown')}")
        else:
            labels_without_segments += 1
    
    # Print summary
    print("\n" + "="*100)
    print("SEGMENT VALIDATION SUMMARY")
    print("="*100)
    print(f"Total labels:              {len(labels)}")
    print(f"Total boxes:               {total_boxes}")
    print(f"Total segments:            {total_segments}")
    print(f"Labels with segments:      {labels_with_segments}")
    print(f"Labels without segments:   {labels_without_segments}")
    print(f"\nISSUES FOUND:")
    print(f"  Mismatched box/seg count: {mismatch_count}")
    print(f"  None segments:            {none_segments_count}")
    print(f"  Empty segments:           {empty_segments_count}")
    print(f"  Invalid shapes:           {invalid_shape_count}")
    print(f"  Type errors (not ndarray):{type_error_count}")
    
    # if segment_shapes:
    #     print(f"\nSegment shape distribution:")
    #     for shape, count in sorted(segment_shapes.items(), key=lambda x: x[1], reverse=True)[:10]:
    #         print(f"  {shape}: {count} segments")
    
    print("="*100)

    print("="*100)


# Main execution
if __name__ == "__main__":
    cache_path="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.engine.segment.cache"
    check_cache_func(cache_path)
    cache_path="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.engine.cache"
    check_cache_func(cache_path)

    cache_path="../datasets/Objects365v1/annotations/objects365_train_segm.engine.segment.cache"
    check_cache_func(cache_path)
    cache_path="../datasets/Objects365v1/annotations/objects365_train_segm.engine.cache"
    check_cache_func(cache_path)

    cache_path="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.engine.segment.cache"
    check_cache_func(cache_path)
    cache_path="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.engine.cache"
    check_cache_func(cache_path)
