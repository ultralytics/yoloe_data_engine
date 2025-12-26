import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

import os 
import numpy as np



def load_labels_from_cache( cache_file):
    from ultralytics.data.utils import load_dataset_cache_file
    cache=load_dataset_cache_file(cache_file)
    labels=cache['labels']
    return labels



def copy_mask_to_cache1(dir, cache1, cache2):
    path1=os.path.join(dir, cache1)
    path2=os.path.join(dir, cache2)

    labels1=load_labels_from_cache(path1)
    labels2=load_labels_from_cache(path2)
    if len(labels1)!=len(labels2):
        print(f"Different number of labels: {len(labels1)} vs {len(labels2)}")
        return


    for label1,label2 in zip(labels1, labels2):

        
        # sort the labels bbox,segments,cls
        segments2=label2.get('segments', None)
        bboxes=label2.get('bboxes', None)
        cls=label2.get('cls', None)





def compare_cache_files(dir, cache1, cache2):
    path1=os.path.join(dir, cache1)
    path2=os.path.join(dir, cache2)

    labels1=load_labels_from_cache(path1)
    labels2=load_labels_from_cache(path2)
    if len(labels1)!=len(labels2):
        print(f"Different number of labels: {len(labels1)} vs {len(labels2)}")
        return


    for label1,label2 in zip(labels1, labels2):
        # print(label1.keys())

        bboxes1=label1.get('bboxes', None)
        bboxes2=label2.get('bboxes', None)
        
        # Use numpy array comparison
        if not np.array_equal(bboxes1, bboxes2):
            print(f"Different bboxes found:")
            print(f"bboxes1: {bboxes1}")
            print(f"bboxes2: {bboxes2}")

            break


    print(f"Endering comparison...")

# random_read_sample_and_compare(dir, cache1, cache2)

def random_read_sample_and_compare(dir, cache1, cache2, num_samples=5):
    import random
    path1=os.path.join(dir, cache1)
    path2=os.path.join(dir, cache2)

    labels1=load_labels_from_cache(path1)
    labels2=load_labels_from_cache(path2)
    if len(labels1)!=len(labels2):
        print(f"Different number of labels: {len(labels1)} vs {len(labels2)}")
        return

    total_samples=len(labels1)
    sample_indices=random.sample(range(total_samples), num_samples)

    def make_hashable(x):
        if isinstance(x, list):
            return tuple(make_hashable(e) for e in x)
        if isinstance(x, dict):
            return tuple((k, make_hashable(v)) for k,v in sorted(x.items()))
        if isinstance(x, set):
            return frozenset(make_hashable(e) for e in x)
        try:
            hash(x)
            return x
        except TypeError:
            return str(x)

    for idx in sample_indices:
        label1=labels1[idx]
        label2=labels2[idx]

        label1_keys=set(list(label1.keys())+["segments" ])
        label2_keys=set(label2.keys())
        if label1_keys != label2_keys:
            print(f"Different keys found at index {idx}:")
            print(f"label1 keys: {label1_keys}")
            print(f"label2 keys: {label2_keys}")
            continue

        for key in label1.keys():

            if key == 'bboxes':
                bboxes1=label1.get('bboxes', None)
                bboxes2=label2.get('bboxes', None)

                if not np.array_equal(bboxes1, bboxes2):
                    print(f"Different bboxes found at index {idx}:")
                    print(f"bboxes1: {bboxes1}")
                    print(f"bboxes2: {bboxes2}")
                else:
                    print(f"Sample index {idx} bboxes are identical.")

            elif key in ["cls","texts"]:
                value1=label1.get(key, None) 
                value2=label2.get(key, None) 
                if value1 is None or value2 is None:
                    if value1 != value2:
                        print(f"Different values for key '{key}' found at index {idx}:")
                        print(f"value1: {value1}")
                        print(f"value2: {value2}")
                    else:
                        print(f"Sample index {idx} key '{key}' values are identical.")
                    # from collections import Counter
                    # c1=Counter(make_hashable(v) for v in value1)
                    # c2=Counter(make_hashable(v) for v in value2)
                    # if c1!=c2:
                    #     print(f"Different values for key '{key}' found at index {idx}:")
                    #     print(f"value1: {value1}")
                    #     print(f"value2: {value2}")
                    # else:
                    #     print(f"Sample index {idx} key '{key}' values are identical.")
            else:
                value1=label1.get(key, None)
                value2=label2.get(key, None)

                equal=False
                try:
                    equal = np.array_equal(np.asarray(value1), np.asarray(value2))
                except Exception:
                    equal = make_hashable(value1) == make_hashable(value2)

                if not equal:
                    print(f"Different values for key '{key}' found at index {idx}:")
                    print(f"value1: {value1}")
                    print(f"value2: {value2}")
                else:
                    print(f"Sample index {idx} key '{key}' values are identical.")


dir="../datasets/mixed_grounding/annotations"



# cache1="final_mixed_train_no_coco_segm.engine.cache"
# cache2="final_mixed_train_no_coco_segm.engine.segment.cache"
# # compare_cache_files(dir, cache1, cache2)
# random_read_sample_and_compare(dir, cache1, cache2, num_samples=10)


dir="../datasets/Objects365v1/annotations"
cache1="objects365_train_segm.engine.cache"
cache2="objects365_train_segm.engine.segment.cache"
# compare_cache_files(dir, cache1, cache2)
# random_read_sample_and_compare(dir, cache1, cache2, num_samples=10)

dir="../datasets/flickr/annotations"
cache1="final_flickr_separateGT_train_segm.engine.cache"
cache2="final_flickr_separateGT_train_segm.engine.segment.cache"
# compare_cache_files(dir, cache1, cache2)
random_read_sample_and_compare(dir, cache1, cache2, num_samples=10)




