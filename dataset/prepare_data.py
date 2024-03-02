
import os
import shutil

import json
import numpy as np
import pickle

from collections import Counter

from tqdm import tqdm

sgfront_dir = "/mnt/hdd1/3D-FRONT/"
instruct_scene_dir = "/mnt/hdd1/commonscenes_workspace/InstructScene/dataset/InstructScene/threed_front_all/"

sg_rels_path = "/mnt/hdd1/3D-FRONT/relationships_all_merged.json"
with open(sg_rels_path, "r") as file:
    sg_rels_raw = json.load(file)

sg_rels = {}
for scan in sg_rels_raw['scans']:
    sg_rels[scan['scan']] = scan

mapping = sgfront_dir + "mapping.json"
with open(mapping, "r") as file:
    mappings_raw = json.load(file)

sg_classes_all = sorted(list(set([cls_name for key, cls_name in mappings_raw.items() if cls_name != '_scene_'])))
num_classes = len(sg_classes_all)
with open(instruct_scene_dir+"/all_classes.txt", "a") as file:
    file.write("\n".join(sg_classes_all))
cls_to_id = {cls_name: i for i, cls_name in enumerate(sg_classes_all)}

obj_boxes_path = "/mnt/hdd1/3D-FRONT/obj_boxes_all_merged.json"
with open(obj_boxes_path, "r") as file:
    obj_boxes = json.load(file)

# Read all directories within
folders = os.listdir(instruct_scene_dir)

# Start keeping dataset stats
dataset_stats = {}
dataset_stats["class_labels"] = sg_classes_all
dataset_stats["class_order"] = cls_to_id
dataset_stats["count_furniture"] = {cls_name: 0 for cls_name in sg_classes_all} # to be filled
dataset_stats["class_frequencies"] = {cls_name: 0 for cls_name in sg_classes_all} # to be filled

for folder in tqdm(folders, desc="Processing"):
    if folder == "dataset_stats.txt":
        continue

    tqdm.write(f"Currently: {folder}")

    # Fetch jid and room name
    try:
        scene_json_id, scene_id = folder.split("_")
    except ValueError:
        print("skipping folder", folder)
        continue


    if scene_id not in obj_boxes.keys():
        print("skipping", scene_id)
        continue

    if scene_id not in sg_rels.keys():
        print("skipping", scene_id)
        continue

    sgfront_boxes = obj_boxes[scene_id]
    sgfront_objects = sg_rels[scene_id]['objects']
    sgfront_rels = sg_rels[scene_id]['relationships']
    # subtract 1 from ids
    for i in range(len(sgfront_rels)):
        sgfront_rels[i][0] -= 1
        sgfront_rels[i][1] -= 1
        sgfront_rels[i][2] -= 1
        sgfront_rels[i][1], sgfront_rels[i][2] = sgfront_rels[i][2], sgfront_rels[i][1]

        # Get box info from sgfront for floor, update boxes.npz
    instruct_boxes_path_orig = instruct_scene_dir + folder + '/boxes_old.npz'
    with np.load(instruct_boxes_path_orig) as data:
        instruct_boxes = {key: data[key] for key in data.files}
    ### instruct_boxes = np.load(instruct_boxes_path, allow_pickle=True)
    # ['uids', 'jids', 'scene_id', 'scene_uid', 'scene_type', 'json_path', 'room_layout', 'floor_plan_vertices', 'floor_plan_faces', 'floor_plan_centroid', 'class_labels', 'translations', 'sizes', 'angles']

    # update translations [Nx3] float32:  instruct_boxes['translations']
    # Update sizes [Nx3] float32: instruct_boxes['sizes']
    # Update angles [Nx1] : instruct_boxes['sizes']
    # Update class labels (one hot vector [NxC]): instruct_boxes['class_labels']
    num_objects = len(obj_boxes[scene_id].keys())-1
    new_sizes = np.zeros((num_objects,3))
    new_translations = np.zeros((num_objects, 3))
    new_angles = np.zeros((num_objects, 1))
    new_class_ids = []
    new_classes = np.zeros((num_objects, num_classes), dtype=np.int32)
    for obj_i in range(num_objects):
        param7 = sgfront_boxes[str(obj_i+1)]['param7']
        new_sizes[obj_i] = np.array(param7[:3])
        new_translations[obj_i] = np.array(param7[3:6])
        new_angles[obj_i] = np.array(param7[-1])

        # get the class names from mapping
        orig_name = sgfront_objects[str(obj_i+1)]
        mapped_name = mappings_raw[orig_name]
        mapped_id = cls_to_id[mapped_name]
        new_class_ids.append(mapped_id)
        new_classes[obj_i, mapped_id] = 1
        dataset_stats["count_furniture"][mapped_name] += 1

    # save instruct_boxes back
    instruct_boxes_path = instruct_scene_dir + folder + '/boxes.npz'
    ### np.savez(instruct_boxes_path_old, **instruct_boxes)
    instruct_boxes['translations'] = new_translations.astype(np.float32)
    instruct_boxes['sizes'] = new_sizes.astype(np.float32)
    instruct_boxes['angles'] = new_angles.astype(np.float32)
    instruct_boxes['class_labels'] = new_classes
    np.savez(instruct_boxes_path, **instruct_boxes)

    # read relations from sgfront
    instruct_rels_path_orig = instruct_scene_dir + folder + '/relations_old.npy'
    instruct_rels = np.load(instruct_rels_path_orig, allow_pickle=True)
    new_instruct_rels = []
    for rel in sgfront_rels:
        new_instruct_rels.append(rel[:3])

    # save relations
    instruct_rels_path = instruct_scene_dir + folder + '/relations.npy'
    ###np.save(instruct_rels_path_old, instruct_rels)
    np.save(instruct_rels_path, np.array(new_instruct_rels))

    # update desc
    instruct_descs_path_orig = instruct_scene_dir + folder + '/descriptions_old.pkl'
    with open(instruct_descs_path_orig, 'rb') as file:
        instruct_descs = pickle.load(file)

    # 'obj_class_ids', 'obj_counts', 'obj_relations'
    new_obj_relations = [tuple(rels) for rels in new_instruct_rels]
    new_obj_class_ids = new_class_ids # [obj_id1, obj_id2]
    new_obj_counts = list(Counter(new_obj_class_ids).items()) # [(obj_id, occurence), (), ..]
    instruct_descs_path = instruct_scene_dir + folder + '/descriptions.pkl'
    ###with open(instruct_descs_path_old, 'wb') as file:
    ###  pickle.dump(instruct_descs, file)
    instruct_descs['obj_class_ids'] = new_obj_class_ids
    instruct_descs['obj_counts'] = new_obj_counts
    instruct_descs['obj_relations'] = new_obj_relations
    with open(instruct_descs_path, 'wb') as file:
      pickle.dump(instruct_descs, file)

# update and save dataset stats
#total_furnitures = sum(dataset_stats["count_furniture"].values())
#dataset_stats["count_furniture"] = {key: 0 for key in sg_classes_all} # to be filled
#dataset_stats["class_frequencies"] = {key: count/total_furnitures for key, count in dataset_stats["count_furniture"].items()}
#with open(instruct_scene_dir+"/dataset_stats.txt", "w") as file:
#    json.dump(dataset_stats, file, indent=4)


