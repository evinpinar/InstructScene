from typing import *
from numpy import ndarray

import math

import numpy as np
from shapely.geometry import Polygon, Point
from nltk.corpus import cmudict

import numpy as np
from cmath import rect, phase
from scipy.spatial import ConvexHull

def compute_loc_rel(corners1: ndarray, corners2: ndarray, name1: str, name2: str) -> Optional[str]:
    assert corners1.shape == corners2.shape == (8, 3), "Shape of corners should be (8, 3)."

    center1 = corners1.mean(axis=0)
    center2 = corners2.mean(axis=0)

    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi
    distance = (d[2]**2 + d[0]**2)**0.5  # center distance on the ground

    box1 = corners1[[0, 1, 4, 5], :][:, [0, 2]]  # 4 corners of the bottom face (0&5, 1&4 are opposite corners)
    box2 = corners2[[0, 1, 4, 5], :][:, [0, 2]]

    # Note that bounding boxes might not be axis-aligned
    polygon1, polygon2 = Polygon(box1[[0, 1, 3, 2], :]), Polygon(box2[[0, 1, 3, 2], :])  # change the order to be convex
    point1, point2 = Point(center1[[0, 2]]), Point(center2[[0, 2]])

    # Initialize the relationship
    p = None

    # Horizontal relationship: "left"/"right"/"front"/"behind"
    if theta >= 3 * math.pi / 4 or theta < -3 * math.pi / 4:
        p = "left of"
    elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = "behind"
    elif -math.pi / 4 <= theta < math.pi / 4:
        p = "right of"
    elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = "in front of"

    # Vertical relationship: "above"/"below"
    if point1.within(polygon2) or point2.within(polygon1):
        delta1 = center1[1] - center2[1]
        delta2 = (
            corners1[:, 1].max() - corners1[:, 1].min() +
            corners2[:, 1].max() - corners2[:, 1].min()
        ) / 2.
        if (delta1 - delta2) >= 0. or "lamp" in name1:
            # Indicate that:
            # (1) delta1 > 0. (because always delta2 > 0.): `center1` is above `center2`
            # (2) delta1 >= delta2: `corners1` and `corners2` not intersect vertically
            # ==> `corners1` is completely above `corners2`
            # Or the subject is a lamp, which is always above other objects
            p = "above"
            return p
        if (-delta1 - delta2) >= 0. or "lamp" in name2:
            # ==> `corners1` is completely below `corners2`
            # Or the object is a lamp, which is always above other objects
            p = "below"
            return p

    if distance > 3.:
        return None  # too far away
    else:
        if distance < 1.:
            p = "closely " + p
        return p


def validate_constrains(triples, pred_boxes, pred_angles, keep, accuracy, strict=True, overlap_threshold=0.3):

    param6 = True
    #param6 = pred_boxes.shape[1] == 6
    #layout_boxes = pred_boxes

    for idx, [s, p, o] in enumerate(triples):
        # if keep is None:
        #     box_s = layout_boxes[s.item()].cpu().detach().numpy()
        #     box_o = layout_boxes[o.item()].cpu().detach().numpy()
        # else:
        #     if keep[s.item()] == 1 and keep[o.item()] == 1: # if both are unchanged we evaluate the normal constraints
        #         box_s = layout_boxes[s.item()].cpu().detach().numpy()
        #         box_o = layout_boxes[o.item()].cpu().detach().numpy()
        #     else:
        #         continue

        box_s = pred_boxes[idx][0] #.cpu().detach().numpy()
        box_o = pred_boxes[idx][1]

        if p == "left":
            # z
            if box_s[5] - box_o[5] > -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['left'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['left'].append(1)
                accuracy['total'].append(1)
        if p == "right":
            if box_s[5] - box_o[5] < 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['right'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['right'].append(1)
                accuracy['total'].append(1)
        if p == "front":
            if box_s[3] - box_o[3] < -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['front'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['front'].append(1)
                accuracy['total'].append(1)
        if p == "behind":
            if box_s[3] - box_o[3] > 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
        # bigger than
        if p == "bigger than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume < 0.15:
                accuracy['bigger'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['bigger'].append(1)
                accuracy['total'].append(1)
        # smaller than
        if p == "smaller than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume > -0.15:
                accuracy['smaller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['smaller'].append(1)
                accuracy['total'].append(1)
        # higher than
        if p == "taller than":
            absheight_s = box_s[4]+box_s[1]
            absheight_o = box_o[4]+box_o[1]
            if (absheight_s - absheight_o) / absheight_s < 0.1:
                accuracy['taller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['taller'].append(1)
                accuracy['total'].append(1)
        # lower than
        if p == "shorter than":
            absheight_s = box_s[4] + box_s[1]
            absheight_o = box_o[4] + box_o[1]
            if (absheight_s - absheight_o) / absheight_s > -0.1:
                accuracy['shorter'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['shorter'].append(1)
                accuracy['total'].append(1)

        # standing on
        if p == "standing on":
            if np.abs(box_s[4] - box_o[4]) < 0.04:
                accuracy['standing on'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['standing on'].append(0)
                accuracy['total'].append(0)
        # close by
        if p == "close by":
            corners_s = corners_from_box(box_s, param6, with_translation=True)
            corners_o = corners_from_box(box_o, param6, with_translation=True)
            c_dist1 = close_dis(corners_s, corners_o)
            if c_dist1 > 0.45:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['close by'].append(1)
                accuracy['total'].append(1)

        # symmetrical to
        if p == "symmetrical to":
            sub_center_in_scene_flip_x = [-box_s[3], box_s[5]]
            sub_center_in_scene_flip_z = [box_s[3], -box_s[5]]
            sub_center_in_scene_flip_xz = [-box_s[3], -box_s[5]]
            obj_center_in_scene = [box_o[3], box_o[5]]
            if cal_l2_distance(sub_center_in_scene_flip_xz, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_x, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_z, obj_center_in_scene) < 0.45:
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)

    return accuracy

def close_dis(corners1,corners2):
    dist = -2 * np.matmul(corners1, corners2.transpose())
    dist += np.sum(corners1 ** 2, axis=-1)[:, None]
    dist += np.sum(corners2 ** 2, axis=-1)[None, :]
    dist = np.sqrt(dist)
    return np.min(dist)

def cal_l2_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def angular_distance(a, b):
    a %= 360.
    b %= 360.

    va = np.matmul(rot2d(a), [1, 0])
    vb = np.matmul(rot2d(b), [1, 0])
    return anglebetween2vecs(va, vb) % 360.


def anglebetween2vecs(va, vb):
    rad = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    return np.rad2deg(rad)


def rot2d(degrees):
    rad = np.deg2rad(degrees)
    return np.asarray([[np.cos(rad), -np.sin(rad)],
                       [np.sin(rad), np.cos(rad)]])


def estimate_angular_mean(deg):
    return np.rad2deg(phase(np.sum(rect(1, np.deg2rad(d)) for d in deg)/len(deg))) % 360.


def estimate_angular_std(degs):
    m = estimate_angular_mean(degs)
    std = np.sqrt(np.sum([angular_distance(d, m)**2 for d in degs]) / len(degs))
    return std

def corners_from_box(box, param6=True, with_translation=False):
    # box given as: [l, h, w, px, py, pz, z]
    # l meansures z axis; h measures y axis; w measures x axis.
    # (px, py, pz) is the bottom center
    if param6:
        l, h, w, px, py, pz = box
    else:
        l, h, w, px, py, pz, _ = box

    (tx, ty, tz) = (px, py, pz) if with_translation else (0,0,0)

    x_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2]
    y_corners = [h,h,h,h,0,0,0,0]
    z_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2]
    corners_3d = np.dot(np.eye(3), np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + tx
    corners_3d[1,:] = corners_3d[1,:] + ty
    corners_3d[2,:] = corners_3d[2,:] + tz
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def box3d_iou(box1, box2, param6=True, with_translation=False):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is positive Y_h
        corners2: numpy array (8,3), assume up direction is positive Y_h
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    corners1 = corners_from_box(box1, param6, with_translation)
    corners2 = corners_from_box(box2, param6, with_translation)

    rect1 = [(corners1[i,2], corners1[i,0]) for i in range(0,4)]
    rect2 = [(corners2[i,2], corners2[i,0]) for i in range(0,4)]

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    volmin = min(vol1, vol2)

    iou = inter_vol / volmin #(vol1 + vol2 - inter_vol)

    return iou, iou_2d

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)


def pointcloud_overlap(pclouds, objs, boxes, angles, triples, vocab, overlap_metric):

    obj_classes = vocab['object_idx_to_name']
    pred_classes = vocab['pred_idx_to_name']
    pair = [(t[0].item(),t[2].item()) for t in triples]
    pred = [t[1].item() for t in triples]
    pair2pred = dict(zip(pair, pred))
    structural = ['floor', 'wall', 'ceiling', '_scene_']
    touching = ['none', 'inside', 'attached to', 'part of', 'cover', 'belonging to', 'build in', 'connected to']
    boxes = torch.cat([boxes.float(), angles.view(-1,1).float()], 1)

    for i in range(len(pclouds) - 1):
        for j in range(i+1, len(pclouds)):
            if obj_classes[objs[i]].split('\n')[0] in structural or \
                    obj_classes[objs[j]].split('\n')[0] in structural:
                # do not consider structural objects
                continue
            if (i, j) in pair2pred.keys() and pred_classes[pair2pred[(i,j)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            if (j, i) in pair2pred.keys() and pred_classes[pair2pred[(j,i)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            pc1 = fit_shapes_to_box(boxes[i].clone(), pclouds[i].clone())
            pc2 = fit_shapes_to_box(boxes[j].clone(), pclouds[j].clone())
            result = pointcloud_overlap_pair(pc1, pc2)
            overlap_metric.append(result)
    return overlap_metric


def pointcloud_overlap_pair(pc1, pc2):
    from sklearn.neighbors import NearestNeighbors
    all_pc = np.concatenate([pc1, pc2], 0)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
    nbrs.fit(all_pc)
    distances, indices = nbrs.kneighbors(pc1)
    # first neighbour will likely be itself other neighbour is a point from the same pc or the other pc
    # two point clouds are overlaping, when the nearest neighbours of one set are from the other set
    overlap = np.sum(indices >= len(pc1))
    return overlap

"""
Taken from https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
"""

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article


################################################################


def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]


def rotate_rel(rel: str, r: float) -> str:
    assert r in [0.0, np.pi * 0.5, np.pi, np.pi * 1.5]

    if rel in ["above", "below"]:
        return rel

    if r == 0.0:
        return rel
    elif r == np.pi * 0.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "right of",
                "behind": "left of",
                "left of": "in front of",
                "right of": "behind"
            }[rel.replace("closely ", "")]
    elif r == np.pi:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "behind",
                "behind": "in front of",
                "left of": "right of",
                "right of": "left of"
            }[rel.replace("closely ", "")]
    elif r == np.pi * 1.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "left of",
                "behind": "right of",
                "left of": "behind",
                "right of": "in front of"
            }[rel.replace("closely ", "")]


def model_desc_from_info(cate: str, style: str, theme: str, material: str, seed=None):
    cate_name = {
        "desk":                                    "desk",
        "nightstand":                              "nightstand",
        "king-size bed":                           "double bed",
        "single bed":                              "single bed",
        "kids bed":                                "kids bed",
        "ceiling lamp":                            "ceiling lamp",
        "pendant lamp":                            "pendant lamp",
        "bookcase/jewelry armoire":                "bookshelf",
        "tv stand":                                "tv stand",
        "wardrobe":                                "wardrobe",
        "lounge chair/cafe chair/office chair":    "lounge chair",
        "dining chair":                            "dining chair",
        "classic chinese chair":                   "classic chinese chair",
        "armchair":                                "armchair",
        "dressing table":                          "dressing table",
        "dressing chair":                          "dressing chair",
        "corner/side table":                       "corner side table",
        "dining table":                            "dining table",
        "round end table":                         "round end table",
        "drawer chest/corner cabinet":             "cabinet",
        "sideboard/side cabinet/console table":    "console table",
        "children cabinet":                        "children cabinet",
        "shelf":                                   "shelf",
        "footstool/sofastool/bed end stool/stool": "stool",
        "coffee table":                            "coffee table",
        "loveseat sofa":                           "loveseat sofa",
        "three-seat/multi-seat sofa":              "multi-seat sofa",
        "l-shaped sofa":                           "l-shaped sofa",
        "lazy sofa":                               "lazy sofa",
        "chaise longue sofa":                      "chaise longue sofa",
        "barstool":                                "barstool",
        "wine cabinet":                            "wine cabinet"
    }[cate.lower().replace(" / ", "/")]

    attrs = []
    if style is not None and style != "Others":
        attrs.append(style.replace(" ", "-").lower())
    if material is not None and material != "Others":
        attrs.append(material.replace(" ", "-").lower())
    if theme is not None:
        attrs.append(theme.replace(" ", "-").lower())

    if seed is not None:
        np.random.seed(seed)
    attr = np.random.choice(attrs) + " " if len(attrs) > 0 else ""

    return attr + cate_name


################################################################


def fill_templates(
    desc: Dict[str, List],
    object_types: List[str], predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int]=None,
    return_obj_ids=False
) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
    if object_descs is None:
        assert object_types is not None

    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

    # Describe the relations between the main objects and others
    selected_relation_indices = np.random.choice(
        len(desc["obj_relations"]),
        min(np.random.choice([1, 2]), len(desc["obj_relations"])),  # select 1 or 2 relations
        replace=False
    )
    selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
    selected_relations = [
        (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
        for s, p, o in selected_relations
    ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
    selected_descs = []
    selected_sentences = []
    selected_object_ids = []  # e.g., [0, ...]; 0 is object id
    for idx in selected_relation_indices:
        s, p, o = desc["obj_relations"][idx]
        s, p, o = int(s), int(p), int(o)
        if object_descs is None:
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
            p_str = predicate_types[p]
            if np.random.rand() > 0.5:
                subject = f"{get_article(s_name).replace('a', 'A')} {s_name}"
                predicate = f" is {p_str} "
                object = f"{get_article(o_name)} {o_name}."
            else:  # 50% of the time to reverse the order
                subject = f"{get_article(o_name).replace('a', 'A')} {o_name}"
                predicate = f" is {reverse_rel(p_str)} "
                object = f"{get_article(s_name)} {s_name}."
        else:
            if np.random.rand() < 0.75:
                s_name = object_descs[s]
            else:  # 25% of the time to use the object type as the description
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
            if np.random.rand() < 0.75:
                o_name = object_descs[o]
            else:
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                o_name = f"{get_article(o_name)} {o_name}"

            p_str = predicate_types[p]
            rev_p_str = reverse_rel(p_str)

            if p_str in ["left of", "right of"]:
                if np.random.rand() < 0.5:
                    p_str = "to the " + p_str
                    rev_p_str = "to the " + rev_p_str
            elif p_str in ["closely left of", "closely right of"]:
                if np.random.rand() < 0.25:
                    p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.5:
                    p_str = "to the close " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.75:
                    p_str = "to the near " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

            if np.random.rand() < 0.5:
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in s_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {s_name}"
                predicate = f" {p_str} "
                object = f"{o_name}."
                selected_descs.append((s_name, o_name))
                selected_object_ids.append(s)
            else:  # 50% of the time to reverse the order
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in o_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {o_name}"
                predicate = f" {rev_p_str} "
                object = f"{s_name}."
                selected_descs.append((o_name, s_name))
                selected_object_ids.append(o)
        selected_sentences.append(subject + predicate + object)

    text = ""
    conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finnally, ", " And ", " "]
    for i, sentence in enumerate(selected_sentences):
        if i == 0:
            text += sentence
        else:
            conjunction = conjunctions[np.random.choice(len(conjunctions))]
            while conjunction == " Finnally, " and i != len(selected_sentences)-1:
                # "Finally" should be used only in the last sentence
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
            if conjunction != " ":
                sentence = sentence[0].lower() + sentence[1:]
            text += conjunction + sentence

    if return_obj_ids:
        return text, selected_relations, selected_descs, selected_object_ids
    else:
        return text, selected_relations, selected_descs  # return `selected_relations`, `selected_descs` for evaluation
