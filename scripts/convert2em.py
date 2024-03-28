import os
import glob
import argparse
import numpy as np
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_idx, get_keypoint_num,
)
from tqdm import tqdm
from easymocap.mytools.file_utils import save_numpy_dict

panoptic15incoco = [1,0,2,3,5,7,9,11,13,4,6,8,10,12,14]
panoptic15inbody25 = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11]

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('meta_path')
# parser.add_argument('data_path')
parser.add_argument('out_path')
args = parser.parse_args()

results = glob.glob(os.path.join(args.path, '*keypoints3d.npz'))
results.sort()
for res in tqdm(results):
    scene = os.path.basename(res).split('_')[0]
    scene_id = scene.replace('scene', '')
    scene_dir = f'scene_{scene_id}'
    meta_dir = os.path.join(args.meta_path, scene_dir)
    gt = np.load(os.path.join(meta_dir, 'keypoints3d_GT.npz'))
    gt_kps3d = gt['keypoints']
    gt_masks = gt['mask']
    # gt_kps3d = Keypoints(
    #     dtype='numpy',
    #     kps=gt['keypoints'],
    #     mask=gt['mask'],
    #     convention=str(gt['convention'])
    # )
    # gt_kps3d = convert_keypoints(gt_kps3d, 'panoptic_15')
    pred = np.load(res)
    # pred_kps3d = Keypoints(
    #     dtype='numpy', 
    #     kps=pred['keypoints'],
    #     mask=pred['mask'],
    #     convention=str(pred['convention'])
    # )
    pred_kps3d = pred['keypoints']
    pred_masks = pred['mask']
    img_list = glob.glob(os.path.join(meta_dir, '*.txt'))
    with open(img_list[0], 'r') as f:
        files = f.readlines()
    assert pred_kps3d.shape[0] == gt_kps3d.shape[0]
    assert pred_kps3d.shape[0] == len(files)
    for i in range(pred_kps3d.shape[0]):
        seq = files[i].split('/')[0] 
        frame = files[i].strip('\n').split('/')[-1].split('.')[0]
        outname = '_'.join([seq, frame]) + '.json'
        pred_kp3d = pred_kps3d[i]
        pred_mask = pred_masks[i]
        valid_kp3d = []
        for n in range(pred_kp3d.shape[0]):
            if pred_mask[n].sum() > 0:
                valid_kp3d.append(pred_kp3d[n])
        if len(valid_kp3d) == 0:
            print('EMPTY PRED!!!')
            pred_kp3d = np.zeros((1, 15, 4))
        else:
            pred_kp3d = np.stack(valid_kp3d)
            pred_neck = 0.5 * (pred_kp3d[:, 5:6] + pred_kp3d[:, 6:7])
            pred_pelvis = 0.5 * (pred_kp3d[:, 11:12] + pred_kp3d[:, 12:13])
            pred_kp3d = np.concatenate([pred_kp3d[:, :1], pred_neck, pred_pelvis, pred_kp3d[:, 5:]], axis=1)
            pred_kp3d = pred_kp3d[:, panoptic15incoco]
        gt_kp3d = gt_kps3d[i]
        gt_mask = gt_masks[i]
        valid_kp3d = []
        for n in range(gt_kp3d.shape[0]):
            if gt_mask[n].sum() > 0:
                valid_kp3d.append(gt_kp3d[n])
        if len(valid_kp3d) == 0:
            print('EMPTY GT!!!')
            gt_kp3d = np.zeros((1, 15, 4))
        else:
            gt_kp3d = np.stack(valid_kp3d)
            gt_kp3d = gt_kp3d[:, panoptic15inbody25]
        json = {
            'pred': pred_kp3d,
            'gt': gt_kp3d
        }
        save_numpy_dict(os.path.join(args.out_path, outname), json)