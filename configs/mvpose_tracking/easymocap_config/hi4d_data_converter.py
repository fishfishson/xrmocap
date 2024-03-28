type = 'EasyMocapDataCovnerter'
data_root = 'data/HI4D_easymocap'
bbox_detector = dict(
    type='MMtrackDetector',
    mmtrack_kwargs=dict(
        config='configs/modules/human_perception/' +
        'mmtrack_deepsort_faster-rcnn_fpn_4e_mot17-private-half.py',
        device='cuda'))
kps2d_estimator = dict(
    type='MMposeTopDownEstimator',
    mmpose_kwargs=dict(
        checkpoint='weight/hrnet_w48_coco_wholebody' +
        '_384x288_dark-f5726563_20200918.pth',
        config='configs/modules/human_perception/mmpose_hrnet_w48_' +
        'coco_wholebody_384x288_dark_plus.py',
        device='cuda'))

batch_size = 1000
scene_names = [
    'pair'
]
view_idxs = [
    '4',
    '16',
    '28',
    '40',
    '52',
    '64',
    '76',
    '88'
]
frame_period = 1
scene_range = 'all'
meta_path = './xrmocap_data/hi4d/xrmocap_meta'
visualize = False
