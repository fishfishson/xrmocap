type = 'EasyMocapDataCovnerter'
data_root = 'data/chi3d_easymocap'
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
        device='cuda'),
    bbox_thr=0.5)

batch_size = 1000
scene_names = [
    's03_Hug 2'
]
view_idxs = [
    '50591643',
    '58860488',
    '60457274',
    '65906101']
frame_period = 1
scene_range = 'all'
meta_path = './xrmocap_data/chi3d_test/xrmocap_meta'
visualize = True
