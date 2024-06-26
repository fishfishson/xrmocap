type = 'TopDownAssociationEvaluation'

__data_root__ = './data/chi3d_easymocap'
__meta_path__ = './xrmocap_data/chi3d_test/xrmocap_meta'
__bbox_thr__ = 0.85

logger = None
output_dir = './output/mvpose_tracking/chi3d_test'
pred_kps3d_convention = 'coco'
eval_kps3d_convention = 'openpose_25'
metric_list = [
    dict(
        type='PredictionMatcher',
        name='matching',
    ),
    dict(type='MPJPEMetric', name='mpjpe', unit_scale=1000),
    dict(type='PAMPJPEMetric', name='pa_mpjpe', unit_scale=1000),
    dict(
        type='PCKMetric',
        name='pck',
        use_pa_mpjpe=True,
        threshold=[50, 100],
    ),
    dict(
        type='PCPMetric',
        name='pcp',
        threshold=0.5,
        show_table=True,
        selected_limbs_names=[
            'left_lower_leg', 'right_lower_leg', 'left_upperarm',
            'right_upperarm', 'left_forearm', 'right_forearm', 'left_thigh',
            'right_thigh'
        ],
        additional_limbs_names=[['jaw', 'headtop']],
    ),
]
pick_dict = dict(
    mpjpe=['mpjpe_mean', 'mpjpe_std'],
    pa_mpjpe=['pa_mpjpe_mean', 'pa_mpjpe_std'],
    pck=['pck@50', 'pck@100'],
    pcp=['pcp_total_mean'],
)

associator = dict(
    type='MvposeAssociator',
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
        logger=logger,
    ),
    affinity_estimator=dict(type='AppearanceAffinityEstimator', init_cfg=None),
    point_selector=dict(
        type='HybridKps2dSelector',
        triangulator=dict(
            type='AniposelibTriangulator', camera_parameters=[],
            logger=logger),
        verbose=False,
        ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'],
        convention=pred_kps3d_convention),
    multi_way_matching=dict(
        type='MultiWayMatching',
        use_dual_stochastic_SVT=True,
        lambda_SVT=50,
        alpha_SVT=0.5,
        n_cam_min=2,
    ),
    kalman_tracking=dict(type='KalmanTracking', n_cam_min=2, logger=logger),
    identity_tracking=dict(
        type='KeypointsDistanceTracking',
        tracking_distance=1.5,
        tracking_kps3d_convention=pred_kps3d_convention,
        tracking_kps3d_name=[
            'left_shoulder', 'right_shoulder', 'left_hip_extra',
            'right_hip_extra'
        ]),
    checkpoint_path='./weight/mvpose/' +
    'resnet50_reid_camstyle-98d61e41_20220921.pth',
    best_distance=1800,
    interval=10,
    bbox_thr=__bbox_thr__,
    device='cuda',
    logger=logger,
)

dataset = dict(
    type='MviewMpersonDataset',
    data_root=__data_root__,
    img_pipeline=[
        dict(type='LoadImagePIL'),
        dict(type='ToTensor'),
        dict(type='RGB2BGR'),
    ],
    meta_path=__meta_path__,
    test_mode=True,
    shuffled=False,
    bbox_convention='xyxy',
    bbox_thr=__bbox_thr__,
    kps2d_convention=pred_kps3d_convention,
    gt_kps3d_convention='openpose_25',
    cam_world2cam=False,
)

dataset_visualization = dict(
    type='MviewMpersonDataVisualization',
    data_root=__data_root__,
    output_dir=output_dir,
    meta_path=__meta_path__,
    pred_kps3d_paths=None,
    bbox_thr=__bbox_thr__,
    vis_percep2d=False,
    kps2d_convention=None,
    vis_gt_kps3d=False,
    gt_kps3d_convention=None,
)
