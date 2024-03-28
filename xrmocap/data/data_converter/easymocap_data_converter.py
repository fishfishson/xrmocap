# yapf: disable
import cv2
import json
import logging
import numpy as np
import os
import re
from json.decoder import JSONDecodeError
from tqdm import tqdm
from typing import List, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.ffmpeg_utils import VideoInfoReader, video_to_array
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.data.data_visualization import MviewMpersonDataVisualization
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.human_perception.builder import MMtrackDetector, build_detector
from xrmocap.transform.convention.keypoints_convention import get_keypoint_num
from .base_data_converter import BaseDataCovnerter

from easymocap.mytools.camera_utils import read_cameras

# yapf: enable


class EasyMocapDataCovnerter(BaseDataCovnerter):

    def __init__(self,
                 data_root: str,
                 bbox_detector: Union[dict, None] = None,
                 kps2d_estimator: Union[dict, None] = None,
                 scene_names: Union[str, List[str]] = 'all',
                 view_idxs: Union[str, List[int]] = 'all',
                 scene_range: Union[str, List[List[int]]] = 'all',
                 frame_period: int = 1,
                 batch_size: int = 500,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'easymocap',
                 visualize: bool = False,
                 verbose: bool = True,
                 kps3d_dir: str = 'body25',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Create a dir at meta_path, split data into scenes according to
        scene_range, and convert GT kps3d, camera parameters and detected
        bbox2d, kps2d into XRLab format.

        Args:
            data_root (str):
                Path to the EasyMocap dataset.
            bbox_detector (Union[dict, None]):
                A human bbox_detector, or its config, or None.
                If None, converting perception 2d will be skipped.
                Defaults to None.
            kps2d_estimator (Union[dict, None]):
                A top-down kps2d estimator, or its config, or None.
                If None, converting perception 2d will be skipped.
                Defaults to None.
            scene_names (List[str], optional):
                A list of scene directory names, like
                ['s02', 's04'].
                Defaults to 'all', all scenes in data_root will be selected.
            view_idxs (List[int], optional):
                A list of selected view indexes in each scene,
                like [[0, 1], [0, 2, 4]].
                Defaults to 'all', all views will be selected.
            scene_range (List[List[int]], optional):
                Frame range of scenes. For instance,
                [[350, 470], [650, 750]]
                will select 120 frames from scene_0 and
                100 frames from scene_1,
                scene_0: 350-470, scene_1 650-750.
                Defaults to 'all', all frames will be selected.
            frame_period (int, optional):
                Sample rate of this converter. This converter will
                select one frame data from every frame_period frames.
                Defaults to 1.
            batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset.
                Defaults to 'panoptic'.
            visualize (bool, optional):
                Whether to visualize perception2d data and
                ground-truth 3d data. Defaults to False.
            verbose (bool, optional):
                Whether to print(logger.info) information during converting.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(
            data_root=data_root,
            meta_path=meta_path,
            dataset_name=dataset_name,
            verbose=verbose,
            logger=logger)
        # if view_idxs == 'all':
        #     self.view_idxs = [i for i in range(31)]
        # else:
        self.view_idxs = view_idxs
        self.n_view = len(self.view_idxs)
        # index: scene idx, value: (start_frame_idx, end_frame_idx)
        if scene_names == 'all':
            self.scene_names = self.list_scene_names()
        else:
            scene_names_ = self.list_scene_names()
            self.scene_names = []
            for x in scene_names_:
                for y in scene_names:
                    if y in x:
                        self.scene_names.append(x)
                        break
        self.scene_names.sort()
        print(self.scene_names)
                
        if scene_range == 'all':
            self.scene_range = [
                None,
            ] * len(self.scene_names)
        else:
            self.scene_range = scene_range

        if batch_size % frame_period != 0:
            batch_size = int(batch_size / frame_period) * frame_period
        self.batch_size = batch_size
        if isinstance(bbox_detector, dict):
            bbox_detector['logger'] = logger
            self.bbox_detector = build_detector(bbox_detector)
        else:
            self.bbox_detector = bbox_detector

        if isinstance(kps2d_estimator, dict):
            kps2d_estimator['logger'] = logger
            self.kps2d_estimator = build_detector(kps2d_estimator)
        else:
            self.kps2d_estimator = kps2d_estimator
        self.frame_period = frame_period
        self.visualize = visualize
        if self.visualize:
            vis_percep2d = self.bbox_detector is not None and \
                    self.kps2d_estimator is not None
            self.visualization = MviewMpersonDataVisualization(
                data_root=data_root,
                meta_path=meta_path,
                vis_percep2d=vis_percep2d,
                vis_gt_kps3d=True,
                output_dir=os.path.join(meta_path, 'visualize'),
                vis_aio_video=False,
                verbose=verbose,
                logger=logger)

        self.kps3d_dir = kps3d_dir
        self.resolution = None

    def run(self, overwrite: bool = False) -> None:
        """Convert xrmocap meta data from source dataset. Scenes will be
        created, and there are image lists, XRPrimer cameras, ground truth
        keypoints3d, perception2d in each of the scene. If any of
        bbox_detector, kps2d_estimator is None, skip converting perception2d.
        If visualize is checked, perception2d and ground truth will be
        visualized.

        Args:
            overwrite (bool, optional):
                Whether replace the files at
                self.meta_path.
                Defaults to False.
        """
        BaseDataCovnerter.run(self, overwrite=overwrite)
        with open(os.path.join(self.meta_path, 'dataset_name.txt'),
                  'w') as f_write:
            f_write.write(f'{self.dataset_name}')
        for scene_idx in range(len(self.scene_range)):
            scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
            os.makedirs(scene_dir, exist_ok=True)
            # self.covnert_video_frames(scene_idx)
            self.covnert_image_list(scene_idx)
            self.convert_cameras(scene_idx)
            self.convert_ground_truth(scene_idx)
            if self.bbox_detector is not None and \
                    self.kps2d_estimator is not None:
                self.convert_perception_2d(scene_idx)
        if self.visualize:
            self.visualization.run(overwrite=overwrite)

    def list_scene_names(self) -> List[str]:
        """Get the list of scene names in self.data_root.

        Returns:
            List[str]: The list of scene names.
        """
        file_list = sorted(os.listdir(self.data_root))
        scene_names = []
        for file_name in file_list:
            file_path = os.path.join(self.data_root, file_name)
            # match_result = pattern.match(file_name)
            # if match_result is not None and \
            #         len(match_result.group(0)) == len(file_name) and \
            #         check_path_existence(file_path, 'dir') == \
            #         Existence.DirectoryExistNotEmpty:
            if os.path.isdir(file_path):
                scene_names.append(file_name)
        return scene_names

    def covnert_video_frames(self, scene_idx: int) -> None:
        """Convert videos in source dataset to extracted jpg pictures.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info('Converting video frames' +
                             f' for scene {scene_idx}.')
        scene_name = self.scene_names[scene_idx]
        video_dir_name = 'hdVideos'
        # auto frame range
        if self.scene_range[scene_idx] is None:
            self.scene_range[scene_idx] = [0, None]
        # number of frames vary among views
        # find their common frame indexes
        min_n_frame = 1e9
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            video_name = f'hd_00_{view_idx:02d}.mp4'
            video_path = os.path.join(self.data_root, scene_name,
                                      video_dir_name, video_name)
            reader = VideoInfoReader(video_path, logger=self.logger)
            min_n_frame = min(int(reader['nb_frames']) - 1, min_n_frame)
        self.scene_range[scene_idx][1] = min(min_n_frame,
                                             self.scene_range[scene_idx][1])
        n_frame = self.scene_range[scene_idx][1] - \
            self.scene_range[scene_idx][0]
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            video_name = f'hd_00_{view_idx:02d}.mp4'
            video_path = os.path.join(self.data_root, scene_name,
                                      video_dir_name, video_name)
            output_folder = os.path.join(scene_dir, f'hd_00_{view_idx:02d}')
            os.makedirs(output_folder, exist_ok=True)
            for start_idx in range(self.scene_range[scene_idx][0],
                                   self.scene_range[scene_idx][1],
                                   self.batch_size):
                end_idx = min(self.scene_range[scene_idx][1],
                              start_idx + self.batch_size)
                if self.batch_size < n_frame:
                    self.logger.info('Decoding frames' +
                                     f'({start_idx}-{end_idx})/' +
                                     f'{self.scene_range[scene_idx][0]}' +
                                     f'-{self.scene_range[scene_idx][1]}')
                img_arr = video_to_array(
                    video_path,
                    start=start_idx,
                    end=end_idx,
                    logger=self.logger)
                for idx in tqdm(
                        range(start_idx, end_idx, self.frame_period),
                        disable=not self.verbose):
                    img = img_arr[idx - start_idx]
                    cv2.imwrite(
                        os.path.join(output_folder, f'{idx:06d}.jpg'), img)

    def covnert_image_list(self, scene_idx: int) -> None:
        """Convert extracted images to image list text file.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info('Converting image relative path list' +
                             f' for scene {scene_idx}.')
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            frame_dir = os.path.join(self.data_root, self.scene_names[scene_idx], 'images', view_idx)
            json_dir = os.path.join(self.data_root, self.scene_names[scene_idx], self.kps3d_dir)
            frame_list = []
            # start_idx = self.scene_range[scene_idx][0]
            # end_idx = self.scene_range[scene_idx][1]
            file_names = os.listdir(json_dir)
            file_names = [x.replace('json', 'jpg') for x in file_names]
            file_names.sort()
            for file_name in file_names[::self.frame_period]:
                # file_name = f'{frame_idx:06d}.jpg'
                abs_file_path = os.path.join(frame_dir, file_name)
                rela_file_path = os.path.relpath(abs_file_path, self.data_root)
                if self.resolution == None:
                    image = cv2.imread(abs_file_path)
                    self.resolution = image.shape[:2]
                if check_path_existence(abs_file_path,
                                        'file') == Existence.FileExist and \
                   check_path_existence(os.path.join(json_dir, file_name.replace('jpg', 'json')),
                                        'file') == Existence.FileExist:
                    frame_list.append(rela_file_path + '\n')
                else:
                    self.logger.error(
                        'Frames extract from dataset' +
                        ' is broken.' + f' Missing file: {abs_file_path}')
                    raise FileNotFoundError
            frame_list[-1] = frame_list[-1][:-1]
            with open(
                    os.path.join(scene_dir,
                                 f'image_list_view_{view_idx}.txt'),
                    'w') as f_write:
                f_write.writelines(frame_list)

    def convert_cameras(self, scene_idx: int) -> None:
        """Convert source data to XRPrimer camera parameters.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info(
                f'Converting camera parameters for scene {scene_idx}.')
        scene_name = self.scene_names[scene_idx]
        # cam_param_path = os.path.join(self.data_root, scene_name,
        #                               f'calibration_{scene_name}.json')
        # with open(cam_param_path, 'r') as f_read:
        #     panoptic_calib_dict = json.load(f_read)
        camera_dict = read_cameras(os.path.join(self.data_root, scene_name))
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        os.makedirs(cam_dir, exist_ok=True)
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            fisheye_param = FisheyeCameraParameter(
                name=f'fisheye_param_{view_idx}', logger=self.logger)
            # cam_key = f'00_{view_idx:02d}'
            # panoptic_cam_dict = None
            # for _, dict_value in enumerate(panoptic_calib_dict['cameras']):
            #     if dict_value['name'] == cam_key:
            #         panoptic_cam_dict = dict_value
            # if panoptic_cam_dict is None:
            #     self.logger.error('Camera calibration not found in json.' +
            #                       f' Missing key: hd_{cam_key}.')
            #     raise KeyError
            fisheye_param.set_resolution(
                width=self.resolution[1],
                height=self.resolution[0])
            translation = np.array(camera_dict[view_idx]['T'])
            fisheye_param.set_KRT(
                K=camera_dict[view_idx]['K'],
                R=camera_dict[view_idx]['R'],
                T=translation,
                world2cam=True)
            dist_list = camera_dict[view_idx]['dist'][0].tolist()
            fisheye_param.set_dist_coeff(
                dist_coeff_k=[dist_list[0], dist_list[1], dist_list[4]],
                dist_coeff_p=[dist_list[2], dist_list[3]])
            # dump the distorted camera
            fisheye_param.dump(
                os.path.join(cam_dir, f'{fisheye_param.name}.json'))

    def convert_ground_truth(self, scene_idx: int) -> None:
        """Convert source data to keypoints3d GT data.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        if self.verbose:
            self.logger.info(
                f'Converting ground truth keypoints3d for scene {scene_idx}.')
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        scene_name = self.scene_names[scene_idx]
        # start_idx = self.scene_range[scene_idx][0]
        # end_idx = self.scene_range[scene_idx][1]
        gt_path = os.path.join(self.data_root, scene_name, self.kps3d_dir)
        frames = os.listdir(gt_path)
        frames.sort()
        n_frame = len(frames)
        n_person = 1
        if self.kps3d_dir == 'body25':
            convention = 'openpose_25'
        else:
            raise NotImplementedError
        n_kps = get_keypoint_num(convention)
        kps3d_arr = np.zeros(shape=(n_frame, n_person, n_kps, 4))
        kps3d_mask = np.zeros(shape=(n_frame, n_person, n_kps), dtype=np.uint8)
        for i, frame in enumerate(frames):
            json_path = os.path.join(gt_path, f'{frame}')
            if check_path_existence(json_path) == Existence.FileExist:
                try:
                    with open(json_path, 'r') as f_read:
                        body3d_list = json.load(f_read)
                    load_success = True
                except JSONDecodeError:
                    self.logger.error(f'Broken json file at {json_path}.')
                    load_success = False
                if load_success:
                    for person_dict in body3d_list:
                        person_id = int(person_dict['id'])
                        if person_id > n_person - 1:
                            new_n_person = person_id - n_person + 1
                            new_kps3d = np.zeros(
                                shape=(n_frame, new_n_person, n_kps, 4))
                            new_mask = np.zeros(
                                shape=(n_frame, new_n_person, n_kps),
                                dtype=np.uint8)
                            kps3d_arr = np.concatenate((kps3d_arr, new_kps3d),
                                                       axis=1)
                            kps3d_mask = np.concatenate((kps3d_mask, new_mask),
                                                        axis=1)
                            n_person = person_id + 1
                        if len(person_dict['keypoints3d'][0]) == 3:
                            kp3d_arr = np.array(
                                person_dict['keypoints3d']).reshape(n_kps, 3)
                            kp3d_arr = np.concatenate([kp3d_arr, np.ones_like(kp3d_arr[:, :1])], axis=-1)
                            kps3d_arr[i, person_id, ...] = kp3d_arr
                        else: 
                            kps3d_arr[i, person_id, ...] = np.array(
                                person_dict['keypoints3d']).reshape(n_kps, 4)
                        kps3d_mask[i, person_id, ...] = 1
        # factor = 0.01  # convert panoptic values to meter
        # kps3d_arr[..., :3] *= factor
        keypoints3d = Keypoints(
            kps=kps3d_arr,
            mask=kps3d_mask,
            convention=convention,
            logger=self.logger)
        keypoints3d.dump(os.path.join(scene_dir, 'keypoints3d_GT.npz'))

    def convert_perception_2d(self, scene_idx: int) -> None:
        """Convert source data to 2D perception data, bbox + keypoints2d +
        track info.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        scene_bbox_list = []
        scene_keypoints_list = []
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            if self.verbose:
                self.logger.info('Inferring perception 2D data for' +
                                 f' scene {self.scene_names[scene_idx]} view {view_idx}')
            list_path = os.path.join(scene_dir,
                                     f'image_list_view_{view_idx}.txt')
            with open(list_path, 'r') as f_read:
                rela_path_list = f_read.readlines()
            frame_list = [
                os.path.join(self.data_root, rela_path.strip())
                for rela_path in rela_path_list
            ]
            bbox_list = self.bbox_detector.infer_frames(
                frame_path_list=frame_list,
                disable_tqdm=not self.verbose,
                multi_person=True,
                load_batch_size=self.batch_size)
            kps2d_list, _, _ = self.kps2d_estimator.infer_frames(
                frame_path_list=frame_list,
                bbox_list=bbox_list,
                disable_tqdm=not self.verbose,
                load_batch_size=self.batch_size)
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            n_person_max = 0
            n_frame = len(bbox_list)
            for frame_idx, frame_bboxes in enumerate(bbox_list):
                if n_person_max < len(frame_bboxes):
                    n_person_max = len(frame_bboxes)
            bbox_arr = np.zeros(shape=(n_frame, n_person_max, 5))
            for frame_idx, frame_bboxes in enumerate(bbox_list):
                for person_idx, person_bbox in enumerate(frame_bboxes):
                    if person_bbox is not None:
                        bbox_arr[frame_idx, person_idx] = person_bbox
            scene_bbox_list.append(bbox_arr)
            scene_keypoints_list.append(keypoints2d)
        dict_to_save = dict(
            bbox_tracked=isinstance(self.bbox_detector, MMtrackDetector),
            bbox_convention='xyxy',
            kps2d_convention=keypoints2d.get_convention(),
        )
        for idx in range(self.n_view):
            view_idx = self.view_idxs[idx]
            dict_to_save[f'bbox2d_view_{view_idx}'] = scene_bbox_list[idx]
            dict_to_save[f'kps2d_view_{view_idx}'] = scene_keypoints_list[
                idx].get_keypoints()
            dict_to_save[
                f'kps2d_mask_view_{view_idx}'] = scene_keypoints_list[
                    idx].get_mask()
        np.savez_compressed(
            file=os.path.join(scene_dir, 'perception_2d.npz'), **dict_to_save)
