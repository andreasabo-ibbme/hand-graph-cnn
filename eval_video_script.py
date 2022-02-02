# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from asyncio import protocols
from copyreg import pickle
from imp import NullImporter
import os
import numpy
import torch
import pickle

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints, save_output_video, save_mesh_overlay, save_3D_plots
from hand_shape_pose.util import renderer

def parse_arguments():
    parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--andrea",
        default="NOT_FOUND",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = os.path.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)
    logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
    logger.info(cfg)

    return output_dir, logger, cfg.EVAL.SAVE_DIR

def main():
    # 0. Config set up
    output_global_dir, logger, local_save_dir = parse_arguments()

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_global_dir)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

    # Process videos one by one
    vid_name_fullpath = r"/home/saboa/data/EDS/SampleVideos/IMG_1692.MOV"

    # Save the output in a separate folder
    vid_name, ext = os.path.splitext(os.path.split(vid_name_fullpath)[-1])
    output_folder_root = os.path.join(local_save_dir, vid_name)
    mkdir(output_folder_root)
    output_2d_video = os.path.join(output_folder_root, "hgCNN_2d_out.MOV")
    output_data_file = os.path.join(output_folder_root, "hgCNN_data.pkl")

    # 2. Load data
    dataset_val = build_dataset(cfg.EVAL.DATASET, vid_name=vid_name_fullpath)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        num_workers=0
    )

    # 3. Inference
    model.eval()
    results_pose_cam_xyz = {}
    all_data = {}
    cpu_device = torch.device("cuda:0")
    logger.info("Evaluate on {} frames:".format(len(dataset_val)))
    for i, batch in enumerate(data_loader_val):
        images, cam_params, bboxes, pose_roots, pose_scales, image_ids = batch
        images, cam_params, bboxes, pose_roots, pose_scales = \
            images.to(device), cam_params.to(device), bboxes.to(device), pose_roots.to(device), pose_scales.to(device)
        with torch.no_grad():
            est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
                model(images, cam_params, bboxes, pose_roots, pose_scales)

            est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
            est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
            est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]


        results_pose_cam_xyz.update({img_id.item(): result for img_id, result in zip(image_ids, est_pose_cam_xyz)})
        all_data.update({img_id.item(): {"pose2D" : pose.cpu().numpy(), "pose3D": pose3D.cpu().numpy(), "mesh3D" : mesh3D.cpu().numpy(), \
            "cam_param": cam_params.cpu().numpy(), "box": bboxes.cpu().numpy()} for\
             img_id, pose, pose3D, mesh3D, cam_params, bboxes in zip(image_ids, est_pose_uv, est_pose_cam_xyz, est_mesh_cam_xyz, cam_params, bboxes)})
        if i % cfg.EVAL.PRINT_FREQ == 0:
            # 4. evaluate pose estimation
            avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, save_results=False)  # cm
            msg = 'Evaluate: [{0}/{1}]\t' 'Average pose estimation error: {2:.2f} (mm)'.format(
                len(results_pose_cam_xyz), len(dataset_val), avg_est_error * 10.0)
            logger.info(msg)

            # 5. visualize mesh and pose estimation
            if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
                file_name = '{}_{}.jpg'.format(os.path.join(output_dir, 'pred'), i)
                logger.info("Saving image: {}".format(file_name))
                save_batch_image_with_mesh_joints(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
                                                  bboxes.to(cpu_device), est_mesh_cam_xyz, est_pose_uv,
                                                  est_pose_cam_xyz, file_name)

    # make output video (optional)
    if cfg.EVAL.SAVE_OUTPUT_VIDEO:
        save_output_video(all_data, output_2d_video, vid_name_fullpath)                 # 2D
        # save_mesh_overlay(all_data, outvideo_name, vid_name_fullpath, mesh_renderer)  # 3D mesh
        # save_3D_plots(all_data, outvideo_name, vid_name_fullpath)                     # 3D points (no underlay)

    # Save all output for later use
    with open(output_data_file, 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
