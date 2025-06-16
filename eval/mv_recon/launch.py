import os
import sys
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import random
import open3d as o3d
import os.path as osp
from dust3r.image_pairs import make_pairs, make_pairs_tri
from torch.utils.data import DataLoader
# from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from dust3r.inference import inference, inference_ttt
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import pdb
import copy
from dust3r.model import AsymmetricCroCo3DStereo
from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
from dust3r.utils.geometry import geotrf
from copy import deepcopy
from dust3r.utils.geometry import inv, geotrf, depthmap_to_pts3d

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )

    parser.add_argument("--index", type=int, default=0)
    # 7 scenes
    parser.add_argument("--epoches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--accum_iter", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=32)


    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--shallow", action="store_true")
    return parser


def main(args):
    from eval.mv_recon.data import SevenScenes, NRGBD, DTU
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError

    print(f"the resolution is {resolution}.")

    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/data5/yuanyuheng/datasets/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=200,
        ),  # 20),

        # 'DTU': DTU(split='test', ROOT="/home/yuanyuheng/idea_ttt/dust3r_ttt/dtu/",
        #            resolution=resolution, num_seq=1, full_video=True, kf_every=5),
        # "NRGBD": NRGBD(
        #     split="test",
        #     ROOT="/data/datasets/neural_rgbd/",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=500,
        # ),
    }

    accelerator = Accelerator()
    device = accelerator.device

    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path, use_prompt = False if args.model_type == 'dust3r' else True, \
                                                                    is_shallow = args.shallow, prompt_size = args.prompt).to(args.device)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data, (str)(resolution[0]))
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):

                    if data_idx != args.index:
                        continue

                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    print(f">> The number of imgs is {len(batch)}")

                    if len(batch) == 1:
                        batch = [batch[0], copy.deepcopy(batch[0])]
                        batch[0]['idx'] = 0
                        batch[1]['idx'] = 1
                    silent = False

                    with torch.enable_grad():

                        if model.use_prompt:
                            print(f"epoches is {args.epoches}, lr is {args.lr}")
                            pairs_tri, idx_tri = make_pairs_tri(batch, prefilter=None, symmetrize=True)
                            model = inference_ttt(pairs_tri, idx_tri, model, device, batch_size=1, 
                                                  epoches = args.epoches, lr =args.lr, accum_iter=args.accum_iter,  verbose=not silent)   

                    model.eval()
                    pairs = make_pairs(batch, prefilter=None, symmetrize=True)
                    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

                    with torch.enable_grad():
                        mode = GlobalAlignerMode.PointCloudOptimizer if len(batch) > 2 else GlobalAlignerMode.PairViewer
                        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
                        lr = 0.01

                        if mode == GlobalAlignerMode.PointCloudOptimizer:
                            loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)

                    preds = []
                    pts3d = scene.get_pts3d()
                    first_pose = scene.get_im_poses()[0]
                    for pts in pts3d:
                        pts_at_first_frame = geotrf(inv(first_pose), pts)
                        preds.append(
                            {'pts3d_in_other_view': pts_at_first_frame.unsqueeze(0).to(device)}
                        )
                    for i, view in enumerate(batch):
                        for k, v in view.items():
                            if isinstance(v, torch.Tensor):
                                batch[i][k] = view[k].to(device)
                        
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}"
                    )

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        pts = pred_pts[j].cpu().numpy()[0]

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        #### Align predicted 3D points to the ground truth
                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)

                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        save_params,
                    )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                        ),
                        pcd,
                    )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                        pcd_gt,
                    )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    print(
                        f"Epoch: {args.epoches}, lr: {args.lr}, accum_iter: {args.accum_iter}, prompts: {args.prompt}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Epoch: {args.epoches}, lr: {args.lr}, accum_iter: {args.accum_iter}, prompts: {args.prompt}",
                        file=open(log_file, "a"),
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    del model, pcd_gt, pcd
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # pdb.set_trace()
    # wandb.init(project='idea_ttt')
    parser = get_args_parser()
    args = parser.parse_args()
    seed_torch()
    main(args)
