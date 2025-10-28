#!/usr/bin/env python3
#python tools/inspect_3dpe.py --pe_dump /root/zjm/ovs/openscene/out/scannet_ptv3_lseg_NCE_pe/result_eval/pe_dump/pe_dump_scene0.npz
import os
import numpy as np
import argparse
from pathlib import Path

def write_ply_xyz_rgb(path, xyz, rgb=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for x,y,z in xyz:
                f.write(f"{x} {y} {z}\n")
        else:
            for (x,y,z),(r,g,b) in zip(xyz, rgb):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

def percentile_color(values, lo_p=1, hi_p=99):
    lo, hi = np.percentile(values, lo_p), np.percentile(values, hi_p)
    hi = max(hi, lo + 1e-6)
    c = np.clip((values - lo) / (hi - lo), 0, 1)
    rgb = (255 * np.stack([c, np.zeros_like(c), 1 - c], axis=1)).astype(np.uint8)
    return rgb

def from_pe_dump(pe_dump_npz, scene_idx=0, cap_views=120000):
    d = np.load(pe_dump_npz, allow_pickle=True)
    coords = d["coords"]      # [N,4] (b,x,y,z)
    mask = d["mask"].astype(bool)  # [N]
    vc = d["view_counts"]     # [N]
    xyz_world = coords[:,1:4].astype(np.float32)

    # 1) 世界坐标 + 视图数着色
    xyz_vis = xyz_world[mask]
    vc_vis = vc[mask].astype(np.float32)
    rgb_vc = percentile_color(vc_vis)
    out1 = Path(pe_dump_npz).with_suffix("").parent / f"world_points_viewcount_scene{scene_idx}.ply"
    write_ply_xyz_rgb(str(out1), xyz_vis, rgb_vc)
    print("Saved:", out1, "N:", xyz_vis.shape[0])

    # 2) 相机系 per-view 坐标（来自 fourier_first3）
    if "fourier_first3" in d.files and d["fourier_first3"] is not None:
        cam_xyz = d["fourier_first3"]
        cam_xyz = cam_xyz[:min(cap_views, cam_xyz.shape[0])].astype(np.float32)
        out2 = Path(pe_dump_npz).with_suffix("").parent / f"cam_xyz_points_scene{scene_idx}.ply"
        write_ply_xyz_rgb(str(out2), cam_xyz, None)
        print("Saved:", out2, "N:", cam_xyz.shape[0])
    else:
        print("fourier_first3 not found in dump; run eval with PE dump enabled to get camera-frame xyz.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dump", type=str, default=None, help="path to pe_dump_sceneX.npz")
    ap.add_argument("--processed_pt", type=str, default=None, help="path to processed feature .pt")
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument("--cap_views", type=int, default=120000)
    args = ap.parse_args()
    if args.pe_dump:
        from_pe_dump(args.pe_dump, args.scene_idx, args.cap_views)
    if args.processed_pt:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("torch is required to read processed .pt files") from e
        data = torch.load(args.processed_pt, map_location='cpu')
        assert 'fourier_pe_packed' in data and 'view_counts' in data, \
            "processed .pt missing fourier_pe_packed/view_counts"
        pe = data['fourier_pe_packed']
        arr = pe if isinstance(pe, np.ndarray) else pe.detach().cpu().numpy()
        cam_xyz = arr[:, :3].astype(np.float32)
        cam_xyz = cam_xyz[:min(args.cap_views, cam_xyz.shape[0])]
        out = Path(args.processed_pt).with_suffix("").parent / (Path(args.processed_pt).stem + "_cam_xyz.ply")
        write_ply_xyz_rgb(str(out), cam_xyz, None)
        vc = data['view_counts']
        vc = vc if isinstance(vc, np.ndarray) else vc.detach().cpu().numpy()
        print("Saved:", out, "cam_xyz:", cam_xyz.shape, "sum(view_counts)=", int(vc.sum()))
    if not args.pe_dump and not args.processed_pt:
        print("Please provide --pe_dump or --processed_pt")

if __name__ == "__main__":
    main()