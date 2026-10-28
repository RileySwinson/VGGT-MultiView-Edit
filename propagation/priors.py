import argparse, glob, os
import numpy as np
import torch
from PIL import Image

# run vggt on a folder of images and save the priors npz. run in the vggt env
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--name", default="priors.npz")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    # select the photos and order them
    paths = sorted(p for p in glob.glob(os.path.join(args.images, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = paths[args.start::args.stride]
    if args.n:
        paths = paths[:args.n]
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images, found {len(paths)}")
    w0, h0 = Image.open(paths[0]).size
    if h0 > w0:
        raise SystemExit("portrait images not supported")
    os.makedirs(args.out, exist_ok=True)

    # load vggt
    dataType = torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(torch.device("cuda"), dtype=dataType).eval()

    # preprocess (assumes all images are the same size). vggt makes them 518 wide
    images = load_and_preprocess_images(paths).to("cuda")

    with torch.no_grad():
        pred = model(images.to(dataType))

    # compute camera pose
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()

    # compute depth and confidence
    depth4 = pred["depth"][0].float().cpu().numpy()  # view, u, v, 0
    confidence = pred["depth_conf"][0].float().cpu().numpy()
    uvToWorld = unproject_depth_map_to_point_map(depth4, extrinsic, intrinsic).astype(np.float32)  # view, u, v, xyz
    depth = depth4[..., 0]

    # export
    out = os.path.join(args.out, args.name)
    np.savez_compressed(
        out,
        images=images.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.float16),
        points=uvToWorld, depth=depth.astype(np.float32), conf=confidence.astype(np.float32),
        extrinsic=extrinsic, intrinsic=intrinsic, paths=np.array([os.path.abspath(p) for p in paths])
    )


if __name__ == "__main__":
    main()
