import os, subprocess
import numpy as np
import torch
from PIL import Image

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.utils.graphics_utils import fov2focal
from propagation.geometry import SIZE, composite, project, visible

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# priors from the scene itself: the cameras and the depth the gaussians render
def scene_priors(cams, gaussian, pipe, background):
    S, H, W = len(cams), cams[0].image_height, cams[0].image_width
    p = {k: np.zeros(shape, np.float32) for k, shape in
         dict(points=(S, H, W, 3), depth=(S, H, W), conf=(S, H, W), extrinsic=(S, 3, 4), intrinsic=(S, 3, 3)).items()}
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ones = torch.ones_like(gaussian.get_xyz)
    for i, cam in enumerate(cams):
        with torch.no_grad():
            z = render(cam, gaussian, pipe, background)["depth_3dgs"][0].cpu()
            alpha = render(cam, gaussian, pipe, background, override_color=ones)["render"][0].cpu()
        z = z / alpha.clamp(min=1e-3)  # the rasteriser sums z*alpha, so divide the alpha back out
        E = cam.world_view_transform.T[:3].cpu()
        fx, fy = fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H)
        inCamera = torch.stack([(u - (W - 1) / 2) / fx * z, (v - (H - 1) / 2) / fy * z, z], -1)
        p["points"][i] = (inCamera - E[:, 3]) @ E[:, :3]
        p["depth"][i], p["conf"][i] = z, 10.0 * (alpha > 0.5)
        p["extrinsic"][i], p["intrinsic"][i] = E, [[fx, 0, (W - 1) / 2], [0, fy, (H - 1) / 2], [0, 0, 1]]
    return p


# a priors.npz from propagation/priors.py, its rows must be this run's camera ids in order
def load_priors(path, ids):
    p = dict(np.load(path, allow_pickle=True))
    names = [os.path.basename(str(x)) for x in p["paths"]]
    if names != [f"{id:04d}.png" for id in ids]:
        raise SystemExit(f"{path} was built from {names[:3]}..., this run edits {ids[:3]}...")
    return p


# vggt on the images in a folder, as a subprocess in the vggt env. returns the npz path
def run_vggt(images, out, name, n, python=""):
    python = python or os.environ.get("VGGT_PYTHON", "")
    if not os.path.exists(python):
        raise SystemExit("set paste_vggt_python or VGGT_PYTHON to the python of the vggt env")
    images, out = os.path.abspath(images), os.path.abspath(out)
    keep = ("HOME", "USER", "LANG", "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "XDG_CACHE_HOME", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}  # the editor's conda toolchain must not leak in
    env["PATH"] = f"{os.path.dirname(python)}:/usr/local/bin:/usr/bin:/bin"
    env["VIRTUAL_ENV"] = os.path.dirname(os.path.dirname(python))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [python, "-m", "propagation.priors", "--images", images, "--out", out, "--name", name, "--n", str(n), "--stride", "1"]
    print("[propagation] " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
        raise SystemExit("vggt failed")
    return os.path.join(out, name)


class Propagation:
    def __init__(
        self,
        cams,
        cfg,
        gaussian,
        pipe,
        background,
        save_dir,
        mask_dir=None,
    ):
        self.ids = sorted(cams)  # camera id to row of the priors
        self.row = {id: i for i, id in enumerate(self.ids)}
        self.cams, self.mask_dir, self.save_dir = cams, mask_dir, save_dir
        self.k, self.release = cfg.paste_k, cfg.paste_release
        self.fullRes, self.colorMatch, self.fill = cfg.paste_fullres, cfg.paste_color_match, cfg.paste_fill
        self.vggtPython = cfg.paste_vggt_python
        self.setPriors(scene_priors([cams[id] for id in self.ids], gaussian, pipe, background), "scene")
        self.count = 0

    # make p the geometry of every warp, visibility test and source ranking from now on
    def setPriors(self, p, name):
        self.p = p
        self.overlap = self.overlapTable(p)
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(f"{self.save_dir}/overlap_{name}.npy", self.overlap)

    # overlap[i, j]: the fraction of view i's object pixels that view j can supply
    def overlapTable(self, p):
        H, W = p["depth"].shape[1:]
        roi = {}
        for i, id in enumerate(self.ids):
            if self.mask_dir:
                m = Image.open(f"{self.mask_dir}/{id:04d}.png").convert("L").resize((W, H), Image.NEAREST)
                roi[i] = torch.from_numpy(np.asarray(m) > 127)
            else:
                roi[i] = torch.ones(H, W, dtype=torch.bool)
        S = len(self.ids)
        table = np.zeros((S, S), np.float32)
        for i in range(S):
            for j in range(S):
                if j != i:
                    grid, z = project(p, i, j)
                    table[i, j] = (visible(p, i, j, grid, z) & roi[i]).sum() / max(roi[i].sum(), 1)
        return table

    # every camera once, always the one that overlaps the edited set most
    def prepassOrder(self):
        done = [self.ids[int(self.overlap.sum(1).argmax())]]
        while len(done) < len(self.ids):
            rest = [id for id in self.ids if id not in done]
            done.append(max(rest, key=lambda id: max(self.overlap[self.row[id], self.row[j]] for j in done)))
        return done

    # the k ids of pool that overlap id most
    def nearest(self, id, pool):
        return sorted(pool, key=lambda j: -self.overlap[self.row[id], self.row[j]])[:self.k]

    # what to paste when view id is refreshed, {} until something has been edited
    def targets(self, id, edit_frames, origin_frames, current):
        pool = self.nearest(id, [j for j in edit_frames if j != id])
        if not pool:
            return {}
        rows = [self.row[j] for j in pool]
        edited = {self.row[j]: edit_frames[j][0].cpu() for j in pool}
        originals = {self.row[j]: origin_frames[j][0].cpu() for j in pool}
        inject, cover = composite(
            self.p, self.row[id], edited, originals, rows,
            maxSources=self.k,
            fullRes=self.fullRes,
            colorMatch=self.colorMatch,
            fill=current if self.fill else None,
        )
        release = torch.full((SIZE, SIZE), self.release)
        strip = torch.cat([inject, cover[..., None].expand(-1, -1, 3)], 1)
        Image.fromarray((strip.numpy() * 255).astype(np.uint8)).save(
            f"{self.save_dir}/{self.count:04d}_v{id:04d}_from{'_'.join(f'{j:04d}' for j in pool)}.png")
        self.count += 1
        return dict(inject=inject, cover=cover, release=release)

    # write every view's current target, run vggt on them, and use that geometry from now on
    def geometryFromTargets(self, edit_frames, step=0):
        tag = "" if step == 0 else f"_step{step:04d}"
        root = os.path.dirname(self.save_dir)  # save/
        images = os.path.join(root, f"targets{tag}")
        os.makedirs(images, exist_ok=True)
        for id in self.ids:
            Image.fromarray((edit_frames[id][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(f"{images}/{id:04d}.png")
        npz = run_vggt(images, root, f"priors_targets{tag}.npz", len(self.ids), self.vggtPython)
        self.setPriors(load_priors(npz, self.ids), f"vggt_targets{tag}")
