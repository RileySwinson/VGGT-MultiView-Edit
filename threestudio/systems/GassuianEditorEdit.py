from dataclasses import dataclass, field

from tqdm import tqdm

import torch
import threestudio
import os
import numpy as np
from PIL import Image

from threestudio.utils.clip_metrics import ClipSimilarity

from threestudio.systems.GassuianEditor import GaussianEditor
from threestudio.systems.propagation import Propagation

@threestudio.register("gsedit-system-edit")
class GaussianEditor_Edit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

        # edit propagation. everything off = stock GaussianEditor
        paste: bool = False  # paste the edited neighbours into the latent of every refreshed target
        paste_k: int = 2  # how many sources to paste from
        paste_release: float = 0.6  # fraction of the denoising steps the paste is held for
        paste_fullres: bool = False  # warp the 512 target directly instead of at the priors' resolution
        paste_color_match: bool = False  # scale each source to the average edit colour of all sources
        paste_fill: bool = False  # fill the unpasted part of the composite with the view's own render
        paste_prepass: bool = False  # edit every view once before training, in overlap order
        paste_prepass_t: int = 900
        paste_prepass_ig: float = 1.5  # image guidance for the pre-pass
        paste_prepass_gs: float = 7.5  # text guidance for the pre-pass
        paste_geometry: str = "vggt_targets"  # scene: rendered depth throughout. vggt_targets: VGGT on the targets once every view has one
        paste_geometry_refresh: int = 0  # re-run VGGT every this many steps, 0 = once
        paste_vggt_python: str = ""  # python of the VGGT env, or set VGGT_PYTHON
        paste_offload: bool = True  # move InstructPix2Pix and LangSAM to the CPU while VGGT runs
        freeze_sh_rest: bool = False  # keep the view-dependent colour (SH degree > 0) fixed
        refresh_t_max: int = -1  # once a view has a target, refreshes draw t at most this high. -1 off
        clamp_to_mask: int = -1  # keep the edit inside the segmentation mask dilated by this many pixels. -1 off

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            self.update_mask()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )

        # one seed per refresh, so the stock and paste arms draw the same t and noise
        self.edit_seed = int(torch.randint(0, 2 ** 31, (1,)))
        mask_dir = os.path.join(self.cache_dir, f"{self.cfg.seg_prompt}_mask_{self.view_num}_view") if self.cfg.seg_prompt else None
        self.view_masks = {}
        if self.cfg.clamp_to_mask >= 0 and mask_dir:
            for id in self.view_list:
                self.view_masks[id] = torch.from_numpy(np.asarray(Image.open(f"{mask_dir}/{id:04d}.png").convert("L"), np.float32) / 255)
        self.propagation = None
        if self.cfg.paste:
            if self.cfg.paste_geometry not in ("scene", "vggt_targets"):
                raise SystemExit("paste_geometry must be scene or vggt_targets")
            cams = {id: self.trainer.datamodule.train_dataset.scene.cameras[id] for id in self.view_list}
            self.propagation = Propagation(cams, self.cfg, self.gaussian, self.pipe, self.background_tensor,
                                           os.path.join(self.get_save_dir(), "paste"), mask_dir)
            if self.cfg.paste_prepass:
                prompt_utils = self.prompt_processor()
                for n, id in enumerate(self.propagation.prepassOrder()):
                    torch.manual_seed(self.edit_seed + id)
                    extra = self.propagation.targets(id, self.edit_frames, self.origin_frames, self.origin_frames[id][0].cpu())
                    result = self.guidance(self.origin_frames[id], self.origin_frames[id], prompt_utils,
                                           t=self.cfg.paste_prepass_t, condition_scale=self.cfg.paste_prepass_ig,
                                           guidance_scale=self.cfg.paste_prepass_gs, **extra)
                    self.edit_frames[id] = self.clamp(id, result["edit_images"].detach().clone())
                    self.save_rgb_image(f"edits/pre{n:04d}_v{id:04d}.png", self.edit_frames[id][0], data_format="HWC")
                    self.log_t(-1, id, result)
            self.geometry_step = None
            self.refresh_geometry(0)

    # once every view has a target, run VGGT on them and paste with that geometry from then on.
    # the RNG state is put back afterwards so the refreshes draw the same t and noise as without it
    def refresh_geometry(self, step):
        if not self.propagation or self.cfg.paste_geometry != "vggt_targets":
            return
        if any(id not in self.edit_frames for id in self.view_list):
            return
        every = self.cfg.paste_geometry_refresh
        if self.geometry_step is not None and (every <= 0 or step - self.geometry_step < every):
            return
        rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        modules = self.offloadable() if self.cfg.paste_offload else []
        for m in modules:
            m.to("cpu")
        torch.cuda.empty_cache()
        try:
            self.propagation.geometryFromTargets(self.edit_frames, step)
        finally:
            for m in modules:
                m.to("cuda")
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
        self.geometry_step = step

    # what can leave the GPU while VGGT runs
    def offloadable(self):
        found = []
        for path in ("guidance.pipe", "text_segmentor.model.sam.model", "text_segmentor.model.groundingdino"):
            obj = self
            for name in path.split("."):
                obj = getattr(obj, name, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "to"):
                found.append(obj)
        return found

    # the noise level each refresh drew, one "step view t" line per refresh
    def log_t(self, step, view, result):
        with open(self.get_save_path("t_sequence.txt"), "a") as f:
            f.write(f"{step}\t{view}\t{result['t']}\n")

    # the edit outside the dilated, softened segmentation mask is replaced by the original render
    def clamp(self, id, frame):
        if id not in self.view_masks:
            return frame
        m = self.view_masks[id].to(frame)[None, None]
        if self.cfg.clamp_to_mask > 0:
            m = torch.nn.functional.max_pool2d(m, 2 * self.cfg.clamp_to_mask + 1, 1, self.cfg.clamp_to_mask)
        m = torch.nn.functional.avg_pool2d(m, 9, 1, 4)
        m = torch.nn.functional.interpolate(m, frame.shape[1:3], mode="bilinear", align_corners=False)[0, 0, ..., None]
        return frame * m + self.origin_frames[id] * (1 - m)

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)
        if self.cfg.freeze_sh_rest:
            for group in self.gaussian.optimizer.param_groups:
                if group["name"] == "f_rest":
                    group["lr"] = 0
        self.refresh_geometry(self.global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]

        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                ):
                    torch.manual_seed(self.edit_seed + 1_000_003 * self.global_step + cur_index)
                    extra = self.propagation.targets(cur_index, self.edit_frames, self.origin_frames,
                                                     images[img_index].detach().cpu()) if self.propagation else {}
                    if cur_index in self.edit_frames and self.cfg.refresh_t_max > 0:
                        extra["t"] = int(torch.randint(self.guidance.min_step, self.cfg.refresh_t_max + 1, (1,)))
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                        **extra,
                    )

                    self.edit_frames[cur_index] = self.clamp(cur_index, result["edit_images"].detach().clone())
                    self.save_rgb_image(f"edits/step{self.global_step:04d}_v{cur_index:04d}.png",
                                        self.edit_frames[cur_index][0], data_format="HWC")
                    self.log_t(self.global_step, cur_index, result)
                    # print("edited image index", cur_index)

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target, total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))

