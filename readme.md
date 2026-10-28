# Edit propagation fork of GaussianEditor

This project is a modified fork of [GaussianEditor](https://github.com/buaacyw/GaussianEditor) licensed under
[S-Lab License 1.0](LICENSE.txt) (non-commercial). It has only been tested on Linux (Ubuntu 22.04, CUDA 11.8),
but should work on other operating systems.

## Abstract

Each target image InstructPix2Pix produces for a camera starts from a composite of its already-edited
neighbouring views, warped into that camera by depth and camera geometry and held in the latent during the early
denoising steps. Before training, every view is edited once in an order chosen by overlap. VGGT is then run on
those first-pass edits, and its depth and confidence drive every paste during training. Three changes keep the
agreement: refreshes draw a capped noise level, the edit is clamped to the dilated segmentation mask, and the
view-dependent colour coefficients are frozen.

## Summary of changes

Added or edited the following files:

```
propagation/
   __init__.py
   geometry.py
   priors.py
threestudio/
   systems/
      propagation.py
      GassuianEditorEdit.py
   models/guidance/instructpix2pix_guidance.py
```

## Setup

This setup requires two Python environments: one for the editor, and one for VGGT. This is because VGGT uses a
newer version of PyTorch.

First, install GaussianEditor as described in its installation process, which requires a conda environment with
CUDA 11.8. This project only adds `bash install.sh`, which builds the rasteriser and simple-knn.

Second, install VGGT in a new environment, as described in its installation process.

The two environments are separate. The editor runs VGGT by calling the VGGT environment's Python as a subprocess,
so you point it there with `system.paste_vggt_python=<vggt env python>` (or the `VGGT_PYTHON` environment
variable) on the run command.

Each scene has the origina images, a COLMAP reconstruction, and a trained 3D Gaussian Splatting representation file. 
These are obtained from:

Instruct-NeRF2NeRF scenes including a `.ply` file and COLMAP output
https://huggingface.co/datasets/Yiwen-ntu/GaussianEditor_Result/tree/main/InstructNeRF2NeRF_PLY_Files

Mip-NeRF 360 scenes, via `bash download.sh` (pretrained `.ply` from
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip and images from
http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

The model weights (InstructPix2Pix, Stable Diffusion and LangSAM for the editor, and VGGT-1B for VGGT) download
automatically on first use.

## Running

The following template is a minimal way to run the method on a scene. Any field in `configs/edit-n2n.yaml` can be
overridden on the command line; see GaussianEditor's
[Command Line](https://github.com/buaacyw/GaussianEditor#command-line) docs for the full set.

```
python launch.py --config configs/edit-n2n.yaml --train --gpu 0 \
  data.source=<colmap scene dir> system.gs_source=<trained .ply> \
  system.prompt_processor.prompt="<instruction>" \
  system.paste=true system.paste_prepass=true \
  system.paste_vggt_python=<vggt env python>
```

Outputs default to `outputs/edit-n2n/<prompt>@<time>/save/`.

This code adds the command-line options below. The descriptions assume familiarity with the process, as described
in the project report.

| option | default | description |
|---|---|---|
| `paste` | false | paste the edited neighbours into the latent of every refreshed target |
| `paste_k` | 2 | number of source views per paste |
| `paste_release` | 0.6 | the fraction of the denoising steps the paste is used for |
| `paste_fill` | false | fill the un-propagated part of the composite image with the view's own render |
| `paste_prepass` | false | edit every view once before training |
| `paste_prepass_t` | 900 | noise level of the first-pass edits |
| `paste_prepass_ig` | 1.5 | image guidance of the first-pass edits |
| `paste_prepass_gs` | 7.5 | text guidance of the first-pass edits |
| `paste_geometry` | vggt_targets | origin of the depth information: `scene` uses the Gaussian renders, `vggt_targets` uses VGGT on the targets |
| `paste_geometry_refresh` | 0 | re-run VGGT every n steps |
| `freeze_sh_rest` | false | freeze the view-dependent colour coefficients |
| `refresh_t_max` | -1 | limit on the noise level of refreshes of a view that already has a target |
| `clamp_to_mask` | -1 | keep the edit inside the segmentation mask dilated by this many pixels |
