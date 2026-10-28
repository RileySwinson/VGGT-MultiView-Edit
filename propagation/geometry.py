import torch
import torch.nn.functional as functional

SIZE = 512

# get pixel p from view i in uv space of view j, and its depth z seen from j
def project(p, i, j, depthScale=1.0):
    points = torch.from_numpy(p["points"][i]).reshape(-1, 3)
    if depthScale != 1.0:
        Ri, Ti = torch.from_numpy(p["extrinsic"][i][:, :3]), torch.from_numpy(p["extrinsic"][i][:, 3])
        centre = -Ri.T @ Ti
        points = centre + depthScale * (points - centre)
    R = torch.from_numpy(p["extrinsic"][j][:, :3])
    T = torch.from_numpy(p["extrinsic"][j][:, 3])
    K = torch.from_numpy(p["intrinsic"][j])
    H, W = p["depth"].shape[1:]

    inCamera = points @ R.T + T
    z = inCamera[:, 2]
    pixels = (inCamera @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]  # clamp to avoid divide by 0
    grid = torch.stack([pixels[:, 0] / (W - 1) * 2 - 1, pixels[:, 1] / (H - 1) * 2 - 1], -1)
    return grid.view(1, H, W, 2), z.view(H, W)

# returns the image from perspective of another camera "grid"
def warp(image, grid):
    return functional.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]

# checks if the pixels of view i are visible to camera j
def visible(
    p, i, j, grid, z,
    confidenceThreshold=2.0,
    depthTolerance=0.05,
):
    confidence = torch.from_numpy(p["conf"][i])
    depthJ = torch.from_numpy(p["depth"][j])
    depthSeenByJ = warp(depthJ[None], grid)[0]
    confidenceSeenByJ = warp(torch.from_numpy(p["conf"][j])[None], grid)[0]

    isConfident = confidence > confidenceThreshold
    isConfidentInJ = confidenceSeenByJ > confidenceThreshold
    isInFront = z > 1e-6
    isInsideFrame = grid.abs().amax(-1)[0] < 1
    isNotOccluded = (depthSeenByJ - z).abs() < depthTolerance * z

    # the depth error from the occlusion test must not move the pixel in j by more than one pixel in the latent space
    shiftedGrid, _ = project(p, i, j, 1 + depthTolerance)
    latentCell = 2 / (SIZE // 8)
    isStable = (shiftedGrid - grid).norm(dim=-1)[0] <= latentCell

    return isConfident & isConfidentInJ & isInFront & isInsideFrame & isNotOccluded & isStable

# how much the edit changed the colour of view j: mean edited colour over mean original colour, where it changed
def editGain(edited, original, changeThreshold):
    changed = (edited - original).abs().mean(-1) > changeThreshold
    if changed.sum() < 100:
        return None
    return edited[changed].mean(0) / original[changed].mean(0).clamp(min=1e-3)

# combine the edited views into one view by checking edit, first source to write a pixel keeps it
def composite(
    p, v, edited, originals, order,
    maxSources=2,
    minimumOverlap=0.05,
    changeThreshold=0.15,
    fullRes=False,
    colorMatch=False,
    fill=None,
):
    H, W = p["depth"].shape[1:]
    R = (SIZE, SIZE) if fullRes else (H, W)
    image = torch.zeros(3, *R)
    filled = torch.zeros(R, dtype=torch.bool)
    enlarge = lambda m: functional.interpolate(m[None, None].float(), R, mode="nearest")[0, 0] > 0.5 if fullRes else m

    if colorMatch:
        gains = [g for g in (editGain(edited[j], originals[j], changeThreshold) for j in edited) if g is not None]
        reference = torch.stack(gains).mean(0) if gains else None

    for j in order[:maxSources]:
        grid, z = project(p, v, j)
        mask = enlarge(visible(p, v, j, grid, z))
        if mask.float().mean() < minimumOverlap:
            continue
        if fullRes:
            grid = functional.interpolate(grid.permute(0, 3, 1, 2), R, mode="bilinear", align_corners=True).permute(0, 2, 3, 1)

        source = edited[j].permute(2, 0, 1)
        if not fullRes:
            source = functional.interpolate(source[None], (H, W), mode="bilinear", align_corners=False, antialias=True)[0]
        if colorMatch and reference is not None:
            gain = editGain(edited[j], originals[j], changeThreshold)
            if gain is not None:
                source = (source * (reference / gain)[:, None, None]).clamp(0, 1)
        warped = warp(source, grid)

        change = (edited[j] - originals[j]).abs().mean(-1)  # amount of change
        change = functional.avg_pool2d(change[None, None], 15, 1, 7)  # 15px blur
        change = functional.interpolate(change, R, mode="bilinear", align_corners=False)[0]
        mask = mask & (warp(change, grid)[0] > changeThreshold)

        take = mask & ~filled
        image[:, take] = warped[:, take]
        filled = filled | take

    if fill is not None:
        fillImage = fill.permute(2, 0, 1)
        if not fullRes:
            fillImage = functional.interpolate(fillImage[None], (H, W), mode="bilinear", align_corners=False, antialias=True)[0]
        image[:, ~filled] = fillImage[:, ~filled]
    if not fullRes:
        image = functional.interpolate(image[None], (SIZE, SIZE), mode="bilinear", align_corners=False)[0]
    cover = functional.interpolate(filled[None, None].float(), (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    return image.permute(1, 2, 0).contiguous(), cover
