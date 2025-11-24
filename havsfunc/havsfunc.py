from __future__ import annotations

from enum import StrEnum, auto, unique

from vsexprtools import norm_expr
from vsrgtools import BlurMatrix
from vstools import (
    Planes,
    check_variable,
    core,
    get_depth,
    get_peak_value,
    get_video_format,
    get_y,
    join,
    normalize_planes,
    scale_delta,
    vs,
)

__all__ = ["fast_line_darken_mod", "overlay", "OverlayMode"]


def fast_line_darken_mod(
    clip: vs.VideoNode,
    strength: int = 48,
    protection: int = 5,
    luma_cap: int = 191,
    threshold: int = 4,
    thinning: int = 0,
) -> vs.VideoNode:
    """
    Args:
        clip: Clip to process.
        strength: Line darkening amount, 0-256. Represents the maximum amount that the luma will be reduced by, weaker
            lines will be reduced by proportionately less.
        protection: Prevents the darkest lines from being darkened. Protection acts as a threshold. Values range from 0
            (no prot) to ~50 (protect everything).
        luma_cap: Value from 0 (black) to 255 (white), used to stop the darkening determination from being 'blinded' by
            bright pixels, and to stop grey lines on white backgrounds being darkened. Any pixels brighter than luma_cap
            are treated as only being as bright as luma_cap. Lowering luma_cap tends to reduce line darkening. 255
            disables capping.
        threshold: Any pixels that were going to be darkened by an amount less than threshold will not be touched.
            Setting this to 0 will disable it, setting it to 4 (default) is recommended, since often a lot of random
            pixels are marked for very slight darkening and a threshold of about 4 should fix them. Note if you set
            threshold too high, some lines will not be darkened.
        thinning: Optional line thinning amount, 0-256. Setting this to 0 will disable it, which gives a big speed
            increase. Note that thinning the lines will inherently darken the remaining pixels in each line a little.
    """
    assert check_variable(clip, fast_line_darken_mod)

    fmt = get_video_format(clip)
    peak = get_peak_value(fmt)

    if fmt.color_family is vs.RGB:
        raise vs.Error("fast_line_darken_mod: RGB format is not supported")

    if fmt.color_family is not vs.GRAY:
        clip_orig = clip
        clip = get_y(clip)
    else:
        clip_orig = None

    Str = strength / 128
    lum = scale_delta(luma_cap, 8, clip)
    thr = scale_delta(threshold, 8, clip)
    thn = thinning / 16

    exin = clip.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = core.std.Expr([clip, exin], f"y {lum} min x {thr} + > x y {lum} min - 0 ? {Str} * x +")
    if thinning == 0:
        last = thick
    else:
        scale_127 = scale_delta(127, 8, clip)
        diff = core.std.Expr([clip, exin], f"y {lum} min x {thr} + > x y {lum} min - 0 ? {scale_127} +")
        linemask = BlurMatrix.MEAN()(diff.std.Minimum().std.Expr(f"x {scale_127} - {thn} * {peak} +"))
        thin = core.std.Expr([clip.std.Maximum(), diff], f"x y {scale_127} - {Str} 1 + * +")
        last = thin.std.MaskedMerge(thick, linemask)

    if clip_orig is not None:
        last = join(last, clip_orig)
    return last


@unique
class OverlayMode(StrEnum):
    ADDITION = auto()
    AVERAGE = auto()
    BLEACH = auto()
    BURN = auto()
    DARKEN = auto()
    DIFFERENCE = auto()
    DIVIDE = auto()
    DODGE = auto()
    EXCLUSION = auto()
    EXTREMITY = auto()
    FREEZE = auto()
    GEOMETRIC = auto()
    GLOW = auto()
    GRAINEXTRACT = auto()
    GRAINMERGE = auto()
    HARDLIGHT = auto()
    HARDMIX = auto()
    HARDOVERLAY = auto()
    HARMONIC = auto()
    HEAT = auto()
    INTERPOLATE = auto()
    LIGHTEN = auto()
    LINEARLIGHT = auto()
    MULTIPLY = auto()
    NEGATION = auto()
    NORMAL = auto()
    OVERLAY = auto()
    PHOENIX = auto()
    PINLIGHT = auto()
    REFLECT = auto()
    SCREEN = auto()
    SOFTDIFFERENCE = auto()
    SOFTLIGHT = auto()
    STAIN = auto()
    SUBTRACT = auto()
    VIVIDLIGHT = auto()


def overlay(
    base_clip: vs.VideoNode,
    overlay_clip: vs.VideoNode,
    x: int = 0,
    y: int = 0,
    mask: vs.VideoNode | None = None,
    opacity: float = 1.0,
    mode: OverlayMode = OverlayMode.NORMAL,
    planes: Planes = None,
    mask_first_plane: bool = True,
) -> vs.VideoNode:
    """
    Puts overlay clip on top of base clip using different blend modes, and with optional positioning, masking and
    opacity.

    Args:
        base_clip: This clip will be the base, determining the size and all other video properties of the result.
        overlay_clip: This is the image that will be placed on top of the base clip.
        x: Defines the x position of the overlay image on the base clip, in pixels. Can be positive or negative.
        y: Defines the y position of the overlay image on the base clip, in pixels. Can be positive or negative.
        mask: Optional transparency mask. Must be the same size as overlay. Where mask is darker, overlay will be more
            transparent.
        opacity: Sets overlay transparency. The value is from 0.0 to 1.0, where 0.0 is transparent and 1.0 is fully
            opaque. This value is multiplied by mask luminance to form the final opacity.
        mode: Defines how your overlay should be blended with your base image.
        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
        mask_first_plane: If True, only the mask's first plane will be used for transparency.
    """
    assert check_variable(base_clip, overlay)
    assert check_variable(overlay_clip, overlay)

    if mask is not None:
        assert check_variable(mask, overlay)

        if (
            mask.width != overlay_clip.width
            or mask.height != overlay_clip.height
            or get_depth(mask) != get_depth(overlay_clip)
        ):
            raise vs.Error("overlay: mask must have the same dimensions and bit depth as overlay")

    opacity = min(max(opacity, 0.0), 1.0)
    planes = normalize_planes(base_clip, planes)

    if base_clip.format.subsampling_w > 0 or base_clip.format.subsampling_h > 0:
        base_orig = base_clip
        base_clip = base_clip.resize.Point(format=base_clip.format.replace(subsampling_w=0, subsampling_h=0))
    else:
        base_orig = None

    if overlay_clip.format.id != base_clip.format.id:
        overlay_clip = overlay_clip.resize.Point(format=base_clip.format)

    if mask is None:
        mask = overlay_clip.std.BlankClip(
            format=overlay_clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0),
            color=get_peak_value(overlay_clip, range_in=vs.ColorRange.RANGE_FULL),
        )
    elif mask.format.id != overlay_clip.format.id and mask.format.color_family is not vs.GRAY:
        mask = mask.resize.Point(format=overlay_clip.format, range_s="full")

    left = x
    right = base_clip.width - overlay_clip.width - x
    top = y
    bottom = base_clip.height - overlay_clip.height - y

    crop_left = min(left, 0) * -1
    crop_right = min(right, 0) * -1
    crop_top = min(top, 0) * -1
    crop_bottom = min(bottom, 0) * -1

    pad_left = max(left, 0)
    pad_right = max(right, 0)
    pad_top = max(top, 0)
    pad_bottom = max(bottom, 0)

    overlay_clip = overlay_clip.std.Crop(crop_left, crop_right, crop_top, crop_bottom)
    overlay_clip = overlay_clip.std.AddBorders(pad_left, pad_right, pad_top, pad_bottom)
    mask = mask.std.Crop(crop_left, crop_right, crop_top, crop_bottom)
    mask = mask.std.AddBorders(pad_left, pad_right, pad_top, pad_bottom, color=[0] * mask.format.num_planes)

    if opacity < 1:
        mask = norm_expr(mask, f"x {opacity} *", 0 if mask_first_plane else planes)

    match mode:
        case OverlayMode.NORMAL:
            pass
        case OverlayMode.ADDITION:
            expr = "x y +"
        case OverlayMode.AVERAGE:
            expr = "x y + 2 /"
        case OverlayMode.BLEACH:
            expr = "plane_max y - plane_max x - + plane_max -"
        case OverlayMode.BURN:
            expr = "x 0 <= x plane_max plane_max y - range_size * x / - ?"
        case OverlayMode.DARKEN:
            expr = "x y min"
        case OverlayMode.DIFFERENCE:
            expr = "x y - abs"
        case OverlayMode.DIVIDE:
            expr = "y 0 <= plane_max plane_max x * y / ?"
        case OverlayMode.DODGE:
            expr = "x plane_max >= x y range_size * plane_max x - / ?"
        case OverlayMode.EXCLUSION:
            expr = "x y + 2 x * y * plane_max / -"
        case OverlayMode.EXTREMITY:
            expr = "plane_max x - y - abs"
        case OverlayMode.FREEZE:
            expr = "y 0 <= 0 plane_max plane_max x - dup * y / plane_max min - ?"
        case OverlayMode.GEOMETRIC:
            expr = "x 0 max y 0 max * sqrt"
        case OverlayMode.GLOW:
            expr = "x plane_max >= x y y * plane_max x - / ?"
        case OverlayMode.GRAINEXTRACT:
            expr = "x y - neutral +"
        case OverlayMode.GRAINMERGE:
            expr = "x y + neutral -"
        case OverlayMode.HARDLIGHT:
            expr = "y neutral < 2 y x * plane_max / * plane_max 2 plane_max y - plane_max x - * plane_max / * - ?"
        case OverlayMode.HARDMIX:
            expr = "x plane_max y - < 0 plane_max ?"
        case OverlayMode.HARDOVERLAY:
            expr = "x plane_max >= plane_max plane_max y * 2 plane_max * 2 x * - / x neutral > * 2 x * y * plane_max / x neutral <= * + ?"
        case OverlayMode.HARMONIC:
            expr = "x 0 <= y 0 <= and 0 2 x * y * x y + / ?"
        case OverlayMode.HEAT:
            expr = "x 0 <= 0 plane_max plane_max y - dup * x / plane_max min - ?"
        case OverlayMode.INTERPOLATE:
            expr = "plane_max 2 x pi * plane_max / cos - y pi * plane_max / cos - * 0.25 *"
        case OverlayMode.LIGHTEN:
            expr = "x y max"
        case OverlayMode.LINEARLIGHT:
            expr = "y neutral < y 2 x * + plane_max - y 2 x neutral - * + ?"
        case OverlayMode.MULTIPLY:
            expr = "x y * plane_max /"
        case OverlayMode.NEGATION:
            expr = "plane_max plane_max x - y - abs -"
        case OverlayMode.OVERLAY:
            expr = "x neutral < 2 x y * plane_max / * plane_max 2 plane_max x - plane_max y - * plane_max / * - ?"
        case OverlayMode.PHOENIX:
            expr = "x y min x y max - plane_max +"
        case OverlayMode.PINLIGHT:
            expr = "y neutral < x 2 y * min x 2 y neutral - * max ?"
        case OverlayMode.REFLECT:
            expr = "y plane_max >= y x x * plane_max y - / ?"
        case OverlayMode.SCREEN:
            expr = "plane_max plane_max x - plane_max y - * plane_max / -"
        case OverlayMode.SOFTDIFFERENCE:
            expr = "x y > y plane_max >= 0 x y - plane_max * plane_max y - / ? y 0 <= 0 y x - plane_max * y / ? ?"
        case OverlayMode.SOFTLIGHT:
            expr = "x x * plane_max / 2 y x plane_max x - * plane_max / * plane_max / * +"
        case OverlayMode.STAIN:
            expr = "2 plane_max * x - y -"
        case OverlayMode.SUBTRACT:
            expr = "x y -"
        case OverlayMode.VIVIDLIGHT:
            expr = "x neutral < x 0 <= 2 x * plane_max plane_max y - range_size * 2 x * / - ? 2 x neutral - * plane_max >= 2 x neutral - * y range_size * plane_max 2 x neutral - * - / ? ?"
        case _:
            raise vs.Error("overlay: invalid mode specified")

    if mode != OverlayMode.NORMAL:
        overlay_clip = norm_expr([overlay_clip, base_clip], expr, planes)

    last = base_clip.std.MaskedMerge(overlay_clip, mask, planes, mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last
