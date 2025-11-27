from __future__ import annotations

from enum import StrEnum, auto, unique

from vsexprtools import norm_expr
from vstools import (
    Planes,
    check_variable,
    get_depth,
    get_neutral_value,
    get_peak_value,
    normalize_planes,
    vs,
)

__all__ = ["overlay", "OverlayMode"]


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

    midgray = 0.5 if base_clip.format.sample_type is vs.FLOAT else get_neutral_value(base_clip)

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
            expr = "x y - {midgray} +"
        case OverlayMode.GRAINMERGE:
            expr = "x y + {midgray} -"
        case OverlayMode.HARDLIGHT:
            expr = "y {midgray} < 2 y x * plane_max / * plane_max 2 plane_max y - plane_max x - * plane_max / * - ?"
        case OverlayMode.HARDMIX:
            expr = "x plane_max y - < 0 plane_max ?"
        case OverlayMode.HARDOVERLAY:
            expr = "x plane_max >= plane_max plane_max y * 2 plane_max * 2 x * - / x {midgray} > * 2 x * y * plane_max / x {midgray} <= * + ?"
        case OverlayMode.HARMONIC:
            expr = "x 0 <= y 0 <= and 0 2 x * y * x y + / ?"
        case OverlayMode.HEAT:
            expr = "x 0 <= 0 plane_max plane_max y - dup * x / plane_max min - ?"
        case OverlayMode.INTERPOLATE:
            expr = "plane_max 2 x pi * plane_max / cos - y pi * plane_max / cos - * 0.25 *"
        case OverlayMode.LIGHTEN:
            expr = "x y max"
        case OverlayMode.LINEARLIGHT:
            expr = "y {midgray} < y 2 x * + plane_max - y 2 x {midgray} - * + ?"
        case OverlayMode.MULTIPLY:
            expr = "x y * plane_max /"
        case OverlayMode.NEGATION:
            expr = "plane_max plane_max x - y - abs -"
        case OverlayMode.OVERLAY:
            expr = "x {midgray} < 2 x y * plane_max / * plane_max 2 plane_max x - plane_max y - * plane_max / * - ?"
        case OverlayMode.PHOENIX:
            expr = "x y min x y max - plane_max +"
        case OverlayMode.PINLIGHT:
            expr = "y {midgray} < x 2 y * min x 2 y {midgray} - * max ?"
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
            expr = "x {midgray} < x 0 <= 2 x * plane_max plane_max y - range_size * 2 x * / - ? 2 x {midgray} - * plane_max >= 2 x {midgray} - * y range_size * plane_max 2 x {midgray} - * - / ? ?"
        case _:
            raise vs.Error("overlay: invalid mode specified")

    if mode != OverlayMode.NORMAL:
        overlay_clip = norm_expr([overlay_clip, base_clip], expr, planes, midgray=midgray)

    last = base_clip.std.MaskedMerge(overlay_clip, mask, planes, mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last
