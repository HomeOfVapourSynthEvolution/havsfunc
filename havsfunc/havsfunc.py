from __future__ import annotations

from typing import Optional, Sequence, Union

from vsrgtools import BlurMatrix
from vstools import (
    check_variable,
    core,
    get_depth,
    get_peak_value,
    get_video_format,
    get_y,
    join,
    scale_delta,
    vs,
)

__all__ = ["fast_line_darken_mod", "Overlay"]


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


def Overlay(
    base: vs.VideoNode,
    overlay: vs.VideoNode,
    x: int = 0,
    y: int = 0,
    mask: Optional[vs.VideoNode] = None,
    opacity: float = 1.0,
    mode: str = 'normal',
    planes: Optional[Union[int, Sequence[int]]] = None,
    mask_first_plane: bool = True,
) -> vs.VideoNode:
    '''
    Puts clip overlay on top of clip base using different blend modes, and with optional x,y positioning, masking and opacity.

    Parameters:
        base: This clip will be the base, determining the size and all other video properties of the result.

        overlay: This is the image that will be placed on top of the base clip.

        x, y: Define the placement of the overlay image on the base clip, in pixels. Can be positive or negative.

        mask: Optional transparency mask. Must be the same size as overlay. Where mask is darker, overlay will be more transparent.

        opacity: Set overlay transparency. The value is from 0.0 to 1.0, where 0.0 is transparent and 1.0 is fully opaque.
            This value is multiplied by mask luminance to form the final opacity.

        mode: Defines how your overlay should be blended with your base image. Available blend modes are:
            addition, average, burn, darken, difference, divide, dodge, exclusion, extremity, freeze, glow, grainextract, grainmerge, hardlight, hardmix, heat,
            lighten, linearlight, multiply, negation, normal, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        mask_first_plane: If true, only the mask's first plane will be used for transparency.
    '''
    if not (isinstance(base, vs.VideoNode) and isinstance(overlay, vs.VideoNode)):
        raise vs.Error('Overlay: this is not a clip')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('Overlay: mask is not a clip')

        if mask.width != overlay.width or mask.height != overlay.height or get_depth(mask) != get_depth(overlay):
            raise vs.Error('Overlay: mask must have the same dimensions and bit depth as overlay')

    if base.format.sample_type == vs.INTEGER:
        bits = get_depth(base)
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
        factor = 1 << bits
    else:
        neutral = 0.5
        peak = factor = 1.0

    plane_range = range(base.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    if base.format.subsampling_w > 0 or base.format.subsampling_h > 0:
        base_orig = base
        base = base.resize.Point(format=base.format.replace(subsampling_w=0, subsampling_h=0))
    else:
        base_orig = None

    if overlay.format.id != base.format.id:
        overlay = overlay.resize.Point(format=base.format)

    if mask is None:
        mask = overlay.std.BlankClip(format=overlay.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
    elif mask.format.id != overlay.format.id and mask.format.color_family != vs.GRAY:
        mask = mask.resize.Point(format=overlay.format, range_s='full')

    opacity = min(max(opacity, 0.0), 1.0)
    mode = mode.lower()

    # Calculate padding sizes
    l, r = x, base.width - overlay.width - x
    t, b = y, base.height - overlay.height - y

    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)

    # Crop and padding
    overlay = overlay.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    overlay = overlay.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb)
    mask = mask.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    mask = mask.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb, color=[0] * mask.format.num_planes)

    if opacity < 1:
        mask = mask.std.Expr(expr=f'x {opacity} *')

    if mode == 'normal':
        pass
    elif mode == 'addition':
        expr = 'x y +'
    elif mode == 'average':
        expr = 'x y + 2 /'
    elif mode == 'burn':
        expr = f'x 0 <= x {peak} {peak} y - {factor} * x / - ?'
    elif mode == 'darken':
        expr = 'x y min'
    elif mode == 'difference':
        expr = 'x y - abs'
    elif mode == 'divide':
        expr = f'y 0 <= {peak} {peak} x * y / ?'
    elif mode == 'dodge':
        expr = f'x {peak} >= x y {factor} * {peak} x - / ?'
    elif mode == 'exclusion':
        expr = f'x y + 2 x * y * {peak} / -'
    elif mode == 'extremity':
        expr = f'{peak} x - y - abs'
    elif mode == 'freeze':
        expr = f'y 0 <= 0 {peak} {peak} x - dup * y / {peak} min - ?'
    elif mode == 'glow':
        expr = f'x {peak} >= x y y * {peak} x - / ?'
    elif mode == 'grainextract':
        expr = f'x y - {neutral} +'
    elif mode == 'grainmerge':
        expr = f'x y + {neutral} -'
    elif mode == 'hardlight':
        expr = f'y {neutral} < 2 y x * {peak} / * {peak} 2 {peak} y - {peak} x - * {peak} / * - ?'
    elif mode == 'hardmix':
        expr = f'x {peak} y - < 0 {peak} ?'
    elif mode == 'heat':
        expr = f'x 0 <= 0 {peak} {peak} y - dup * x / {peak} min - ?'
    elif mode == 'lighten':
        expr = 'x y max'
    elif mode == 'linearlight':
        expr = f'y {neutral} < y 2 x * + {peak} - y 2 x {neutral} - * + ?'
    elif mode == 'multiply':
        expr = f'x y * {peak} /'
    elif mode == 'negation':
        expr = f'{peak} {peak} x - y - abs -'
    elif mode == 'overlay':
        expr = f'x {neutral} < 2 x y * {peak} / * {peak} 2 {peak} x - {peak} y - * {peak} / * - ?'
    elif mode == 'phoenix':
        expr = f'x y min x y max - {peak} +'
    elif mode == 'pinlight':
        expr = f'y {neutral} < x 2 y * min x 2 y {neutral} - * max ?'
    elif mode == 'reflect':
        expr = f'y {peak} >= y x x * {peak} y - / ?'
    elif mode == 'screen':
        expr = f'{peak} {peak} x - {peak} y - * {peak} / -'
    elif mode == 'softlight':
        expr = f'x {neutral} > y {peak} y - x {neutral} - * {neutral} / 0.5 y {neutral} - abs {peak} / - * + y y {neutral} x - {neutral} / * 0.5 y {neutral} - abs {peak} / - * - ?'
    elif mode == 'subtract':
        expr = 'x y -'
    elif mode == 'vividlight':
        expr = f'x {neutral} < x 0 <= 2 x * {peak} {peak} y - {factor} * 2 x * / - ? 2 x {neutral} - * {peak} >= 2 x {neutral} - * y {factor} * {peak} 2 x {neutral} - * - / ? ?'
    else:
        raise vs.Error('Overlay: invalid mode specified')

    if mode != 'normal':
        overlay = core.std.Expr([overlay, base], expr=[expr if i in planes else '' for i in plane_range])

    # Return padded clip
    last = core.std.MaskedMerge(base, overlay, mask, planes=planes, first_plane=mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last
