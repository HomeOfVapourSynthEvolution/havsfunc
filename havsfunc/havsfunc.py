from __future__ import annotations

from functools import partial
from typing import Any, Optional, Sequence, Union

from vsexprtools import norm_expr
from vsrgtools import BlurMatrix, repair
from vstools import (
    ColorRange,
    PlanesT,
    check_ref_clip,
    check_variable,
    core,
    cround,
    get_depth,
    get_peak_value,
    get_video_format,
    join,
    normalize_planes,
    plane,
    scale_delta,
    scale_value,
    shift_clip,
    vs,
)

__all__ = [
    "daa",
    "daa3mod",
    "fast_line_darken_mod",
    "lut_decrawl",
    "lut_derainbow",
    "mcdaa3",
    "mt_clamp",
    "Overlay",
    "scdetect",
    "SmoothLevels",
    "Stab",
    "STPresso",
]


def daa(clip: vs.VideoNode, opencl: bool = False, device: int | None = None, **kwargs: Any) -> vs.VideoNode:
    """
    Anti-aliasing with contra-sharpening by Didée.

    It averages two independent interpolations, where each interpolation set works between odd-distanced pixels. This on
    its own provides sufficient amount of blurring. Enough blurring that the script uses a contra-sharpening step to
    counteract the blurring.

    :param clip:    Clip to process.
    :param opencl:  Whether to use OpenCL version of NNEDI3.
    :param device:  Device ordinal of OpenCL device.
    """
    assert check_variable(clip, daa)

    if opencl:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, device=device)
    else:
        nnedi3 = core.znedi3.nnedi3

    blurtype = BlurMatrix.MEAN() if clip.width > 1100 else BlurMatrix.BINOMIAL()

    nn = nnedi3(clip, field=3, **kwargs)
    dbl = nn[::2].std.Merge(nn[1::2])
    dblD = clip.std.MakeDiff(dbl)
    shrpD = dbl.std.MakeDiff(blurtype(dbl))
    DD = repair(shrpD, dblD, 13)
    return dbl.std.MergeDiff(DD)


def daa3mod(clip: vs.VideoNode, opencl: bool = False, device: int | None = None, **kwargs: Any) -> vs.VideoNode:
    """
    :param clip:    Clip to process.
    :param opencl:  Whether to use OpenCL version of NNEDI3.
    :param device:  Device ordinal of OpenCL device.
    """
    assert check_variable(clip, daa3mod)

    c = clip.resize.Spline36(clip.width, clip.height * 3 // 2)
    return daa(c, opencl, device, **kwargs).resize.Spline36(clip.width, clip.height)


def fast_line_darken_mod(
    clip: vs.VideoNode,
    strength: int = 48,
    protection: int = 5,
    luma_cap: int = 191,
    threshold: int = 4,
    thinning: int = 0,
) -> vs.VideoNode:
    """
    :param clip:        Clip to process.
    :param strength:    Line darkening amount, 0-256. Represents the maximum amount that the luma will be reduced by,
                        weaker lines will be reduced by proportionately less.
    :param protection:  Prevents the darkest lines from being darkened. Protection acts as a threshold. Values range
                        from 0 (no prot) to ~50 (protect everything).
    :param luma_cap:    Value from 0 (black) to 255 (white), used to stop the darkening determination from being
                        'blinded' by bright pixels, and to stop grey lines on white backgrounds being darkened. Any
                        pixels brighter than luma_cap are treated as only being as bright as luma_cap. Lowering luma_cap
                        tends to reduce line darkening. 255 disables capping.
    :param threshold:   Any pixels that were going to be darkened by an amount less than threshold will not be touched.
                        Setting this to 0 will disable it, setting it to 4 (default) is recommended, since often a lot
                        of random pixels are marked for very slight darkening and a threshold of about 4 should fix
                        them. Note if you set threshold too high, some lines will not be darkened.
    :param thinning:    Optional line thinning amount, 0-256. Setting this to 0 will disable it, which gives a big speed
                        increase. Note that thinning the lines will inherently darken the remaining pixels in each line
                        a little.
    """
    assert check_variable(clip, fast_line_darken_mod)

    fmt = get_video_format(clip)
    peak = get_peak_value(fmt)

    if fmt.color_family == vs.RGB:
        raise vs.Error("fast_line_darken_mod: RGB format is not supported")

    if fmt.color_family != vs.GRAY:
        clip_orig = clip
        clip = plane(clip, 0)
    else:
        clip_orig = None

    # parameters
    Str = strength / 128
    lum = scale_delta(luma_cap, 8, clip)
    thr = scale_delta(threshold, 8, clip)
    thn = thinning / 16

    # filtering
    exin = clip.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = core.std.Expr([clip, exin], f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +")
    if thinning == 0:
        last = thick
    else:
        scale_127 = scale_delta(127, 8, clip)
        diff = core.std.Expr([clip, exin], f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale_127} +")
        linemask = BlurMatrix.MEAN()(diff.std.Minimum().std.Expr(f"x {scale_127} - {thn} * {peak} +"))
        thin = core.std.Expr([clip.std.Maximum(), diff], f"x y {scale_127} - {Str} 1 + * +")
        last = thin.std.MaskedMerge(thick, linemask)

    if clip_orig is not None:
        last = join(last, clip_orig)
    return last


def lut_decrawl(
    clip: vs.VideoNode,
    ythresh: int = 10,
    cthresh: int = 10,
    maxdiff: int = 50,
    scnchg: int = 25,
    usemaxdiff: bool = True,
    mask: bool = False,
) -> vs.VideoNode:
    """
    :param clip:        Clip to process.
    :param ythresh:     This determines how close the luma values of the pixel in the previous and next frames have to
                        be for the pixel to be hit. Higher values (within reason) should catch more dot crawl, but may
                        introduce unwanted artifacts. Probably shouldn't be set above 20 or so.
    :param cthresh:     This determines how close the chroma values of the pixel in the previous and next frames have to
                        be for the pixel to be hit. Just as with ythresh.
    :param maxdiff:     This is the maximum difference allowed between the luma values of the pixel in the CURRENT frame
                        and in each of its neighbour frames (so, the upper limit to what fluctuations are considered dot
                        crawl). Lower values will reduce artifacts but may cause the filter to miss some dot crawl.
                        Obviously, this should never be lower than ythresh. Meaningless if usemaxdiff=False.
    :param scnchg:      Scene change detection threshold. Any frame with total luma difference between it and the
                        previous/next frame greater than this value will not be processed.
    :param usemaxdiff:  Whether or not to reject luma fluctuations higher than maxdiff. Setting this to False is not
                        recommended, as it may introduce artifacts; but on the other hand, it produces a 30% speed
                        boost. Test on your particular source.
    :param mask:        When set True, the function will return the mask instead of the image. Use to find the best
                        values of ythresh, cthresh, and maxdiff.
    """

    def _scene_change(n: int, f: vs.VideoFrame, clips: list[vs.VideoNode]) -> vs.VideoNode:
        if f.props["_SceneChangePrev"] or f.props["_SceneChangeNext"]:
            return clips[0]
        else:
            return clips[1]

    assert check_variable(clip, lut_decrawl)

    fmt = get_video_format(clip)
    peak = get_peak_value(fmt)

    if fmt.color_family != vs.YUV:
        raise vs.Error("lut_decrawl: only YUV format is supported")

    ythresh = scale_value(ythresh, 8, clip, ColorRange.FULL)
    cthresh = scale_value(cthresh, 8, clip, ColorRange.FULL, chroma=True)
    maxdiff = scale_value(maxdiff, 8, clip, ColorRange.FULL)

    clip_minus = shift_clip(clip, -1)
    clip_plus = shift_clip(clip, 1)

    clip_y = plane(clip, 0)
    clip_minus_y, clip_minus_u, clip_minus_v = clip_minus.std.SplitPlanes()
    clip_plus_y, clip_plus_u, clip_plus_v = clip_plus.std.SplitPlanes()

    average_y = core.std.Expr([clip_minus_y, clip_plus_y], f"x y - abs {ythresh} < x y + 2 / 0 ?")
    average_u = core.std.Expr([clip_minus_u, clip_plus_u], f"x y - abs {cthresh} < {peak} 0 ?")
    average_v = core.std.Expr([clip_minus_v, clip_plus_v], f"x y - abs {cthresh} < {peak} 0 ?")

    ymask = average_y.std.Binarize(scale_value(1, 8, clip, ColorRange.FULL))
    if usemaxdiff:
        diffplus_y = core.std.Expr([clip_plus_y, clip_y], f"x y - abs {maxdiff} < {peak} 0 ?")
        diffminus_y = core.std.Expr([clip_minus_y, clip_y], f"x y - abs {maxdiff} < {peak} 0 ?")
        diffs_y = core.std.Expr([diffplus_y, diffminus_y], f"x y + {peak + 1} < 0 {peak} ?")
        ymask = core.std.Expr([ymask, diffs_y], f"x y + {peak + 1} < 0 {peak} ?")
    cmask = core.std.Expr([average_u, average_v], f"x y + {peak + 1} < 0 {peak} ?")
    cmask = cmask.resize.Point(clip.width, clip.height)

    themask = core.std.Expr([ymask, cmask], f"x y + {peak + 1} < 0 {peak} ?")

    fixed_y = average_y.std.Merge(clip_y)

    output = clip_y.std.MaskedMerge(fixed_y, themask)
    output = join(output, clip)
    sc = scdetect(clip, scnchg / 255)
    output = output.std.FrameEval(partial(_scene_change, clips=[clip, output]), prop_src=sc, clip_src=[clip, output])

    return themask if mask else output


def lut_derainbow(
    clip: vs.VideoNode, cthresh: int = 10, ythresh: int = 10, y: bool = True, linkUV: bool = True, mask: bool = False
) -> vs.VideoNode:
    """
    :param clip:    Clip to process.
    :param cthresh: This determines how close the chroma values of the pixel in the previous and next frames have to be
                    for the pixel to be hit. Higher values (within reason) should catch more rainbows, but may introduce
                    unwanted artifacts. Probably shouldn't be set above 20 or so.
    :param ythresh: If the y parameter is set True, then this determines how close the luma values of the pixel in the
                    previous and next frames have to be for the pixel to be hit. Just as with cthresh.
    :param y:       Determines whether luma difference will be considered in determining which pixels to hit and which
                    to leave alone.
    :param linkUV:  Determines whether both chroma channels are considered in determining which pixels in each channel
                    to hit. When set True, only pixels that meet the thresholds for both U and V will be hit; when set
                    False, the U and V channels are masked separately (so a pixel could have its U hit but not its V, or
                    vice versa).
    :param mask:    When set True, the function will return the mask (for combined U/V) instead of the image. Formerly
                    used to find the best values of cthresh and ythresh. If linkUV=False, then this mask won't actually
                    be used anyway (because each chroma channel will have its own mask).
    """
    assert check_variable(clip, lut_derainbow)

    fmt = get_video_format(clip)
    peak = get_peak_value(fmt)

    if fmt.color_family != vs.YUV:
        raise vs.Error("lut_derainbow: only YUV format is supported")

    cthresh = scale_value(cthresh, 8, clip, ColorRange.FULL, chroma=True)
    ythresh = scale_value(ythresh, 8, clip, ColorRange.FULL)

    clip_minus = shift_clip(clip, -1)
    clip_plus = shift_clip(clip, 1)

    clip_u = plane(clip, 1)
    clip_v = plane(clip, 2)
    clip_minus_y, clip_minus_u, clip_minus_v = clip_minus.std.SplitPlanes()
    clip_plus_y, clip_plus_u, clip_plus_v = clip_plus.std.SplitPlanes()

    average_y = core.std.Expr([clip_minus_y, clip_plus_y], f"x y - abs {ythresh} < {peak} 0 ?")
    average_y = average_y.resize.Bilinear(clip_u.width, clip_u.height)
    average_u = core.std.Expr([clip_minus_u, clip_plus_u], f"x y - abs {cthresh} < x y + 2 / 0 ?")
    average_v = core.std.Expr([clip_minus_v, clip_plus_v], f"x y - abs {cthresh} < x y + 2 / 0 ?")

    scale_21 = scale_value(21, 8, clip, ColorRange.FULL, chroma=True)
    umask = average_u.std.Binarize(scale_21)
    vmask = average_v.std.Binarize(scale_21)
    themask = core.std.Expr([umask, vmask], f"x y + {peak + 1} < 0 {peak} ?")
    if y:
        blank = average_y.std.BlankClip(keep=True)
        umask = blank.std.MaskedMerge(average_y, umask)
        vmask = blank.std.MaskedMerge(average_y, vmask)
        themask = blank.std.MaskedMerge(average_y, themask)

    fixed_u = average_u.std.Merge(clip_u)
    fixed_v = average_v.std.Merge(clip_v)

    output_u = clip_u.std.MaskedMerge(fixed_u, themask if linkUV else umask)
    output_v = clip_v.std.MaskedMerge(fixed_v, themask if linkUV else vmask)

    output = join(clip, output_u, output_v)

    if mask:
        return themask.resize.Point(clip.width, clip.height)
    return output


def mcdaa3(clip: vs.VideoNode, opencl: bool = False, device: int | None = None, **kwargs: Any) -> vs.VideoNode:
    """
    :param clip:    Clip to process.
    :param opencl:  Whether to use OpenCL version of NNEDI3.
    :param device:  Device ordinal of OpenCL device.
    """
    assert check_variable(clip, mcdaa3)

    sup = clip.hqdn3d.Hqdn3d().fft3dfilter.FFT3DFilter().mv.Super(sharp=1)
    fv1 = sup.mv.Analyse(isb=False, truemotion=False, dct=2)
    fv2 = sup.mv.Analyse(isb=True, truemotion=True, dct=2)
    csaa = daa3mod(clip, opencl, device, **kwargs)
    momask1 = clip.mv.Mask(fv1, ml=2, kind=1)
    momask2 = clip.mv.Mask(fv2, ml=3, kind=1)
    momask = momask1.std.Merge(momask2)
    return clip.std.MaskedMerge(csaa, momask)


def mt_clamp(
    clip: vs.VideoNode,
    bright: vs.VideoNode,
    dark: vs.VideoNode,
    overshoot: int = 0,
    undershoot: int = 0,
    planes: PlanesT = None,
) -> vs.VideoNode:
    """clamp the value of the clip between bright + overshoot and dark - undershoot"""
    check_ref_clip(clip, bright, mt_clamp)
    check_ref_clip(clip, dark, mt_clamp)
    planes = normalize_planes(clip, planes)

    expr = f"x z {undershoot} - y {overshoot} + clamp"
    return norm_expr([clip, bright, dark], expr, planes)


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


def scdetect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    def _copy_property(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props["_SceneChangePrev"] = f[1].props["_SceneChangePrev"]
        fout.props["_SceneChangeNext"] = f[1].props["_SceneChangeNext"]
        return fout

    assert check_variable(clip, scdetect)

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s="709")

    sc = sc.misc.SCDetect(threshold)
    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame([clip, sc], _copy_property)

    return sc


#########################################################################################
###                                                                                   ###
###                      function Smooth Levels : SmoothLevels()                      ###
###                                                                                   ###
###                                v1.02 by "LaTo INV."                               ###
###                                                                                   ###
###                                  28 January 2009                                  ###
###                                                                                   ###
#########################################################################################
###
###
### /!\ Needed filters : RGVS, neo_f3kdb
### --------------------
###
###
###
### +---------+
### | GENERAL |
### +---------+
###
### Levels options:
### ---------------
### input_low, gamma, input_high, output_low, output_high [default: 0, 1.0, maximum value of input format, 0, maximum value of input format]
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### chroma [default: 50]
### ---------------------
### 0   = no chroma processing     (similar as Ylevels)
### xx  = intermediary
### 100 = normal chroma processing (similar as Levels)
###
### limiter [default: 0]
### --------------------
### 0 = no limiter             (similar as Ylevels)
### 1 = input limiter
### 2 = output limiter         (similar as Levels: coring=false)
### 3 = input & output limiter (similar as Levels: coring=true)
###
###
###
### +----------+
### | LIMITING |
### +----------+
###
### Lmode [default: 0]
### ------------------
### 0 = no limit
### 1 = limit conversion on dark & bright areas (apply conversion @0%   at luma=0 & @100% at luma=Ecenter & @0% at luma=255)
### 2 = limit conversion on dark areas          (apply conversion @0%   at luma=0 & @100% at luma=255)
### 3 = limit conversion on bright areas        (apply conversion @100% at luma=0 & @0%   at luma=255)
###
### DarkSTR [default: 100]
### ----------------------
### Strength for limiting: the higher, the more conversion are reduced on dark areas (for Lmode=1&2)
###
### BrightSTR [default: 100]
### ------------------------
### Strength for limiting: the higher, the more conversion are reduced on bright areas (for Lmode=1&3)
###
### Ecenter [default: median value of input format]
### ----------------------
### Center of expression for Lmode=1
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### protect [default: -1]
### ---------------------
### -1  = protect off
### >=0 = pure black protection
###       ---> don't apply conversion on pixels egal or below this value
###            (ex: with 16, the black areas like borders and generic are untouched so they don't look washed out)
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### Ecurve [default: 0]
### -------------------
### Curve used for limit & protect:
### 0 = use sine curve
### 1 = use linear curve
###
###
###
### +-----------+
### | SMOOTHING |
### +-----------+
###
### Smode [default: -2]
### -------------------
### 2  = smooth on, maxdiff must be < to "255/Mfactor"
### 1  = smooth on, maxdiff must be < to "128/Mfactor"
### 0  = smooth off
### -1 = smooth on if maxdiff < "128/Mfactor", else off
### -2 = smooth on if maxdiff < "255/Mfactor", else off
###
### Mfactor [default: 2]
### --------------------
### The higher, the more precise but the less maxdiff allowed:
### maxdiff=128/Mfactor for Smode1&-1 and maxdiff=255/Mfactor for Smode2&-2
###
### RGmode [default: 12]
### --------------------
### In strength order: + 19 > 12 >> 20 > 11 -
###
### useDB [default: false]
### ---------------------
### Use neo_f3kdb on top of removegrain: prevent posterize when doing levels conversion
###
###
#########################################################################################
def SmoothLevels(input, input_low=0, gamma=1.0, input_high=None, output_low=0, output_high=None, chroma=50, limiter=0, Lmode=0, DarkSTR=100, BrightSTR=100, Ecenter=None, protect=-1, Ecurve=0,
                 Smode=-2, Mfactor=2, RGmode=12, useDB=False):
    # sin(pi x / 2) for -1 < x < 1 using Taylor series
    def _sine_expr(var):
        return f'{-3.5988432352121e-6} {var} * {var} * {0.00016044118478736} + {var} * {var} * {-0.0046817541353187} + {var} * {var} * {0.079692626246167} + {var} * {var} * {-0.64596409750625} + {var} * {var} * {1.5707963267949} + {var} *'

    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SmoothLevels: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('SmoothLevels: RGB format is not supported')

    isGray = (input.format.color_family == vs.GRAY)

    if input.format.sample_type == vs.INTEGER:
        neutral = [1 << (input.format.bits_per_sample - 1)] * 2
        peak = (1 << input.format.bits_per_sample) - 1
    else:
        neutral = [0.5, 0.0]
        peak = 1.0

    if chroma <= 0 and not isGray:
        input_orig = input
        input = plane(input, 0)
    else:
        input_orig = None

    if input_high is None:
        input_high = peak

    if output_high is None:
        output_high = peak

    if Ecenter is None:
        Ecenter = neutral[0]

    if gamma <= 0:
        raise vs.Error('SmoothLevels: gamma must be greater than 0.0')

    if Ecenter <= 0 or Ecenter >= peak:
        raise vs.Error('SmoothLevels: Ecenter must be greater than 0 and less than maximum value of input format')

    if Mfactor <= 0:
        raise vs.Error('SmoothLevels: Mfactor must be greater than 0')

    if RGmode == 4:
        RemoveGrain = partial(core.std.Median)
    elif RGmode in [11, 12]:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif RGmode == 19:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif RGmode == 20:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        RemoveGrain = partial(core.rgvs.RemoveGrain, mode=[RGmode])

    ### EXPRESSION
    exprY = f'x {input_low} - {input_high - input_low + (input_high == input_low)} / {1 / gamma} pow {output_high - output_low} * {output_low} +'

    if chroma > 0 and not isGray:
        scaleC = ((output_high - output_low) / (input_high - input_low + (input_high == input_low)) + 100 / chroma - 1) / (100 / chroma)
        exprC = f'x {neutral[1]} - {scaleC} * {neutral[1]} +'

    Dstr = DarkSTR / 100
    Bstr = BrightSTR / 100

    if Lmode <= 0:
        exprL = '1'
    elif Ecurve <= 0:
        if Lmode == 1:
            var_d = f'x {Ecenter} /'
            var_b = f'{peak} x - {peak} {Ecenter} - /'
            exprL = f'x {Ecenter} < ' + _sine_expr(var_d) + f' {Dstr} pow x {Ecenter} > ' + _sine_expr(var_b) + f' {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            var_d = f'x {peak} /'
            exprL = _sine_expr(var_d) + f' {Dstr} pow'
        else:
            var_b = f'{peak} x - {peak} /'
            exprL = _sine_expr(var_b) + f' {Bstr} pow'
    else:
        if Lmode == 1:
            exprL = f'x {Ecenter} < x {Ecenter} / abs {Dstr} pow x {Ecenter} > 1 x {Ecenter} - {peak - Ecenter} / abs - {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            exprL = f'1 x {peak} - {peak} / abs - {Dstr} pow'
        else:
            exprL = f'x {peak} - {peak} / abs {Bstr} pow'

    if protect <= -1:
        exprP = '1'
    elif Ecurve <= 0:
        scale_16 = scale_value(16, 8, input, ColorRange.FULL)
        var_p = f'x {protect} - {scale_16} /'
        exprP = f'x {protect} <= 0 x {protect + scale_16} >= 1 ' + _sine_expr(var_p) + ' ? ?'
    else:
        scale_16 = scale_value(16, 8, input, ColorRange.FULL)
        exprP = f'x {protect} <= 0 x {protect + scale_16} >= 1 x {protect} - {scale_16} / abs ? ?'

    ### PROCESS
    if limiter == 1 or limiter >= 3:
        limitI = input.std.Expr(expr=[f'x {input_low} max {input_high} min'])
    else:
        limitI = input

    expr = exprL + ' ' + exprP + ' * ' + exprY + ' x - * x +'
    level = limitI.std.Expr(expr=[expr] if chroma <= 0 or isGray else [expr, exprC])
    diff = core.std.Expr([limitI, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process = RemoveGrain(diff)
    if useDB:
        process = process.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        smth = core.std.MakeDiff(limitI, process)
    else:
        smth = core.std.Expr([limitI, process], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    level2 = core.std.Expr([limitI, diff], expr=[f'x y {neutral[1]} - {Mfactor} / -'])
    diff2 = core.std.Expr([level2, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process2 = RemoveGrain(diff2)
    if useDB:
        process2 = process2.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        smth2 = core.std.MakeDiff(smth, process2)
    else:
        smth2 = core.std.Expr([smth, process2], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    mask1 = core.std.Expr([limitI, level], expr=[f'x y - abs {neutral[0] / Mfactor} >= {peak} 0 ?'])
    mask2 = core.std.Expr([limitI, level], expr=[f'x y - abs {peak / Mfactor} >= {peak} 0 ?'])

    if Smode >= 2:
        Slevel = smth2
    elif Smode == 1:
        Slevel = smth
    elif Smode == -1:
        Slevel = core.std.MaskedMerge(smth, level, mask1)
    elif Smode <= -2:
        Slevel = core.std.MaskedMerge(core.std.MaskedMerge(smth, smth2, mask1), level, mask2)
    else:
        Slevel = level

    if limiter >= 2:
        limitO = Slevel.std.Expr(expr=[f'x {output_low} max {output_high} min'])
    else:
        limitO = Slevel

    if input_orig is not None:
        limitO = core.std.ShufflePlanes([limitO, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return limitO


##############################################################################
# Original script by g-force converted into a stand alone script by McCauley #
# latest version from December 10, 2008                                      #
##############################################################################
def Stab(clp, dxmax=4, dymax=4, mirror=0):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Stab: this is not a clip')

    clp = scdetect(clp, 25 / 255)
    temp = clp.misc.AverageFrames([1] * 15, scenechange=True)
    inter = core.std.Interleave([core.rgvs.Repair(temp, clp.misc.AverageFrames([1] * 3, scenechange=True), mode=[1]), clp])
    mdata = inter.mv.DepanEstimate(trust=0, dxmax=dxmax, dymax=dymax)
    last = inter.mv.DepanCompensate(data=mdata, offset=-1, mirror=mirror)
    return last[::2]


def STPresso(
    clp: vs.VideoNode,
    limit: int = 3,
    bias: int = 24,
    RGmode: Union[int, vs.VideoNode] = 4,
    tthr: int = 12,
    tlimit: int = 3,
    tbias: int = 49,
    back: int = 1,
    planes: Optional[Union[int, Sequence[int]]] = None,
) -> vs.VideoNode:
    """
    Dampen the grain just a little, to keep the original look.

    Parameters:
        clp: Clip to process.

        limit: The spatial part won't change a pixel more than this.

        bias: The percentage of the spatial filter that will apply.

        RGmode: The spatial filter is RemoveGrain, this is its mode. It also accepts loading your personal prefiltered clip.

        tthr: Temporal threshold for fluxsmooth. Can be set "a good bit bigger" than usually.

        tlimit: The temporal filter won't change a pixel more than this.

        tbias: The percentage of the temporal filter that will apply.

        back: After all changes have been calculated, reduce all pixel changes by this value. (shift "back" towards original value)

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    """
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('STPresso: this is not a clip')

    plane_range = range(clp.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    bits = get_depth(clp)
    limit = scale_value(limit, 8, bits)
    tthr = scale_value(tthr, 8, bits)
    tlimit = scale_value(tlimit, 8, bits)
    back = scale_value(back, 8, bits)

    LIM = cround(limit * 100 / bias - 1) if limit > 0 else cround(scale_value(100 / bias, 8, bits))
    TLIM = cround(tlimit * 100 / tbias - 1) if tlimit > 0 else cround(scale_value(100 / tbias, 8, bits))

    if limit < 0:
        expr = f'x y - abs {LIM} < x x 1 x y - dup abs / * - ?'
    else:
        expr = f'x y - abs {scale_value(1, 8, bits)} < x x {LIM} + y < x {limit} + x {LIM} - y > x {limit} - x {100 - bias} * y {bias} * + 100 / ? ? ?'
    if tlimit < 0:
        texpr = f'x y - abs {TLIM} < x x 1 x y - dup abs / * - ?'
    else:
        texpr = f'x y - abs {scale_value(1, 8, bits)} < x x {TLIM} + y < x {tlimit} + x {TLIM} - y > x {tlimit} - x {100 - tbias} * y {tbias} * + 100 / ? ? ?'

    if isinstance(RGmode, vs.VideoNode):
        bzz = RGmode
    else:
        if RGmode == 4:
            bzz = clp.std.Median(planes=planes)
        elif RGmode in [11, 12]:
            bzz = clp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
        elif RGmode == 19:
            bzz = clp.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=planes)
        elif RGmode == 20:
            bzz = clp.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=planes)
        else:
            bzz = clp.rgvs.RemoveGrain(mode=RGmode)

    last = core.std.Expr([clp, bzz], expr=[expr if i in planes else '' for i in plane_range])

    if tthr > 0:
        analyse_args = dict(truemotion=False, delta=1, blksize=16, overlap=8)

        mvSuper = bzz.mv.Super(sharp=1)
        bv1 = mvSuper.mv.Analyse(isb=True, **analyse_args)
        fv1 = mvSuper.mv.Analyse(isb=False, **analyse_args)
        bc1 = core.mv.Compensate(bzz, mvSuper, bv1)
        fc1 = core.mv.Compensate(bzz, mvSuper, fv1)

        interleave = core.std.Interleave([fc1, bzz, bc1])
        smooth = interleave.flux.SmoothT(temporal_threshold=tthr, planes=planes)
        smooth = smooth.std.SelectEvery(cycle=3, offsets=1)

        diff = core.std.MakeDiff(bzz, smooth, planes=planes)
        diff = core.std.MakeDiff(last, diff, planes=planes)
        last = core.std.Expr([last, diff], expr=[texpr if i in planes else '' for i in plane_range])

    if back > 0:
        expr = f'x {back} + y < x {back} + x {back} - y > x {back} - y ? ?'
        last = core.std.Expr([last, clp], expr=[expr if i in planes else '' for i in plane_range])

    return last


def bbmod(*args, **kwargs):
    raise vs.Error("havsfunc.bbmod outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def ChangeFPS(*args, **kwargs):
    raise vs.Error("havsfunc.ChangeFPS outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def ContraSharpening(*args, **kwargs):
    raise vs.Error(
        "havsfunc.ContraSharpening outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def deblock_qed(*args, **kwargs):
    raise vs.Error(
        "havsfunc.deblock_qed outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def dec_txt60mc(*args, **kwargs):
    raise vs.Error(
        "havsfunc.dec_txt60mc outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def DeHalo_alpha(*args, **kwargs):
    raise vs.Error(
        "havsfunc.DeHalo_alpha outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def DitherLumaRebuild(*args, **kwargs):
    raise vs.Error(
        "havsfunc.DitherLumaRebuild outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def EdgeCleaner(*args, **kwargs):
    raise vs.Error(
        "havsfunc.EdgeCleaner outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def FineDehalo(*args, **kwargs):
    raise vs.Error("havsfunc.FineDehalo outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def FineDehalo_contrasharp(*args, **kwargs):
    raise vs.Error(
        "havsfunc.FineDehalo_contrasharp outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def FineDehalo2(*args, **kwargs):
    raise vs.Error(
        "havsfunc.FineDehalo2 outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def FixColumnBrightness(*args, **kwargs):
    raise vs.Error("havsfunc.FixColumnBrightness outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def FixRowBrightness(*args, **kwargs):
    raise vs.Error("havsfunc.FixRowBrightness outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def FixColumnBrightnessProtect(*args, **kwargs):
    raise vs.Error("havsfunc.FixColumnBrightnessProtect outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def FixRowBrightnessProtect(*args, **kwargs):
    raise vs.Error("havsfunc.FixRowBrightnessProtect outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def FixColumnBrightnessProtect2(*args, **kwargs):
    raise vs.Error("havsfunc.FixColumnBrightnessProtect2 outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def FixRowBrightnessProtect2(*args, **kwargs):
    raise vs.Error("havsfunc.FixRowBrightnessProtect2 outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def Gauss(*args, **kwargs):
    raise vs.Error("havsfunc.Gauss outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def GrainFactory3(*args, **kwargs):
    raise vs.Error(
        "havsfunc.GrainFactory3 outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def HQDeringmod(*args, **kwargs):
    raise vs.Error(
        "havsfunc.HQDeringmod outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def ivtc_txt30mc(*args, **kwargs):
    raise vs.Error(
        "havsfunc.ivtc_txt30mc outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def ivtc_txt60mc(*args, **kwargs):
    raise vs.Error(
        "havsfunc.ivtc_txt60mc outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def KNLMeansCL(*args, **kwargs):
    raise vs.Error(
        "havsfunc.KNLMeansCL outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def m4(*args, **kwargs):
    raise vs.Error("havsfunc.m4 outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def MCTemporalDenoise(*args, **kwargs):
    raise vs.Error(
        "havsfunc.MCTemporalDenoise outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def MinBlur(*args, **kwargs):
    raise vs.Error("havsfunc.MinBlur outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def mt_expand_multi(*args, **kwargs):
    raise vs.Error(
        "havsfunc.mt_expand_multi outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def mt_inpand_multi(*args, **kwargs):
    raise vs.Error(
        "havsfunc.mt_inpand_multi outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def mt_inflate_multi(*args, **kwargs):
    raise vs.Error(
        "havsfunc.mt_inflate_multi outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def mt_deflate_multi(*args, **kwargs):
    raise vs.Error(
        "havsfunc.mt_deflate_multi outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def Padding(*args, **kwargs):
    raise vs.Error("havsfunc.Padding outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def QTGMC(*args, **kwargs):
    raise vs.Error(
        "havsfunc.QTGMC outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def santiag(*args, **kwargs):
    raise vs.Error("havsfunc.santiag outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def scale(*args, **kwargs):
    raise vs.Error("havsfunc.scale outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def smartfademod(*args, **kwargs):
    raise vs.Error(
        "havsfunc.smartfademod outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def SMDegrain(*args, **kwargs):
    raise vs.Error("havsfunc.SMDegrain outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")


def srestore(*args, **kwargs):
    raise vs.Error("havsfunc.srestore outdated. Use https://github.com/WolframRhodium/muvsfunc instead.")


def Vinverse(*args, **kwargs):
    raise vs.Error(
        "havsfunc.Vinverse outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def Vinverse2(*args, **kwargs):
    raise vs.Error(
        "havsfunc.Vinverse2 outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead."
    )


def YAHR(*args, **kwargs):
    raise vs.Error("havsfunc.YAHR outdated. Use https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack instead.")
