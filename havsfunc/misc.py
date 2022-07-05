from __future__ import annotations

from typing import Optional, Sequence, Union

import vapoursynth as vs
from vsutil import get_depth, join, plane, scale_value

from .helpers import AverageFrames, m4, scale

core = vs.core


def FixChromaBleedingMod(input: vs.VideoNode, cx: int = 4, cy: int = 4, thr: float = 4.0,
                         strength: float = 0.8, blur: bool = False) -> vs.VideoNode:
    """
    FixChromaBleedingMod v1.36
    A script to reduce color bleeding, over-saturation, and color shifting mainly in red and blue areas.

    Parameters:
        input: Clip to process.

        cx: Horizontal chroma shift. Positive value shifts chroma to the left,
            negative value shifts chroma to the right.

        cy: Vertical chroma shift. Positive value shifts chroma upwards, negative value shifts chroma downwards.

        thr: Masking threshold, higher values treat more areas as color bleed.

        strength: Saturation strength in clip to be merged with the original chroma.
            Values below 1.0 reduce the saturation, a value of 1.0 leaves the saturation intact.

        blur: Set to true to blur the mask clip.
    """
    from adjust import Tweak

    if not isinstance(input, vs.VideoNode):
        raise vs.Error('FixChromaBleedingMod: this is not a clip')

    if input.format.color_family != vs.YUV or input.format.sample_type != vs.INTEGER:
        raise vs.Error('FixChromaBleedingMod: only YUV format with integer sample type is supported')

    # prepare to work on the V channel and filter noise
    vch = plane(Tweak(input, sat=thr), 2)
    if blur:
        area = vch.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    else:
        area = vch

    bits = get_depth(input)
    i16 = scale_value(16, 8, bits)
    i25 = scale_value(25, 8, bits)
    i231 = scale_value(231, 8, bits)
    i235 = scale_value(235, 8, bits)
    i240 = scale_value(240, 8, bits)

    # select and normalize both extremes of the scale
    red = area.std.Levels(min_in=i235, max_in=i235, min_out=i235, max_out=i16)
    blue = area.std.Levels(min_in=i16, max_in=i16, min_out=i16, max_out=i235)

    # merge both masks
    mask = core.std.Merge(red, blue)
    if not blur:
        mask = mask.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    mask = mask.std.Levels(min_in=i231, max_in=i231, min_out=i235, max_out=i16)

    # expand to cover beyond the bleeding areas and shift to compensate the resizing
    mask = mask.std.Convolution(matrix=[0, 0, 0, 1, 0, 0, 0, 0, 0], divisor=1, saturate=False).std.Convolution(
        matrix=[1, 1, 1, 1, 1, 1, 0, 0, 0], divisor=8, saturate=False
    )

    # binarize (also a trick to expand)
    mask = mask.std.Levels(min_in=i25, max_in=i25, min_out=i16, max_out=i240).std.Inflate()

    # prepare a version of the image that has its chroma shifted and less saturated
    input_c = Tweak(input.resize.Spline16(src_left=cx, src_top=cy), sat=strength)

    # combine both images using the mask
    fu = core.std.MaskedMerge(plane(input, 1), plane(input_c, 1), mask)
    fv = core.std.MaskedMerge(plane(input, 2), plane(input_c, 2), mask)
    return join([input, fu, fv])


# Parameters:
#  g1str (float)       - [0.0 - ???] strength of grain / for dark areas. Default is 7.0
#  g2str (float)       - [0.0 - ???] strength of grain / for midtone areas. Default is 5.0
#  g3str (float)       - [0.0 - ???] strength of grain / for bright areas. Default is 3.0
#  g1shrp (int)        - [0 - 100] sharpness of grain / for dark areas (NO EFFECT when g1size=1.0 !!). Default is 60
#  g2shrp (int)        - [0 - 100] sharpness of grain / for midtone areas (NO EFFECT when g2size=1.0 !!). Default is 66
#  g3shrp (int)        - [0 - 100] sharpness of grain / for bright areas (NO EFFECT when g3size=1.0 !!). Default is 80
#  g1size (float)      - [0.5 - 4.0] size of grain / for dark areas. Default is 1.5
#  g2size (float)      - [0.5 - 4.0] size of grain / for midtone areas. Default is 1.2
#  g3size (float)      - [0.5 - 4.0] size of grain / for bright areas. Default is 0.9
#  temp_avg (int)      - [0 - 100] percentage of noise's temporal averaging. Default is 0
#  ontop_grain (float) - [0 - ???] additional grain to put on top of prev. generated grain. Default is 0.0
#  seed (int)          - specifies a repeatable grain sequence. Set to at least 0 to use.
#  th1 (int)           - start of dark->midtone mixing zone. Default is 24
#  th2 (int)           - end of dark->midtone mixing zone. Default is 56
#  th3 (int)           - start of midtone->bright mixing zone. Default is 128
#  th4 (int)           - end of midtone->bright mixing zone. Default is 160
def GrainFactory3(clp, g1str=7.0, g2str=5.0, g3str=3.0, g1shrp=60, g2shrp=66, g3shrp=80, g1size=1.5, g2size=1.2,
                  g3size=0.9, temp_avg=0, ontop_grain=0.0, seed=-1, th1=24, th2=56, th3=128, th4=160):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('GrainFactory3: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('GrainFactory3: RGB format is not supported')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = plane(clp, 0)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height
    sx1 = m4(ox / g1size)
    sy1 = m4(oy / g1size)
    sx1a = m4((ox + sx1) / 2)
    sy1a = m4((oy + sy1) / 2)
    sx2 = m4(ox / g2size)
    sy2 = m4(oy / g2size)
    sx2a = m4((ox + sx2) / 2)
    sy2a = m4((oy + sy2) / 2)
    sx3 = m4(ox / g3size)
    sy3 = m4(oy / g3size)
    sx3a = m4((ox + sx3) / 2)
    sy3a = m4((oy + sy3) / 2)

    b1 = g1shrp / -50 + 1
    b2 = g2shrp / -50 + 1
    b3 = g3shrp / -50 + 1
    b1a = b1 / 2
    b2a = b2 / 2
    b3a = b3 / 2
    c1 = (1 - b1) / 2
    c2 = (1 - b2) / 2
    c3 = (1 - b3) / 2
    c1a = (1 - b1a) / 2
    c2a = (1 - b2a) / 2
    c3a = (1 - b3a) / 2
    tmpavg = temp_avg / 100
    th1 = scale(th1, peak)
    th2 = scale(th2, peak)
    th3 = scale(th3, peak)
    th4 = scale(th4, peak)

    grainlayer1 = clp.std.BlankClip(width=sx1, height=sy1, color=[neutral]).grain.Add(var=g1str, seed=seed)
    if g1size != 1 and (sx1 != ox or sy1 != oy):
        if g1size > 1.5:
            grainlayer1 = grainlayer1.resize.Bicubic(sx1a, sy1a, filter_param_a=b1a, filter_param_b=c1a) \
                .resize.Bicubic(ox, oy, filter_param_a=b1a, filter_param_b=c1a)
        else:
            grainlayer1 = grainlayer1.resize.Bicubic(ox, oy, filter_param_a=b1, filter_param_b=c1)

    grainlayer2 = clp.std.BlankClip(width=sx2, height=sy2, color=[neutral]).grain.Add(var=g2str, seed=seed)
    if g2size != 1 and (sx2 != ox or sy2 != oy):
        if g2size > 1.5:
            grainlayer2 = grainlayer2.resize.Bicubic(sx2a, sy2a, filter_param_a=b2a, filter_param_b=c2a) \
                .resize.Bicubic(ox, oy, filter_param_a=b2a, filter_param_b=c2a)
        else:
            grainlayer2 = grainlayer2.resize.Bicubic(ox, oy, filter_param_a=b2, filter_param_b=c2)

    grainlayer3 = clp.std.BlankClip(width=sx3, height=sy3, color=[neutral]).grain.Add(var=g3str, seed=seed)
    if g3size != 1 and (sx3 != ox or sy3 != oy):
        if g3size > 1.5:
            grainlayer3 = grainlayer3.resize.Bicubic(sx3a, sy3a, filter_param_a=b3a, filter_param_b=c3a) \
                .resize.Bicubic(ox, oy, filter_param_a=b3a, filter_param_b=c3a)
        else:
            grainlayer3 = grainlayer3.resize.Bicubic(ox, oy, filter_param_a=b3, filter_param_b=c3)

    expr1 = f'x {th1} < 0 x {th2} > {peak} {peak} {th2 - th1} / x {th1} - * ? ?'
    expr2 = f'x {th3} < 0 x {th4} > {peak} {peak} {th4 - th3} / x {th3} - * ? ?'
    grainlayer = core.std.MaskedMerge(core.std.MaskedMerge(grainlayer1, grainlayer2, clp.std.Expr(expr=[expr1])),
                                      grainlayer3, clp.std.Expr(expr=[expr2]))

    if temp_avg > 0:
        grainlayer = core.std.Merge(grainlayer, AverageFrames(grainlayer, weights=[1] * 3), weight=[tmpavg])
    if ontop_grain > 0:
        grainlayer = grainlayer.grain.Add(var=ontop_grain, seed=seed)

    result = core.std.MakeDiff(clp, grainlayer)

    if clp_orig is not None:
        result = core.std.ShufflePlanes([result, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return result


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
    """
    Puts clip overlay on top of clip base using different blend modes, and with optional x,y positioning,
    masking and opacity.

    Parameters:
        base: This clip will be the base, determining the size and all other video properties of the result.

        overlay: This is the image that will be placed on top of the base clip.

        x, y: Define the placement of the overlay image on the base clip, in pixels. Can be positive or negative.

        mask: Optional transparency mask. Must be the same size as overlay.
            Where mask is darker, overlay will be more transparent.

        opacity: Set overlay transparency. The value is from 0.0 to 1.0,
            where 0.0 is transparent and 1.0 is fully opaque.
            This value is multiplied by mask luminance to form the final opacity.

        mode: Defines how your overlay should be blended with your base image. Available blend modes are:
            addition, average, burn, darken, difference, divide, dodge, exclusion, extremity, freeze, glow,
            grainextract, grainmerge, hardlight, hardmix, heat, lighten, linearlight, multiply, negation, normal,
            overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        mask_first_plane: If true, only the mask's first plane will be used for transparency.
    """
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
        mask = overlay.std.BlankClip(format=overlay.format.replace(
            color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
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
        expr = f'x {neutral} > y {peak} y - x {neutral} - * {neutral} / 0.5 y {neutral} - abs {peak} / - * + y y ' \
               + f'{neutral} x - {neutral} / * 0.5 y {neutral} - abs {peak} / - * - ?'
    elif mode == 'subtract':
        expr = 'x y -'
    elif mode == 'vividlight':
        expr = f'x {neutral} < x 0 <= 2 x * {peak} {peak} y - {factor} * 2 x * / - ? 2 x {neutral} - * {peak} >= ' \
               + f'2 x {neutral} - * y {factor} * {peak} 2 x {neutral} - * - / ? ?'
    else:
        raise vs.Error('Overlay: invalid mode specified')

    if mode != 'normal':
        overlay = core.std.Expr([overlay, base], expr=[expr if i in planes else '' for i in plane_range])

    # Return padded clip
    last = core.std.MaskedMerge(base, overlay, mask, planes=planes, first_plane=mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last
