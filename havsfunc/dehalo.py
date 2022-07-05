from __future__ import annotations

from typing import Optional, Sequence, Union

import vapoursynth as vs
from vsutil import fallback, get_depth, get_y, scale_value

from .blur import MinBlur
from .helpers import Padding, cround, m4
from .mask import AvsPrewitt, mt_expand_multi, mt_inflate_multi, mt_inpand_multi

core = vs.core


def DeHalo_alpha(
    clp: vs.VideoNode,
    rx: float = 2.0,
    ry: float = 2.0,
    darkstr: float = 1.0,
    brightstr: float = 1.0,
    lowsens: float = 50.0,
    highsens: float = 50.0,
    ss: float = 1.5,
) -> vs.VideoNode:
    """
    Reduce halo artifacts that can occur when sharpening.

    Parameters:
        clp: Clip to process.

        rx, ry: As usual, the radii for halo removal. This function is rather sensitive to the radius settings.
            Set it as low as possible! If radius is set too high, it will start missing small spots.

        darkstr, brightstr: The strength factors for processing dark and bright halos.
            Default 1.0 both for symmetrical processing.
            On Comic/Anime, darkstr=0.4~0.8 sometimes might be better ... sometimes.
            In General, the function seems to preserve dark lines rather good.

        lowsens, highsens: Sensitivity settings, not that easy to describe them exactly ...
            In a sense, they define a window between how weak an achieved effect has to be to get fully accepted,
            and how strong an achieved effect has to be to get fully discarded.

        ss: Supersampling factor, to avoid creation of aliasing.
    """
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('DeHalo_alpha: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('DeHalo_alpha: RGB format is not supported')

    bits = get_depth(clp)

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = get_y(clp)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height

    halos = clp.resize.Bicubic(m4(ox / rx), m4(oy / ry), filter_param_a=1 / 3, filter_param_b=1 / 3) \
        .resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    are = core.std.Expr([clp.std.Maximum(), clp.std.Minimum()], expr='x y -')
    ugly = core.std.Expr([halos.std.Maximum(), halos.std.Minimum()], expr='x y -')
    so = core.std.Expr(
        [ugly, are],
        expr=f'y x - y 0.000001 + / {scale_value(255, 8, bits)} * {scale_value(lowsens, 8, bits)} - y '
             f'{scale_value(256, 8, bits)} + {scale_value(512, 8, bits)} / {highsens / 100} + *',
    )
    if clp.format.sample_type == vs.FLOAT:
        so = so.std.Limiter()
    lets = core.std.MaskedMerge(halos, clp, so)
    if ss <= 1:
        remove = core.rgvs.Repair(clp, lets, mode=1)
    else:
        remove = core.std.Expr(
            [
                core.std.Expr(
                    [
                        clp.resize.Lanczos(m4(ox * ss), m4(oy * ss)),
                        lets.std.Maximum().resize.Bicubic(m4(ox * ss), m4(oy * ss),
                                                          filter_param_a=1 / 3, filter_param_b=1 / 3),
                    ],
                    expr='x y min',
                ),
                lets.std.Minimum().resize.Bicubic(m4(ox * ss), m4(oy * ss), filter_param_a=1 / 3, filter_param_b=1 / 3),
            ],
            expr='x y max',
        ).resize.Lanczos(ox, oy)
    them = core.std.Expr([clp, remove], expr=f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')

    if clp_orig is not None:
        them = core.std.ShufflePlanes([them, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return them


def EdgeCleaner(c: vs.VideoNode,
                strength: int = 10,
                rep: bool = True,
                rmode: int = 17,
                smode: int = 0,
                hot: bool = False) -> vs.VideoNode:
    """
    EdgeCleaner v1.04
    A simple edge cleaning and weak dehaloing function.

    Parameters:
        c: Clip to process.

        strength: Specifies edge denoising strength.

        rep: Activates Repair for the aWarpSharped clip.

        rmode: Specifies the Repair mode.
            1 is very mild and good for halos,
            16 and 18 are good for edge structure preserval on strong settings but keep more halos and edge noise,
            17 is similar to 16 but keeps much less haloing, other modes are not recommended.

        smode: Specifies what method will be used for finding small particles, ie stars.
            0 is disabled, 1 uses RemoveGrain.

        hot: Specifies whether removal of hot pixels should take place.
    """
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('EdgeCleaner: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('EdgeCleaner: RGB format is not supported')

    bits = get_depth(c)
    peak = (1 << bits) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = get_y(c)
    else:
        c_orig = None

    if smode > 0:
        strength += 4

    main = Padding(c, 6, 6, 6, 6).warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    if rep:
        main = core.rgvs.Repair(main, c, mode=rmode)

    mask = (
        AvsPrewitt(c)
        .std.Expr(expr=f'x {scale_value(4, 8, bits)} < 0 x {scale_value(32, 8, bits)} > {peak} x ? ?')
        .std.InvertMask()
        .std.Convolution(matrix=[1] * 9)
    )

    final = core.std.MaskedMerge(c, main, mask)
    if hot:
        final = core.rgvs.Repair(final, c, mode=2)
    if smode > 0:
        clean = c.rgvs.RemoveGrain(mode=17)
        diff = core.std.MakeDiff(c, clean)
        mask = AvsPrewitt(
            diff.std.Levels(min_in=scale_value(40, 8, bits), max_in=scale_value(168, 8, bits), gamma=0.35)
            .rgvs.RemoveGrain(mode=7)) \
            .std.Expr(expr=f'x {scale_value(4, 8, bits)} < 0 x {scale_value(16, 8, bits)} > {peak} x ? ?')
        final = core.std.MaskedMerge(final, c, mask)

    if c_orig is not None:
        final = core.std.ShufflePlanes([final, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return final


def FineDehalo(
    src: vs.VideoNode,
    rx: float = 2.0,
    ry: Optional[float] = None,
    thmi: int = 80,
    thma: int = 128,
    thlimi: int = 50,
    thlima: int = 100,
    darkstr: float = 1.0,
    brightstr: float = 1.0,
    showmask: int = 0,
    contra: float = 0.0,
    excl: bool = True,
    edgeproc: float = 0.0,
    mask: Optional[vs.VideoNode] = None,
) -> vs.VideoNode:
    """
    Halo removal script that uses DeHalo_alpha with a few masks and optional contra-sharpening
    to try remove halos without removing important details.

    Parameters:
        src: Clip to process.

        rx, ry: The radii for halo removal in DeHalo_alpha.

        thmi, thma: Minimum and maximum threshold for sharp edges; keep only the sharpest edges (line edges).
            To see the effects of these settings take a look at the strong mask (showmask=4).

        thlimi, thlima: Minimum and maximum limiting threshold; includes more edges than previously,
            but ignores simple details.

        darkstr, brightstr: The strength factors for processing dark and bright halos in DeHalo_alpha.

        showmask: Shows mask; useful for adjusting settings.
            0 = none
            1 = outside mask
            2 = shrink mask
            3 = edge mask
            4 = strong mask

        contra: Contra-sharpening.

        excl: Activates an additional step (exclusion zones) to make sure that the main edges are really excluded.

        mask: Basic edge mask to apply the threshold instead of applying to the mask created by AvsPrewitt.
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('FineDehalo: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo: RGB format is not supported')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('FineDehalo: mask is not a clip')

        if mask.format.color_family != vs.GRAY:
            raise vs.Error('FineDehalo: mask must be Gray format')

    is_float = src.format.sample_type == vs.FLOAT

    bits = get_depth(src)

    if src.format.color_family != vs.GRAY:
        src_orig = src
        src = get_y(src)
    else:
        src_orig = None

    ry = fallback(ry, rx)

    rx_i = cround(rx)
    ry_i = cround(ry)

    # Dehaloing #

    dehaloed = DeHalo_alpha(src, rx=rx, ry=ry, darkstr=darkstr, brightstr=brightstr)

    # Contrasharpening
    if contra > 0:
        dehaloed = FineDehalo_contrasharp(dehaloed, src, contra)

    # Main edges #

    # Basic edge detection, thresholding will be applied later
    edges = fallback(mask, AvsPrewitt(src))

    # Keeps only the sharpest edges (line edges)
    strong = edges.std.Expr(expr=f'x {scale_value(thmi, 8, bits)} - {thma - thmi} / 255 *')
    if is_float:
        strong = strong.std.Limiter()

    # Extends them to include the potential halos
    large = mt_expand_multi(strong, sw=rx_i, sh=ry_i)

    # Exclusion zones #

    # When two edges are close from each other (both edges of a single line or multiple parallel color bands),
    # the halo removal oversmoothes them or makes seriously bleed the bands, producing annoying artifacts.
    # Therefore we have to produce a mask to exclude these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = edges.std.Expr(expr=f'x {scale_value(thlimi, 8, bits)} - {thlima - thlimi} / 255 *')
    if is_float:
        light = light.std.Limiter()

    # To build the exclusion zone, we make grow the edge mask, then shrink it to its original shape.
    # During the growing stage, close adjacent edge masks will join and merge, forming a solid area,
    # which will remain solid even after the shrinking stage.

    # Mask growing
    shrink = mt_expand_multi(light, mode='ellipse', sw=rx_i, sh=ry_i)

    # At this point, because the mask was made of a shades of grey, we may end up with large areas
    # of dark grey after shrinking.
    # To avoid this, we amplify and saturate the mask here (actually we could even binarize it).
    shrink = shrink.std.Expr(expr='x 4 *')
    if is_float:
        shrink = shrink.std.Limiter()

    # Mask shrinking
    shrink = mt_inpand_multi(shrink, mode='ellipse', sw=rx_i, sh=ry_i)

    # This mask is almost binary, which will produce distinct discontinuities once applied. Then we have to smooth it.
    shrink = shrink.std.Convolution(matrix=[1] * 9).std.Convolution(matrix=[1] * 9)

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure that the main edges are really excluded.
    # We do not want them to be smoothed by the halo removal.
    if excl:
        shr_med = core.std.Expr([strong, shrink], expr='x y max')
    else:
        shr_med = strong

    # Subtracts masks and amplifies the difference to be sure we get 255 on the areas to be processed
    outside = core.std.Expr([large, shr_med], expr='x y - 2 *')
    if is_float:
        outside = outside.std.Limiter()

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        outside = core.std.Expr([outside, strong], expr=f'x y {edgeproc * 0.66} * +')
        if is_float:
            outside = outside.std.Limiter()

    # Smooth again and amplify to grow the mask a bit, otherwise the halo parts sticking to the edges could be missed
    outside = outside.std.Convolution(matrix=[1] * 9).std.Expr(expr='x 2 *')
    if is_float:
        outside = outside.std.Limiter()

    # Masking #

    if showmask <= 0:
        last = core.std.MaskedMerge(src, dehaloed, outside)

    if src_orig is not None:
        if showmask <= 0:
            return core.std.ShufflePlanes([last, src_orig], planes=[0, 1, 2], colorfamily=src_orig.format.color_family)
        elif showmask == 1:
            return outside.resize.Bicubic(format=src_orig.format)
        elif showmask == 2:
            return shrink.resize.Bicubic(format=src_orig.format)
        elif showmask == 3:
            return edges.resize.Bicubic(format=src_orig.format)
        else:
            return strong.resize.Bicubic(format=src_orig.format)
    else:
        if showmask <= 0:
            return last
        elif showmask == 1:
            return outside
        elif showmask == 2:
            return shrink
        elif showmask == 3:
            return edges
        else:
            return strong


def FineDehalo_contrasharp(dehaloed: vs.VideoNode, src: vs.VideoNode, level: float) -> vs.VideoNode:
    """level == 1.0 : normal contrasharp"""
    if not (isinstance(dehaloed, vs.VideoNode) and isinstance(src, vs.VideoNode)):
        raise vs.Error('FineDehalo_contrasharp: this is not a clip')

    if dehaloed.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo_contrasharp: RGB format is not supported')

    if dehaloed.format.id != src.format.id:
        raise vs.Error('FineDehalo_contrasharp: clips must have the same format')

    neutral = 1 << (get_depth(dehaloed) - 1) if dehaloed.format.sample_type == vs.INTEGER else 0.0

    if dehaloed.format.color_family != vs.GRAY:
        dehaloed_orig = dehaloed
        dehaloed = get_y(dehaloed)
        src = get_y(src)
    else:
        dehaloed_orig = None

    bb = dehaloed.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    bb2 = core.rgvs.Repair(bb, core.rgvs.Repair(bb, bb.ctmf.CTMF(radius=2), mode=1), mode=1)
    xd = core.std.MakeDiff(bb, bb2)
    xd = xd.std.Expr(expr=f'x {neutral} - 2.49 * {level} * {neutral} +')
    xdd = core.std.Expr(
        [xd, core.std.MakeDiff(src, dehaloed)], expr=f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} '
                                                     f'- abs y {neutral} - abs < x y ? ?'
    )
    last = core.std.MergeDiff(dehaloed, xdd)

    if dehaloed_orig is not None:
        last = core.std.ShufflePlanes([last, dehaloed_orig], planes=[0, 1, 2],
                                      colorfamily=dehaloed_orig.format.color_family)
    return last


def FineDehalo2(
    src: vs.VideoNode,
    hconv: Sequence[int] = [-1, -2, 0, 0, 40, 0, 0, -2, -1],
    vconv: Sequence[int] = [-2, -1, 0, 0, 40, 0, 0, -1, -2],
    showmask: bool = False
) -> vs.VideoNode:
    """
    Try to remove 2nd order halos.

    Parameters:
        src: Clip to process.

        hconv, vconv: Horizontal and vertical convolutions.

        showmask: Shows mask.
    """

    def grow_mask(mask: vs.VideoNode, coordinates: Sequence[int]) -> vs.VideoNode:
        mask = mask.std.Maximum(coordinates=coordinates).std.Minimum(coordinates=coordinates)
        mask_1 = mask.std.Maximum(coordinates=coordinates)
        mask_2 = mask_1.std.Maximum(coordinates=coordinates).std.Maximum(coordinates=coordinates)
        mask = core.std.Expr([mask_2, mask_1], expr='x y -')
        return mask.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Expr(expr='x 1.8 *')

    if not isinstance(src, vs.VideoNode):
        raise vs.Error('FineDehalo2: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo2: RGB format is not supported')

    is_float = src.format.sample_type == vs.FLOAT

    if src.format.color_family != vs.GRAY:
        src_orig = src
        src = get_y(src)
    else:
        src_orig = None

    fix_h = src.std.Convolution(matrix=vconv, mode='v')
    fix_v = src.std.Convolution(matrix=hconv, mode='h')
    mask_h = src.std.Convolution(matrix=[1, 2, 1, 0, 0, 0, -1, -2, -1], divisor=4, saturate=False)
    mask_v = src.std.Convolution(matrix=[1, 0, -1, 2, 0, -2, 1, 0, -1], divisor=4, saturate=False)
    temp_h = core.std.Expr([mask_h, mask_v], expr='x 3 * y -')
    temp_v = core.std.Expr([mask_v, mask_h], expr='x 3 * y -')
    if is_float:
        temp_h = temp_h.std.Limiter()
        temp_v = temp_v.std.Limiter()
    mask_h = grow_mask(temp_h, [0, 1, 0, 0, 0, 0, 1, 0])
    mask_v = grow_mask(temp_v, [0, 0, 0, 1, 1, 0, 0, 0])
    if is_float:
        mask_h = mask_h.std.Limiter()
        mask_v = mask_v.std.Limiter()

    if not showmask:
        last = core.std.MaskedMerge(src, fix_h, mask_h)
        last = core.std.MaskedMerge(last, fix_v, mask_v)
    else:
        last = core.std.Expr([mask_h, mask_v], expr='x y max')

    if src_orig is not None:
        if not showmask:
            last = core.std.ShufflePlanes([last, src_orig], planes=[0, 1, 2], colorfamily=src_orig.format.color_family)
        else:
            last = last.resize.Bicubic(format=src_orig.format)
    return last


def YAHR(clp: vs.VideoNode, blur: int = 2, depth: int = 32) -> vs.VideoNode:
    """
    Y'et A'nother H'alo R'educing script

    Parameters:
        clp: Clip to process.

        blur: "blur" parameter of AWarpSharp2.

        depth: "depth" parameter of AWarpSharp2.
    """
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('YAHR: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('YAHR: RGB format is not supported')

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = get_y(clp)
    else:
        clp_orig = None

    b1 = MinBlur(clp, 2).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    b1D = core.std.MakeDiff(clp, b1)
    w1 = Padding(clp, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)
    w1b1 = MinBlur(w1, 2).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    w1b1D = core.std.MakeDiff(w1, w1b1)
    DD = core.rgvs.Repair(b1D, w1b1D, mode=13)
    DD2 = core.std.MakeDiff(b1D, DD)
    last = core.std.MakeDiff(clp, DD2)

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last


def HQDeringmod(
    input: vs.VideoNode,
    smoothed: Optional[vs.VideoNode] = None,
    ringmask: Optional[vs.VideoNode] = None,
    mrad: int = 1,
    msmooth: int = 1,
    incedge: bool = False,
    mthr: int = 60,
    minp: int = 1,
    nrmode: Optional[int] = None,
    sigma: float = 128.0,
    sigma2: Optional[float] = None,
    sbsize: Optional[int] = None,
    sosize: Optional[int] = None,
    sharp: int = 1,
    drrep: Optional[int] = None,
    thr: float = 12.0,
    elast: float = 2.0,
    darkthr: Optional[float] = None,
    planes: Union[int, Sequence[int]] = 0,
    show: bool = False,
) -> vs.VideoNode:
    """
    HQDering mod v1.8
    Applies deringing by using a smart smoother near edges (where ringing occurs) only.

    Parameters:
        input: Clip to process.

        mrad: Expanding of edge mask, higher value means more aggressive processing.

        msmooth: Inflate of edge mask, smooth boundaries of mask.

        incedge: Whether to include edge in ring mask, by default ring mask only include area near edges.

        mthr: Threshold of prewitt edge mask, lower value means more aggressive processing.
            But for strong ringing, lower value will treat some ringing as edge,
            which protects this ringing from being processed.

        minp: Inpanding of prewitt edge mask, higher value means more aggressive processing.

        nrmode: Kernel of deringing.
            0 = DFTTest
            1 = MinBlur(r=1)
            2 = MinBlur(r=2)
            3 = MinBlur(r=3)

        sigma: Sigma for medium frequecies in DFTTest.

        sigma2: Sigma for low & high frequecies in DFTTest.

        sbsize: Length of the sides of the spatial window in DFTTest.

        sosize: Spatial overlap amount in DFTTest.

        sharp: Whether to use contra-sharpening to resharp deringed clip, 1-3 represents radius, 0 means no sharpening.

        drrep: Use repair for details retention, recommended values are 24/23/13/12/1.

        thr: The same meaning with "thr" in LimitFilter.

        elast: The same meaning with "elast" in LimitFilter.

        darkthr: Threshold for darker area near edges, by default equals to thr/4.
            Set it lower if you think de-ringing destroys too much lines, etc.
            When "darkthr" is not equal to "thr", "thr" limits darkening while "darkthr" limits brightening.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        show: Whether to output mask clip instead of filtered clip.
    """
    from mvsfunc import LimitFilter

    if not isinstance(input, vs.VideoNode):
        raise vs.Error('HQDeringmod: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('HQDeringmod: RGB format is not supported')

    if smoothed is not None:
        if not isinstance(smoothed, vs.VideoNode):
            raise vs.Error('HQDeringmod: smoothed is not a clip')

        if smoothed.format.id != input.format.id:
            raise vs.Error("HQDeringmod: smoothed must have the same format as input")

    if ringmask is not None and not isinstance(ringmask, vs.VideoNode):
        raise vs.Error("HQDeringmod: ringmask is not a clip")

    is_gray = input.format.color_family == vs.GRAY

    bits = get_depth(input)
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    plane_range = range(input.format.num_planes)

    if isinstance(planes, int):
        planes = [planes]

    HD = input.width > 1024 or input.height > 576

    nrmode = fallback(nrmode, 2 if HD else 1)
    sigma2 = fallback(sigma2, sigma / 16)
    sbsize = fallback(sbsize, 8 if HD else 6)
    sosize = fallback(sosize, 6 if HD else 4)
    drrep = fallback(drrep, 24 if nrmode > 0 else 0)
    darkthr = fallback(darkthr, thr / 4)

    # Kernel: Smoothing
    if smoothed is None:
        if nrmode <= 0:
            smoothed = input.dfttest.DFTTest(
                sbsize=sbsize, sosize=sosize, tbsize=1, slocation=[
                    0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0
                ], planes=planes
            )
        else:
            smoothed = MinBlur(input, nrmode, planes)

    # Post-Process: Contra-Sharpening
    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1] * 9

    if sharp <= 0:
        sclp = smoothed
    else:
        pre = smoothed.std.Median(planes=planes)
        if sharp == 1:
            method = pre.std.Convolution(matrix=matrix1, planes=planes)
        elif sharp == 2:
            method = pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        else:
            method = (
                pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
                .std.Convolution(matrix=matrix2, planes=planes)
            )
        sharpdiff = core.std.MakeDiff(pre, method, planes=planes)
        allD = core.std.MakeDiff(input, smoothed, planes=planes)
        ssDD = core.rgvs.Repair(sharpdiff, allD, mode=[1 if i in planes else 0 for i in plane_range])
        ssDD = core.std.Expr(
            [ssDD, sharpdiff], expr=[f'x {neutral} - abs y {neutral} - abs <= x y ?'
                                     if i in planes else '' for i in plane_range]
        )
        sclp = core.std.MergeDiff(smoothed, ssDD, planes=planes)

    # Post-Process: Repairing
    if drrep <= 0:
        repclp = sclp
    else:
        repclp = core.rgvs.Repair(input, sclp, mode=[drrep if i in planes else 0 for i in plane_range])

    # Post-Process: Limiting
    if (thr <= 0 and darkthr <= 0) or (thr >= 255 and darkthr >= 255):
        limitclp = repclp
    else:
        limitclp = LimitFilter(repclp, input, thr=thr, elast=elast, brighten_thr=darkthr, planes=planes)

    # Post-Process: Ringing Mask Generating
    if ringmask is None:
        expr = f'x {scale_value(mthr, 8, bits)} < 0 x ?'
        prewittm = AvsPrewitt(input, planes=0).std.Expr(expr=expr if is_gray else [expr, ''])
        fmask = core.misc.Hysteresis(prewittm.std.Median(planes=0), prewittm, planes=0)
        if mrad > 0:
            omask = mt_expand_multi(fmask, planes=0, sw=mrad, sh=mrad)
        else:
            omask = fmask
        if msmooth > 0:
            omask = mt_inflate_multi(omask, planes=0, radius=msmooth)
        if incedge:
            ringmask = omask
        else:
            if minp > 3:
                imask = fmask.std.Minimum(planes=0).std.Minimum(planes=0)
            elif minp > 2:
                imask = fmask.std.Inflate(planes=0).std.Minimum(planes=0).std.Minimum(planes=0)
            elif minp > 1:
                imask = fmask.std.Minimum(planes=0)
            elif minp > 0:
                imask = fmask.std.Inflate(planes=0).std.Minimum(planes=0)
            else:
                imask = fmask
            expr = f'x {peak} y - * {peak} /'
            ringmask = core.std.Expr([omask, imask], expr=expr if is_gray else [expr, ''])

    # Mask Merging & Output
    if show:
        if is_gray:
            return ringmask
        else:
            return ringmask.std.Expr(expr=['', repr(neutral)])
    else:
        return core.std.MaskedMerge(input, limitclp, ringmask, planes=planes, first_plane=True)
