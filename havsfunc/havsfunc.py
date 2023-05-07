from __future__ import annotations

import math
from fractions import Fraction
from functools import partial
from typing import Any, Mapping, Optional, Sequence, Union

from vsdenoise import nl_means, prefilter_to_full_range
from vsexprtools import complexpr_available, norm_expr
from vsmasktools import Morpho
from vsrgtools import gauss_blur, min_blur, repair, sbr
from vsrgtools.util import mean_matrix, wmean_matrix
from vstools import (
    DitherType,
    PlanesT,
    change_fps,
    check_ref_clip,
    check_variable,
    core,
    cround,
    depth,
    fallback,
    get_depth,
    join,
    normalize_planes,
    padder,
    plane,
    scale_8bit,
    scale_value,
    vs,
)

__all__ = [
    "average_frames",
    "avs_prewitt",
    "daa",
    "daa3mod",
    "Deblock_QED",
    "FastLineDarkenMOD",
    "FixChromaBleedingMod",
    "GSMC",
    "LSFmod",
    "LUTDeCrawl",
    "LUTDeRainbow",
    "mcdaa3",
    "MCTemporalDenoise",
    "mt_clamp",
    "Overlay",
    "QTGMC",
    "scdetect",
    "smartfademod",
    "SmoothLevels",
    "srestore",
    "Stab",
    "STPresso",
    "Toon",
]


def average_frames(
    clip: vs.VideoNode, weights: float | Sequence[float], scenechange: float | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, average_frames)
    planes = normalize_planes(clip, planes)

    if scenechange:
        clip = scdetect(clip, scenechange)
    return clip.std.AverageFrames(weights=weights, scenechange=scenechange, planes=planes)


def avs_prewitt(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    assert check_variable(clip, avs_prewitt)
    planes = normalize_planes(clip, planes)

    matrices = [
        [1, 1, 0, 1, 0, -1, 0, -1, -1],
        [1, 1, 1, 0, 0, 0, -1, -1, -1],
        [1, 0, -1, 1, 0, -1, 1, 0, -1],
        [0, -1, -1, 1, 0, -1, 1, 1, 0],
    ]
    clips = [clip.std.Convolution(matrix=matrix, planes=planes, saturate=False) for matrix in matrices]
    return norm_expr(clips, "x y max z max a max", planes)


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

    nn = nnedi3(clip, field=3, **kwargs)
    dbl = nn[::2].std.Merge(nn[1::2])
    dblD = clip.std.MakeDiff(dbl)
    shrpD = dbl.std.MakeDiff(dbl.std.Convolution(matrix=mean_matrix if clip.width > 1100 else wmean_matrix))
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


def Deblock_QED(
    clp: vs.VideoNode, quant1: int = 24, quant2: int = 26, aOff1: int = 1, bOff1: int = 2, aOff2: int = 1, bOff2: int = 2, uv: int = 3
) -> vs.VideoNode:
    '''
    A postprocessed Deblock: Uses full frequencies of Deblock's changes on block borders, but DCT-lowpassed changes on block interiours.

    Parameters:
        clp: Clip to process.

        quant1: Strength of block edge deblocking.

        quant2: Strength of block internal deblocking.

        aOff1: Halfway "sensitivity" and halfway a strength modifier for borders.

        bOff1: "Sensitivity to detect blocking" for borders.

        aOff2: Halfway "sensitivity" and halfway a strength modifier for block interiors.

        bOff2: "Sensitivity to detect blocking" for block interiors.

        uv:
            3 = use proposed method for chroma deblocking
            2 = no chroma deblocking at all (fastest method)
            1 = directly use chroma debl. from the normal Deblock()
            -1 = directly use chroma debl. from the strong Deblock()
    '''
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Deblock_QED: this is not a clip')

    is_gray = clp.format.color_family == vs.GRAY
    planes = [0, 1, 2] if uv > 2 and not is_gray else 0

    if clp.format.sample_type == vs.INTEGER:
        bits = get_depth(clp)
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
    else:
        neutral = 0.0
        peak = 1.0

    # add borders if clp is not mod 8
    w = clp.width
    h = clp.height
    padX = 8 - w % 8 if w & 7 else 0
    padY = 8 - h % 8 if h & 7 else 0
    if padX or padY:
        clp = clp.resize.Point(w + padX, h + padY, src_width=w + padX, src_height=h + padY)

    # block
    block = clp.std.BlankClip(width=6, height=6, format=clp.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), length=1, color=0)
    block = block.std.AddBorders(1, 1, 1, 1, color=peak)
    block = core.std.StackHorizontal([block for _ in range(clp.width // 8)])
    block = core.std.StackVertical([block for _ in range(clp.height // 8)])
    if not is_gray:
        blockc = block.std.CropAbs(width=clp.width >> clp.format.subsampling_w, height=clp.height >> clp.format.subsampling_h)
        block = core.std.ShufflePlanes([block, blockc], planes=[0, 0, 0], colorfamily=clp.format.color_family)
    block = block.std.Loop(times=clp.num_frames)

    # create normal deblocking (for block borders) and strong deblocking (for block interiour)
    normal = clp.deblock.Deblock(quant=quant1, aoffset=aOff1, boffset=bOff1, planes=[0, 1, 2] if uv != 2 and not is_gray else 0)
    strong = clp.deblock.Deblock(quant=quant2, aoffset=aOff2, boffset=bOff2, planes=[0, 1, 2] if uv != 2 and not is_gray else 0)

    # build difference maps of both
    normalD = core.std.MakeDiff(clp, normal, planes=planes)
    strongD = core.std.MakeDiff(clp, strong, planes=planes)

    # separate border values of the difference maps, and set the interiours to '128'
    expr = f'y {peak} = x {neutral} ?'
    normalD2 = core.std.Expr([normalD, block], expr=expr if uv > 2 or is_gray else [expr, ''])
    strongD2 = core.std.Expr([strongD, block], expr=expr if uv > 2 or is_gray else [expr, ''])

    # interpolate the border values over the whole block: DCTFilter can do it. (Kiss to Tom Barry!)
    # (Note: this is not fully accurate, but a reasonable approximation.)
    # add borders if clp is not mod 16
    sw = strongD2.width
    sh = strongD2.height
    remX = 16 - sw % 16 if sw & 15 else 0
    remY = 16 - sh % 16 if sh & 15 else 0
    if remX or remY:
        strongD2 = strongD2.resize.Point(sw + remX, sh + remY, src_width=sw + remX, src_height=sh + remY)
    expr = f'x {neutral} - 1.01 * {neutral} +'
    strongD3 = (
        strongD2.std.Expr(expr=expr if uv > 2 or is_gray else [expr, ''])
        .dctf.DCTFilter(factors=[1, 1, 0, 0, 0, 0, 0, 0], planes=planes)
        .std.Crop(right=remX, bottom=remY)
    )

    # apply compensation from "normal" deblocking to the borders of the full-block-compensations calculated from "strong" deblocking ...
    expr = f'y {neutral} = x y ?'
    strongD4 = core.std.Expr([strongD3, normalD2], expr=expr if uv > 2 or is_gray else [expr, ''])

    # ... and apply it.
    deblocked = core.std.MakeDiff(clp, strongD4, planes=planes)

    # simple decisions how to treat chroma
    if not is_gray:
        if uv < 0:
            deblocked = core.std.ShufflePlanes([deblocked, strong], planes=[0, 1, 2], colorfamily=clp.format.color_family)
        elif uv < 2:
            deblocked = core.std.ShufflePlanes([deblocked, normal], planes=[0, 1, 2], colorfamily=clp.format.color_family)

    # remove mod 8 borders
    return deblocked.std.Crop(right=padX, bottom=padY)


##############################
# FastLineDarken 1.4x MT MOD #
##############################
#
# Written by Vectrangle    (http://forum.doom9.org/showthread.php?t=82125)
# Didée: - Speed Boost, Updated: 11th May 2007
# Dogway - added protection option. 12-May-2011
#
# Parameters are:
#  strength (integer)   - Line darkening amount, 0-256. Default 48. Represents the _maximum_ amount
#                         that the luma will be reduced by, weaker lines will be reduced by
#                         proportionately less.
#  protection (integer) - Prevents the darkest lines from being darkened. Protection acts as a threshold.
#                         Values range from 0 (no prot) to ~50 (protect everything)
#  luma_cap (integer)   - value from 0 (black) to 255 (white), used to stop the darkening
#                         determination from being 'blinded' by bright pixels, and to stop grey
#                         lines on white backgrounds being darkened. Any pixels brighter than
#                         luma_cap are treated as only being as bright as luma_cap. Lowering
#                         luma_cap tends to reduce line darkening. 255 disables capping. Default 191.
#  threshold (integer)  - any pixels that were going to be darkened by an amount less than
#                         threshold will not be touched. setting this to 0 will disable it, setting
#                         it to 4 (default) is recommended, since often a lot of random pixels are
#                         marked for very slight darkening and a threshold of about 4 should fix
#                         them. Note if you set threshold too high, some lines will not be darkened
#  thinning (integer)   - optional line thinning amount, 0-256. Setting this to 0 will disable it,
#                         which is gives a _big_ speed increase. Note that thinning the lines will
#                         inherently darken the remaining pixels in each line a little. Default 0.
def FastLineDarkenMOD(c, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FastLineDarkenMOD: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FastLineDarkenMOD: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1 if c.format.sample_type == vs.INTEGER else 1.0

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    ## parameters ##
    Str = strength / 128
    lum = scale_8bit(c, luma_cap)
    thr = scale_8bit(c, threshold)
    thn = thinning / 16

    ## filtering ##
    exin = c.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = core.std.Expr([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +'])
    if thinning <= 0:
        last = thick
    else:
        diff = core.std.Expr([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale_8bit(c, 127)} +'])
        linemask = diff.std.Minimum().std.Expr(expr=[f'x {scale_8bit(c, 127)} - {thn} * {peak} +']).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        thin = core.std.Expr([c.std.Maximum(), diff], expr=[f'x y {scale_8bit(c, 127)} - {Str} 1 + * +'])
        last = core.std.MaskedMerge(thin, thick, linemask)

    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


def FixChromaBleedingMod(input: vs.VideoNode, cx: int = 4, cy: int = 4, thr: float = 4.0, strength: float = 0.8, blur: bool = False) -> vs.VideoNode:
    '''
    FixChromaBleedingMod v1.36
    A script to reduce color bleeding, over-saturation, and color shifting mainly in red and blue areas.

    Parameters:
        input: Clip to process.

        cx: Horizontal chroma shift. Positive value shifts chroma to the left, negative value shifts chroma to the right.

        cy: Vertical chroma shift. Positive value shifts chroma upwards, negative value shifts chroma downwards.

        thr: Masking threshold, higher values treat more areas as color bleed.

        strength: Saturation strength in clip to be merged with the original chroma.
            Values below 1.0 reduce the saturation, a value of 1.0 leaves the saturation intact.

        blur: Set to true to blur the mask clip.
    '''
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


######
###
### GrainStabilizeMC v1.0      by mawen1250      2014.03.22
###
### Requirements: MVTools, RGVS
###
### Temporal-only on-top grain stabilizer
### Only stabilize the difference ( on-top grain ) between source clip and spatial-degrained clip
###
### Parameters:
###  nrmode (int)   - Mode to get grain/noise from input clip. 0: 3x3 Average Blur, 1: 3x3 SBR, 2: 5x5 SBR, 3: 7x7 SBR. Or define your own denoised clip "p". Default is 2 for HD / 1 for SD
###  radius (int)   - Temporal radius of MDegrain for grain stabilize (1-3). Default is 1
###  adapt (int)    - Threshold for luma-adaptative mask. -1: off, 0: source, 255: invert. Or define your own luma mask clip "Lmask". Default is -1
###  rep (int)      - Mode of repair to avoid artifacts, set 0 to turn off this operation. Default is 13
###  planes (int[]) - Whether to process the corresponding plane. The other planes will be passed through unchanged.
###
######
def GSMC(input, p=None, Lmask=None, nrmode=None, radius=1, adapt=-1, rep=13, planes=None, thSAD=300, thSADC=None, thSCD1=300, thSCD2=100, limit=None, limitc=None):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('GSMC: this is not a clip')

    if p is not None and (not isinstance(p, vs.VideoNode) or p.format.id != input.format.id):
        raise vs.Error("GSMC: 'p' must be the same format as input")

    if Lmask is not None and not isinstance(Lmask, vs.VideoNode):
        raise vs.Error("GSMC: 'Lmask' is not a clip")

    neutral = 1 << (input.format.bits_per_sample - 1)
    peak = (1 << input.format.bits_per_sample) - 1

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    HD = input.width > 1024 or input.height > 576

    if nrmode is None:
        nrmode = 2 if HD else 1
    if thSADC is None:
        thSADC = thSAD // 2
    if limit is not None:
        limit = scale_8bit(input, limit)
    if limitc is not None:
        limitc = scale_8bit(input, limitc)

    Y = 0 in planes
    U = 1 in planes
    V = 2 in planes

    chromamv = U or V
    blksize = 32 if HD else 16
    overlap = blksize // 4
    if not Y:
        if not U:
            plane = 2
        elif not V:
            plane = 1
        else:
            plane = 3
    elif not (U or V):
        plane = 0
    else:
        plane = 4

    # Kernel: Spatial Noise Dumping
    if p is not None:
        pre_nr = p
    elif nrmode <= 0:
        pre_nr = input.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=planes)
    else:
        pre_nr = sbr(input, nrmode, planes=planes)
    dif_nr = core.std.MakeDiff(input, pre_nr, planes=planes)

    # Kernel: MC Grain Stabilize
    psuper = prefilter_to_full_range(pre_nr, 2, planes).mv.Super(pel=1, chroma=chromamv)
    difsuper = dif_nr.mv.Super(pel=1, levels=1, chroma=chromamv)

    analyse_args = dict(blksize=blksize, chroma=chromamv, truemotion=False, global_=True, overlap=overlap)
    fv1 = psuper.mv.Analyse(isb=False, delta=1, **analyse_args)
    bv1 = psuper.mv.Analyse(isb=True, delta=1, **analyse_args)
    if radius >= 2:
        fv2 = psuper.mv.Analyse(isb=False, delta=2, **analyse_args)
        bv2 = psuper.mv.Analyse(isb=True, delta=2, **analyse_args)
    if radius >= 3:
        fv3 = psuper.mv.Analyse(isb=False, delta=3, **analyse_args)
        bv3 = psuper.mv.Analyse(isb=True, delta=3, **analyse_args)

    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
    if radius <= 1:
        dif_sb = core.mv.Degrain1(dif_nr, difsuper, bv1, fv1, **degrain_args)
    elif radius == 2:
        dif_sb = core.mv.Degrain2(dif_nr, difsuper, bv1, fv1, bv2, fv2, **degrain_args)
    else:
        dif_sb = core.mv.Degrain3(dif_nr, difsuper, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)

    # Post-Process: Luma-Adaptive Mask Merging & Repairing
    stable = core.std.MergeDiff(pre_nr, dif_sb, planes=planes)
    if rep > 0:
        stable = core.rgvs.Repair(stable, input, mode=[rep if i in planes else 0 for i in range(input.format.num_planes)])

    if Lmask is not None:
        return core.std.MaskedMerge(input, stable, Lmask, planes=planes)
    elif adapt <= -1:
        return stable
    else:
        input_y = plane(input, 0)
        if adapt == 0:
            Lmask = input_y.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        elif adapt >= 255:
            Lmask = input_y.std.Invert().std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        else:
            expr = 'x {adapt} - abs {peak} * {adapt} {neutral} - abs {neutral} + /'.format(adapt=scale_8bit(input, adapt), peak=peak, neutral=neutral)
            Lmask = input_y.std.Expr(expr=[expr]).std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        return core.std.MaskedMerge(input, stable, Lmask, planes=planes)


################################################################################################
###                                                                                          ###
###                       LimitedSharpenFaster MOD : function LSFmod()                       ###
###                                                                                          ###
###                                Modded Version by LaTo INV.                               ###
###                                                                                          ###
###                                  v1.9 - 05 October 2009                                  ###
###                                                                                          ###
################################################################################################
###
### +--------------+
### | DEPENDENCIES |
### +--------------+
###
### -> RGVS
### -> CAS
###
###
###
### +---------+
### | GENERAL |
### +---------+
###
### strength [int]
### --------------
### Strength of the sharpening
###
### Smode [int: 1,2,3]
### ----------------------
### Sharpen mode:
###    =1 : Range sharpening
###    =2 : Nonlinear sharpening (corrected version)
###    =3 : Contrast Adaptive Sharpening
###
### Smethod [int: 1,2,3]
### --------------------
### Sharpen method: (only used in Smode=1,2)
###    =1 : 3x3 kernel
###    =2 : Min/Max
###    =3 : Min/Max + 3x3 kernel
###
### kernel [int: 11,12,19,20]
### -------------------------
### Kernel used in Smethod=1&3
### In strength order: + 19 > 12 >> 20 > 11 -
###
###
###
### +---------+
### | SPECIAL |
### +---------+
###
### preblur [int: 0,1,2,3]
### --------------------------------
### Mode to avoid noise sharpening & ringing:
###    =-1 : No preblur
###    = 0 : MinBlur(0)
###    = 1 : MinBlur(1)
###    = 2 : MinBlur(2)
###    = 3 : DFTTest
###
### secure [bool]
### -------------
### Mode to avoid banding & oil painting (or face wax) effect of sharpening
###
### source [clip]
### -------------
### If source is defined, LSFmod doesn't sharp more a denoised clip than this source clip
### In this mode, you can safely set Lmode=0 & PP=off
###    Usage:   denoised.LSFmod(source=source)
###    Example: last.FFT3DFilter().LSFmod(source=last,Lmode=0,soft=0)
###
###
###
### +----------------------+
### | NONLINEAR SHARPENING |
### +----------------------+
###
### Szrp [int]
### ----------
### Zero Point:
###    - differences below Szrp are amplified (overdrive sharpening)
###    - differences above Szrp are reduced   (reduced sharpening)
###
### Spwr [int]
### ----------
### Power: exponent for sharpener
###
### SdmpLo [int]
### ------------
### Damp Low: reduce sharpening for small changes [0:disable]
###
### SdmpHi [int]
### ------------
### Damp High: reduce sharpening for big changes [0:disable]
###
###
###
### +----------+
### | LIMITING |
### +----------+
###
### Lmode [int: ...,0,1,2,3,4]
### --------------------------
### Limit mode:
###    <0 : Limit with repair (ex: Lmode=-1 --> repair(1), Lmode=-5 --> repair(5)...)
###    =0 : No limit
###    =1 : Limit to over/undershoot
###    =2 : Limit to over/undershoot on edges and no limit on not-edges
###    =3 : Limit to zero on edges and to over/undershoot on not-edges
###    =4 : Limit to over/undershoot on edges and to over/undershoot2 on not-edges
###
### overshoot [int]
### ---------------
### Limit for pixels that get brighter during sharpening
###
### undershoot [int]
### ----------------
### Limit for pixels that get darker during sharpening
###
### overshoot2 [int]
### ----------------
### Same as overshoot, only for Lmode=4
###
### undershoot2 [int]
### -----------------
### Same as undershoot, only for Lmode=4
###
###
###
### +-----------------+
### | POST-PROCESSING |
### +-----------------+
###
### soft [int: -2,-1,0...100]
### -------------------------
### Soft the sharpening effect (-1 = old autocalculate, -2 = new autocalculate)
###
### soothe [bool]
### -------------
###    =True  : Enable soothe temporal stabilization
###    =False : Disable soothe temporal stabilization
###
### keep [int: 0...100]
### -------------------
### Minimum percent of the original sharpening to keep (only with soothe=True)
###
###
###
### +-------+
### | EDGES |
### +-------+
###
### edgemode [int: -1,0,1,2]
### ------------------------
###    =-1 : Show edgemask
###    = 0 : Sharpening all
###    = 1 : Sharpening only edges
###    = 2 : Sharpening only not-edges
###
### edgemaskHQ [bool]
### -----------------
###    =True  : Original edgemask
###    =False : Faster edgemask
###
###
###
### +------------+
### | UPSAMPLING |
### +------------+
###
### ss_x ; ss_y [float]
### -------------------
### Supersampling factor (reduce aliasing on edges)
###
### dest_x ; dest_y [int]
### ---------------------
### Output resolution after sharpening (avoid a resizing step)
###
###
###
### +----------+
### | SETTINGS |
### +----------+
###
### defaults [string: "old" or "slow" or "fast"]
### --------------------------------------------
###    = "old"  : Reset settings to original version (output will be THE SAME AS LSF)
###    = "slow" : Enable SLOW modded version settings
###    = "fast" : Enable FAST modded version settings
###  --> /!\ [default:"fast"]
###
###
### defaults="old" :  - strength    = 100
### ----------------  - Smode       = 1
###                   - Smethod     = Smode==1?2:1
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = false
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 2
###                   - SdmpLo      = strength/25
###                   - SdmpHi      = 0
###
###                   - Lmode       = 1
###                   - overshoot   = 1
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 25
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==1?1.50:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="slow" : - strength    = 100
### ----------------- - Smode       = 2
###                   - Smethod     = 3
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 4
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = -2
###                   - soothe      = true
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==3?1.00:1.50
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="fast" : - strength    = 80
### ----------------- - Smode       = 3
###                   - Smethod     = 2
###                   - kernel      = 11
###
###                   - preblur     = 0
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 0
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = false
###
###                   - ss_x        = Smode==3?1.00:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
################################################################################################
def LSFmod(input, strength=None, Smode=None, Smethod=None, kernel=11, preblur=None, secure=None, source=None, Szrp=16, Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None,
           overshoot2=None, undershoot2=None, soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ss_x=None, ss_y=None, dest_x=None, dest_y=None, defaults='fast'):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('LSFmod: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('LSFmod: RGB format is not supported')

    if source is not None and (not isinstance(source, vs.VideoNode) or source.format.id != input.format.id):
        raise vs.Error("LSFmod: 'source' must be the same format as input")

    isGray = (input.format.color_family == vs.GRAY)
    isInteger = (input.format.sample_type == vs.INTEGER)

    if isInteger:
        neutral = 1 << (input.format.bits_per_sample - 1)
        peak = (1 << input.format.bits_per_sample) - 1
        factor = 1 << (input.format.bits_per_sample - 8)
    else:
        neutral = 0.0
        peak = 1.0
        factor = 255.0

    ### DEFAULTS
    try:
        num = ['old', 'slow', 'fast'].index(defaults.lower())
    except:
        raise vs.Error('LSFmod: defaults must be "old" or "slow" or "fast"')

    ox = input.width
    oy = input.height

    if strength is None:
        strength = [100, 100, 80][num]
    if Smode is None:
        Smode = [1, 2, 3][num]
    if Smethod is None:
        Smethod = [2 if Smode == 1 else 1, 3, 2][num]
    if preblur is None:
        preblur = [-1, -1, 0][num]
    if secure is None:
        secure = [False, True, True][num]
    if Spwr is None:
        Spwr = [2, 4, 4][num]
    if SdmpLo is None:
        SdmpLo = [strength // 25, 4, 4][num]
    if SdmpHi is None:
        SdmpHi = [0, 48, 48][num]
    if Lmode is None:
        Lmode = [1, 4, 0][num]
    if overshoot is None:
        overshoot = [1, strength // 100, strength // 100][num]
    if undershoot is None:
        undershoot = overshoot
    if overshoot2 is None:
        overshoot2 = overshoot * 2
    if undershoot2 is None:
        undershoot2 = overshoot2
    if soft is None:
        soft = [0, -2, 0][num]
    if soothe is None:
        soothe = [False, True, False][num]
    if keep is None:
        keep = [25, 20, 20][num]
    if edgemaskHQ is None:
        edgemaskHQ = [True, True, False][num]
    if ss_x is None:
        ss_x = [1.5 if Smode == 1 else 1.25, 1.0 if Smode == 3 else 1.5, 1.0 if Smode == 3 else 1.25][num]
    if ss_y is None:
        ss_y = ss_x
    if dest_x is None:
        dest_x = ox
    if dest_y is None:
        dest_y = oy

    if kernel == 4:
        RemoveGrain = partial(core.std.Median)
    elif kernel in [11, 12]:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif kernel == 19:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif kernel == 20:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        RemoveGrain = partial(core.rgvs.RemoveGrain, mode=[kernel])

    if soft == -1:
        soft = math.sqrt(((ss_x + ss_y) / 2 - 1) * 100) * 10
    elif soft <= -2:
        soft = int((1 + 2 / (ss_x + ss_y)) * math.sqrt(strength))
    soft = min(soft, 100)

    xxs = cround(ox * ss_x / 8) * 8
    yys = cround(oy * ss_y / 8) * 8

    Str = strength / 100

    ### SHARP
    if ss_x > 1 or ss_y > 1:
        tmp = input.resize.Spline36(xxs, yys)
    else:
        tmp = input

    if not isGray:
        tmp_orig = tmp
        tmp = plane(tmp, 0)

    if preblur <= -1:
        pre = tmp
    elif preblur >= 3:
        expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale_8bit(input, 16), j=scale_8bit(input, 75), peak=peak)
        pre = core.std.MaskedMerge(tmp.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0]), tmp, tmp.std.Expr(expr=[expr]))
    else:
        pre = min_blur(tmp, preblur)

    dark_limit = pre.std.Minimum()
    bright_limit = pre.std.Maximum()

    if Smode < 3:
        if Smethod <= 1:
            method = RemoveGrain(pre)
        elif Smethod == 2:
            method = core.std.Merge(dark_limit, bright_limit)
        else:
            method = RemoveGrain(core.std.Merge(dark_limit, bright_limit))

        if secure:
            method = core.std.Expr([method, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale_8bit(input, 1))])

        if preblur > -1:
            method = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, method))

        if Smode <= 1:
            normsharp = core.std.Expr([tmp, method], expr=[f'x x y - {Str} * +'])
        else:
            tmpScaled = tmp.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=tmp.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            methodScaled = method.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=method.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            expr = f'x y = x x x y - abs {Szrp} / {1 / Spwr} pow {Szrp} * {Str} * x y - dup abs / * x y - dup * {Szrp * Szrp} {SdmpLo} + * x y - dup * {SdmpLo} + {Szrp * Szrp} * / * 1 {SdmpHi} 0 = 0 {(Szrp / SdmpHi) ** 4} ? + 1 {SdmpHi} 0 = 0 x y - abs {SdmpHi} / 4 pow ? + / * + ? {factor if isInteger else 1 / factor} *'
            normsharp = core.std.Expr([tmpScaled, methodScaled], expr=[expr], format=tmp.format)
    else:
        normsharp = pre.cas.CAS(sharpness=min(Str, 1))

        if secure:
            normsharp = core.std.Expr([normsharp, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale_8bit(input, 1))])

        if preblur > -1:
            normsharp = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, normsharp))

    ### LIMIT
    normal = mt_clamp(normsharp, bright_limit, dark_limit, scale_8bit(input, overshoot), scale_8bit(input, undershoot))
    second = mt_clamp(normsharp, bright_limit, dark_limit, scale_8bit(input, overshoot2), scale_8bit(input, undershoot2))
    zero = mt_clamp(normsharp, bright_limit, dark_limit, 0, 0)

    if edgemaskHQ:
        edge = tmp.std.Sobel(scale=2)
    else:
        edge = core.std.Expr([tmp.std.Maximum(), tmp.std.Minimum()], expr=['x y -'])
    edge = edge.std.Expr(expr=[f'x {1 / factor if isInteger else factor} * {128 if edgemaskHQ else 32} / 0.86 pow 255 * {factor if isInteger else 1 / factor} *'])

    if Lmode < 0:
        limit1 = core.rgvs.Repair(normsharp, tmp, mode=[abs(Lmode)])
    elif Lmode == 0:
        limit1 = normsharp
    elif Lmode == 1:
        limit1 = normal
    elif Lmode == 2:
        limit1 = core.std.MaskedMerge(normsharp, normal, edge.std.Inflate())
    elif Lmode == 3:
        limit1 = core.std.MaskedMerge(normal, zero, edge.std.Inflate())
    else:
        limit1 = core.std.MaskedMerge(second, normal, edge.std.Inflate())

    if edgemode <= 0:
        limit2 = limit1
    elif edgemode == 1:
        limit2 = core.std.MaskedMerge(tmp, limit1, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    else:
        limit2 = core.std.MaskedMerge(limit1, tmp, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))

    ### SOFT
    if soft == 0:
        PP1 = limit2
    else:
        sharpdiff = core.std.MakeDiff(tmp, limit2)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])], expr=[f'x {neutral} - abs y {neutral} - abs > y {soft} * x {100 - soft} * + 100 / x ?'])
        PP1 = core.std.MakeDiff(tmp, sharpdiff)

    ### SOOTHE
    if soothe:
        diff = core.std.MakeDiff(tmp, PP1)
        diff = core.std.Expr([diff, average_frames(diff, weights=[1] * 3, scenechange=32 / 255)],
                             expr=[f'x {neutral} - y {neutral} - * 0 < x {neutral} - 100 / {keep} * {neutral} + x {neutral} - abs y {neutral} - abs > x {keep} * y {100 - keep} * + 100 / x ? ?'])
        PP2 = core.std.MakeDiff(tmp, diff)
    else:
        PP2 = PP1

    ### OUTPUT
    if dest_x != ox or dest_y != oy:
        if not isGray:
            PP2 = core.std.ShufflePlanes([PP2, tmp_orig], planes=[0, 1, 2], colorfamily=input.format.color_family)
        out = PP2.resize.Spline36(dest_x, dest_y)
    elif ss_x > 1 or ss_y > 1:
        out = PP2.resize.Spline36(dest_x, dest_y)
        if not isGray:
            out = core.std.ShufflePlanes([out, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    elif not isGray:
        out = core.std.ShufflePlanes([PP2, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    else:
        out = PP2

    if edgemode <= -1:
        return edge.resize.Spline36(dest_x, dest_y, format=input.format)
    elif source is not None:
        if dest_x != ox or dest_y != oy:
            src = source.resize.Spline36(dest_x, dest_y)
            In = input.resize.Spline36(dest_x, dest_y)
        else:
            src = source
            In = input

        shrpD = core.std.MakeDiff(In, out, planes=[0])
        expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
        shrpL = core.std.Expr([core.rgvs.Repair(shrpD, core.std.MakeDiff(In, src, planes=[0]), mode=[1] if isGray else [1, 0]), shrpD], expr=[expr] if isGray else [expr, ''])
        return core.std.MakeDiff(In, shrpL, planes=[0])
    else:
        return out


########################################################
#                                                      #
# LUTDeCrawl, a dot crawl removal script by Scintilla  #
# Created 10/3/08                                      #
# Last updated 10/3/08                                 #
#                                                      #
########################################################
#
# Requires YUV input, frame-based only.
# Is of average speed (faster than VagueDenoiser, slower than HQDN3D).
# Suggestions for improvement welcome: scintilla@aquilinestudios.org
#
# Arguments:
#
# ythresh (int, default=10) - This determines how close the luma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Higher values (within reason) should catch more dot crawl,
#   but may introduce unwanted artifacts.  Probably shouldn't be set
#   above 20 or so.
#
# cthresh (int, default=10) - This determines how close the chroma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Just as with ythresh.
#
# maxdiff (int, default=50) - This is the maximum difference allowed between the
#   luma values of the pixel in the CURRENT frame and in each of its
#   neighbour frames (so, the upper limit to what fluctuations are
#   considered dot crawl).  Lower values will reduce artifacts but may
#   cause the filter to miss some dot crawl.  Obviously, this should
#   never be lower than ythresh.  Meaningless if usemaxdiff = false.
#
# scnchg (int, default=25) - Scene change detection threshold.  Any frame with
#   total luma difference between it and the previous/next frame greater
#   than this value will not be processed.
#
# usemaxdiff (bool, default=True) - Whether or not to reject luma fluctuations
#   higher than maxdiff.  Setting this to false is not recommended, as
#   it may introduce artifacts; but on the other hand, it produces a
#   30% speed boost.  Test on your particular source.
#
# mask (bool, default=False) - When set true, the function will return the mask
#   instead of the image.  Use to find the best values of cthresh,
#   ythresh, and maxdiff.
#   (The scene change threshold, scnchg, is not reflected in the mask.)
#
###################
def LUTDeCrawl(input, ythresh=10, cthresh=10, maxdiff=50, scnchg=25, usemaxdiff=True, mask=False):
    def YDifferenceFromPrevious(n, f, clips):
        if f.props['_SceneChangePrev']:
            return clips[0]
        else:
            return clips[1]

    def YDifferenceToNext(n, f, clips):
        if f.props['_SceneChangeNext']:
            return clips[0]
        else:
            return clips[1]

    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or input.format.bits_per_sample > 10:
        raise vs.Error('LUTDeCrawl: This is not an 8-10 bit YUV clip')

    shift = input.format.bits_per_sample - 8
    peak = (1 << input.format.bits_per_sample) - 1

    ythresh = scale_8bit(input, ythresh)
    cthresh = scale_8bit(input, cthresh)
    maxdiff = scale_8bit(input, maxdiff)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_y = plane(input, 0)
    input_minus_y = plane(input_minus, 0)
    input_minus_u = plane(input_minus, 1)
    input_minus_v = plane(input_minus, 2)
    input_plus_y = plane(input_plus, 0)
    input_plus_u = plane(input_plus, 1)
    input_plus_v = plane(input_plus, 2)

    average_y = core.std.Expr([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < x y + 2 / 0 ?'])
    average_u = core.std.Expr([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])
    average_v = core.std.Expr([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])

    ymask = average_y.std.Binarize(threshold=1 << shift)
    if usemaxdiff:
        diffplus_y = core.std.Expr([input_plus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffminus_y = core.std.Expr([input_minus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffs_y = core.std.Lut2(diffplus_y, diffminus_y, function=lambda x, y: x & y)
        ymask = core.std.Lut2(ymask, diffs_y, function=lambda x, y: x & y)
    cmask = core.std.Lut2(average_u.std.Binarize(threshold=129 << shift), average_v.std.Binarize(threshold=129 << shift), function=lambda x, y: x & y)
    cmask = cmask.resize.Point(input.width, input.height)

    themask = core.std.Lut2(ymask, cmask, function=lambda x, y: x & y)

    fixed_y = core.std.Merge(average_y, input_y)

    output = core.std.ShufflePlanes([core.std.MaskedMerge(input_y, fixed_y, themask), input], planes=[0, 1, 2], colorfamily=input.format.color_family)

    input = scdetect(input, scnchg / 255)
    output = output.std.FrameEval(eval=partial(YDifferenceFromPrevious, clips=[input, output]), prop_src=input)
    output = output.std.FrameEval(eval=partial(YDifferenceToNext, clips=[input, output]), prop_src=input)

    if mask:
        return themask
    else:
        return output


#####################################################
#                                                   #
# LUTDeRainbow, a derainbowing script by Scintilla  #
# Last updated 2022-10-08                           #
#                                                   #
#####################################################
#
# Requires YUV input, frame-based only.
# Is of reasonable speed (faster than aWarpSharp, slower than DeGrainMedian).
# Suggestions for improvement welcome: scintilla@aquilinestudios.org
#
# Arguments:
#
# cthresh (int, default=10) - This determines how close the chroma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Higher values (within reason) should catch more rainbows,
#   but may introduce unwanted artifacts.  Probably shouldn't be set
#   above 20 or so.
#
# ythresh (int, default=10) - If the y parameter is set true, then this
#   determines how close the luma values of the pixel in the previous
#   and next frames have to be for the pixel to be hit.  Just as with
#   cthresh.
#
# y (bool, default=True) - Determines whether luma difference will be considered
#   in determining which pixels to hit and which to leave alone.
#
# linkUV (bool, default=True) - Determines whether both chroma channels are
#   considered in determining which pixels in each channel to hit.
#   When set true, only pixels that meet the thresholds for both U and
#   V will be hit; when set false, the U and V channels are masked
#   separately (so a pixel could have its U hit but not its V, or vice
#   versa).
#
# mask (bool, default=False) - When set true, the function will return the mask
#   (for combined U/V) instead of the image.  Formerly used to find the
#   best values of cthresh and ythresh.  If linkUV=false, then this
#   mask won't actually be used anyway (because each chroma channel
#   will have its own mask).
#
###################
def LUTDeRainbow(input, cthresh=10, ythresh=10, y=True, linkUV=True, mask=False):
    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or input.format.bits_per_sample > 16:
        raise vs.Error('LUTDeRainbow: This is not an 8-16 bit YUV clip')

    # Since LUT2 can't handle clips with more than 10 bits, we default to using
    # Expr and MaskedMerge to handle the same logic for higher bit depths.
    useExpr = input.format.bits_per_sample > 10

    shift = input.format.bits_per_sample - 8
    peak = (1 << input.format.bits_per_sample) - 1

    cthresh = scale_8bit(input, cthresh)
    ythresh = scale_8bit(input, ythresh)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_u = plane(input, 1)
    input_v = plane(input, 2)
    input_minus_y = plane(input_minus, 0)
    input_minus_u = plane(input_minus, 1)
    input_minus_v = plane(input_minus, 2)
    input_plus_y = plane(input_plus, 0)
    input_plus_u = plane(input_plus, 1)
    input_plus_v = plane(input_plus, 2)

    average_y = core.std.Expr([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < {peak} 0 ?']).resize.Bilinear(input_u.width, input_u.height)
    average_u = core.std.Expr([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])
    average_v = core.std.Expr([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])

    umask = average_u.std.Binarize(threshold=21 << shift)
    vmask = average_v.std.Binarize(threshold=21 << shift)

    if useExpr:
        themask = core.std.Expr([umask, vmask], expr=[f'x y + {peak + 1} < 0 {peak} ?'])
        if y:
            umask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, umask)
            vmask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, vmask)
            themask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, themask)
    else:
        themask = core.std.Lut2(umask, vmask, function=lambda x, y: x & y)
        if y:
            umask = core.std.Lut2(umask, average_y, function=lambda x, y: x & y)
            vmask = core.std.Lut2(vmask, average_y, function=lambda x, y: x & y)
            themask = core.std.Lut2(themask, average_y, function=lambda x, y: x & y)

    fixed_u = core.std.Merge(average_u, input_u)
    fixed_v = core.std.Merge(average_v, input_v)

    output_u = core.std.MaskedMerge(input_u, fixed_u, themask if linkUV else umask)
    output_v = core.std.MaskedMerge(input_v, fixed_v, themask if linkUV else vmask)

    output = core.std.ShufflePlanes([input, output_u, output_v], planes=[0, 0, 0], colorfamily=input.format.color_family)

    if mask:
        return themask.resize.Point(input.width, input.height)
    else:
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


####################################################################################################################################
###                                                                                                                              ###
###                                   Motion-Compensated Temporal Denoise: MCTemporalDenoise()                                   ###
###                                                                                                                              ###
###                                                     v1.4.20 by "LaTo INV."                                                   ###
###                                                                                                                              ###
###                                                           2 July 2010                                                        ###
###                                                                                                                              ###
####################################################################################################################################
###
###
###
### /!\ Needed filters: MVTools, DFTTest, FFT3DFilter, TTempSmooth, RGVS, Deblock, DCTFilter
### -------------------
###
###
###
### USAGE: MCTemporalDenoise(i, radius, pfMode, sigma, twopass, useTTmpSm, limit, limit2, post, chroma, refine,
###                          deblock, useQED, quant1, quant2,
###                          edgeclean, ECrad, ECthr,
###                          stabilize, maxr, TTstr,
###                          bwbh, owoh, blksize, overlap,
###                          bt, ncpu,
###                          thSAD, thSADC, thSAD2, thSADC2, thSCD1, thSCD2,
###                          truemotion, MVglobal, pel, pelsearch, search, searchparam, MVsharp, DCT,
###                          p, settings)
###
###
###
### PARAMETERS:
### -----------
###
### +---------+
### | DENOISE |
### +---------+--------------------------------------------------------------------------------------+
### | radius    : Temporal radius [1...6]                                                            |
### | pfMode    : Pre-filter mode [-1=off,0=FFT3DFilter,1=MinBlur(1),2=MinBlur(2),3=DFTTest]         |
### | sigma     : FFT3D sigma for the pre-filtering clip (if pfMode=0)                               |
### | twopass   : Do the denoising job in 2 stages (stronger but very slow)                          |
### | useTTmpSm : Use MDegrain (faster) or MCompensate+TTempSmooth (stronger)                        |
### | limit     : Limit the effect of the first denoising [-1=auto,0=off,1...255]                    |
### | limit2    : Limit the effect of the second denoising (if twopass=true) [-1=auto,0=off,1...255] |
### | post      : Sigma value for post-denoising with FFT3D [0=off,...]                              |
### | chroma    : Process or not the chroma plane                                                    |
### | refine    : Refine and recalculate motion data of previously estimated motion vectors          |
### +------------------------------------------------------------------------------------------------+
###
###
### +---------+
### | DEBLOCK |
### +---------+-----------------------------------------------------------------------------------+
### | deblock : Enable deblocking before the denoising                                            |
### | useQED  : If true, use Deblock_QED, else use Deblock (faster & stronger)                    |
### | quant1  : Deblock_QED "quant1" parameter (Deblock "quant" parameter is "(quant1+quant2)/2") |
### | quant2  : Deblock_QED "quant2" parameter (Deblock "quant" parameter is "(quant1+quant2)/2") |
### +---------------------------------------------------------------------------------------------+
###
###
### +------------------------------+
### | EDGECLEAN: DERING, DEHALO... |
### +------------------------------+-----------------------------------------------------------------------------------------------------+
### | edgeclean : Enable safe edgeclean process after the denoising (only on edges which are in non-detailed areas, so less detail loss) |
### | ECrad     : Radius for mask (the higher, the greater distance from the edge is filtered)                                           |
### | ECthr     : Threshold for mask (the higher, the less "small edges" are process) [0...255]                                          |
### +------------------------------------------------------------------------------------------------------------------------------------+
###
###
### +-----------+
### | STABILIZE |
### +-----------+------------------------------------------------------------------------------------------------+
### | stabilize : Enable TTempSmooth post processing to stabilize flat areas (background will be less "nervous") |
### | maxr      : Temporal radius (the higher, the more stable image)                                            |
### | TTstr     : Strength (see TTempSmooth docs)                                                                |
### +------------------------------------------------------------------------------------------------------------+
###
###
### +---------------------+
### | BLOCKSIZE / OVERLAP |
### +---------------------+----------------+
### | bwbh    : FFT3D blocksize            |
### | owoh    : FFT3D overlap              |
### |             - for speed:   bwbh/4    |
### |             - for quality: bwbh/2    |
### | blksize : MVTools blocksize          |
### | overlap : MVTools overlap            |
### |             - for speed:   blksize/4 |
### |             - for quality: blksize/2 |
### +--------------------------------------+
###
###
### +-------+
### | FFT3D |
### +-------+--------------------------+
### | bt   : FFT3D block temporal size |
### | ncpu : FFT3DFilter ncpu          |
### +----------------------------------+
###
###
### +---------+
### | MVTOOLS |
### +---------+------------------------------------------------------+
### | thSAD   : MVTools thSAD for the first pass                     |
### | thSADC  : MVTools thSADC for the first pass                    |
### | thSAD2  : MVTools thSAD for the second pass (if twopass=true)  |
### | thSADC2 : MVTools thSADC for the second pass (if twopass=true) |
### | thSCD1  : MVTools thSCD1                                       |
### | thSCD2  : MVTools thSCD2                                       |
### +-----------------------------------+----------------------------+
### | truemotion  : MVTools truemotion  |
### | MVglobal    : MVTools global      |
### | pel         : MVTools pel         |
### | pelsearch   : MVTools pelsearch   |
### | search      : MVTools search      |
### | searchparam : MVTools searchparam |
### | MVsharp     : MVTools sharp       |
### | DCT         : MVTools DCT         |
### +-----------------------------------+
###
###
### +--------+
### | GLOBAL |
### +--------+-----------------------------------------------------+
### | p        : Set an external prefilter clip                    |
### | settings : Global MCTemporalDenoise settings [default="low"] |
### |             - "very low"                                     |
### |             - "low"                                          |
### |             - "medium"                                       |
### |             - "high"                                         |
### |             - "very high"                                    |
### +--------------------------------------------------------------+
###
###
###
### DEFAULTS:
### ---------
###
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+
### | SETTINGS    |      VERY LOW        |      LOW             |      MEDIUM          |      HIGH            |      VERY HIGH       |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | radius      |      1               |      2               |      3               |      2               |      3               |
### | pfMode      |      3               |      3               |      3               |      3               |      3               |
### | sigma       |      2               |      4               |      8               |      12              |      16              |
### | twopass     |      false           |      false           |      false           |      true            |      true            |
### | useTTmpSm   |      false           |      false           |      false           |      false           |      false           |
### | limit       |      -1              |      -1              |      -1              |      -1              |      0               |
### | limit2      |      -1              |      -1              |      -1              |      0               |      0               |
### | post        |      0               |      0               |      0               |      0               |      0               |
### | chroma      |      false           |      false           |      true            |      true            |      true            |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | deblock     |      false           |      false           |      false           |      false           |      false           |
### | useQED      |      true            |      true            |      true            |      false           |      false           |
### | quant1      |      10              |      20              |      30              |      30              |      40              |
### | quant2      |      20              |      40              |      60              |      60              |      80              |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | edgeclean   |      false           |      false           |      false           |      false           |      false           |
### | ECrad       |      1               |      2               |      3               |      4               |      5               |
### | ECthr       |      64              |      32              |      32              |      16              |      16              |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | stabilize   |      false           |      false           |      false           |      true            |      true            |
### | maxr        |      1               |      1               |      2               |      2               |      2               |
### | TTstr       |      1               |      1               |      1               |      2               |      2               |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | bwbh        |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |
### | owoh        |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |
### | blksize     |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |
### | overlap     |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | bt          |      1               |      3               |      3               |      3               |      4               |
### | ncpu        |      1               |      1               |      1               |      1               |      1               |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | thSAD       |      200             |      300             |      400             |      500             |      600             |
### | thSADC      |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |
### | thSAD2      |      200             |      300             |      400             |      500             |      600             |
### | thSADC2     |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |
### | thSCD1      |      200             |      300             |      400             |      500             |      600             |
### | thSCD2      |      90              |      100             |      100             |      130             |      130             |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | truemotion  |      false           |      false           |      false           |      false           |      false           |
### | MVglobal    |      true            |      true            |      true            |      true            |      true            |
### | pel         |      1               |      2               |      2               |      2               |      2               |
### | pelsearch   |      1               |      2               |      2               |      2               |      2               |
### | search      |      4               |      4               |      4               |      4               |      4               |
### | searchparam |      2               |      2               |      2               |      2               |      2               |
### | MVsharp     |      2               |      2               |      2               |      1               |      0               |
### | DCT         |      0               |      0               |      0               |      0               |      0               |
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+
###
####################################################################################################################################
def MCTemporalDenoise(i, radius=None, pfMode=3, sigma=None, twopass=None, useTTmpSm=False, limit=None, limit2=None, post=0, chroma=None, refine=False, deblock=False, useQED=None, quant1=None,
                      quant2=None, edgeclean=False, ECrad=None, ECthr=None, stabilize=None, maxr=None, TTstr=None, bwbh=None, owoh=None, blksize=None, overlap=None, bt=None, ncpu=1, thSAD=None,
                      thSADC=None, thSAD2=None, thSADC2=None, thSCD1=None, thSCD2=None, truemotion=False, MVglobal=True, pel=None, pelsearch=None, search=4, searchparam=2, MVsharp=None, DCT=0, p=None,
                      settings='low'):
    if not isinstance(i, vs.VideoNode):
        raise vs.Error('MCTemporalDenoise: this is not a clip')

    if p is not None and (not isinstance(p, vs.VideoNode) or p.format.id != i.format.id):
        raise vs.Error("MCTemporalDenoise: 'p' must be the same format as input")

    isGray = (i.format.color_family == vs.GRAY)

    neutral = 1 << (i.format.bits_per_sample - 1)
    peak = (1 << i.format.bits_per_sample) - 1

    ### DEFAULTS
    try:
        settings_num = ['very low', 'low', 'medium', 'high', 'very high'].index(settings.lower())
    except:
        raise vs.Error('MCTemporalDenoise: these settings do not exist')

    HD = i.width > 1024 or i.height > 576

    if radius is None:
        radius = [1, 2, 3, 2, 3][settings_num]
    if sigma is None:
        sigma = [2, 4, 8, 12, 16][settings_num]
    if twopass is None:
        twopass = [False, False, False, True, True][settings_num]
    if limit is None:
        limit = [-1, -1, -1, -1, 0][settings_num]
    if limit2 is None:
        limit2 = [-1, -1, -1, 0, 0][settings_num]
    if chroma is None:
        chroma = [False, False, True, True, True][settings_num]
    if useQED is None:
        useQED = [True, True, True, False, False][settings_num]
    if quant1 is None:
        quant1 = [10, 20, 30, 30, 40][settings_num]
    if quant2 is None:
        quant2 = [20, 40, 60, 60, 80][settings_num]
    if ECrad is None:
        ECrad = [1, 2, 3, 4, 5][settings_num]
    if ECthr is None:
        ECthr = [64, 32, 32, 16, 16][settings_num]
    if stabilize is None:
        stabilize = [False, False, False, True, True][settings_num]
    if maxr is None:
        maxr = [1, 1, 2, 2, 2][settings_num]
    if TTstr is None:
        TTstr = [1, 1, 1, 2, 2][settings_num]
    if bwbh is None:
        bwbh = 16 if HD else 8
    if owoh is None:
        owoh = 8 if HD else 4
    if blksize is None:
        blksize = 16 if HD else 8
    if overlap is None:
        overlap = 8 if HD else 4
    if bt is None:
        bt = [1, 3, 3, 3, 4][settings_num]
    if thSAD is None:
        thSAD = [200, 300, 400, 500, 600][settings_num]
    if thSADC is None:
        thSADC = thSAD // 2
    if thSAD2 is None:
        thSAD2 = [200, 300, 400, 500, 600][settings_num]
    if thSADC2 is None:
        thSADC2 = thSAD2 // 2
    if thSCD1 is None:
        thSCD1 = [200, 300, 400, 500, 600][settings_num]
    if thSCD2 is None:
        thSCD2 = [90, 100, 100, 130, 130][settings_num]
    if pel is None:
        pel = [1, 2, 2, 2, 2][settings_num]
    if pelsearch is None:
        pelsearch = [1, 2, 2, 2, 2][settings_num]
    if MVsharp is None:
        MVsharp = [2, 2, 2, 1, 0][settings_num]

    sigma *= peak / 255
    limit = scale_8bit(i, limit)
    limit2 = scale_8bit(i, limit2)
    post *= peak / 255
    ECthr = scale_8bit(i, ECthr)
    planes = [0, 1, 2] if chroma and not isGray else [0]

    ### INPUT
    mod = bwbh if bwbh >= blksize else blksize
    xi = i.width
    xf = math.ceil(xi / mod) * mod - xi + mod
    xn = int(xi + xf)
    yi = i.height
    yf = math.ceil(yi / mod) * mod - yi + mod
    yn = int(yi + yf)

    pointresize_args = dict(width=xn, height=yn, src_left=-xf / 2, src_top=-yf / 2, src_width=xn, src_height=yn)
    i = i.resize.Point(**pointresize_args)

    ### PREFILTERING
    fft3d_args = dict(planes=planes, bw=bwbh, bh=bwbh, bt=bt, ow=owoh, oh=owoh, ncpu=ncpu)
    if p is not None:
        p = p.resize.Point(**pointresize_args)
    elif pfMode <= -1:
        p = i
    elif pfMode == 0:
        p = i.fft3dfilter.FFT3DFilter(sigma=sigma * 0.8, sigma2=sigma * 0.6, sigma3=sigma * 0.4, sigma4=sigma * 0.2, **fft3d_args)
    elif pfMode >= 3:
        p = i.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes)
    else:
        p = min_blur(i, pfMode, planes)

    pD = core.std.MakeDiff(i, p, planes=planes)
    p = prefilter_to_full_range(p, 2, planes)

    ### DEBLOCKING
    crop_args = dict(left=xf // 2, right=xf // 2, top=yf // 2, bottom=yf // 2)
    if not deblock:
        d = i
    elif useQED:
        d = Deblock_QED(i.std.Crop(**crop_args), quant1=quant1, quant2=quant2, uv=3 if chroma else 2).resize.Point(**pointresize_args)
    else:
        d = i.std.Crop(**crop_args).deblock.Deblock(quant=(quant1 + quant2) // 2, planes=planes).resize.Point(**pointresize_args)

    ### PREPARING
    super_args = dict(hpad=0, vpad=0, pel=pel, chroma=chroma, sharp=MVsharp)
    pMVS = p.mv.Super(rfilter=4 if refine else 2, **super_args)
    if refine:
        rMVS = p.mv.Super(levels=1, **super_args)

    analyse_args = dict(blksize=blksize, search=search, searchparam=searchparam, pelsearch=pelsearch, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=DCT)
    recalculate_args = dict(thsad=thSAD // 2, blksize=max(blksize // 2, 4), search=search, chroma=chroma, truemotion=truemotion, overlap=max(overlap // 2, 2), dct=DCT)
    f1v = pMVS.mv.Analyse(isb=False, delta=1, **analyse_args)
    b1v = pMVS.mv.Analyse(isb=True, delta=1, **analyse_args)
    if refine:
        f1v = core.mv.Recalculate(rMVS, f1v, **recalculate_args)
        b1v = core.mv.Recalculate(rMVS, b1v, **recalculate_args)
    if radius > 1:
        f2v = pMVS.mv.Analyse(isb=False, delta=2, **analyse_args)
        b2v = pMVS.mv.Analyse(isb=True, delta=2, **analyse_args)
        if refine:
            f2v = core.mv.Recalculate(rMVS, f2v, **recalculate_args)
            b2v = core.mv.Recalculate(rMVS, b2v, **recalculate_args)
    if radius > 2:
        f3v = pMVS.mv.Analyse(isb=False, delta=3, **analyse_args)
        b3v = pMVS.mv.Analyse(isb=True, delta=3, **analyse_args)
        if refine:
            f3v = core.mv.Recalculate(rMVS, f3v, **recalculate_args)
            b3v = core.mv.Recalculate(rMVS, b3v, **recalculate_args)
    if radius > 3:
        f4v = pMVS.mv.Analyse(isb=False, delta=4, **analyse_args)
        b4v = pMVS.mv.Analyse(isb=True, delta=4, **analyse_args)
        if refine:
            f4v = core.mv.Recalculate(rMVS, f4v, **recalculate_args)
            b4v = core.mv.Recalculate(rMVS, b4v, **recalculate_args)
    if radius > 4:
        f5v = pMVS.mv.Analyse(isb=False, delta=5, **analyse_args)
        b5v = pMVS.mv.Analyse(isb=True, delta=5, **analyse_args)
        if refine:
            f5v = core.mv.Recalculate(rMVS, f5v, **recalculate_args)
            b5v = core.mv.Recalculate(rMVS, b5v, **recalculate_args)
    if radius > 5:
        f6v = pMVS.mv.Analyse(isb=False, delta=6, **analyse_args)
        b6v = pMVS.mv.Analyse(isb=True, delta=6, **analyse_args)
        if refine:
            f6v = core.mv.Recalculate(rMVS, f6v, **recalculate_args)
            b6v = core.mv.Recalculate(rMVS, b6v, **recalculate_args)

    # if useTTmpSm or stabilize:
        # mask_args = dict(ml=thSAD, gamma=0.999, kind=1, ysc=255)
        # SAD_f1m = core.mv.Mask(d, f1v, **mask_args)
        # SAD_b1m = core.mv.Mask(d, b1v, **mask_args)

    def MCTD_MVD(i, iMVS, thSAD, thSADC):
        degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=4 if chroma else 0, thscd1=thSCD1, thscd2=thSCD2)
        if radius <= 1:
            sm = core.mv.Degrain1(i, iMVS, b1v, f1v, **degrain_args)
        elif radius == 2:
            sm = core.mv.Degrain2(i, iMVS, b1v, f1v, b2v, f2v, **degrain_args)
        elif radius == 3:
            sm = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
        elif radius == 4:
            mv12 = core.mv.Degrain2(i, iMVS, b1v, f1v, b2v, f2v, **degrain_args)
            mv34 = core.mv.Degrain2(i, iMVS, b3v, f3v, b4v, f4v, **degrain_args)
            sm = core.std.Merge(mv12, mv34, weight=[0.4444])
        elif radius == 5:
            mv123 = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
            mv45 = core.mv.Degrain2(i, iMVS, b4v, f4v, b5v, f5v, **degrain_args)
            sm = core.std.Merge(mv123, mv45, weight=[0.4545])
        else:
            mv123 = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
            mv456 = core.mv.Degrain3(i, iMVS, b4v, f4v, b5v, f5v, b6v, f6v, **degrain_args)
            sm = core.std.Merge(mv123, mv456, weight=[0.4615])

        return sm

    def MCTD_TTSM(i, iMVS, thSAD):
        compensate_args = dict(thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
        f1c = core.mv.Compensate(i, iMVS, f1v, **compensate_args)
        b1c = core.mv.Compensate(i, iMVS, b1v, **compensate_args)
        if radius > 1:
            f2c = core.mv.Compensate(i, iMVS, f2v, **compensate_args)
            b2c = core.mv.Compensate(i, iMVS, b2v, **compensate_args)
            # SAD_f2m = core.mv.Mask(i, f2v, **mask_args)
            # SAD_b2m = core.mv.Mask(i, b2v, **mask_args)
        if radius > 2:
            f3c = core.mv.Compensate(i, iMVS, f3v, **compensate_args)
            b3c = core.mv.Compensate(i, iMVS, b3v, **compensate_args)
            # SAD_f3m = core.mv.Mask(i, f3v, **mask_args)
            # SAD_b3m = core.mv.Mask(i, b3v, **mask_args)
        if radius > 3:
            f4c = core.mv.Compensate(i, iMVS, f4v, **compensate_args)
            b4c = core.mv.Compensate(i, iMVS, b4v, **compensate_args)
            # SAD_f4m = core.mv.Mask(i, f4v, **mask_args)
            # SAD_b4m = core.mv.Mask(i, b4v, **mask_args)
        if radius > 4:
            f5c = core.mv.Compensate(i, iMVS, f5v, **compensate_args)
            b5c = core.mv.Compensate(i, iMVS, b5v, **compensate_args)
            # SAD_f5m = core.mv.Mask(i, f5v, **mask_args)
            # SAD_b5m = core.mv.Mask(i, b5v, **mask_args)
        if radius > 5:
            f6c = core.mv.Compensate(i, iMVS, f6v, **compensate_args)
            b6c = core.mv.Compensate(i, iMVS, b6v, **compensate_args)
            # SAD_f6m = core.mv.Mask(i, f6v, **mask_args)
            # SAD_b6m = core.mv.Mask(i, b6v, **mask_args)

        # b = i.std.BlankClip(color=[0] if isGray else [0, neutral, neutral])
        if radius <= 1:
            c = core.std.Interleave([f1c, i, b1c])
            # SAD_m = core.std.Interleave([SAD_f1m, b, SAD_b1m])
        elif radius == 2:
            c = core.std.Interleave([f2c, f1c, i, b1c, b2c])
            # SAD_m = core.std.Interleave([SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m])
        elif radius == 3:
            c = core.std.Interleave([f3c, f2c, f1c, i, b1c, b2c, b3c])
            # SAD_m = core.std.Interleave([SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m])
        elif radius == 4:
            c = core.std.Interleave([f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c])
            # SAD_m = core.std.Interleave([SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m])
        elif radius == 5:
            c = core.std.Interleave([f5c, f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c, b5c])
            # SAD_m = core.std.Interleave([SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m, SAD_b5m])
        else:
            c = core.std.Interleave([f6c, f5c, f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c, b5c, b6c])
            # SAD_m = core.std.Interleave([SAD_f6m, SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m, SAD_b5m, SAD_b6m])

        # sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False, pfclip=SAD_m, planes=planes)
        sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False, planes=planes)
        return sm.std.SelectEvery(cycle=radius * 2 + 1, offsets=[radius])

    ### DENOISING: FIRST PASS
    dMVS = d.mv.Super(levels=1, **super_args)
    sm = MCTD_TTSM(d, dMVS, thSAD) if useTTmpSm else MCTD_MVD(d, dMVS, thSAD, thSADC)

    if limit <= -1:
        smD = core.std.MakeDiff(i, sm, planes=planes)
        expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
        DD = core.std.Expr([pD, smD], expr=[expr] if chroma or isGray else [expr, ''])
        smL = core.std.MakeDiff(i, DD, planes=planes)
    elif limit > 0:
        expr = f'x y - abs {limit} <= x x y - 0 < y {limit} - y {limit} + ? ?'
        smL = core.std.Expr([sm, i], expr=[expr] if chroma or isGray else [expr, ''])
    else:
        smL = sm

    ### DENOISING: SECOND PASS
    if twopass:
        smLMVS = smL.mv.Super(levels=1, **super_args)
        sm = MCTD_TTSM(smL, smLMVS, thSAD2) if useTTmpSm else MCTD_MVD(smL, smLMVS, thSAD2, thSADC2)

        if limit2 <= -1:
            smD = core.std.MakeDiff(i, sm, planes=planes)
            expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
            DD = core.std.Expr([pD, smD], expr=[expr] if chroma or isGray else [expr, ''])
            smL = core.std.MakeDiff(i, DD, planes=planes)
        elif limit2 > 0:
            expr = f'x y - abs {limit2} <= x x y - 0 < y {limit2} - y {limit2} + ? ?'
            smL = core.std.Expr([sm, i], expr=[expr] if chroma or isGray else [expr, ''])
        else:
            smL = sm

    ### POST-DENOISING: FFT3D
    if post <= 0:
        smP = smL
    else:
        smP = smL.fft3dfilter.FFT3DFilter(sigma=post * 0.8, sigma2=post * 0.6, sigma3=post * 0.4, sigma4=post * 0.2, **fft3d_args)

    ### EDGECLEANING
    if edgeclean:
        mP = avs_prewitt(plane(smP, 0))
        mS = Morpho.expand(mP, ECrad).std.Inflate()
        mD = core.std.Expr([mS, mP.std.Inflate()], expr=[f'x y - {ECthr} <= 0 x y - ?']).std.Inflate().std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        smP = core.std.MaskedMerge(smP, DeHalo_alpha(smP.dfttest.DFTTest(tbsize=1, planes=planes), darkstr=0), mD, planes=planes)

    ### STABILIZING
    if stabilize:
        # mM = core.std.Merge(plane(SAD_f1m, 0), plane(SAD_b1m, 0)).std.Lut(function=lambda x: min(cround(x ** 1.6), peak))
        mE = avs_prewitt(plane(smP, 0)).std.Lut(function=lambda x: min(cround(x ** 1.8), peak)).std.Median().std.Inflate()
        # mF = core.std.Expr([mM, mE], expr=['x y max']).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        mF = mE.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        TTc = smP.ttmpsm.TTempSmooth(maxr=maxr, mdiff=[255], strength=TTstr, planes=planes)
        smP = core.std.MaskedMerge(TTc, smP, mF, planes=planes)

    ### OUTPUT
    return smP.std.Crop(**crop_args)


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

    if complexpr_available:
        expr = f"x z {undershoot} - y {overshoot} + clamp"
    else:
        expr = f"x z {undershoot} - max y {overshoot} + min"
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
        expr = f'x y +'
    elif mode == 'average':
        expr = f'x y + 2 /'
    elif mode == 'burn':
        expr = f'x 0 <= x {peak} {peak} y - {factor} * x / - ?'
    elif mode == 'darken':
        expr = f'x y min'
    elif mode == 'difference':
        expr = f'x y - abs'
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
        expr = f'x y max'
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
        expr = f'x y -'
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


QTGMC_globals = {}


def QTGMC(
    Input: vs.VideoNode,
    Preset: str = 'Slower',
    TR0: Optional[int] = None,
    TR1: Optional[int] = None,
    TR2: Optional[int] = None,
    Rep0: Optional[int] = None,
    Rep1: int = 0,
    Rep2: Optional[int] = None,
    EdiMode: Optional[str] = None,
    RepChroma: bool = True,
    NNSize: Optional[int] = None,
    NNeurons: Optional[int] = None,
    EdiQual: int = 1,
    EdiMaxD: Optional[int] = None,
    ChromaEdi: str = '',
    EdiExt: Optional[vs.VideoNode] = None,
    Sharpness: Optional[float] = None,
    SMode: Optional[int] = None,
    SLMode: Optional[int] = None,
    SLRad: Optional[int] = None,
    SOvs: int = 0,
    SVThin: float = 0.0,
    Sbb: Optional[int] = None,
    SrchClipPP: Optional[int] = None,
    SubPel: Optional[int] = None,
    SubPelInterp: int = 2,
    BlockSize: Optional[int] = None,
    Overlap: Optional[int] = None,
    Search: Optional[int] = None,
    SearchParam: Optional[int] = None,
    PelSearch: Optional[int] = None,
    ChromaMotion: Optional[bool] = None,
    TrueMotion: bool = False,
    Lambda: Optional[int] = None,
    LSAD: Optional[int] = None,
    PNew: Optional[int] = None,
    PLevel: Optional[int] = None,
    GlobalMotion: bool = True,
    DCT: int = 0,
    ThSAD1: int = 640,
    ThSAD2: int = 256,
    ThSCD1: int = 180,
    ThSCD2: int = 98,
    SourceMatch: int = 0,
    MatchPreset: Optional[str] = None,
    MatchEdi: Optional[str] = None,
    MatchPreset2: Optional[str] = None,
    MatchEdi2: Optional[str] = None,
    MatchTR2: int = 1,
    MatchEnhance: float = 0.5,
    Lossless: int = 0,
    NoiseProcess: Optional[int] = None,
    EZDenoise: Optional[float] = None,
    EZKeepGrain: Optional[float] = None,
    NoisePreset: str = 'Fast',
    Denoiser: Optional[str] = None,
    FftThreads: int = 1,
    DenoiseMC: Optional[bool] = None,
    NoiseTR: Optional[int] = None,
    Sigma: Optional[float] = None,
    ChromaNoise: bool = False,
    ShowNoise: Union[bool, float] = 0.0,
    GrainRestore: Optional[float] = None,
    NoiseRestore: Optional[float] = None,
    NoiseDeint: Optional[str] = None,
    StabilizeNoise: Optional[bool] = None,
    InputType: int = 0,
    ProgSADMask: Optional[float] = None,
    FPSDivisor: int = 1,
    ShutterBlur: int = 0,
    ShutterAngleSrc: float = 180.0,
    ShutterAngleOut: float = 180.0,
    SBlurLimit: int = 4,
    Border: bool = False,
    Precise: Optional[bool] = None,
    Tuning: str = 'None',
    ShowSettings: bool = False,
    GlobalNames: str = 'QTGMC',
    PrevGlobals: str = 'Replace',
    ForceTR: int = 0,
    Str: float = 2.0,
    Amp: float = 0.0625,
    FastMA: bool = False,
    ESearchP: bool = False,
    RefineMotion: bool = False,
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    QTGMC 3.33

    A very high quality deinterlacer with a range of features for both quality and convenience. These include a simple presets system, extensive noise
    processing capabilities, support for repair of progressive material, precision source matching, shutter speed simulation, etc. Originally based on
    TempGaussMC_beta2 by Didée.

    Parameters:
        Input: Clip to process.

        Preset: Sets a range of defaults for different encoding speeds.
            Select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast" & "Draft".

        TR0: Temporal binomial smoothing radius used to create motion search clip. In general 2=quality, 1=speed, 0=don't use.

        TR1: Temporal binomial smoothing radius used on interpolated clip for initial output. In general 2=quality, 1=speed, 0=don't use.

        TR2: Temporal linear smoothing radius used for final stablization / denoising. Increase for smoother output.

        Rep0: Repair motion search clip (0=off): repair unwanted blur after temporal smooth TR0 (see QTGMC_KeepOnlyBobShimmerFixes function for details).

        Rep1: Repair initial output clip (0=off): repair unwanted blur after temporal smooth TR1.

        Rep2: Repair final output clip (0=off): unwanted blur after temporal smooth TR2 (will also repair TR1 blur if Rep1 not used).

        EdiMode: Interpolation method, from "NNEDI3", "EEDI3+NNEDI3" (EEDI3 with sclip from NNEDI3), "EEDI3" or "Bwdif", anything else uses "Bob".

        RepChroma: Whether the repair modes affect chroma.

        NNSize: Area around each pixel used as predictor for NNEDI3. A larger area is slower with better quality, read the NNEDI3 docs to see the area choices.
            Note: area sizes are not in increasing order (i.e. increased value doesn't always mean increased quality).

        NNeurons: Controls number of neurons in NNEDI3, larger = slower and better quality but improvements are small.

        EdiQual: Quality setting for NNEDI3. Higher values for better quality - but improvements are marginal.

        EdiMaxD: Spatial search distance for finding connecting edges in EEDI3.

        ChromaEdi: Interpolation method used for chroma. Set to "" to use EdiMode above (default). Otherwise choose from "NNEDI3", "Bwdif" or "Bob" - all high
            speed variants. This can give a minor speed-up if using a very slow EdiMode (i.e. one of the EEDIx modes).

        EdiExt: Provide externally created interpolated clip rather than use one of the above modes.

        Sharpness: How much to resharpen the temporally blurred clip (default is always 1.0 unlike original TGMC).

        SMode: Resharpening mode.
            0 = none
            1 = difference from 3x3 blur kernel
            2 = vertical max/min average + 3x3 kernel

        SLMode: Sharpness limiting.
            0 = off
            [1 = spatial, 2 = temporal]: before final temporal smooth
            [3 = spatial, 4 = temporal]: after final temporal smooth

        SLRad: Temporal or spatial radius used with sharpness limiting (depends on SLMode). Temporal radius can only be 0, 1 or 3.

        SOvs: Amount of overshoot allowed with temporal sharpness limiting (SLMode=2,4), i.e. allow some oversharpening.

        SVThin: How much to thin down 1-pixel wide lines that have been widened due to interpolation into neighboring field lines.

        Sbb: Back blend (blurred) difference between pre & post sharpened clip (minor fidelity improvement).
            0 = off
            1 = before (1st) sharpness limiting
            2 = after (1st) sharpness limiting
            3 = both

        SrchClipPP: Pre-filtering for motion search clip.
            0 = none
            1 = simple blur
            2 = Gauss blur
            3 = Gauss blur + edge soften

        SubPel: Sub-pixel accuracy for motion analysis.
            1 = 1 pixel
            2 = 1/2 pixel
            4 = 1/4 pixel

        SubPelInterp: Interpolation used for sub-pixel motion analysis.
            0 = bilinear (soft)
            1 = bicubic (sharper)
            2 = Weiner (sharpest)

        BlockSize: Size of blocks that are matched during motion analysis.

        Overlap: How much to overlap motion analysis blocks (requires more blocks, but essential to smooth block edges in motion compenstion).

        Search: Search method used for matching motion blocks - see MVTools2 documentation for available algorithms.

        SearchParam: Parameter for search method chosen. For default search method (hexagon search) it is the search range.

        PelSearch: Search parameter (as above) for the finest sub-pixel level (see SubPel).

        ChromaMotion: Whether to consider chroma when analyzing motion. Setting to false gives good speed-up,
            but may very occasionally make incorrect motion decision.

        TrueMotion: Whether to use the 'truemotion' defaults from MAnalyse (see MVTools2 documentation).

        Lambda: Motion vector field coherence - how much the motion analysis favors similar motion vectors for neighboring blocks.
            Should be scaled by BlockSize*BlockSize/64.

        LSAD: How much to reduce need for vector coherence (i.e. Lambda above) if prediction of motion vector from neighbors is poor,
            typically in areas of complex motion. This value is scaled in MVTools (unlike Lambda).

        PNew: Penalty for choosing a new motion vector for a block over an existing one - avoids choosing new vectors for minor gain.

        PLevel: Mode for scaling lambda across different sub-pixel levels - see MVTools2 documentation for choices.

        GlobalMotion: Whether to estimate camera motion to assist in selecting block motion vectors.

        DCT: Modes to use DCT (frequency analysis) or SATD as part of the block matching process - see MVTools2 documentation for choices.

        ThSAD1: SAD threshold for block match on shimmer-removing temporal smooth (TR1). Increase to reduce bob-shimmer more (may smear/blur).

        ThSAD2: SAD threshold for block match on final denoising temporal smooth (TR2). Increase to strengthen final smooth (may smear/blur).

        ThSCD1: Scene change detection parameter 1 - see MVTools documentation.

        ThSCD2: Scene change detection parameter 2 - see MVTools documentation.

        SourceMatch:
            0 = source-matching off (standard algorithm)
            1 = basic source-match
            2 = refined match
            3 = twice refined match

        MatchPreset: Speed/quality for basic source-match processing, select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
            "Very Fast", "Super Fast", "Ultra Fast" ("Draft" is not supported). Ideal choice is the same as main preset,
            but can choose a faster setting (but not a slower setting). Default is 3 steps faster than main Preset.

        MatchEdi: Override default interpolation method for basic source-match. Default method is same as main EdiMode setting (usually NNEDI3).
            Only need to override if using slow method for main interpolation (e.g. EEDI3) and want a faster method for source-match.

        MatchPreset2: Speed/quality for refined source-match processing, select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
            "Very Fast", "Super Fast", "Ultra Fast" ("Draft" is not supported). Default is 2 steps faster than MatchPreset.
            Faster settings are usually sufficient but can use slower settings if you get extra aliasing in this mode.

        MatchEdi2: Override interpolation method for refined source-match. Can be a good idea to pick MatchEdi2="Bob" for speed.

        MatchTR2: Temporal radius for refined source-matching. 2=smoothness, 1=speed/sharper, 0=not recommended. Differences are very marginal.
            Basic source-match doesn't need this setting as its temporal radius must match TR1 core setting (i.e. there is no MatchTR1).

        MatchEnhance: Enhance the detail found by source-match modes 2 & 3. A slight cheat - will enhance noise if set too strong. Best set < 1.0.

        Lossless: Puts exact source fields into result & cleans any artefacts. 0=off, 1=after final temporal smooth, 2=before resharpening.
            Adds some extra detail but: mode 1 gets shimmer / minor combing, mode 2 is more stable/tweakable but not exactly lossless.

        NoiseProcess: Bypass mode.
            0 = disable
            1 = denoise source & optionally restore some noise back at end of script [use for stronger denoising]
            2 = identify noise only & optionally restore some after QTGMC smoothing [for grain retention / light denoising]

        EZDenoise: Automatic setting to denoise source. Set > 0.0 to enable. Higher values denoise more. Can use ShowNoise to help choose value.

        EZKeepGrain: Automatic setting to retain source grain/detail. Set > 0.0 to enable. Higher values retain more grain. A good starting point = 1.0.

        NoisePreset: Automatic setting for quality of noise processing. Choices: "Slower", "Slow", "Medium", "Fast", and "Faster".

        Denoiser: Select denoiser to use for noise bypass / denoising. Select from "bm3d", "dfttest", "fft3dfilter" or "knlmeanscl".
            Unknown value selects "fft3dfilter".

        FftThreads: Number of threads to use if using "fft3dfilter" for Denoiser.

        DenoiseMC: Whether to provide a motion-compensated clip to the denoiser for better noise vs detail detection (will be a little slower).

        NoiseTR: Temporal radius used when analyzing clip for noise extraction. Higher values better identify noise vs detail but are slower.

        Sigma: Amount of noise known to be in the source, sensible values vary by source and denoiser, so experiment. Use ShowNoise to help.

        ChromaNoise: When processing noise (NoiseProcess > 0), whether to process chroma noise or not (luma noise is always processed).

        ShowNoise: Display extracted and "deinterlaced" noise rather than normal output. Set to true or false, or set a value (around 4 to 16) to specify
            contrast for displayed noise. Visualising noise helps to determine suitable value for Sigma or EZDenoise - want to see noise and noisy detail,
            but not too much clean structure or edges - fairly subjective.

        GrainRestore: How much removed noise/grain to restore before final temporal smooth. Retain "stable" grain and some detail (effect depends on TR2).

        NoiseRestore: How much removed noise/grain to restore after final temporal smooth. Retains any kind of noise.

        NoiseDeint: When noise is taken from interlaced source, how to 'deinterlace' it before restoring.
            "Bob" & "DoubleWeave" are fast but with minor issues: "Bob" is coarse and "Doubleweave" lags by one frame.
            "Generate" is a high quality mode that generates fresh noise lines, but it is slower. Unknown value selects "DoubleWeave".

        StabilizeNoise: Use motion compensation to limit shimmering and strengthen detail within the restored noise. Recommended for "Generate" mode.

        InputType: Default = 0 for interlaced input. Settings 1, 2 & 3 accept progressive input for deshimmer or repair. Frame rate of progressive source is not
            doubled. Mode 1 is for general progressive material. Modes 2 & 3 are designed for badly deinterlaced material.

        ProgSADMask: Only applies to InputType=2,3. If ProgSADMask > 0.0 then blend InputType modes 1 and 2/3 based on block motion SAD.
            Higher values help recover more detail, but repair less artefacts. Reasonable range about 2.0 to 20.0, or 0.0 for no blending.

        FPSDivisor: 1=Double-rate output, 2=Single-rate output. Higher values can be used too (e.g. 60fps & FPSDivisor=3 gives 20fps output).

        ShutterBlur: 0=Off, 1=Enable, 2,3=Higher precisions (slower). Higher precisions reduce blur "bleeding" into static areas a little.

        ShutterAngleSrc: Shutter angle used in source. If necessary, estimate from motion blur seen in a single frame.
            0=pin-sharp, 360=fully blurred from frame to frame.

        ShutterAngleOut: Shutter angle to simulate in output. Extreme values may be rejected (depends on other settings).
            Cannot reduce motion blur already in the source.

        SBlurLimit: Limit motion blur where motion lower than given value. Increase to reduce blur "bleeding". 0=Off. Sensible range around 2-12.

        Border: Pad a little vertically while processing (doesn't affect output size) - set true you see flickering on the very top or bottom line of the
            output. If you have wider edge effects than that, you should crop afterwards instead.

        Precise: Set to false to use faster algorithms with *very* slight imprecision in places.

        Tuning: Tweaks the defaults for different source types. Choose from "None", "DV-SD", "DV-HD".

        ShowSettings: Display all the current parameter values - useful to find preset defaults.

        GlobalNames: The name used to expose intermediate clips to calling script. QTGMC now exposes its motion vectors and other intermediate clips to the
            calling script through global variables. These globals are uniquely named. By default they begin with the prefix "QTGMC_". The available clips are:
                Backward motion vectors                 bVec1, bVec2, bVec3 (temporal radius 1 to 3)
                Forward motion vectors                  fVec1, fVec2, fVec3
                Filtered clip used for motion analysis  srchClip
                MVTools "super" clip for filtered clip  srchSuper
            Not all these clips are necessarily created - it depends on your QTGMC settings. To ensure motion vector creation to radius X, set ForceTR=X
            Clips can be accessed from other scripts with havsfunc.QTGMC_globals['Prefix_Name']

        PrevGlobals: What to do with global variables from earlier QTGMC call that match above name. Either "Replace", or "Reuse" (for a speed-up).
            Set PrevGlobals="Reuse" to reuse existing similar named globals for this run & not recalculate motion vectors etc. This will improve performance.
            Set PrevGlobals="Replace" to overwrite similar named globals from a previous run. This is the default and easiest option for most use cases.

        ForceTR: Ensure globally exposed motion vectors are calculated to this radius even if not needed by QTGMC.

        Str: With this parameter you control the strength of the brightening of the prefilter clip for motion analysis.
            This is good when problems with dark areas arise.

        Amp: Use this together with Str (active when Str is different from 1.0). This defines the amplitude of the brightening in the luma range,
            for example by using 1.0 all the luma range will be used and the brightening will find its peak at luma value 128 in the original.

        FastMA: Use 8-bit for faster motion analysis when using high bit depth input.

        ESearchP: Use wider search range for hex and umh search method.

        RefineMotion: Refines and recalculates motion data of previously estimated motion vectors with new parameters set (e.g. lesser block size).
            The two-stage method may be also useful for more stable (robust) motion estimation.

        TFF: Since VapourSynth only has a weak notion of field order internally, TFF may have to be set. Setting TFF to true means top field first and false
            means bottom field first. Note that the _FieldBased frame property, if present, takes precedence over TFF.

        nnedi3_args: Additional arguments to pass to NNEDI3.

        eedi3_args: Additional arguments to pass to EEDI3.

        opencl: Whether to use the OpenCL version of NNEDI3 and EEDI3.

        device: Sets target OpenCL device.
    '''
    if not isinstance(Input, vs.VideoNode):
        raise vs.Error('QTGMC: this is not a clip')

    if EdiExt is not None:
        if not isinstance(EdiExt, vs.VideoNode):
            raise vs.Error('QTGMC: EdiExt is not a clip')

        if EdiExt.format.id != Input.format.id:
            raise vs.Error('QTGMC: EdiExt must have the same format as input')

    if InputType != 1 and TFF is None:
        with Input.get_frame(0) as f:
            if (field_based := f.props.get('_FieldBased')) not in [1, 2]:
                raise vs.Error('QTGMC: TFF was not specified and field order could not be determined from frame properties')

        TFF = field_based == 2

    is_gray = Input.format.color_family == vs.GRAY

    bits = get_depth(Input)
    neutral = 1 << (bits - 1)

    SOvs = scale_value(SOvs, 8, bits)

    # ---------------------------------------
    # Presets

    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'very slow', 'slower', 'slow', 'medium', 'fast', 'faster', 'very fast', 'super fast', 'ultra fast', 'draft']
    try:
        pNum = presets.index(Preset)
    except ValueError:
        raise vs.Error("QTGMC: 'Preset' choice is invalid")

    if MatchPreset is None:
        mpNum1 = min(pNum + 3, 9)
        MatchPreset = presets[mpNum1]
    else:
        try:
            mpNum1 = presets[:10].index(MatchPreset.lower())
        except ValueError:
            raise vs.Error("QTGMC: 'MatchPreset' choice is invalid/unsupported")

    if MatchPreset2 is None:
        mpNum2 = min(mpNum1 + 2, 9)
        MatchPreset2 = presets[mpNum2]
    else:
        try:
            mpNum2 = presets[:10].index(MatchPreset2.lower())
        except ValueError:
            raise vs.Error("QTGMC: 'MatchPreset2' choice is invalid/unsupported")

    try:
        npNum = presets[2:7].index(NoisePreset.lower())
    except ValueError:
        raise vs.Error("QTGMC: 'NoisePreset' choice is invalid")

    try:
        tNum = ['none', 'dv-sd', 'dv-hd'].index(Tuning.lower())
    except ValueError:
        raise vs.Error("QTGMC: 'Tuning' choice is invalid")

    # Tunings only affect blocksize in this version
    bs = [16, 16, 32][tNum]
    bs2 = 32

    # fmt: off
    #                                                 Very                                                        Very      Super     Ultra
    # Preset groups:                        Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast      Fast      Draft
    TR0 = fallback(TR0,                   [ 2,        2,        2,        2,        2,        2,        1,        1,        1,        1,        0      ][pNum])
    TR1 = fallback(TR1,                   [ 2,        2,        2,        1,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    TR2X = fallback(TR2,                  [ 3,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0      ][pNum])
    Rep0 = fallback(Rep0,                 [ 4,        4,        4,        4,        3,        3,        0,        0,        0,        0,        0      ][pNum])
    Rep2 = fallback(Rep2,                 [ 4,        4,        4,        4,        4,        4,        4,        4,        3,        3,        0      ][pNum])
    EdiMode = fallback(EdiMode,           ['NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'Bwdif',  'Bob'   ][pNum]).lower()
    NNSize = fallback(NNSize,             [ 1,        1,        1,        1,        5,        5,        4,        4,        4,        4,        4      ][pNum])
    NNeurons = fallback(NNeurons,         [ 2,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0      ][pNum])
    EdiMaxD = fallback(EdiMaxD,           [ 12,       10,       8,        7,        7,        6,        6,        5,        4,        4,        4      ][pNum])
    SMode = fallback(SMode,               [ 2,        2,        2,        2,        2,        2,        2,        2,        2,        2,        0      ][pNum])
    SLModeX = fallback(SLMode,            [ 2,        2,        2,        2,        2,        2,        2,        2,        0,        0,        0      ][pNum])
    SLRad = fallback(SLRad,               [ 3,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    Sbb = fallback(Sbb,                   [ 3,        1,        1,        0,        0,        0,        0,        0,        0,        0,        0      ][pNum])
    SrchClipPP = fallback(SrchClipPP,     [ 3,        3,        3,        3,        3,        2,        2,        2,        1,        1,        0      ][pNum])
    SubPel = fallback(SubPel,             [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    BlockSize = fallback(BlockSize,       [ bs,       bs,       bs,       bs,       bs,       bs,       bs2,      bs2,      bs2,      bs2,      bs2    ][pNum])
    bs = BlockSize
    Overlap = fallback(Overlap,           [ bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 4,  bs // 4,  bs // 4,  bs // 4][pNum])
    Search = fallback(Search,             [ 5,        4,        4,        4,        4,        4,        4,        4,        0,        0,        0      ][pNum])
    SearchParam = fallback(SearchParam,   [ 2,        2,        2,        2,        2,        2,        2,        1,        1,        1,        1      ][pNum])
    PelSearch = fallback(PelSearch,       [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    ChromaMotion = fallback(ChromaMotion, [ True,     True,     True,     False,    False,    False,    False,    False,    False,    False,    False  ][pNum])
    Precise = fallback(Precise,           [ True,     True,     False,    False,    False,    False,    False,    False,    False,    False,    False  ][pNum])
    ProgSADMask = fallback(ProgSADMask,   [ 10.0,     10.0,     10.0,     10.0,     10.0,     0.0,      0.0,      0.0,      0.0,      0.0,      0.0    ][pNum])

    if ESearchP and Search in [4, 5]:
        if pNum < 4:
            SearchParam = 24
        elif pNum < 8:
            SearchParam = 16

    # Noise presets                             Slower      Slow       Medium     Fast      Faster
    Denoiser = fallback(Denoiser,             ['dfttest',  'dfttest', 'dfttest', 'fft3df', 'fft3df'][npNum]).lower()
    DenoiseMC = fallback(DenoiseMC,           [ True,       True,      False,     False,    False  ][npNum])
    NoiseTR = fallback(NoiseTR,               [ 2,          1,         1,         1,        0      ][npNum])
    NoiseDeint = fallback(NoiseDeint,         ['Generate', 'Bob',      '',        '',       ''     ][npNum]).lower()
    StabilizeNoise = fallback(StabilizeNoise, [ True,       True,      True,      False,    False  ][npNum])
    # fmt: on

    # The basic source-match step corrects and re-runs the interpolation of the input clip. So it initially uses same interpolation settings as the main preset
    MatchNNSize = NNSize
    MatchNNeurons = NNeurons
    MatchEdiMaxD = EdiMaxD
    MatchEdiQual = EdiQual

    # However, can use a faster initial interpolation when using source-match allowing the basic source-match step to "correct" it with higher quality settings
    if SourceMatch > 0 and mpNum1 < pNum:
        raise vs.Error("QTGMC: 'MatchPreset' cannot use a slower setting than 'Preset'")
    # Basic source-match presets
    if SourceMatch > 0:
        # fmt: off
        #                    Very                                      Very  Super  Ultra
        #           Placebo  Slow  Slower  Slow  Medium  Fast  Faster  Fast  Fast   Fast
        NNSize =   [1,       1,    1,      1,    5,      5,    4,      4,    4,     4    ][mpNum1]
        NNeurons = [2,       2,    1,      1,    1,      0,    0,      0,    0,     0    ][mpNum1]
        EdiMaxD =  [12,      10,   8,      7,    7,      6,    6,      5,    4,     4    ][mpNum1]
        EdiQual =  [1,       1,    1,      1,    1,      1,    1,      1,    1,     1    ][mpNum1]
        # fmt: on
    TempEdi = EdiMode  # Main interpolation is actually done by basic-source match step when enabled, so a little swap and wriggle is needed
    if SourceMatch > 0:
        EdiMode = fallback(MatchEdi, EdiMode if mpNum1 < 9 else 'Bwdif').lower()  # Force Bwdif for "Ultra Fast" basic source match
    MatchEdi = TempEdi

    # fmt: off
    #                                           Very                                                        Very      Super    Ultra
    # Refined source-match presets    Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast     Fast
    MatchEdi2 = fallback(MatchEdi2, ['NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', ''   ][mpNum2]).lower()
    MatchNNSize2 =                  [ 1,        1,        1,        1,        5,        5,        4,        4,        4,       4    ][mpNum2]
    MatchNNeurons2 =                [ 2,        2,        1,        1,        1,        0,        0,        0,        0,       0    ][mpNum2]
    MatchEdiMaxD2 =                 [ 12,       10,       8,        7,        7,        6,        6,        5,        4,       4    ][mpNum2]
    MatchEdiQual2 =                 [ 1,        1,        1,        1,        1,        1,        1,        1,        1,       1    ][mpNum2]
    # fmt: on

    # ---------------------------------------
    # Settings

    # Core defaults
    TR2 = fallback(TR2, max(TR2X, 1)) if SourceMatch > 0 else TR2X  # ***TR2 defaults always at least 1 when using source-match***

    # Source-match defaults
    MatchTR1 = TR1

    # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match), but adjusted to give roughly same sharpness for all settings
    if Sharpness is not None and Sharpness <= 0:
        SMode = 0
    SLMode = fallback(SLMode, 0) if SourceMatch > 0 else SLModeX  # ***Sharpness limiting disabled by default for source-match***
    if SLRad <= 0:
        SLMode = 0
    spatialSL = SLMode in [1, 3]
    temporalSL = SLMode in [2, 4]
    Sharpness = fallback(Sharpness, 0.0 if SMode <= 0 else 0.2 if SourceMatch > 0 else 1.0)  # Default sharpness is 1.0, or 0.2 if using source-match
    sharpMul = 2 if temporalSL else 1.5 if spatialSL else 1  # Adjust sharpness based on other settings
    sharpAdj = Sharpness * (sharpMul * (0.2 + TR1 * 0.15 + TR2 * 0.25) + (0.1 if SMode == 1 else 0))  # [This needs a bit more refinement]
    if SMode <= 0:
        Sbb = 0

    # Noise processing settings
    if EZDenoise is not None and EZDenoise > 0 and EZKeepGrain is not None and EZKeepGrain > 0:
        raise vs.Error("QTGMC: EZDenoise and EZKeepGrain cannot be used together")
    if NoiseProcess is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseProcess = 1
        elif (EZKeepGrain is not None and EZKeepGrain > 0) or Preset in ['placebo', 'very slow']:
            NoiseProcess = 2
        else:
            NoiseProcess = 0
    if GrainRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            GrainRestore = 0.0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            GrainRestore = 0.3 * math.sqrt(EZKeepGrain)
        else:
            GrainRestore = [0.0, 0.7, 0.3][NoiseProcess]
    if NoiseRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseRestore = 0.0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            NoiseRestore = 0.1 * math.sqrt(EZKeepGrain)
        else:
            NoiseRestore = [0.0, 0.3, 0.1][NoiseProcess]
    if Sigma is None:
        if EZDenoise is not None and EZDenoise > 0:
            Sigma = EZDenoise
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            Sigma = 4.0 * EZKeepGrain
        else:
            Sigma = 2.0
    if isinstance(ShowNoise, bool):
        ShowNoise = 10.0 if ShowNoise else 0.0
    if ShowNoise > 0:
        NoiseProcess = 2
        NoiseRestore = 1.0
    if NoiseProcess <= 0:
        NoiseTR = 0
        GrainRestore = 0.0
        NoiseRestore = 0.0
    totalRestore = GrainRestore + NoiseRestore
    if totalRestore <= 0:
        StabilizeNoise = False
    noiseTD = [1, 3, 5][NoiseTR]
    noiseCentre = scale_value(128.5, 8, bits) if Denoiser in ['fft3df', 'fft3dfilter'] else neutral

    # MVTools settings
    Lambda = fallback(Lambda, (1000 if TrueMotion else 100) * BlockSize * BlockSize // 64)
    LSAD = fallback(LSAD, 1200 if TrueMotion else 400)
    PNew = fallback(PNew, 50 if TrueMotion else 25)
    PLevel = fallback(PLevel, 1 if TrueMotion else 0)

    # Motion blur settings
    if ShutterAngleOut * FPSDivisor == ShutterAngleSrc:  # If motion blur output is same as input
        ShutterBlur = 0

    # Miscellaneous
    PrevGlobals = PrevGlobals.lower()
    ReplaceGlobals = PrevGlobals in ['replace', 'reuse']  # If reusing existing globals put them back afterwards - simplifies logic later
    ReuseGlobals = PrevGlobals == 'reuse'
    if InputType < 2:
        ProgSADMask = 0.0

    # Get maximum temporal radius needed
    maxTR = max(SLRad if temporalSL else 0, MatchTR2, TR1, TR2, NoiseTR)
    if (ProgSADMask > 0 or StabilizeNoise or ShutterBlur > 0) and maxTR < 1:
        maxTR = 1
    maxTR = max(ForceTR, maxTR)

    # ---------------------------------------
    # Pre-Processing

    w = Input.width
    h = Input.height

    # Reverse "field" dominance for progressive repair mode 3 (only difference from mode 2)
    if InputType >= 3:
        TFF = not TFF

    # Pad vertically during processing (to prevent artefacts at top & bottom edges)
    if Border:
        h += 8
        clip = Input.resize.Point(w, h, src_top=-4, src_height=h)
    else:
        clip = Input

    hpad = vpad = BlockSize

    # ---------------------------------------
    # Motion Analysis

    # Bob the input as a starting point for motion search clip
    if InputType <= 0:
        bobbed = clip.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
    elif InputType == 1:
        bobbed = clip
    else:
        bobbed = clip.std.Convolution(matrix=[1, 2, 1], mode='v')

    # If required, get any existing global clips with a matching "GlobalNames" setting. Unmatched values get None
    if ReuseGlobals:
        srchClip = QTGMC_GetUserGlobal(GlobalNames, 'srchClip')
        srchSuper = QTGMC_GetUserGlobal(GlobalNames, 'srchSuper')
        bVec1 = QTGMC_GetUserGlobal(GlobalNames, 'bVec1')
        fVec1 = QTGMC_GetUserGlobal(GlobalNames, 'fVec1')
        bVec2 = QTGMC_GetUserGlobal(GlobalNames, 'bVec2')
        fVec2 = QTGMC_GetUserGlobal(GlobalNames, 'fVec2')
        bVec3 = QTGMC_GetUserGlobal(GlobalNames, 'bVec3')
        fVec3 = QTGMC_GetUserGlobal(GlobalNames, 'fVec3')
    else:
        srchClip = srchSuper = bVec1 = fVec1 = bVec2 = fVec2 = bVec3 = fVec3 = None

    CMplanes = [0, 1, 2] if ChromaMotion and not is_gray else [0]

    # The bobbed clip will shimmer due to being derived from alternating fields. Temporally smooth over the neighboring frames using a binomial kernel. Binomial
    # kernels give equal weight to even and odd frames and hence average away the shimmer. The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
    # These kernels are approximately Gaussian kernels, which work well as a prefilter before motion analysis (hence the original name for this script)
    # Create linear weightings of neighbors first                                                  -2    -1    0     1     2
    if not isinstance(srchClip, vs.VideoNode):
        if TR0 > 0:
            ts1 = average_frames(bobbed, weights=[1] * 3, scenechange=28 / 255, planes=CMplanes)  # 0.00  0.33  0.33  0.33  0.00
        if TR0 > 1:
            ts2 = average_frames(bobbed, weights=[1] * 5, scenechange=28 / 255, planes=CMplanes)  # 0.20  0.20  0.20  0.20  0.20

    # Combine linear weightings to give binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
    if isinstance(srchClip, vs.VideoNode):
        binomial0 = None
    elif TR0 <= 0:
        binomial0 = bobbed
    elif TR0 == 1:
        binomial0 = core.std.Merge(ts1, bobbed, weight=0.25 if ChromaMotion or is_gray else [0.25, 0])
    else:
        binomial0 = core.std.Merge(
            core.std.Merge(ts1, ts2, weight=0.357 if ChromaMotion or is_gray else [0.357, 0]), bobbed, weight=0.125 if ChromaMotion or is_gray else [0.125, 0]
        )

    # Remove areas of difference between temporal blurred motion search clip and bob that are not due to bob-shimmer - removes general motion blur
    if isinstance(srchClip, vs.VideoNode) or Rep0 <= 0:
        repair0 = binomial0
    else:
        repair0 = QTGMC_KeepOnlyBobShimmerFixes(binomial0, bobbed, Rep0, RepChroma and ChromaMotion)

    matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
    # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
    if not isinstance(srchClip, vs.VideoNode):
        if SrchClipPP == 1:
            spatialBlur = repair0.resize.Bilinear(w // 2, h // 2).std.Convolution(matrix=matrix, planes=CMplanes).resize.Bilinear(w, h)
        elif SrchClipPP >= 2:
            spatialBlur = gauss_blur(repair0.std.Convolution(matrix=matrix, planes=CMplanes), 1.75)
            spatialBlur = core.std.Merge(spatialBlur, repair0, weight=0.1 if ChromaMotion or is_gray else [0.1, 0])
        if SrchClipPP <= 0:
            srchClip = repair0
        elif SrchClipPP < 3:
            srchClip = spatialBlur
        else:
            expr = 'x {i3} + y < x {i3} + x {i3} - y > x {i3} - y ? ?'.format(i3=scale_value(3, 8, bits))
            tweaked = core.std.Expr([repair0, bobbed], expr=expr if ChromaMotion or is_gray else [expr, ''])
            expr = 'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?'.format(i7=scale_value(7, 8, bits), i2=scale_value(2, 8, bits))
            srchClip = core.std.Expr([spatialBlur, tweaked], expr=expr if ChromaMotion or is_gray else [expr, ''])
        srchClip = prefilter_to_full_range(srchClip, Str, CMplanes)
        if bits > 8 and FastMA:
            srchClip = depth(srchClip, 8, dither_type=DitherType.NONE)

    super_args = dict(pel=SubPel, hpad=hpad, vpad=vpad)
    analyse_args = dict(
        blksize=BlockSize,
        overlap=Overlap,
        search=Search,
        searchparam=SearchParam,
        pelsearch=PelSearch,
        truemotion=TrueMotion,
        lambda_=Lambda,
        lsad=LSAD,
        pnew=PNew,
        plevel=PLevel,
        global_=GlobalMotion,
        dct=DCT,
        chroma=ChromaMotion,
    )
    recalculate_args = dict(
        thsad=ThSAD1 // 2,
        blksize=max(BlockSize // 2, 4),
        search=Search,
        searchparam=SearchParam,
        chroma=ChromaMotion,
        truemotion=TrueMotion,
        pnew=PNew,
        overlap=max(Overlap // 2, 2),
        dct=DCT,
    )

    # Calculate forward and backward motion vectors from motion search clip
    if maxTR > 0:
        if not isinstance(srchSuper, vs.VideoNode):
            srchSuper = srchClip.mv.Super(sharp=SubPelInterp, chroma=ChromaMotion, **super_args)
        if not isinstance(bVec1, vs.VideoNode):
            bVec1 = srchSuper.mv.Analyse(isb=True, delta=1, **analyse_args)
            if RefineMotion:
                bVec1 = core.mv.Recalculate(srchSuper, bVec1, **recalculate_args)
        if not isinstance(fVec1, vs.VideoNode):
            fVec1 = srchSuper.mv.Analyse(isb=False, delta=1, **analyse_args)
            if RefineMotion:
                fVec1 = core.mv.Recalculate(srchSuper, fVec1, **recalculate_args)
    if maxTR > 1:
        if not isinstance(bVec2, vs.VideoNode):
            bVec2 = srchSuper.mv.Analyse(isb=True, delta=2, **analyse_args)
            if RefineMotion:
                bVec2 = core.mv.Recalculate(srchSuper, bVec2, **recalculate_args)
        if not isinstance(fVec2, vs.VideoNode):
            fVec2 = srchSuper.mv.Analyse(isb=False, delta=2, **analyse_args)
            if RefineMotion:
                fVec2 = core.mv.Recalculate(srchSuper, fVec2, **recalculate_args)
    if maxTR > 2:
        if not isinstance(bVec3, vs.VideoNode):
            bVec3 = srchSuper.mv.Analyse(isb=True, delta=3, **analyse_args)
            if RefineMotion:
                bVec3 = core.mv.Recalculate(srchSuper, bVec3, **recalculate_args)
        if not isinstance(fVec3, vs.VideoNode):
            fVec3 = srchSuper.mv.Analyse(isb=False, delta=3, **analyse_args)
            if RefineMotion:
                fVec3 = core.mv.Recalculate(srchSuper, fVec3, **recalculate_args)

    # Expose search clip, motion search super clip and motion vectors to calling script through globals
    if ReplaceGlobals:
        QTGMC_SetUserGlobal(GlobalNames, 'srchClip', srchClip)
        QTGMC_SetUserGlobal(GlobalNames, 'srchSuper', srchSuper)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec1', bVec1)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec1', fVec1)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec2', bVec2)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec2', fVec2)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec3', bVec3)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec3', fVec3)

    # ---------------------------------------
    # Noise Processing

    # Expand fields to full frame size before extracting noise (allows use of motion vectors which are frame-sized)
    if NoiseProcess > 0:
        if InputType > 0:
            fullClip = clip
        else:
            fullClip = clip.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=1)
    if NoiseTR > 0:
        fullSuper = fullClip.mv.Super(levels=1, chroma=ChromaNoise, **super_args)  # TEST chroma OK?

    CNplanes = [0, 1, 2] if ChromaNoise and not is_gray else [0]

    if NoiseProcess > 0:
        # Create a motion compensated temporal window around current frame and use to guide denoisers
        if not DenoiseMC or NoiseTR <= 0:
            noiseWindow = fullClip
        elif NoiseTR == 1:
            noiseWindow = core.std.Interleave(
                [
                    core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    fullClip,
                    core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                ]
            )
        else:
            noiseWindow = core.std.Interleave(
                [
                    core.mv.Compensate(fullClip, fullSuper, fVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                    core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    fullClip,
                    core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    core.mv.Compensate(fullClip, fullSuper, bVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                ]
            )
        if Denoiser == 'bm3d':
            import mvsfunc as mvf
            dnWindow = mvf.BM3D(noiseWindow, radius1=NoiseTR, sigma=[Sigma if plane in CNplanes else 0 for plane in range(3)])
        elif Denoiser == 'dfttest':
            dnWindow = noiseWindow.dfttest.DFTTest(sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)
        elif Denoiser in ['knlm', 'knlmeanscl']:
            dnWindow = nl_means(noiseWindow, strength=Sigma, tr=NoiseTR, planes=CNplanes)
        else:
            dnWindow = noiseWindow.fft3dfilter.FFT3DFilter(sigma=Sigma, planes=CNplanes, bt=noiseTD, ncpu=FftThreads)

        # Rework denoised clip to match source format - various code paths here: discard the motion compensation window, discard doubled lines (from point resize)
        # Also reweave to get interlaced noise if source was interlaced (could keep the full frame of noise, but it will be poor quality from the point resize)
        if not DenoiseMC:
            if InputType > 0:
                denoised = dnWindow
            else:
                denoised = dnWindow.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]).std.DoubleWeave(TFF)[::2]
        elif InputType > 0:
            if NoiseTR <= 0:
                denoised = dnWindow
            else:
                denoised = dnWindow.std.SelectEvery(cycle=noiseTD, offsets=NoiseTR)
        else:
            denoised = dnWindow.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=noiseTD * 4, offsets=[NoiseTR * 2, NoiseTR * 6 + 3]).std.DoubleWeave(TFF)[::2]

        if totalRestore > 0:
            # Get actual noise from difference. Then 'deinterlace' where we have weaved noise - create the missing lines of noise in various ways
            noise = core.std.MakeDiff(clip, denoised, planes=CNplanes)
            if InputType > 0:
                deintNoise = noise
            elif NoiseDeint == 'bob':
                deintNoise = noise.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
            elif NoiseDeint == 'generate':
                deintNoise = QTGMC_Generate2ndFieldNoise(noise, denoised, ChromaNoise, TFF)
            else:
                deintNoise = noise.std.SeparateFields(tff=TFF).std.DoubleWeave(tff=TFF)

            # Motion-compensated stabilization of generated noise
            if StabilizeNoise:
                noiseSuper = deintNoise.mv.Super(sharp=SubPelInterp, levels=1, chroma=ChromaNoise, **super_args)
                mcNoise = core.mv.Compensate(deintNoise, noiseSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
                expr = f'x {neutral} - abs y {neutral} - abs > x y ? 0.6 * x y + 0.2 * +'
                finalNoise = core.std.Expr([deintNoise, mcNoise], expr=expr if ChromaNoise or is_gray else [expr, ''])
            else:
                finalNoise = deintNoise

    # If NoiseProcess=1 denoise input clip. If NoiseProcess=2 leave noise in the clip and let the temporal blurs "denoise" it for a stronger effect
    innerClip = denoised if NoiseProcess == 1 else clip

    # ---------------------------------------
    # Interpolation

    # Support badly deinterlaced progressive content - drop half the fields and reweave to get 1/2fps interlaced stream appropriate for QTGMC processing
    if InputType > 1:
        ediInput = innerClip.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]).std.DoubleWeave(TFF)[::2]
    else:
        ediInput = innerClip

    # Create interpolated image as starting point for output
    if EdiExt is not None:
        edi1 = EdiExt.resize.Point(w, h, src_top=(EdiExt.height - h) // 2, src_height=h)
    else:
        edi1 = QTGMC_Interpolate(
            ediInput, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, bobbed, ChromaEdi.lower(), TFF, nnedi3_args, eedi3_args, opencl, device
        )

    # InputType=2,3: use motion mask to blend luma between original clip & reweaved clip based on ProgSADMask setting. Use chroma from original clip in any case
    if InputType < 2:
        edi = edi1
    elif ProgSADMask <= 0:
        if not is_gray:
            edi = core.std.ShufflePlanes([edi1, innerClip], planes=[0, 1, 2], colorfamily=Input.format.color_family)
        else:
            edi = edi1
    else:
        inputTypeBlend = core.mv.Mask(srchClip, bVec1, kind=1, ml=ProgSADMask)
        edi = core.std.MaskedMerge(innerClip, edi1, inputTypeBlend, planes=0)

    # Get the max/min value for each pixel over neighboring motion-compensated frames - used for temporal sharpness limiting
    if TR1 > 0 or temporalSL:
        ediSuper = edi.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
    if temporalSL:
        bComp1 = core.mv.Compensate(edi, ediSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        fComp1 = core.mv.Compensate(edi, ediSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        tMax = core.std.Expr([core.std.Expr([edi, fComp1], expr='x y max'), bComp1], expr='x y max')
        tMin = core.std.Expr([core.std.Expr([edi, fComp1], expr='x y min'), bComp1], expr='x y min')
        if SLRad > 1:
            bComp3 = core.mv.Compensate(edi, ediSuper, bVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            fComp3 = core.mv.Compensate(edi, ediSuper, fVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            tMax = core.std.Expr([core.std.Expr([tMax, fComp3], expr='x y max'), bComp3], expr='x y max')
            tMin = core.std.Expr([core.std.Expr([tMin, fComp3], expr='x y min'), bComp3], expr='x y min')

    # ---------------------------------------
    # Create basic output

    # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames. As above, this is done to remove shimmer from
    # alternate frames so the same binomial kernels are used. However, by using motion-compensated smoothing this time we avoid motion blur. The use of
    # MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different, but they evaluate to the same values
    # Create linear weightings of neighbors first                                                               -2    -1    0     1     2
    if TR1 > 0:
        degrain1 = core.mv.Degrain1(edi, ediSuper, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)  # 0.00  0.33  0.33  0.33  0.00
    if TR1 > 1:
        degrain2 = core.mv.Degrain1(edi, ediSuper, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)  # 0.33  0.00  0.33  0.00  0.33

    # Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
    if TR1 <= 0:
        binomial1 = edi
    elif TR1 == 1:
        binomial1 = core.std.Merge(degrain1, edi, weight=0.25)
    else:
        binomial1 = core.std.Merge(core.std.Merge(degrain1, degrain2, weight=0.2), edi, weight=0.0625)

    # Remove areas of difference between smoothed image and interpolated image that are not bob-shimmer fixes: repairs residual motion blur from temporal smooth
    if Rep1 <= 0:
        repair1 = binomial1
    else:
        repair1 = QTGMC_KeepOnlyBobShimmerFixes(binomial1, edi, Rep1, RepChroma)

    # Apply source match - use difference between output and source to succesively refine output [extracted to function to clarify main code path]
    if SourceMatch <= 0:
        match = repair1
    else:
        match = QTGMC_ApplySourceMatch(
            repair1,
            InputType,
            ediInput,
            bVec1 if maxTR > 0 else None,
            fVec1 if maxTR > 0 else None,
            bVec2 if maxTR > 1 else None,
            fVec2 if maxTR > 1 else None,
            SubPel,
            SubPelInterp,
            hpad,
            vpad,
            ThSAD1,
            ThSCD1,
            ThSCD2,
            SourceMatch,
            MatchTR1,
            MatchEdi,
            MatchNNSize,
            MatchNNeurons,
            MatchEdiQual,
            MatchEdiMaxD,
            MatchTR2,
            MatchEdi2,
            MatchNNSize2,
            MatchNNeurons2,
            MatchEdiQual2,
            MatchEdiMaxD2,
            MatchEnhance,
            TFF,
            nnedi3_args,
            eedi3_args,
            opencl,
            device,
        )

    # Lossless=2 - after preparing an interpolated, de-shimmered clip, restore the original source fields into it and clean up any artefacts
    # This mode will not give a true lossless result because the resharpening and final temporal smooth are still to come, but it will add further detail
    # However, it can introduce minor combing. This setting is best used together with source-match (it's effectively the final source-match stage)
    if Lossless >= 2:
        lossed1 = QTGMC_MakeLossless(match, innerClip, InputType, TFF)
    else:
        lossed1 = match

    # ---------------------------------------
    # Resharpen / retouch output

    # Resharpen to counteract temporal blurs. Little sharpening needed for source-match mode since it has already recovered sharpness from source
    if SMode <= 0:
        resharp = lossed1
    elif SMode == 1:
        resharp = core.std.Expr([lossed1, lossed1.std.Convolution(matrix=matrix)], expr=f'x x y - {sharpAdj} * +')
    else:
        vresharp1 = core.std.Merge(lossed1.std.Maximum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), lossed1.std.Minimum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]))
        if Precise:  # Precise mode: reduce tiny overshoot
            vresharp = core.std.Expr([vresharp1, lossed1], expr='x y < x {i1} + x y > x {i1} - x ? ?'.format(i1=scale_value(1, 8, bits)))
        else:
            vresharp = vresharp1
        resharp = core.std.Expr([lossed1, vresharp.std.Convolution(matrix=matrix)], expr=f'x x y - {sharpAdj} * +')

    # Slightly thin down 1-pixel high horizontal edges that have been widened into neighboring field lines by the interpolator
    SVThinSc = SVThin * 6.0
    if SVThin > 0:
        expr = f'y x - {SVThinSc} * {neutral} +'
        vertMedD = core.std.Expr([lossed1, lossed1.rgvs.VerticalCleaner(mode=1 if is_gray else [1, 0])], expr=expr if is_gray else [expr, ''])
        vertMedD = vertMedD.std.Convolution(matrix=[1, 2, 1], planes=0, mode='h')
        expr = f'y {neutral} - abs x {neutral} - abs > y {neutral} ?'
        neighborD = core.std.Expr([vertMedD, vertMedD.std.Convolution(matrix=matrix, planes=0)], expr=expr if is_gray else [expr, ''])
        thin = core.std.MergeDiff(resharp, neighborD, planes=0)
    else:
        thin = resharp

    # Back blend the blurred difference between sharpened & unsharpened clip, before (1st) sharpness limiting (Sbb == 1,3). A small fidelity improvement
    if Sbb not in [1, 3]:
        backBlend1 = thin
    else:
        backBlend1 = core.std.MakeDiff(thin, gauss_blur(core.std.MakeDiff(thin, lossed1, planes=0).std.Convolution(matrix=matrix, planes=0), 1.2), planes=0)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (before final temporal smooth) if SLMode == 1,2. This location will restrict sharpness more, but any artefacts introduced will be smoothed
    if SLMode == 1:
        if SLRad <= 1:
            sharpLimit1 = core.rgvs.Repair(backBlend1, edi, mode=1)
        else:
            sharpLimit1 = core.rgvs.Repair(backBlend1, core.rgvs.Repair(backBlend1, edi, mode=12), mode=1)
    elif SLMode == 2:
        sharpLimit1 = mt_clamp(backBlend1, tMax, tMin, SOvs, SOvs)
    else:
        sharpLimit1 = backBlend1

    # Back blend the blurred difference between sharpened & unsharpened clip, after (1st) sharpness limiting (Sbb == 2,3). A small fidelity improvement
    if Sbb < 2:
        backBlend2 = sharpLimit1
    else:
        backBlend2 = core.std.MakeDiff(
            sharpLimit1, gauss_blur(core.std.MakeDiff(sharpLimit1, lossed1, planes=0).std.Convolution(matrix=matrix, planes=0), 1.2), planes=0
        )

    # Add back any extracted noise, prior to final temporal smooth - this will restore detail that was removed as "noise" without restoring the noise itself
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if GrainRestore <= 0:
        addNoise1 = backBlend2
    else:
        expr = f'x {noiseCentre} - {GrainRestore} * {neutral} +'
        addNoise1 = core.std.MergeDiff(backBlend2, finalNoise.std.Expr(expr=expr if ChromaNoise or is_gray else [expr, '']), planes=CNplanes)

    # Final light linear temporal smooth for denoising
    if TR2 > 0:
        stableSuper = addNoise1.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
    if TR2 <= 0:
        stable = addNoise1
    elif TR2 == 1:
        stable = core.mv.Degrain1(addNoise1, stableSuper, bVec1, fVec1, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    elif TR2 == 2:
        stable = core.mv.Degrain2(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    else:
        stable = core.mv.Degrain3(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)

    # Remove areas of difference between final output & basic interpolated image that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
    if Rep2 <= 0:
        repair2 = stable
    else:
        repair2 = QTGMC_KeepOnlyBobShimmerFixes(stable, edi, Rep2, RepChroma)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (after final temporal smooth) if SLMode == 3,4. Allows more sharpening here, but more prone to introducing minor artefacts
    if SLMode == 3:
        if SLRad <= 1:
            sharpLimit2 = core.rgvs.Repair(repair2, edi, mode=1)
        else:
            sharpLimit2 = core.rgvs.Repair(repair2, core.rgvs.Repair(repair2, edi, mode=12), mode=1)
    elif SLMode >= 4:
        sharpLimit2 = mt_clamp(repair2, tMax, tMin, SOvs, SOvs)
    else:
        sharpLimit2 = repair2

    # Lossless=1 - inject source fields into result and clean up inevitable artefacts. Provided NoiseRestore=0.0 or 1.0, this mode will make the script result
    # properly lossless, but this will retain source artefacts and cause some combing (where the smoothed deinterlace doesn't quite match the source)
    if Lossless == 1:
        lossed2 = QTGMC_MakeLossless(sharpLimit2, innerClip, InputType, TFF)
    else:
        lossed2 = sharpLimit2

    # Add back any extracted noise, after final temporal smooth. This will appear as noise/grain in the output
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if NoiseRestore <= 0:
        addNoise2 = lossed2
    else:
        expr = f'x {noiseCentre} - {NoiseRestore} * {neutral} +'
        addNoise2 = core.std.MergeDiff(lossed2, finalNoise.std.Expr(expr=expr if ChromaNoise or is_gray else [expr, '']), planes=CNplanes)

    # ---------------------------------------
    # Post-Processing

    # Shutter motion blur - get level of blur depending on output framerate and blur already in source
    blurLevel = (ShutterAngleOut * FPSDivisor - ShutterAngleSrc) * 100 / 360
    if blurLevel < 0:
        raise vs.Error('QTGMC: cannot reduce motion blur already in source: increase ShutterAngleOut or FPSDivisor')
    if blurLevel > 200:
        raise vs.Error('QTGMC: exceeded maximum motion blur level: decrease ShutterAngleOut or FPSDivisor')

    # ShutterBlur mode 2,3 - get finer resolution motion vectors to reduce blur "bleeding" into static areas
    rBlockDivide = [1, 1, 2, 4][ShutterBlur]
    rBlockSize = max(BlockSize // rBlockDivide, 4)
    rOverlap = max(Overlap // rBlockDivide, 2)
    rBlockDivide = BlockSize // rBlockSize
    rLambda = Lambda // (rBlockDivide * rBlockDivide)
    if ShutterBlur > 1:
        recalculate_args = dict(
            thsad=ThSAD1,
            blksize=rBlockSize,
            overlap=rOverlap,
            search=Search,
            searchparam=SearchParam,
            truemotion=TrueMotion,
            lambda_=rLambda,
            pnew=PNew,
            dct=DCT,
            chroma=ChromaMotion,
        )
        sbBVec1 = core.mv.Recalculate(srchSuper, bVec1, **recalculate_args)
        sbFVec1 = core.mv.Recalculate(srchSuper, fVec1, **recalculate_args)
    elif ShutterBlur > 0:
        sbBVec1 = bVec1
        sbFVec1 = fVec1

    # Shutter motion blur - use MFlowBlur to blur along motion vectors
    if ShutterBlur > 0:
        sblurSuper = addNoise2.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
        sblur = core.mv.FlowBlur(addNoise2, sblurSuper, sbBVec1, sbFVec1, blur=blurLevel, thscd1=ThSCD1, thscd2=ThSCD2)

    # Shutter motion blur - use motion mask to reduce blurring in areas of low motion - also helps reduce blur "bleeding" into static areas, then select blur type
    if ShutterBlur <= 0:
        sblurred = addNoise2
    elif SBlurLimit <= 0:
        sblurred = sblur
    else:
        sbMotionMask = core.mv.Mask(srchClip, bVec1, kind=0, ml=SBlurLimit)
        sblurred = core.std.MaskedMerge(addNoise2, sblur, sbMotionMask)

    # Reduce frame rate
    if FPSDivisor > 1:
        decimated = sblurred.std.SelectEvery(cycle=FPSDivisor, offsets=0)
    else:
        decimated = sblurred

    # Crop off temporary vertical padding
    if Border:
        cropped = decimated.std.Crop(top=4, bottom=4)
    else:
        cropped = decimated

    # Show output of choice + settings
    if ShowNoise <= 0:
        output = cropped
    else:
        expr = f'x {neutral} - {ShowNoise} * {neutral} +'
        output = finalNoise.std.Expr(expr=expr if ChromaNoise or is_gray else [expr, repr(neutral)])
    output = output.std.SetFieldBased(value=0)
    if not ShowSettings:
        return output
    else:
        text = (
            f'{TR0=} | {TR1=} | {TR2=} | {Rep0=} | {Rep1=} | {Rep2=} | {RepChroma=} | {EdiMode=} | {NNSize=} | {NNeurons=} | {EdiQual=} | {EdiMaxD=} | '
            + f'{ChromaEdi=} | {Sharpness=} | {SMode=} | {SLMode=} | {SLRad=} | {SOvs=} | {SVThin=} | {Sbb=} | {SrchClipPP=} | {SubPel=} | {SubPelInterp=} | '
            + f'{BlockSize=} | {Overlap=} | {Search=} | {SearchParam=} | {PelSearch=} | {ChromaMotion=} | {TrueMotion=} | {Lambda=} | {LSAD=} | {PNew=} | '
            + f'{PLevel=} | {GlobalMotion=} | {DCT=} | {ThSAD1=} | {ThSAD2=} | {ThSCD1=} | {ThSCD2=} | {SourceMatch=} | {MatchPreset=} | {MatchEdi=} | '
            + f'{MatchPreset2=} | {MatchEdi2=} | {MatchTR2=} | {MatchEnhance=} | {Lossless=} | {NoiseProcess=} | {Denoiser=} | {FftThreads=} | {DenoiseMC=} | '
            + f'{NoiseTR=} | {Sigma=} | {ChromaNoise=} | {ShowNoise=} | {GrainRestore=} | {NoiseRestore=} | {NoiseDeint=} | {StabilizeNoise=} | {InputType=} | '
            + f'{ProgSADMask=} | {FPSDivisor=} | {ShutterBlur=} | {ShutterAngleSrc=} | {ShutterAngleOut=} | {SBlurLimit=} | {Border=} | {Precise=} | '
            + f'{Preset=} | {Tuning=} | {GlobalNames=} | {PrevGlobals=} | {ForceTR=} | {Str=} | {Amp=} | {FastMA=} | {ESearchP=} | {RefineMotion=}'
        )
        return output.text.Text(text=text)


def QTGMC_Interpolate(
    Input: vs.VideoNode,
    InputType: int,
    EdiMode: str,
    NNSize: int,
    NNeurons: int,
    EdiQual: int,
    EdiMaxD: int,
    Fallback: Optional[vs.VideoNode] = None,
    ChromaEdi: str = '',
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Interpolate input clip using method given in EdiMode. Use Fallback or Bob as result if mode not in list. If ChromaEdi string if set then interpolate chroma
    separately with that method (only really useful for EEDIx). The function is used as main algorithm starting point and for first two source-match stages
    '''
    is_gray = Input.format.color_family == vs.GRAY
    if is_gray:
        ChromaEdi = ''
    planes = [0, 1, 2] if ChromaEdi == '' and not is_gray else [0]

    field = 3 if TFF else 2

    if opencl:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, field=field, device=device, **nnedi3_args)
        eedi3 = partial(core.eedi3m.EEDI3CL, field=field, planes=planes, mdis=EdiMaxD, device=device, **eedi3_args)
    else:
        nnedi3 = partial(core.znedi3.nnedi3, field=field, **nnedi3_args)
        eedi3 = partial(core.eedi3m.EEDI3, field=field, planes=planes, mdis=EdiMaxD, **eedi3_args)

    if InputType == 1:
        return Input
    elif EdiMode == 'nnedi3':
        interp = nnedi3(Input, planes=planes, nsize=NNSize, nns=NNeurons, qual=EdiQual)
    elif EdiMode == 'eedi3+nnedi3':
        interp = eedi3(Input, sclip=nnedi3(Input, planes=planes, nsize=NNSize, nns=NNeurons, qual=EdiQual))
    elif EdiMode == 'eedi3':
        interp = eedi3(Input)
    elif EdiMode == 'bwdif':
        interp = Input.bwdif.Bwdif(field=field)
    else:
        interp = fallback(Fallback, Input.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5))

    if ChromaEdi == 'nnedi3':
        interpuv = nnedi3(Input, planes=[1, 2], nsize=4, nns=0, qual=1)
    elif ChromaEdi == 'bwdif':
        interpuv = Input.bwdif.Bwdif(field=field)
    elif ChromaEdi == 'bob':
        interpuv = Input.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
    else:
        return interp

    return core.std.ShufflePlanes([interp, interpuv], planes=[0, 1, 2], colorfamily=Input.format.color_family)


def QTGMC_KeepOnlyBobShimmerFixes(Input: vs.VideoNode, Ref: vs.VideoNode, Rep: int = 1, Chroma: bool = True) -> vs.VideoNode:
    '''
    Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference, i.e. bob shimmer fixes
    Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed
                     by this process. Restore remaining areas of difference back to as they were in reference clip
    '''
    is_gray = Input.format.color_family == vs.GRAY
    planes = [0, 1, 2] if Chroma and not is_gray else [0]

    bits = get_depth(Input)
    neutral = 1 << (bits - 1)

    # ed is the erosion distance - how much to deflate then reflate to remove thin areas of interest: 0 = minimum to 6 = maximum
    # od is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:  0 = none to 3 = one full pixel
    # If Rep < 10, then ed = Rep and od = 0, otherwise ed = 10s digit and od = 1s digit (nasty method, but kept for compatibility with original TGMC)
    ed = Rep if Rep < 10 else Rep // 10
    od = 0 if Rep < 10 else Rep % 10

    diff = core.std.MakeDiff(Ref, Input)

    coordinates = [0, 1, 0, 0, 0, 0, 1, 0]

    # Areas of positive difference
    choke1 = diff.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke1 = choke1.std.Deflate(planes=planes)
    if ed in [2, 5]:
        choke1 = choke1.std.Median(planes=planes)
    choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)

    # Over-dilation - extra reflation up to about 1 pixel
    if od == 1:
        choke1 = choke1.std.Inflate(planes=planes)
    elif od == 2:
        choke1 = choke1.std.Inflate(planes=planes).std.Inflate(planes=planes)
    elif od >= 3:
        choke1 = choke1.std.Maximum(planes=planes)

    # Areas of negative difference (similar to above)
    choke2 = diff.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke2 = choke2.std.Inflate(planes=planes)
    if ed in [2, 5]:
        choke2 = choke2.std.Median(planes=planes)
    choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)

    if od == 1:
        choke2 = choke2.std.Deflate(planes=planes)
    elif od == 2:
        choke2 = choke2.std.Deflate(planes=planes).std.Deflate(planes=planes)
    elif od >= 3:
        choke2 = choke2.std.Minimum(planes=planes)

    # Combine above areas to find those areas of difference to restore
    expr1 = f'x {scale_value(129, 8, bits)} < x y {neutral} < {neutral} y ? ?'
    expr2 = f'x {scale_value(127, 8, bits)} > x y {neutral} > {neutral} y ? ?'
    restore = core.std.Expr(
        [core.std.Expr([diff, choke1], expr=expr1 if Chroma or is_gray else [expr1, '']), choke2], expr=expr2 if Chroma or is_gray else [expr2, '']
    )
    return core.std.MergeDiff(Input, restore, planes=planes)


def QTGMC_Generate2ndFieldNoise(Input: vs.VideoNode, InterleavedClip: vs.VideoNode, ChromaNoise: bool = False, TFF: Optional[bool] = None) -> vs.VideoNode:
    '''
    Given noise extracted from an interlaced source (i.e. the noise is interlaced), generate "progressive" noise with a new "field" of noise injected. The new
    noise is centered on a weighted local average and uses the difference between local min & max as an estimate of local variance
    '''
    is_gray = Input.format.color_family == vs.GRAY
    planes = [0, 1, 2] if ChromaNoise and not is_gray else [0]

    bits = get_depth(Input)
    neutral = 1 << (bits - 1)

    origNoise = Input.std.SeparateFields(tff=TFF)
    noiseMax = origNoise.std.Maximum(planes=planes).std.Maximum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    noiseMin = origNoise.std.Minimum(planes=planes).std.Minimum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    random = (
        InterleavedClip.std.SeparateFields(tff=TFF)
        .std.BlankClip(color=[neutral] * Input.format.num_planes)
        .grain.Add(var=1800, uvar=1800 if ChromaNoise else 0)
    )
    expr = f'x {neutral} - y * {scale_value(256, 8, bits)} / {neutral} +'
    varRandom = core.std.Expr([core.std.MakeDiff(noiseMax, noiseMin, planes=planes), random], expr=expr if ChromaNoise or is_gray else [expr, ''])
    newNoise = core.std.MergeDiff(noiseMin, varRandom, planes=planes)
    return core.std.Interleave([origNoise, newNoise]).std.DoubleWeave(TFF)[::2]


def QTGMC_MakeLossless(Input: vs.VideoNode, Source: vs.VideoNode, InputType: int, TFF: Optional[bool] = None) -> vs.VideoNode:
    '''
    Insert the source lines into the result to create a true lossless output. However, the other lines in the result have had considerable processing and won't
    exactly match source lines. There will be some slight residual combing. Use vertical medians to clean a little of this away
    '''
    if InputType == 1:
        raise vs.Error('QTGMC: lossless modes are incompatible with InputType=1')

    neutral = 1 << (get_depth(Input) - 1)

    # Weave the source fields and the "new" fields that have generated in the input
    if InputType <= 0:
        srcFields = Source.std.SeparateFields(tff=TFF)
    else:
        srcFields = Source.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3])
    newFields = Input.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[1, 2])
    processed = core.std.Interleave([srcFields, newFields]).std.SelectEvery(cycle=4, offsets=[0, 1, 3, 2]).std.DoubleWeave(TFF)[::2]

    # Clean some of the artefacts caused by the above - creating a second version of the "new" fields
    vertMedian = processed.rgvs.VerticalCleaner(mode=1)
    vertMedDiff = core.std.MakeDiff(processed, vertMedian)
    vmNewDiff1 = vertMedDiff.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[1, 2])
    vmNewDiff2 = core.std.Expr(
        [vmNewDiff1.rgvs.VerticalCleaner(mode=1), vmNewDiff1], expr=f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} - abs y {neutral} - abs < x y ? ?'
    )
    vmNewDiff3 = core.rgvs.Repair(vmNewDiff2, vmNewDiff2.rgvs.RemoveGrain(mode=2), mode=1)

    # Reweave final result
    return core.std.Interleave([srcFields, core.std.MakeDiff(newFields, vmNewDiff3)]).std.SelectEvery(cycle=4, offsets=[0, 1, 3, 2]).std.DoubleWeave(TFF)[::2]


def QTGMC_ApplySourceMatch(
    Deinterlace: vs.VideoNode,
    InputType: int,
    Source: vs.VideoNode,
    bVec1: Union[vs.VideoNode, None],
    fVec1: Union[vs.VideoNode, None],
    bVec2: Union[vs.VideoNode, None],
    fVec2: Union[vs.VideoNode, None],
    SubPel: int,
    SubPelInterp: int,
    hpad: int,
    vpad: int,
    ThSAD1: int,
    ThSCD1: int,
    ThSCD2: int,
    SourceMatch: int,
    MatchTR1: int,
    MatchEdi: str,
    MatchNNSize: int,
    MatchNNeurons: int,
    MatchEdiQual: int,
    MatchEdiMaxD: int,
    MatchTR2: int,
    MatchEdi2: str,
    MatchNNSize2: int,
    MatchNNeurons2: int,
    MatchEdiQual2: int,
    MatchEdiMaxD2: int,
    MatchEnhance: float,
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Source-match, a three stage process that takes the difference between deinterlaced input and the original interlaced source, to shift the input more towards
    the source without introducing shimmer. All other arguments defined in main script
    '''

    # Basic source-match. Find difference between source clip & equivalent fields in interpolated/smoothed clip (called the "error" in formula below). Ideally
    # there should be no difference, we want the fields in the output to be as close as possible to the source whilst remaining shimmer-free. So adjust the
    # *source* in such a way that smoothing it will give a result closer to the unadjusted source. Then rerun the interpolation (edi) and binomial smooth with
    # this new source. Result will still be shimmer-free and closer to the original source.
    # Formula used for correction is P0' = P0 + (P0-P1)/(k+S(1-k)), where P0 is original image, P1 is the 1st attempt at interpolation/smoothing , P0' is the
    # revised image to use as new source for interpolation/smoothing, k is the weighting given to the current frame in the smooth, and S is a factor indicating
    # "temporal similarity" of the error from frame to frame, i.e. S = average over all pixels of [neighbor frame error / current frame error] . Decreasing
    # S will make the result sharper, sensible range is about -0.25 to 1.0. Empirically, S=0.5 is effective [will do deeper analysis later]
    errorTemporalSimilarity = 0.5  # S in formula described above
    errorAdjust1 = [1.0, 2.0 / (1.0 + errorTemporalSimilarity), 8.0 / (3.0 + 5.0 * errorTemporalSimilarity)][MatchTR1]
    if SourceMatch < 1 or InputType == 1:
        match1Clip = Deinterlace
    else:
        match1Clip = Deinterlace.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]).std.DoubleWeave(TFF)[::2]
    if SourceMatch < 1 or MatchTR1 <= 0:
        match1Update = Source
    else:
        match1Update = core.std.Expr([Source, match1Clip], expr=f'x {errorAdjust1 + 1} * y {errorAdjust1} * -')
    if SourceMatch > 0:
        match1Edi = QTGMC_Interpolate(
            match1Update,
            InputType,
            MatchEdi,
            MatchNNSize,
            MatchNNeurons,
            MatchEdiQual,
            MatchEdiMaxD,
            TFF=TFF,
            nnedi3_args=nnedi3_args,
            eedi3_args=eedi3_args,
            opencl=opencl,
            device=device,
        )
        if MatchTR1 > 0:
            match1Super = match1Edi.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match1Degrain1 = core.mv.Degrain1(match1Edi, match1Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR1 > 1:
            match1Degrain2 = core.mv.Degrain1(match1Edi, match1Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 1:
        match1 = Deinterlace
    elif MatchTR1 <= 0:
        match1 = match1Edi
    elif MatchTR1 == 1:
        match1 = core.std.Merge(match1Degrain1, match1Edi, weight=0.25)
    else:
        match1 = core.std.Merge(core.std.Merge(match1Degrain1, match1Degrain2, weight=0.2), match1Edi, weight=0.0625)

    if SourceMatch < 2:
        return match1

    # Enhance effect of source-match stages 2 & 3 by sharpening clip prior to refinement (source-match tends to underestimate so this will leave result sharper)
    if SourceMatch > 1 and MatchEnhance > 0:
        match1Shp = core.std.Expr([match1, match1.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])], expr=f'x x y - {MatchEnhance} * +')
    else:
        match1Shp = match1

    # Source-match refinement. Find difference between source clip & equivalent fields in (updated) interpolated/smoothed clip. Interpolate & binomially smooth
    # this difference then add it back to output. Helps restore differences that the basic match missed. However, as this pass works on a difference rather than
    # the source image it can be prone to occasional artefacts (difference images are not ideal for interpolation). In fact a lower quality interpolation such
    # as a simple bob often performs nearly as well as advanced, slower methods (e.g. NNEDI3)
    if SourceMatch < 2 or InputType == 1:
        match2Clip = match1Shp
    else:
        match2Clip = match1Shp.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]).std.DoubleWeave(TFF)[::2]
    if SourceMatch > 1:
        match2Diff = core.std.MakeDiff(Source, match2Clip)
        match2Edi = QTGMC_Interpolate(
            match2Diff,
            InputType,
            MatchEdi2,
            MatchNNSize2,
            MatchNNeurons2,
            MatchEdiQual2,
            MatchEdiMaxD2,
            TFF=TFF,
            nnedi3_args=nnedi3_args,
            eedi3_args=eedi3_args,
            opencl=opencl,
            device=device,
        )
        if MatchTR2 > 0:
            match2Super = match2Edi.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match2Degrain1 = core.mv.Degrain1(match2Edi, match2Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match2Degrain2 = core.mv.Degrain1(match2Edi, match2Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 2:
        match2 = match1
    elif MatchTR2 <= 0:
        match2 = match2Edi
    elif MatchTR2 == 1:
        match2 = core.std.Merge(match2Degrain1, match2Edi, weight=0.25)
    else:
        match2 = core.std.Merge(core.std.Merge(match2Degrain1, match2Degrain2, weight=0.2), match2Edi, weight=0.0625)

    # Source-match second refinement - correct error introduced in the refined difference by temporal smoothing. Similar to error correction from basic step
    errorAdjust2 = [1.0, 2.0 / (1.0 + errorTemporalSimilarity), 8.0 / (3.0 + 5.0 * errorTemporalSimilarity)][MatchTR2]
    if SourceMatch < 3 or MatchTR2 <= 0:
        match3Update = match2Edi
    else:
        match3Update = core.std.Expr([match2Edi, match2], expr=f'x {errorAdjust2 + 1} * y {errorAdjust2} * -')
    if SourceMatch > 2:
        if MatchTR2 > 0:
            match3Super = match3Update.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match3Degrain1 = core.mv.Degrain1(match3Update, match3Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match3Degrain2 = core.mv.Degrain1(match3Update, match3Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 3:
        match3 = match2
    elif MatchTR2 <= 0:
        match3 = match3Update
    elif MatchTR2 == 1:
        match3 = core.std.Merge(match3Degrain1, match3Update, weight=0.25)
    else:
        match3 = core.std.Merge(core.std.Merge(match3Degrain1, match3Degrain2, weight=0.2), match3Update, weight=0.0625)

    # Apply difference calculated in source-match refinement
    return core.std.MergeDiff(match1Shp, match3)


def QTGMC_SetUserGlobal(Prefix: str, Name: str, Value: Union[vs.VideoNode, None]) -> None:
    '''Set global variable called "Prefix_Name" to "Value".'''
    global QTGMC_globals
    QTGMC_globals[f'{Prefix}_{Name}'] = Value


def QTGMC_GetUserGlobal(Prefix: str, Name: str) -> Union[vs.VideoNode, None]:
    '''Return value of global variable called "Prefix_Name". Returns None if it doesn't exist'''
    global QTGMC_globals
    return QTGMC_globals.get(f'{Prefix}_{Name}')


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


def smartfademod(clip: vs.VideoNode, threshold: float = 0.4, show: bool = False, tff: Optional[bool] = None) -> vs.VideoNode:
    '''
    Aimed at removing interlaced fades in anime. Uses luma difference between two fields as activation threshold.

    Parameters:
        clip: Clip to process.

        threshold: Threshold for fade detection.

        show: Displays luma difference between fields without processing anything.

        tff: Since VapourSynth only has a weak notion of field order internally, tff may have to be set. Setting tff to true means top field first and false
            means bottom field first. Note that the _FieldBased frame property, if present, takes precedence over tff.
    '''

    def frame_eval(n: int, f: Sequence[vs.VideoFrame], orig: vs.VideoNode, defade: vs.VideoNode) -> vs.VideoNode:
        diff = abs(f[0].props['PlaneStatsAverage'] - f[1].props['PlaneStatsAverage']) * 255
        if show:
            return orig.text.Text(text=diff)
        return defade if diff > threshold else orig

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('smartfademod: this is not a clip')

    if tff is None:
        with clip.get_frame(0) as f:
            if f.props.get('_FieldBased') not in [1, 2]:
                raise vs.Error('smartfademod: tff was not specified and field order could not be determined from frame properties')

    sep = clip.std.SeparateFields(tff=tff)
    even = sep[::2].std.PlaneStats()
    odd = sep[1::2].std.PlaneStats()
    defade = daa(clip)
    return clip.std.FrameEval(eval=partial(frame_eval, orig=clip, defade=defade), prop_src=[even, odd], clip_src=[clip, defade])


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
        var_p = f'x {protect} - {scale_8bit(input, 16)} /'
        exprP = f'x {protect} <= 0 x {protect + scale_8bit(input, 16)} >= 1 ' + _sine_expr(var_p) + f' ? ?'
    else:
        exprP = f'x {protect} <= 0 x {protect + scale_8bit(input, 16)} >= 1 x {protect} - {scale_8bit(input, 16)} / abs ? ?'

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


###### srestore v2.7e ######
def srestore(source, frate=None, omode=6, speed=None, mode=2, thresh=16, dclip=None):
    if not isinstance(source, vs.VideoNode):
        raise vs.Error('srestore: this is not a clip')

    if source.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    if dclip is None:
        dclip = source
    elif not isinstance(dclip, vs.VideoNode):
        raise vs.Error("srestore: 'dclip' is not a clip")
    elif dclip.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    neutral = 1 << (source.format.bits_per_sample - 1)
    peak = (1 << source.format.bits_per_sample) - 1

    ###### parameters & other necessary vars ######
    srad = math.sqrt(abs(speed)) * 4 if speed is not None and abs(speed) >= 1 else 12
    irate = source.fps_num / source.fps_den
    bsize = 16 if speed is not None and speed > 0 else 32
    bom = isinstance(omode, str)
    thr = abs(thresh) + 0.01

    if bom or abs(omode - 3) < 2.5:
        frfac = 1
    elif frate is not None:
        if frate * 5 < irate or frate > irate:
            frfac = 1
        else:
            frfac = abs(frate) / irate
    elif cround(irate * 10010) % 30000 == 0:
        frfac = 1001 / 2400
    else:
        frfac = 480 / 1001

    if abs(frfac * 1001 - cround(frfac * 1001)) < 0.01:
        numr = cround(frfac * 1001)
    elif abs(1001 / frfac - cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = cround(frfac * 9000)
    if frate is not None and abs(irate * numr / cround(numr / frfac) - frate) > abs(irate * cround(frate * 100) / cround(irate * 100) - frate):
        numr = cround(frate * 100)
    denm = cround(numr / frfac)

    ###### source preparation & lut ######
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(core.std.Merge(source, source.std.Trim(first=1), weight=[0, 0.5]), source.std.Trim(first=1), weight=[0.5, 0])

    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)
    dclip = dclip.resize.Point(dclip.width if srad == 4 else int(dclip.width / 2 / srad + 4) * 4, dclip.height if srad == 4 else int(dclip.height / 2 / srad + 4) * 4).std.Trim(first=2)
    if mode < 0:
        dclip = core.std.StackVertical([core.std.StackHorizontal([plane(dclip, 1), plane(dclip, 2)]), plane(dclip, 0)])
    else:
        dclip = plane(dclip, 0)
    if bom:
        dclip = dclip.std.Expr(expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - * ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = core.std.Expr([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = core.std.Expr([diff.std.Trim(first=1), core.std.MergeDiff(diff, diff.std.Trim(first=2))], expr=[expr2]).resize.Bilinear(bsize, bsize)
    dclp = diff.std.Trim(first=1).std.Lut(function=lambda x: max(cround(abs(x - 128) ** 1.1 - 1), 0)).resize.Bilinear(bsize, bsize)

    ###### postprocessing ######
    if bom:
        sourceDuplicate = source.std.DuplicateFrames(frames=[0])
        sourceTrim1 = source.std.Trim(first=1)
        sourceTrim2 = source.std.Trim(first=2)

        unblend1 = core.std.Expr([sourceDuplicate, source], expr=['y 2 * x -'])
        unblend2 = core.std.Expr([sourceTrim1, sourceTrim2], expr=['x 2 * y -'])

        qmask1 = core.std.MakeDiff(unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]), unblend1, planes=[0])
        qmask2 = core.std.MakeDiff(unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]), unblend2, planes=[0])
        diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
        bmask = core.std.Expr([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = 'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'.format(i=scale_8bit(source, 4), peak=peak, j=scale_8bit(source, 200), k=scale_8bit(source, 28))
        dmask = core.std.Expr([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = core.std.Expr([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = core.std.Expr([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['y x 2 / - z a 2 / - +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(unblend1, unblend2, dmask.std.Convolution(matrix=matrix, planes=[0]).std.Expr(expr=['', repr(neutral)]))
        elif omode == 'pp2':
            fin = core.std.MaskedMerge(unblend1, unblend2, bmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True)
        elif omode == 'pp3':
            fin = core.std.MaskedMerge(unblend1, unblend2, pmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True).std.Convolution(matrix=matrix, planes=[1, 2])
        else:
            raise vs.Error('srestore: unexpected value for omode')

    ###### initialise variables ######
    lfr = -100
    offs = 0
    ldet = -100
    lpos = 0
    d32 = d21 = d10 = d01 = d12 = d23 = d34 = None
    m42 = m31 = m20 = m11 = m02 = m13 = m24 = None
    bp2 = bp1 = bn0 = bn1 = bn2 = bn3 = None
    cp2 = cp1 = cn0 = cn1 = cn2 = cn3 = None

    def srestore_inside(n, f):
        nonlocal lfr, offs, ldet, lpos, d32, d21, d10, d01, d12, d23, d34, m42, m31, m20, m11, m02, m13, m24, bp2, bp1, bn0, bn1, bn2, bn3, cp2, cp1, cn0, cn1, cn2, cn3

        ### preparation ###
        jmp = lfr + 1 == n
        cfo = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
        bfo = cfo > -numr and cfo <= numr
        lfr = n
        offs = offs + 2 * denm if bfo and offs <= -4 * numr else offs - 2 * denm if bfo and offs >= 4 * numr else offs
        pos = 0 if frfac == 1 else -cround((cfo + offs) / (2 * numr)) if bfo else lpos
        cof = cfo + offs + 2 * numr * pos
        ldet = -1 if n + pos == ldet else n + pos

        ### diff value shifting ###
        d_v = f[1].props['PlaneStatsMax'] + 0.015625
        if jmp:
            d43 = d32
            d32 = d21
            d21 = d10
            d10 = d01
            d01 = d12
            d12 = d23
            d23 = d34
        else:
            d43 = d32 = d21 = d10 = d01 = d12 = d23 = d_v
        d34 = d_v

        m_v = f[2].props['PlaneStatsDiff'] * 255 + 0.015625 if not bom and abs(omode) > 5 else 1
        if jmp:
            m53 = m42
            m42 = m31
            m31 = m20
            m20 = m11
            m11 = m02
            m02 = m13
            m13 = m24
        else:
            m53 = m42 = m31 = m20 = m11 = m02 = m13 = m_v
        m24 = m_v

        ### get blend and clear values ###
        b_v = 128 - f[0].props['PlaneStatsMin']
        if b_v < 1:
            b_v = 0.125
        c_v = f[0].props['PlaneStatsMax'] - 128
        if c_v < 1:
            c_v = 0.125

        ### blend value shifting ###
        if jmp:
            bp3 = bp2
            bp2 = bp1
            bp1 = bn0
            bn0 = bn1
            bn1 = bn2
            bn2 = bn3
        else:
            bp3 = b_v - c_v if bom else b_v
            bp2 = bp1 = bn0 = bn1 = bn2 = bp3
        bn3 = b_v - c_v if bom else b_v

        ### clear value shifting ###
        if jmp:
            cp3 = cp2
            cp2 = cp1
            cp1 = cn0
            cn0 = cn1
            cn1 = cn2
            cn2 = cn3
        else:
            cp3 = cp2 = cp1 = cn0 = cn1 = cn2 = c_v
        cn3 = c_v

        ### used detection values ###
        bb = [bp3, bp2, bp1, bn0, bn1][pos + 2]
        bc = [bp2, bp1, bn0, bn1, bn2][pos + 2]
        bn = [bp1, bn0, bn1, bn2, bn3][pos + 2]

        cb = [cp3, cp2, cp1, cn0, cn1][pos + 2]
        cc = [cp2, cp1, cn0, cn1, cn2][pos + 2]
        cn = [cp1, cn0, cn1, cn2, cn3][pos + 2]

        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]
        dn2 = [d01, d12, d23, d34, d34][pos + 2]

        mb1 = [m53, m42, m31, m20, m11][pos + 2]
        mb = [m42, m31, m20, m11, m02][pos + 2]
        mc = [m31, m20, m11, m02, m13][pos + 2]
        mn = [m20, m11, m02, m13, m24][pos + 2]
        mn1 = [m11, m02, m13, m24, 0.01][pos + 2]

        ### basic calculation ###
        bbool = 0.8 * bc * cb > bb * cc and 0.8 * bc * cn > bn * cc and bc * bc > cc
        blend = bbool and bc * 5 > cc and dbc + dcn > 1.5 * thr and (dbb < 7 * dbc or dbb < 8 * dcn) and (dnn < 8 * dcn or dnn < 7 * dbc) and (mb < mb1 and mb < mc or mn < mn1 and mn < mc or (dbb + dnn) * 4 < dbc + dcn or (bb * cc * 5 < bc * cb or mb > thr) and (bn * cc * 5 < bc * cn or mn > thr) and bc > thr)
        clear = dbb + dbc > thr and dcn + dnn > thr and (bc < 2 * bb or bc < 2 * bn) and (dbb + dnn) * 2 > dbc + dcn and (mc < 0.96 * mb and mc < 0.96 * mn and (bb * 2 > cb or bn * 2 > cn) and cc > cb and cc > cn or frfac > 0.45 and frfac < 0.55 and 0.8 * mc > mb1 and 0.8 * mc > mn1 and mb > 0.8 * mn and mn > 0.8 * mb)
        highd = dcn > 5 * dbc and dcn > 5 * dnn and dcn > thr and dbc < thr and dnn < thr
        lowd = dcn * 5 < dbc and dcn * 5 < dnn and dbc > thr and dnn > thr and dcn < thr and frfac > 0.35 and (frfac < 0.51 or dcn * 5 < dbb)
        res = d43 < thr and d32 < thr and d21 < thr and d10 < thr and d01 < thr and d12 < thr and d23 < thr and d34 < thr or dbc * 4 < dbb and dcn * 4 < dbb and dnn * 4 < dbb and dn2 * 4 < dbb or dcn * 4 < dbc and dnn * 4 < dbc and dn2 * 4 < dbc

        ### offset calculation ###
        if blend:
            odm = denm
        elif clear:
            odm = 0
        elif highd:
            odm = denm - numr
        elif lowd:
            odm = 2 * denm - numr
        else:
            odm = cof
        odm += cround((cof - odm) / (2 * denm)) * 2 * denm

        if blend:
            odr = denm - numr
        elif clear or highd:
            odr = numr
        elif frfac < 0.5:
            odr = 2 * numr
        else:
            odr = 2 * (denm - numr)
        odr *= 0.9

        if ldet >= 0:
            if cof > odm + odr:
                if cof - offs - odm - odr > denm and res:
                    cof = odm + 2 * denm - odr
                else:
                    cof = odm + odr
            elif cof < odm - odr:
                if offs > denm and res:
                    cof = odm - 2 * denm + odr
                else:
                    cof = odm - odr
            elif offs < -1.15 * denm and res:
                cof += 2 * denm
            elif offs > 1.25 * denm and res:
                cof -= 2 * denm

        offs = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
        lpos = pos
        opos = 0 if frfac == 1 else -cround((cfo + offs + (denm if bfo and offs <= -4 * numr else 0)) / (2 * numr))
        pos = min(max(opos, -2), 2)

        ### frame output calculation - resync - dup ###
        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]

        ### dup_hq - merge ###
        if opos != pos or abs(mode) < 2 or abs(mode) == 3:
            dup = 0
        elif dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == lpos) or (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == lpos and dnn < 0.9 * dbc or dnn * 9 < dbc and dcn * 3 < dbc:
            dup = 1
        elif (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == lpos and dbb < 0.9 * dcn or dbb * 9 < dcn and dbc * 3 < dcn or dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == lpos):
            dup = -1
        else:
            dup = 0
        mer = not bom and opos == pos and dup == 0 and abs(mode) > 2 and (dbc * 8 < dcn or dbc * 8 < dbb or dcn * 8 < dbc or dcn * 8 < dnn or dbc * 2 < thr or dcn * 2 < thr or dnn * 9 < dbc and dcn * 3 < dbc or dbb * 9 < dcn and dbc * 3 < dcn)

        ### deblend - doubleblend removal - postprocessing ###
        add = bp1 * cn2 > bn2 * cp1 * (1 + thr * 0.01) and bn0 * cn2 > bn2 * cn0 * (1 + thr * 0.01) and cn2 * bn1 > cn1 * bn2 * (1 + thr * 0.01)
        if bom:
            if bn0 > bp2 and bn0 >= bp1 and bn0 > bn1 and bn0 > bn2 and cn0 < 125:
                if d12 * d12 < d10 or d12 * 9 < d10:
                    dup = 1
                elif d10 * d10 < d12 or d10 * 9 < d12:
                    dup = 0
                else:
                    dup = 4
            elif bp1 > bp3 and bp1 >= bp2 and bp1 > bn0 and bp1 > bn1:
                dup = 1
            else:
                dup = 0
        elif dup == 0:
            if omode > 0 and omode < 5:
                if not bbool:
                    dup = 0
                elif omode == 4 and bp1 * cn1 < bn1 * cp1 or omode == 3 and d10 < d01 or omode == 1:
                    dup = -1
                else:
                    dup = 1
            elif omode == 5:
                if bp1 * cp2 > bp2 * cp1 * (1 + thr * 0.01) and bn0 * cp2 > bp2 * cn0 * (1 + thr * 0.01) and cp2 * bn1 > cn1 * bp2 * (1 + thr * 0.01) and (not add or cp2 * bn2 > cn2 * bp2):
                    dup = -2
                elif add:
                    dup = 2
                elif bn0 * cp1 > bp1 * cn0 and (bn0 * cn1 < bn1 * cn0 or cp1 * bn1 > cn1 * bp1):
                    dup = -1
                elif bn0 * cn1 > bn1 * cn0:
                    dup = 1
                else:
                    dup = 0
            else:
                dup = 0

        ### output clip ###
        if dup == 4:
            return fin
        else:
            oclp = mec if mer and dup == 0 else source
            opos += dup - (1 if dup == 0 and mer and dbc < dcn else 0)
            if opos < 0:
                return oclp.std.DuplicateFrames(frames=[0] * -opos)
            else:
                return oclp.std.Trim(first=opos)

    ###### evaluation call & output calculation ######
    bclpYStats = bclp.std.PlaneStats()
    dclpYStats = dclp.std.PlaneStats()
    dclipYStats = core.std.PlaneStats(dclip, dclip.std.Trim(first=2))
    last = source.std.FrameEval(eval=srestore_inside, prop_src=[bclpYStats, dclpYStats, dclipYStats])

    ###### final decimation ######
    return change_fps(last.std.Cache(make_linear=True), Fraction(source.fps_num * numr, source.fps_den * denm))


##############################################################################
# Original script by g-force converted into a stand alone script by McCauley #
# latest version from December 10, 2008                                      #
##############################################################################
def Stab(clp, dxmax=4, dymax=4, mirror=0):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Stab: this is not a clip')

    temp = average_frames(clp, weights=[1] * 15, scenechange=25 / 255)
    inter = core.std.Interleave([core.rgvs.Repair(temp, average_frames(clp, weights=[1] * 3, scenechange=25 / 255), mode=[1]), clp])
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


#####################
## Toon v0.82 edit ##
#####################
#
# function created by mf
#   support by Soulhunter ;-)
#   ported to masktools v2 and optimized by Didee (0.82)
#   added parameters and smaller changes by MOmonster (0.82 edited)
#
# toon v0.8 is the newest light-weight build of mf´s nice line darken function mf_toon
#
# Parameters:
#  str (float) - Strength of the line darken. Default is 1.0
#  l_thr (int) - Lower threshold for the linemask. Default is 2
#  u_thr (int) - Upper threshold for the linemask. Default is 12
#  blur (int)  - "blur" parameter of AWarpSharp2. Default is 2
#  depth (int) - "depth" parameter of AWarpSharp2. Default is 32
def Toon(input, str=1.0, l_thr=2, u_thr=12, blur=2, depth=32):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('Toon: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('Toon: RGB format is not supported')

    neutral = 1 << (input.format.bits_per_sample - 1)
    peak = (1 << input.format.bits_per_sample) - 1
    multiple = peak / 255

    if input.format.color_family != vs.GRAY:
        input_orig = input
        input = plane(input, 0)
    else:
        input_orig = None

    lthr = neutral + scale_8bit(input, l_thr)
    lthr8 = lthr / multiple
    uthr = neutral + scale_8bit(input, u_thr)
    uthr8 = uthr / multiple
    ludiff = u_thr - l_thr

    last = core.std.MakeDiff(input.std.Maximum().std.Minimum(), input)
    last = core.std.Expr([last, padder(last, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)], expr=['x y min'])
    expr = f'y {lthr} <= {neutral} y {uthr} >= x {uthr8} y {multiple} / - 128 * x {multiple} / y {multiple} / {lthr8} - * + {ludiff} / {multiple} * ? {neutral} - {str} * {neutral} + ?'
    last = core.std.MakeDiff(input, core.std.Expr([last, last.std.Maximum()], expr=[expr]))

    if input_orig is not None:
        last = core.std.ShufflePlanes([last, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return last


def bbmod(*args, **kwargs):
    raise vs.Error("havsfunc.bbmod outdated. Use https://github.com/OpusGang/awsmfunc instead.")


def ChangeFPS(*args, **kwargs):
    raise vs.Error("havsfunc.ChangeFPS outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-tools instead.")


def ContraSharpening(*args, **kwargs):
    raise vs.Error("havsfunc.ContraSharpening outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-rgtools instead.")


def dec_txt60mc(*args, **kwargs):
    raise vs.Error("havsfunc.dec_txt60mc outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deinterlace instead.")


def DeHalo_alpha(*args, **kwargs):
    raise vs.Error("havsfunc.DeHalo_alpha outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")


def DitherLumaRebuild(*args, **kwargs):
    raise vs.Error("havsfunc.DitherLumaRebuild outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-denoise instead.")


def EdgeCleaner(*args, **kwargs):
    raise vs.Error("havsfunc.EdgeCleaner outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")


def FineDehalo(*args, **kwargs):
    raise vs.Error("havsfunc.FineDehalo outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")


def FineDehalo_contrasharp(*args, **kwargs):
    raise vs.Error("havsfunc.FineDehalo_contrasharp outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-rgtools instead.")


def FineDehalo2(*args, **kwargs):
    raise vs.Error("havsfunc.FineDehalo2 outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")


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
    raise vs.Error("havsfunc.Gauss outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-rgtools instead.")


def GrainFactory3(*args, **kwargs):
    raise vs.Error("havsfunc.GrainFactory3 outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deband instead.")


def HQDeringmod(*args, **kwargs):
    raise vs.Error("havsfunc.HQDeringmod outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")


def ivtc_txt30mc(*args, **kwargs):
    raise vs.Error("havsfunc.ivtc_txt30mc outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deinterlace instead.")


def ivtc_txt60mc(*args, **kwargs):
    raise vs.Error("havsfunc.ivtc_txt60mc outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deinterlace instead.")


def KNLMeansCL(*args, **kwargs):
    raise vs.Error("havsfunc.KNLMeansCL outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-denoise instead.")


def m4(*args, **kwargs):
    raise vs.Error("havsfunc.m4 outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-tools instead.")


def MinBlur(*args, **kwargs):
    raise vs.Error("havsfunc.MinBlur outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-rgtools instead.")


def mt_expand_multi(*args, **kwargs):
    raise vs.Error("havsfunc.mt_expand_multi outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-masktools instead.")


def mt_inpand_multi(*args, **kwargs):
    raise vs.Error("havsfunc.mt_inpand_multi outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-masktools instead.")


def mt_inflate_multi(*args, **kwargs):
    raise vs.Error("havsfunc.mt_inflate_multi outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-masktools instead.")


def mt_deflate_multi(*args, **kwargs):
    raise vs.Error("havsfunc.mt_deflate_multi outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-masktools instead.")


def Padding(*args, **kwargs):
    raise vs.Error("havsfunc.Padding outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-tools instead.")


def santiag(*args, **kwargs):
    raise vs.Error("havsfunc.santiag outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-aa instead.")


def scale(*args, **kwargs):
    raise vs.Error("havsfunc.scale outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-tools instead.")


def SMDegrain(*args, **kwargs):
    raise vs.Error("havsfunc.SMDegrain outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-denoise instead.")


def Vinverse(*args, **kwargs):
    raise vs.Error("havsfunc.Vinverse outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deinterlace instead.")


def Vinverse2(*args, **kwargs):
    raise vs.Error("havsfunc.Vinverse2 outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-deinterlace instead.")


def YAHR(*args, **kwargs):
    raise vs.Error("havsfunc.YAHR outdated. Use https://github.com/Irrational-Encoding-Wizardry/vs-dehalo instead.")
