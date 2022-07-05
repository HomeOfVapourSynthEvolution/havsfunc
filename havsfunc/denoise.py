from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import vapoursynth as vs
from vsutil import get_depth
from vsutil import plane as vsu_plane
from vsutil import scale_value

from .blur import MinBlur, sbr
from .dehalo import DeHalo_alpha
from .helpers import AverageFrames, Weave, cround, scale
from .levels import DitherLumaRebuild
from .mask import AvsPrewitt, mt_expand_multi
from .misc import Overlay
from .sharp import ContraSharpening, LSFmod

core = vs.core


def Deblock_QED(
    clp: vs.VideoNode,
    quant1: int = 24, quant2: int = 26,
    aOff1: int = 1, aOff2: int = 1,
    bOff1: int = 2, bOff2: int = 2,
    uv: int = 3
) -> vs.VideoNode:
    """
    A postprocessed Deblock: Uses full frequencies of Deblock's changes on block borders,
    but DCT-lowpassed changes on block interiours.

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
    """
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
    block = clp.std.BlankClip(width=6, height=6,
                              format=clp.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0),
                              length=1, color=0)
    block = block.std.AddBorders(1, 1, 1, 1, color=peak)
    block = core.std.StackHorizontal([block for _ in range(clp.width // 8)])
    block = core.std.StackVertical([block for _ in range(clp.height // 8)])
    if not is_gray:
        blockc = block.std.CropAbs(width=clp.width >> clp.format.subsampling_w,
                                   height=clp.height >> clp.format.subsampling_h)
        block = core.std.ShufflePlanes([block, blockc], planes=[0, 0, 0], colorfamily=clp.format.color_family)
    block = block.std.Loop(times=clp.num_frames)

    # create normal deblocking (for block borders) and strong deblocking (for block interiour)
    normal = clp.deblock.Deblock(quant=quant1, aoffset=aOff1, boffset=bOff1, planes=[0, 1, 2]
                                 if uv != 2 and not is_gray else 0)
    strong = clp.deblock.Deblock(quant=quant2, aoffset=aOff2, boffset=bOff2, planes=[0, 1, 2]
                                 if uv != 2 and not is_gray else 0)

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

    # apply compensation from "normal" deblocking to the borders of the full-block-compensations calculated
    # from "strong" deblocking ...
    expr = f'y {neutral} = x y ?'
    strongD4 = core.std.Expr([strongD3, normalD2], expr=expr if uv > 2 or is_gray else [expr, ''])

    # ... and apply it.
    deblocked = core.std.MakeDiff(clp, strongD4, planes=planes)

    # simple decisions how to treat chroma
    if not is_gray:
        if uv < 0:
            deblocked = core.std.ShufflePlanes([deblocked, strong], planes=[0, 1, 2],
                                               colorfamily=clp.format.color_family)
        elif uv < 2:
            deblocked = core.std.ShufflePlanes([deblocked, normal], planes=[0, 1, 2],
                                               colorfamily=clp.format.color_family)

    # remove mod 8 borders
    return deblocked.std.Crop(right=padX, bottom=padY)


#################################################
###                                           ###
###                  logoNR                   ###
###                                           ###
###      by 06_taro - astrataro@gmail.com     ###
###                                           ###
###            v0.1 - 22 March 2012           ###
###                                           ###
#################################################
###
### Post-denoise filter of EraseLogo.
### Only process logo areas in logo frames, even if l/t/r/b are not set. Non-logo areas are left untouched.
###
###
### +---------+
### |  USAGE  |
### +---------+
###
### dlg [clip]
### ------------------
###    Clip after delogo.
###
### src [clip]
### ------------------
###    Clip before delogo.
###
### chroma [bool, default: True]
### ------------------
###    Process chroma plane or not.
###
### l/t/r/b [int, default: 0]
### ------------------
###    left/top/right/bottom pixels to be cropped for logo area.
###    Have the same restriction as Crop, e.g., no odd value for YV12.
###    logoNR only filters the logo areas in logo frames, no matter l/t/r/b are set or not.
###    So if you have other heavy filters running in a pipeline and don't care much about the speed of logoNR,
###    it is safe to left these values unset.
###    Setting these values only makes logoNR run faster, with rarely noticeable difference in result,
###    unless you set wrong values and the logo is not covered in your cropped target area.
###
### d/a/s/h [int, default: 1/2/2/3]
### ------------------
###    The same parameters of KNLMeansCL.
###
### +----------------+
### |  REQUIREMENTS  |
### +----------------+
###
### -> KNLMeansCL
### -> RGVS
def logoNR(dlg, src, chroma=True, l=0, t=0, r=0, b=0, d=1, a=2, s=2, h=3):
    if not (isinstance(dlg, vs.VideoNode) and isinstance(src, vs.VideoNode)):
        raise vs.Error('logoNR: this is not a clip')

    if dlg.format.id != src.format.id:
        raise vs.Error('logoNR: clips must have the same format')

    if dlg.format.color_family == vs.GRAY:
        chroma = False

    if not chroma and dlg.format.color_family != vs.GRAY:
        dlg_orig = dlg
        dlg = vsu_plane(dlg, 0)
        src = vsu_plane(src, 0)
    else:
        dlg_orig = None

    b_crop = (l != 0) or (t != 0) or (r != 0) or (b != 0)
    if b_crop:
        src = src.std.Crop(left=l, right=r, top=t, bottom=b)
        last = dlg.std.Crop(left=l, right=r, top=t, bottom=b)
    else:
        last = dlg

    if chroma:
        clp_nr = KNLMeansCL(last, d=d, a=a, s=s, h=h)
    else:
        clp_nr = last.knlm.KNLMeansCL(d=d, a=a, s=s, h=h)
    logoM = mt_expand_multi(core.std.Expr([last, src], expr=['x y - abs 16 *']), mode='losange', sw=3, sh=3) \
        .std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]).std.Deflate()
    clp_nr = core.std.MaskedMerge(last, clp_nr, logoM)
    if b_crop:
        clp_nr = Overlay(dlg, clp_nr, x=l, y=t)

    if dlg_orig is not None:
        clp_nr = core.std.ShufflePlanes([clp_nr, dlg_orig], planes=[0, 1, 2], colorfamily=dlg_orig.format.color_family)
    return clp_nr


##############################################################################
# Original script by g-force converted into a stand alone script by McCauley #
# latest version from December 10, 2008                                      #
##############################################################################
def Stab(clp, dxmax=4, dymax=4, mirror=0):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Stab: this is not a clip')

    temp = AverageFrames(clp, weights=[1] * 15, scenechange=25 / 255)
    inter = core.std.Interleave([core.rgvs.Repair(temp, AverageFrames(clp, weights=[1] * 3, scenechange=25 / 255),
                                                  mode=[1]), clp])
    mdata = inter.mv.DepanEstimate(trust=0, dxmax=dxmax, dymax=dymax)
    last = inter.mv.DepanCompensate(data=mdata, offset=-1, mirror=mirror)
    return last[::2]


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
###  nrmode (int)   - Mode to get grain/noise from input clip. 0: 3x3 Average Blur, 1: 3x3 SBR, 2: 5x5 SBR, 3: 7x7 SBR.
###                   Or define your own denoised clip "p". Default is 2 for HD / 1 for SD
###  radius (int)   - Temporal radius of MDegrain for grain stabilize (1-3). Default is 1
###  adapt (int)    - Threshold for luma-adaptative mask. -1: off, 0: source, 255: invert.
###                   Or define your own luma mask clip "Lmask". Default is -1
###  rep (int)      - Mode of repair to avoid artifacts, set 0 to turn off this operation. Default is 13
###  planes (int[]) - Whether to process the corresponding plane. The other planes will be passed through unchanged.
###
######
def GSMC(
    input,
    p=None,
    Lmask=None,
    nrmode=None,
    radius=1,
    adapt=-1,
    rep=13,
    planes=None,
    thSAD=300,
    thSADC=None,
    thSCD1=300,
    thSCD2=100,
    limit=None,
    limitc=None
) -> vs.VideoNode:
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
        limit = scale(limit, peak)
    if limitc is not None:
        limitc = scale(limitc, peak)

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
        pre_nr = input.std.Convolution(matrix=[1] * 9, planes=planes)
    else:
        pre_nr = sbr(input, r=nrmode, planes=planes)
    dif_nr = core.std.MakeDiff(input, pre_nr, planes=planes)

    # Kernel: MC Grain Stabilize
    psuper = DitherLumaRebuild(pre_nr, s0=1, chroma=chromamv).mv.Super(pel=1, chroma=chromamv)
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

    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane,
                        limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
    if radius <= 1:
        dif_sb = core.mv.Degrain1(dif_nr, difsuper, bv1, fv1, **degrain_args)
    elif radius == 2:
        dif_sb = core.mv.Degrain2(dif_nr, difsuper, bv1, fv1, bv2, fv2, **degrain_args)
    else:
        dif_sb = core.mv.Degrain3(dif_nr, difsuper, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)

    # Post-Process: Luma-Adaptive Mask Merging & Repairing
    stable = core.std.MergeDiff(pre_nr, dif_sb, planes=planes)
    if rep > 0:
        stable = core.rgvs.Repair(stable, input,
                                  mode=[rep if i in planes else 0 for i in range(input.format.num_planes)])

    if Lmask is not None:
        return core.std.MaskedMerge(input, stable, Lmask, planes=planes)
    elif adapt <= -1:
        return stable
    else:
        input_y = vsu_plane(input, 0)
        if adapt == 0:
            Lmask = input_y.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        elif adapt >= 255:
            Lmask = input_y.std.Invert().std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        else:
            expr = f'x {scale(adapt, peak)} - abs {peak} * {adapt} {neutral} - abs {neutral} + /'
            Lmask = input_y.std.Expr(expr=[expr]).std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
        return core.std.MaskedMerge(input, stable, Lmask, planes=planes)


########################################################################################################################
###                                                                                                                  ###
###                                   Motion-Compensated Temporal Denoise: MCTemporalDenoise()                       ###
###                                                                                                                  ###
###                                                     v1.4.20 by "LaTo INV."                                       ###
###                                                                                                                  ###
###                                                           2 July 2010                                            ###
###                                                                                                                  ###
########################################################################################################################
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
### +------------------------------+-----------------------------------------------------------------------------------+
### | edgeclean : Enable safe edgeclean process after the denoising                                                    |
###               (only on edges which are in non-detailed areas, so less detail loss)                                 |
### | ECrad     : Radius for mask (the higher, the greater distance from the edge is filtered)                         |
### | ECthr     : Threshold for mask (the higher, the less "small edges" are process) [0...255]                        |
### +------------------------------------------------------------------------------------------------------------------+
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
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+  # noqa
### | SETTINGS    |      VERY LOW        |      LOW             |      MEDIUM          |      HIGH            |      VERY HIGH       |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | radius      |      1               |      2               |      3               |      2               |      3               |  # noqa
### | pfMode      |      3               |      3               |      3               |      3               |      3               |  # noqa
### | sigma       |      2               |      4               |      8               |      12              |      16              |  # noqa
### | twopass     |      false           |      false           |      false           |      true            |      true            |  # noqa
### | useTTmpSm   |      false           |      false           |      false           |      false           |      false           |  # noqa
### | limit       |      -1              |      -1              |      -1              |      -1              |      0               |  # noqa
### | limit2      |      -1              |      -1              |      -1              |      0               |      0               |  # noqa
### | post        |      0               |      0               |      0               |      0               |      0               |  # noqa
### | chroma      |      false           |      false           |      true            |      true            |      true            |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | deblock     |      false           |      false           |      false           |      false           |      false           |  # noqa
### | useQED      |      true            |      true            |      true            |      false           |      false           |  # noqa
### | quant1      |      10              |      20              |      30              |      30              |      40              |  # noqa
### | quant2      |      20              |      40              |      60              |      60              |      80              |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | edgeclean   |      false           |      false           |      false           |      false           |      false           |  # noqa
### | ECrad       |      1               |      2               |      3               |      4               |      5               |  # noqa
### | ECthr       |      64              |      32              |      32              |      16              |      16              |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | stabilize   |      false           |      false           |      false           |      true            |      true            |  # noqa
### | maxr        |      1               |      1               |      2               |      2               |      2               |  # noqa
### | TTstr       |      1               |      1               |      1               |      2               |      2               |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | bwbh        |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |  # noqa
### | owoh        |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |  # noqa
### | blksize     |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |  # noqa
### | overlap     |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | bt          |      1               |      3               |      3               |      3               |      4               |  # noqa
### | ncpu        |      1               |      1               |      1               |      1               |      1               |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | thSAD       |      200             |      300             |      400             |      500             |      600             |  # noqa
### | thSADC      |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |  # noqa
### | thSAD2      |      200             |      300             |      400             |      500             |      600             |  # noqa
### | thSADC2     |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |  # noqa
### | thSCD1      |      200             |      300             |      400             |      500             |      600             |  # noqa
### | thSCD2      |      90              |      100             |      100             |      130             |      130             |  # noqa
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|  # noqa
### | truemotion  |      false           |      false           |      false           |      false           |      false           |  # noqa
### | MVglobal    |      true            |      true            |      true            |      true            |      true            |  # noqa
### | pel         |      1               |      2               |      2               |      2               |      2               |  # noqa
### | pelsearch   |      1               |      2               |      2               |      2               |      2               |  # noqa
### | search      |      4               |      4               |      4               |      4               |      4               |  # noqa
### | searchparam |      2               |      2               |      2               |      2               |      2               |  # noqa
### | MVsharp     |      2               |      2               |      2               |      1               |      0               |  # noqa
### | DCT         |      0               |      0               |      0               |      0               |      0               |  # noqa
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+  # noqa
###
########################################################################################################################
def MCTemporalDenoise(
    i, radius=None, pfMode=3, sigma=None, twopass=None, useTTmpSm=False, limit=None, limit2=None, post=0, chroma=None,
    refine=False, deblock=False, useQED=None, quant1=None, quant2=None, edgeclean=False, ECrad=None, ECthr=None,
    stabilize=None, maxr=None, TTstr=None, bwbh=None, owoh=None, blksize=None, overlap=None, bt=None, ncpu=1,
    thSAD=None, thSADC=None, thSAD2=None, thSADC2=None, thSCD1=None, thSCD2=None, truemotion=False, MVglobal=True,
    pel=None, pelsearch=None, search=4, searchparam=2, MVsharp=None, DCT=0, p=None, settings='low'
) -> vs.VideoNode:
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
    except ValueError:
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
    limit = scale(limit, peak)
    limit2 = scale(limit2, peak)
    post *= peak / 255
    ECthr = scale(ECthr, peak)
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
        p = i.fft3dfilter.FFT3DFilter(sigma=sigma * 0.8, sigma2=sigma * 0.6, sigma3=sigma * 0.4, sigma4=sigma * 0.2,
                                      **fft3d_args)
    elif pfMode >= 3:
        p = i.dfttest.DFTTest(tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0], planes=planes)
    else:
        p = MinBlur(i, r=pfMode, planes=planes)

    pD = core.std.MakeDiff(i, p, planes=planes)
    p = DitherLumaRebuild(p, s0=1, chroma=chroma)

    ### DEBLOCKING
    crop_args = dict(left=xf // 2, right=xf // 2, top=yf // 2, bottom=yf // 2)
    if not deblock:
        d = i
    elif useQED:
        d = Deblock_QED(i.std.Crop(**crop_args), quant1=quant1, quant2=quant2, uv=3 if chroma else 2) \
            .resize.Point(**pointresize_args)
    else:
        d = i.std.Crop(**crop_args).deblock.Deblock(quant=(quant1 + quant2) // 2, planes=planes) \
            .resize.Point(**pointresize_args)

    ### PREPARING
    super_args = dict(hpad=0, vpad=0, pel=pel, chroma=chroma, sharp=MVsharp)
    pMVS = p.mv.Super(rfilter=4 if refine else 2, **super_args)
    if refine:
        rMVS = p.mv.Super(levels=1, **super_args)

    analyse_args = dict(blksize=blksize, search=search, searchparam=searchparam, pelsearch=pelsearch, chroma=chroma,
                        truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=DCT)
    recalculate_args = dict(thsad=thSAD // 2, blksize=max(blksize // 2, 4), search=search, chroma=chroma,
                            truemotion=truemotion, overlap=max(overlap // 2, 2), dct=DCT)
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
            # SAD_m = core.std.Interleave([SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m,
            #                              SAD_b4m, SAD_b5m])
        else:
            c = core.std.Interleave([f6c, f5c, f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c, b5c, b6c])
            # SAD_m = core.std.Interleave([SAD_f6m, SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m,
            #                              SAD_b3m, SAD_b4m, SAD_b5m, SAD_b6m])

        # sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False,
        #                           pfclip=SAD_m, planes=planes)
        sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False,
                                  planes=planes)
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
        smP = smL.fft3dfilter.FFT3DFilter(sigma=post * 0.8, sigma2=post * 0.6, sigma3=post * 0.4, sigma4=post * 0.2,
                                          **fft3d_args)

    ### EDGECLEANING
    if edgeclean:
        mP = AvsPrewitt(vsu_plane(smP, 0))
        mS = mt_expand_multi(mP, sw=ECrad, sh=ECrad).std.Inflate()
        mD = core.std.Expr([mS, mP.std.Inflate()], expr=[f'x y - {ECthr} <= 0 x y - ?']).std.Inflate() \
            .std.Convolution(matrix=[1] * 9)
        smP = core.std.MaskedMerge(smP, DeHalo_alpha(smP.dfttest.DFTTest(tbsize=1, planes=planes), darkstr=0), mD,
                                   planes=planes)

    ### STABILIZING
    if stabilize:
        # mM = core.std.Merge(vsu_plane(SAD_f1m, 0), vsu_plane(SAD_b1m, 0)) \
        #     .std.Lut(function=lambda x: min(cround(x ** 1.6), peak))
        mE = AvsPrewitt(vsu_plane(smP, 0)).std.Lut(function=lambda x: min(cround(x ** 1.8), peak)) \
            .std.Median().std.Inflate()
        # mF = core.std.Expr([mM, mE], expr=['x y max']).std.Convolution(matrix=[1] * 9)
        mF = mE.std.Convolution(matrix=[1] * 9)
        TTc = smP.ttmpsm.TTempSmooth(maxr=maxr, mdiff=[255], strength=TTstr, planes=planes)
        smP = core.std.MaskedMerge(TTc, smP, mF, planes=planes)

    ### OUTPUT
    return smP.std.Crop(**crop_args)


################################################################################################
###                                                                                          ###
###                           Simple MDegrain Mod - SMDegrain()                              ###
###                                                                                          ###
###                       Mod by Dogway - Original idea by Caroliano                         ###
###                                                                                          ###
###          Special Thanks: Sagekilla, Didée, cretindesalpes, Gavino and MVtools people     ###
###                                                                                          ###
###                       v3.1.2d (Dogway's mod) - 21 July 2015                              ###
###                                                                                          ###
################################################################################################
###
### General purpose simple degrain function. Pure temporal denoiser. Basically a wrapper(function)/frontend
### of mvtools2+mdegrain with some added common related options.
###
### Goal is accessibility and quality but not targeted to any specific kind of source.
### The reason behind is to keep it simple so aside masktools2 you will only need MVTools2.
###
### Check documentation for deep explanation on settings and defaults.
### VideoHelp thread: (http://forum.videohelp.com/threads/369142)
###
################################################################################################

# Globals
bv6 = bv4 = bv3 = bv2 = bv1 = fv1 = fv2 = fv3 = fv4 = fv6 = None


def SMDegrain(input, tr=2, thSAD=300, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False,
              tff=None, plane=4, Globals=0, pel=None, subpixel=2, prefilter=-1, mfilter=None, blksize=None,
              overlap=None, search=4, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None, thSCD1=400,
              thSCD2=130, chroma=True, hpad=None, vpad=None, Str=1.0, Amp=0.0625):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SMDegrain: this is not a clip')

    if input.format.color_family == vs.GRAY:
        plane = 0
        chroma = False

    peak = (1 << input.format.bits_per_sample) - 1

    # Defaults & Conditionals
    thSAD2 = thSAD // 2
    if thSADC is None:
        thSADC = thSAD2

    GlobalR = (Globals == 1)
    GlobalO = (Globals >= 3)
    if1 = CClip is not None

    if contrasharp is None:
        contrasharp = not GlobalO and if1

    w = input.width
    h = input.height
    preclip = isinstance(prefilter, vs.VideoNode)
    ifC = isinstance(contrasharp, bool)
    if0 = contrasharp if ifC else contrasharp > 0
    if4 = w > 1024 or h > 576

    if pel is None:
        pel = 1 if if4 else 2
    if pel < 2:
        subpixel = min(subpixel, 2)
    pelclip = pel > 1 and subpixel >= 3

    if blksize is None:
        blksize = 16 if if4 else 8
    blk2 = blksize // 2
    if overlap is None:
        overlap = blk2
    ovl2 = overlap // 2
    if truemotion is None:
        truemotion = not if4
    if MVglobal is None:
        MVglobal = truemotion

    planes = [0, 1, 2] if chroma else [0]
    plane0 = (plane != 0)

    if hpad is None:
        hpad = blksize
    if vpad is None:
        vpad = blksize
    limit = scale(limit, peak)
    if limitc is None:
        limitc = limit
    else:
        limitc = scale(limitc, peak)

    # Error Report
    if not (ifC or isinstance(contrasharp, int)):
        raise vs.Error("SMDegrain: 'contrasharp' only accepts bool and integer inputs")
    if if1 and (not isinstance(CClip, vs.VideoNode) or CClip.format.id != input.format.id):
        raise vs.Error("SMDegrain: 'CClip' must be the same format as input")
    if interlaced and h & 3:
        raise vs.Error('SMDegrain: interlaced source requires mod 4 height sizes')
    if interlaced and not isinstance(tff, bool):
        raise vs.Error("SMDegrain: 'tff' must be set if source is interlaced. "
                       "Setting tff to true means top field first and false means bottom field first")
    if not (isinstance(prefilter, int) or preclip):
        raise vs.Error("SMDegrain: 'prefilter' only accepts integer and clip inputs")
    if preclip and prefilter.format.id != input.format.id:
        raise vs.Error("SMDegrain: 'prefilter' must be the same format as input")
    if mfilter is not None and (not isinstance(mfilter, vs.VideoNode) or mfilter.format.id != input.format.id):
        raise vs.Error("SMDegrain: 'mfilter' must be the same format as input")
    if RefineMotion and blksize < 8:
        raise vs.Error('SMDegrain: for RefineMotion you need a blksize of at least 8')
    if not chroma and plane != 0:
        raise vs.Error('SMDegrain: denoising chroma with luma only vectors is bugged in mvtools and thus unsupported')

    # RefineMotion Variables
    if RefineMotion:
        # MRecalculate works with half block size
        halfblksize = blk2
        # Halve the overlap to suit the halved block size
        halfoverlap = overlap if overlap <= 2 else ovl2 + ovl2 % 2
        # MRecalculate uses a more strict thSAD, which defaults to 150 (half of function's default of 300)
        halfthSAD = thSAD2

    # Input preparation for Interlacing
    if not interlaced:
        inputP = input
    else:
        inputP = input.std.SeparateFields(tff=tff)

    # Prefilter & Motion Filter
    if mfilter is None:
        mfilter = inputP

    if not GlobalR:
        if preclip:
            pref = prefilter
        elif prefilter <= -1:
            pref = inputP
        elif prefilter == 3:
            expr = f'x {scale(16, peak)} < {peak} x {scale(75, peak)} > 0 {peak} x {scale(16, peak)} - {peak} ' \
                   + f'{scale(75, peak)} {scale(16, peak)}'
            pref = core.std.MaskedMerge(inputP.dfttest.DFTTest(tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0],
                                                               planes=planes),
                                        inputP,
                                        vsu_plane(inputP, 0).std.Expr(expr=[expr]),
                                        planes=planes)
        elif prefilter >= 4:
            if chroma:
                pref = KNLMeansCL(inputP, d=1, a=1, h=7)
            else:
                pref = inputP.knlm.KNLMeansCL(d=1, a=1, h=7)
        else:
            pref = MinBlur(inputP, r=prefilter, planes=planes)
    else:
        pref = inputP

    # Default Auto-Prefilter - Luma expansion TV->PC (up to 16% more values for motion estimation)
    if not GlobalR:
        pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)

    # Subpixel 3
    if pelclip:
        import nnedi3_resample as nnrs
        cshift = 0.25 if pel == 2 else 0.375
        pclip = nnrs.nnedi3_resample(pref, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode='znedi3')
        if not GlobalR:
            pclip2 = nnrs.nnedi3_resample(inputP, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4,
                                          mode='znedi3')

    # Motion vectors search
    global bv6, bv4, bv3, bv2, bv1, fv1, fv2, fv3, fv4, fv6
    super_args = dict(hpad=hpad, vpad=vpad, pel=pel)
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal,
                        overlap=overlap, dct=dct)
    if RefineMotion:
        recalculate_args = dict(thsad=halfthSAD, blksize=halfblksize, search=search, chroma=chroma,
                                truemotion=truemotion, overlap=halfoverlap, dct=dct)

    if pelclip:
        super_search = pref.mv.Super(chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
        super_search = pref.mv.Super(chroma=chroma, sharp=subpixel, rfilter=4, **super_args)

    if not GlobalR:
        if pelclip:
            super_render = inputP.mv.Super(levels=1, chroma=plane0, pelclip=pclip2, **super_args)
            if RefineMotion:
                Recalculate = pref.mv.Super(levels=1, chroma=chroma, pelclip=pclip, **super_args)
        else:
            super_render = inputP.mv.Super(levels=1, chroma=plane0, sharp=subpixel, **super_args)
            if RefineMotion:
                Recalculate = pref.mv.Super(levels=1, chroma=chroma, sharp=subpixel, **super_args)

        if interlaced:
            if tr > 2:
                bv6 = super_search.mv.Analyse(isb=True, delta=6, **analyse_args)
                fv6 = super_search.mv.Analyse(isb=False, delta=6, **analyse_args)
                if RefineMotion:
                    bv6 = core.mv.Recalculate(Recalculate, bv6, **recalculate_args)
                    fv6 = core.mv.Recalculate(Recalculate, fv6, **recalculate_args)
            if tr > 1:
                bv4 = super_search.mv.Analyse(isb=True, delta=4, **analyse_args)
                fv4 = super_search.mv.Analyse(isb=False, delta=4, **analyse_args)
                if RefineMotion:
                    bv4 = core.mv.Recalculate(Recalculate, bv4, **recalculate_args)
                    fv4 = core.mv.Recalculate(Recalculate, fv4, **recalculate_args)
        else:
            if tr > 2:
                bv3 = super_search.mv.Analyse(isb=True, delta=3, **analyse_args)
                fv3 = super_search.mv.Analyse(isb=False, delta=3, **analyse_args)
                if RefineMotion:
                    bv3 = core.mv.Recalculate(Recalculate, bv3, **recalculate_args)
                    fv3 = core.mv.Recalculate(Recalculate, fv3, **recalculate_args)
            bv1 = super_search.mv.Analyse(isb=True, delta=1, **analyse_args)
            fv1 = super_search.mv.Analyse(isb=False, delta=1, **analyse_args)
            if RefineMotion:
                bv1 = core.mv.Recalculate(Recalculate, bv1, **recalculate_args)
                fv1 = core.mv.Recalculate(Recalculate, fv1, **recalculate_args)
        if interlaced or tr > 1:
            bv2 = super_search.mv.Analyse(isb=True, delta=2, **analyse_args)
            fv2 = super_search.mv.Analyse(isb=False, delta=2, **analyse_args)
            if RefineMotion:
                bv2 = core.mv.Recalculate(Recalculate, bv2, **recalculate_args)
                fv2 = core.mv.Recalculate(Recalculate, fv2, **recalculate_args)
    else:
        super_render = super_search

    # Finally, MDegrain
    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc,
                        thscd1=thSCD1, thscd2=thSCD2)
    if not GlobalO:
        if interlaced:
            if tr >= 3:
                output = core.mv.Degrain3(mfilter, super_render, bv2, fv2, bv4, fv4, bv6, fv6, **degrain_args)
            elif tr == 2:
                output = core.mv.Degrain2(mfilter, super_render, bv2, fv2, bv4, fv4, **degrain_args)
            else:
                output = core.mv.Degrain1(mfilter, super_render, bv2, fv2, **degrain_args)
        else:
            if tr >= 3:
                output = core.mv.Degrain3(mfilter, super_render, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)
            elif tr == 2:
                output = core.mv.Degrain2(mfilter, super_render, bv1, fv1, bv2, fv2, **degrain_args)
            else:
                output = core.mv.Degrain1(mfilter, super_render, bv1, fv1, **degrain_args)

    # Contrasharp (only sharpens luma)
    if not GlobalO and if0:
        if if1:
            if interlaced:
                CClip = CClip.std.SeparateFields(tff=tff)
        else:
            CClip = inputP

    # Output
    if not GlobalO:
        if if0:
            if interlaced:
                if ifC:
                    return Weave(ContraSharpening(output, CClip, planes=planes), tff=tff)
                else:
                    return Weave(LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False,
                                        defaults='slow'), tff=tff)
            elif ifC:
                return ContraSharpening(output, CClip, planes=planes)
            else:
                return LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow')
        elif interlaced:
            return Weave(output, tff=tff)
        else:
            return output
    else:
        return input


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

        RGmode: The spatial filter is RemoveGrain, this is its mode.
            It also accepts loading your personal prefiltered clip.

        tthr: Temporal threshold for fluxsmooth. Can be set "a good bit bigger" than usually.

        tlimit: The temporal filter won't change a pixel more than this.

        tbias: The percentage of the temporal filter that will apply.

        back: After all changes have been calculated, reduce all pixel changes by this value.
            (shift "back" towards original value)

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
        expr = f'x y - abs {scale_value(1, 8, bits)} < x x {LIM} + y < x {limit} + x {LIM} - y > x {limit} - ' \
               + f'x {100 - bias} * y {bias} * + 100 / ? ? ?'
    if tlimit < 0:
        texpr = f'x y - abs {TLIM} < x x 1 x y - dup abs / * - ?'
    else:
        texpr = f'x y - abs {scale_value(1, 8, bits)} < x x {TLIM} + y < x {tlimit} + x {TLIM} - y > x {tlimit} - ' \
                + f'x {100 - tbias} * y {tbias} * + 100 / ? ? ?'

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
            bzz = clp.std.Convolution(matrix=[1] * 9, planes=planes)
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


def KNLMeansCL(
    clip: vs.VideoNode,
    d: Optional[int] = None,
    a: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[float] = None,
    wmode: Optional[int] = None,
    wref: Optional[float] = None,
    device_type: Optional[str] = None,
    device_id: Optional[int] = None,
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('KNLMeansCL: this is not a clip')

    if clip.format.color_family != vs.YUV:
        raise vs.Error('KNLMeansCL: this wrapper is intended to be used only for YUV format')

    if clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0:
        den = clip.knlm.KNLMeansCL(d=d, a=a, s=s, h=h, wmode=wmode, wref=wref,
                                   device_type=device_type, device_id=device_id)
        return den.knlm.KNLMeansCL(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref,
                                   device_type=device_type, device_id=device_id)
    else:
        return clip.knlm.KNLMeansCL(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref,
                                    device_type=device_type, device_id=device_id)
