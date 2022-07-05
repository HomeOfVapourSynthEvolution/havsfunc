from __future__ import annotations

import math
from functools import partial
from typing import Optional, Sequence, Union

import vapoursynth as vs
from vsutil import get_depth, plane

from .blur import MinBlur
from .helpers import AverageFrames, Padding, cround, scale
from .mask import mt_clamp

core = vs.core


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
    lum = scale(luma_cap, peak)
    thr = scale(threshold, peak)
    thn = thinning / 16

    ## filtering ##
    exin = c.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = core.std.Expr([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +'])
    if thinning <= 0:
        last = thick
    else:
        diff = core.std.Expr([c, exin],
                             expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale(127, peak)} +'])
        linemask = diff.std.Minimum().std.Expr(expr=[f'x {scale(127, peak)} - {thn} * {peak} +']) \
            .std.Convolution(matrix=[1] * 9)
        thin = core.std.Expr([c.std.Maximum(), diff], expr=[f'x y {scale(127, peak)} - {Str} 1 + * +'])
        last = core.std.MaskedMerge(thin, thick, linemask)

    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
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

    lthr = neutral + scale(l_thr, peak)
    lthr8 = lthr / multiple
    uthr = neutral + scale(u_thr, peak)
    uthr8 = uthr / multiple
    ludiff = u_thr - l_thr

    last = core.std.MakeDiff(input.std.Maximum().std.Minimum(), input)
    last = core.std.Expr(
        [last, Padding(last, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth) .std.Crop(6, 6, 6, 6)],
        ['x y min'])
    expr = f'y {lthr} <= {neutral} y {uthr} >= x {uthr8} y {multiple} / - 128 * x {multiple} / y {multiple} / ' \
           + f'{lthr8} - * + {ludiff} / {multiple} * ? {neutral} - {str} * {neutral} + ?'
    last = core.std.MakeDiff(input, core.std.Expr([last, last.std.Maximum()], expr=[expr]))

    if input_orig is not None:
        last = core.std.ShufflePlanes([last, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return last


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
def LSFmod(input, strength=None, Smode=None, Smethod=None, kernel=11, preblur=None, secure=None, source=None, Szrp=16,
           Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None, overshoot2=None,
           undershoot2=None, soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ss_x=None, ss_y=None,
           dest_x=None, dest_y=None, defaults='fast') -> vs.VideoNode:
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
    except ValueError:
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
        RemoveGrain = partial(core.std.Convolution, matrix=[1] * 9)
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
        expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?' \
            .format(i=scale(16, peak), j=scale(75, peak), peak=peak)
        pre = core.std.MaskedMerge(tmp.dfttest.DFTTest(tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0]),
                                   tmp, tmp.std.Expr(expr=[expr]))
    else:
        pre = MinBlur(tmp, r=preblur)

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
            method = core.std.Expr([method, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale(1, peak))])

        if preblur > -1:
            method = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, method))

        if Smode <= 1:
            normsharp = core.std.Expr([tmp, method], expr=[f'x x y - {Str} * +'])
        else:
            tmpScaled = tmp.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'],
                                     format=tmp.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            methodScaled = method.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'],
                                           format=method.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            expr = f'x y = x x x y - abs {Szrp} / {1 / Spwr} pow {Szrp} * {Str} * x y - dup abs / * x y - dup * ' \
                   + f'{Szrp * Szrp} {SdmpLo} + * x y - dup * {SdmpLo} + {Szrp * Szrp} * / * 1 {SdmpHi} 0 = 0 ' \
                   + f'{(Szrp / SdmpHi) ** 4} ? + 1 {SdmpHi} 0 = 0 x y - abs {SdmpHi} / 4 pow ? + / * + ? ' \
                   + f'{factor if isInteger else 1 / factor} *'
            normsharp = core.std.Expr([tmpScaled, methodScaled], expr=[expr], format=tmp.format)
    else:
        normsharp = pre.cas.CAS(sharpness=min(Str, 1))

        if secure:
            normsharp = core.std.Expr([normsharp, pre],
                                      [f'x y < x {scale(1, peak)} + x y > x {scale(1, peak)} - x ? ?'])

        if preblur > -1:
            normsharp = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, normsharp))

    ### LIMIT
    normal = mt_clamp(normsharp, bright_limit, dark_limit, scale(overshoot, peak), scale(undershoot, peak))
    second = mt_clamp(normsharp, bright_limit, dark_limit, scale(overshoot2, peak), scale(undershoot2, peak))
    zero = mt_clamp(normsharp, bright_limit, dark_limit, 0, 0)

    if edgemaskHQ:
        edge = tmp.std.Sobel(scale=2)
    else:
        edge = core.std.Expr([tmp.std.Maximum(), tmp.std.Minimum()], expr=['x y -'])
    edge = edge.std.Expr([f'x {1 / factor if isInteger else factor} * {128 if edgemaskHQ else 32} '
                         + f'/ 0.86 pow 255 * {factor if isInteger else 1 / factor} *'])

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
        limit2 = core.std.MaskedMerge(tmp, limit1, edge.std.Inflate().std.Inflate()
                                      .std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    else:
        limit2 = core.std.MaskedMerge(limit1, tmp, edge.std.Inflate().std.Inflate()
                                      .std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))

    ### SOFT
    if soft == 0:
        PP1 = limit2
    else:
        sharpdiff = core.std.MakeDiff(tmp, limit2)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])],
                                  [f'x {neutral} - abs y {neutral} - abs > y {soft} * x {100 - soft} * + 100 / x ?'])
        PP1 = core.std.MakeDiff(tmp, sharpdiff)

    ### SOOTHE
    if soothe:
        diff = core.std.MakeDiff(tmp, PP1)
        diff = core.std.Expr([diff, AverageFrames(diff, weights=[1] * 3, scenechange=32 / 255)],
                             [f'x {neutral} - y {neutral} - * 0 < x {neutral} - 100 / {keep} * {neutral} + x  '
                              f'{neutral} - abs y {neutral} - abs > x {keep} * y {100 - keep} * + 100 / x ? ?'])
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
        shrpL = core.std.Expr([core.rgvs.Repair(
                              shrpD, core.std.MakeDiff(In, src, planes=[0]), mode=[1] if isGray else [1, 0]), shrpD],
                              expr=[expr] if isGray else [expr, ''])
        return core.std.MakeDiff(In, shrpL, planes=[0])
    else:
        return out


def ContraSharpening(
    denoised: vs.VideoNode, original: vs.VideoNode, radius: int = 1, rep: int = 1,
    planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.

    Parameters:
        denoised: Denoised clip to sharpen.

        original: Original clip before denoising.

        radius: Spatial radius for contra-sharpening.

        rep: Mode of repair to limit the difference.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
            By default only luma plane will be processed for non-RGB formats.
    """
    if not (isinstance(denoised, vs.VideoNode) and isinstance(original, vs.VideoNode)):
        raise vs.Error('ContraSharpening: this is not a clip')

    if denoised.format.id != original.format.id:
        raise vs.Error('ContraSharpening: clips must have the same format')

    neutral = 1 << (get_depth(denoised) - 1)

    plane_range = range(denoised.format.num_planes)

    if planes is None:
        planes = [0] if denoised.format.color_family != vs.RGB else [0, 1, 2]
    elif isinstance(planes, int):
        planes = [planes]

    pad = 2 if radius < 3 else 4
    denoised = Padding(denoised, pad, pad, pad, pad)
    original = Padding(original, pad, pad, pad, pad)

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1] * 9

    # damp down remaining spots of the denoised clip
    s = MinBlur(denoised, radius, planes)
    # the difference achieved by the denoising
    allD = core.std.MakeDiff(original, denoised, planes=planes)

    RG11 = s.std.Convolution(matrix=matrix1, planes=planes)
    if radius >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if radius >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    # the difference of a simple kernel blur
    ssD = core.std.MakeDiff(s, RG11, planes=planes)
    # limit the difference to the max of what the denoising removed locally
    ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in plane_range])
    # abs(diff) after limiting may not be bigger than before
    ssDD = core.std.Expr([ssDD, ssD],
                         [f'x {neutral} - abs y {neutral} - abs < x y ?' if i in planes else '' for i in plane_range])
    # apply the limited difference (sharpening is just inverse blurring)
    last = core.std.MergeDiff(denoised, ssDD, planes=planes)
    return last.std.Crop(pad, pad, pad, pad)
