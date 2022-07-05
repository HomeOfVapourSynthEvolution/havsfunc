from __future__ import annotations

import math
from functools import partial

import vapoursynth as vs
from vsutil import get_depth, plane, scale_value

from .helpers import scale, sine_expr
from .misc import Overlay

core = vs.core


# a.k.a. BalanceBordersMod
def bbmod(c, cTop, cBottom, cLeft, cRight, thresh=128, blur=999):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('bbmod: this is not a clip')

    if c.format.color_family in [vs.GRAY, vs.RGB]:
        raise vs.Error('bbmod: Gray and RGB formats are not supported')

    if thresh <= 0:
        raise vs.Error('bbmod: thresh must be greater than 0')

    if blur <= 0:
        raise vs.Error('bbmod: blur must be greater than 0')

    neutral = 1 << (c.format.bits_per_sample - 1)
    peak = (1 << c.format.bits_per_sample) - 1

    BicubicResize = partial(core.resize.Bicubic, filter_param_a=1, filter_param_b=0)

    def btb(c, cTop):
        cWidth = c.width
        cHeight = c.height
        cTop = min(cTop, cHeight - 1)
        blurWidth = max(8, math.floor(cWidth / blur))

        c2 = c.resize.Point(cWidth * 2, cHeight * 2)

        last = c2.std.CropAbs(width=cWidth * 2, height=2, top=cTop * 2)
        last = last.resize.Point(cWidth * 2, cTop * 2)
        referenceBlurChroma = BicubicResize(BicubicResize(last.std.Expr(expr=[f'x {neutral} - abs 2 *', '']),
                                                          blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)
        referenceBlur = BicubicResize(BicubicResize(last, blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)

        original = c2.std.CropAbs(width=cWidth * 2, height=cTop * 2)

        last = BicubicResize(original, blurWidth * 2, cTop * 2)
        originalBlurChroma = BicubicResize(BicubicResize(last.std.Expr(expr=[f'x {neutral} - abs 2 *', '']),
                                                         blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)
        originalBlur = BicubicResize(BicubicResize(last, blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)

        balancedChroma = core.std.Expr([original, originalBlurChroma, referenceBlurChroma],
                                       expr=['', f'z y / 8 min 0.4 max x {neutral} - * {neutral} +'])
        expr = 'z {i} - y {i} - / 8 min 0.4 max x {i} - * {i} +'.format(i=scale(16, peak))
        balancedLuma = core.std.Expr([balancedChroma, originalBlur, referenceBlur], expr=[expr, 'z y - x +'])

        difference = core.std.MakeDiff(balancedLuma, original)
        difference = difference.std.Expr(expr=[f'x {scale(128 + thresh, peak)} min {scale(128 - thresh, peak)} max'])

        last = core.std.MergeDiff(original, difference)
        return core.std.StackVertical(
            [last, c2.std.CropAbs(width=cWidth * 2, height=(cHeight - cTop) * 2, top=cTop * 2)]) \
            .resize.Point(cWidth, cHeight)

    if cTop > 0:
        c = btb(c, cTop)
    c = c.std.Transpose().std.FlipHorizontal()
    if cLeft > 0:
        c = btb(c, cLeft)
    c = c.std.Transpose().std.FlipHorizontal()
    if cBottom > 0:
        c = btb(c, cBottom)
    c = c.std.Transpose().std.FlipHorizontal()
    if cRight > 0:
        c = btb(c, cRight)
    return c.std.Transpose().std.FlipHorizontal()


# column is the column you want to work on.
def FixColumnBrightness(c, column, input_low, input_high, output_low, output_high):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixColumnBrightness: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixColumnBrightness: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    input_low = scale(input_low, peak)
    input_high = scale(input_high, peak)
    output_low = scale(output_low, peak)
    output_high = scale(output_high, peak)

    last = SmoothLevels(c, input_low, 1, input_high, output_low, output_high, Smode=0)
    last = last.std.CropAbs(width=1, height=c.height, left=column)
    last = Overlay(c, last, x=column)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


# row is the row you want to work on.
def FixRowBrightness(c, row, input_low, input_high, output_low, output_high):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixRowBrightness: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixRowBrightness: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    input_low = scale(input_low, peak)
    input_high = scale(input_high, peak)
    output_low = scale(output_low, peak)
    output_high = scale(output_high, peak)

    last = SmoothLevels(c, input_low, 1, input_high, output_low, output_high, Smode=0)
    last = last.std.CropAbs(width=c.width, height=1, top=row)
    last = Overlay(c, last, y=row)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


# protect_value determines which pixels wouldn't be affected by the filter.
# Increasing the value, you protect the pixels with lower luma.
def FixColumnBrightnessProtect(c, column, input_low, input_high, output_low, output_high, protect_value=20):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixColumnBrightnessProtect: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixColumnBrightnessProtect: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    input_low = scale(255 - input_low, peak)
    input_high = scale(255 - input_high, peak)
    output_low = scale(255 - output_low, peak)
    output_high = scale(255 - output_high, peak)
    protect_value = scale(protect_value, peak)

    last = SmoothLevels(c.std.Invert(), input_low, 1, input_high, output_low, output_high,
                        protect=protect_value, Smode=0).std.Invert()
    last = last.std.CropAbs(width=1, height=c.height, left=column)
    last = Overlay(c, last, x=column)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


def FixRowBrightnessProtect(c, row, input_low, input_high, output_low, output_high, protect_value=20):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixRowBrightnessProtect: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixRowBrightnessProtect: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    input_low = scale(255 - input_low, peak)
    input_high = scale(255 - input_high, peak)
    output_low = scale(255 - output_low, peak)
    output_high = scale(255 - output_high, peak)
    protect_value = scale(protect_value, peak)

    last = SmoothLevels(c.std.Invert(), input_low, 1, input_high, output_low, output_high,
                        protect=protect_value, Smode=0).std.Invert()
    last = last.std.CropAbs(width=c.width, height=1, top=row)
    last = Overlay(c, last, y=row)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


# adj_val should be a number x where -100 < x < 100. This parameter decides
# how much the brightness should be affected. Numbers below 0 will make it darker
# and number above 0 will make it brighter.
#
# prot_val is the protect value. This is what makes it behave differently than the
# normal FixBrightness. Any luma above (255-prot_val) will not be affected which is
# the basic idea of the protect script.
def FixColumnBrightnessProtect2(c, column, adj_val, prot_val=16):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixColumnBrightnessProtect2: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixColumnBrightnessProtect2: RGB format is not supported')

    if not (-100 < adj_val < 100):
        raise vs.Error('FixColumnBrightnessProtect2: adj_val must be greater than -100 and less than 100')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    expr = f'x {scale(16, peak)} - {100 - adj_val} / 100 * {scale(16, peak)} + x {scale(255 - prot_val, peak)} - ' \
           + f'-10 / 0 max 1 min * x x {scale(245 - prot_val, peak)} - 10 / 0 max 1 min * +'
    last = c.std.Expr(expr=[expr])
    last = last.std.CropAbs(width=1, height=c.height, left=column)
    last = Overlay(c, last, x=column)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


def FixRowBrightnessProtect2(c, row, adj_val, prot_val=16):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FixRowBrightnessProtect2: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FixRowBrightnessProtect2: RGB format is not supported')

    if not (-100 < adj_val < 100):
        raise vs.Error('FixRowBrightnessProtect2: adj_val must be greater than -100 and less than 100')

    peak = (1 << c.format.bits_per_sample) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = plane(c, 0)
    else:
        c_orig = None

    expr = f'x {scale(16, peak)} - {100 - adj_val} / 100 * {scale(16, peak)} + x {scale(255 - prot_val, peak)} - ' \
           + f'-10 / 0 max 1 min * x x {scale(245 - prot_val, peak)} - 10 / 0 max 1 min * +'
    last = c.std.Expr(expr=[expr])
    last = last.std.CropAbs(width=c.width, height=1, top=row)
    last = Overlay(c, last, y=row)
    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


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
### /!\ Needed filters : RGVS, f3kdb
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
### input_low, gamma, input_high, output_low, output_high [default: 0, 1.0, maximum value of input format, 0,
### maximum value of input format]
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth
### of input format manually by users
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
### 1 = limit conversion on dark & bright areas (apply conversion @0%   at luma=0 & @100% at luma=Ecenter
###                                                                                                 & @0% at luma=255)
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
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format
### manually by users
###
### protect [default: -1]
### ---------------------
### -1  = protect off
### >=0 = pure black protection
###       ---> don't apply conversion on pixels egal or below this value
###            (ex: with 16, the black areas like borders and generic are untouched so they don't look washed out)
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format
### manually by users
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
### Use f3kdb on top of removegrain: prevent posterize when doing levels conversion
###
###
#########################################################################################
def SmoothLevels(input, input_low=0, gamma=1.0, input_high=None, output_low=0, output_high=None, chroma=50, limiter=0,
                 Lmode=0, DarkSTR=100, BrightSTR=100, Ecenter=None, protect=-1, Ecurve=0, Smode=-2, Mfactor=2,
                 RGmode=12, useDB=False) -> vs.VideoNode:
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
        RemoveGrain = partial(core.std.Convolution, matrix=[1] * 9)
    else:
        RemoveGrain = partial(core.rgvs.RemoveGrain, mode=[RGmode])

    ### EXPRESSION
    exprY = f'x {input_low} - {input_high - input_low + (input_high == input_low)} / {1 / gamma} pow ' \
            + f'{output_high - output_low} * {output_low} +'

    if chroma > 0 and not isGray:
        scaleC = ((output_high - output_low) / (input_high - input_low + (input_high == input_low))
                  + 100 / chroma - 1) / (100 / chroma)
        exprC = f'x {neutral[1]} - {scaleC} * {neutral[1]} +'

    Dstr = DarkSTR / 100
    Bstr = BrightSTR / 100

    if Lmode <= 0:
        exprL = '1'
    elif Ecurve <= 0:
        if Lmode == 1:
            var_d = f'x {Ecenter} /'
            var_b = f'{peak} x - {peak} {Ecenter} - /'
            exprL = f'x {Ecenter} < ' + sine_expr(var_d) + f' {Dstr} pow x {Ecenter} > ' + sine_expr(var_b) \
                    + f' {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            var_d = f'x {peak} /'
            exprL = sine_expr(var_d) + f' {Dstr} pow'
        else:
            var_b = f'{peak} x - {peak} /'
            exprL = sine_expr(var_b) + f' {Bstr} pow'
    else:
        if Lmode == 1:
            exprL = f'x {Ecenter} < x {Ecenter} / abs {Dstr} pow x {Ecenter} > 1 x {Ecenter} - {peak - Ecenter} / ' \
                    + f'abs - {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            exprL = f'1 x {peak} - {peak} / abs - {Dstr} pow'
        else:
            exprL = f'x {peak} - {peak} / abs {Bstr} pow'

    if protect <= -1:
        exprP = '1'
    elif Ecurve <= 0:
        var_p = f'x {protect} - {scale(16, peak)} /'
        exprP = f'x {protect} <= 0 x {protect + scale(16, peak)} >= 1 ' + sine_expr(var_p) + ' ? ?'
    else:
        exprP = f'x {protect} <= 0 x {protect + scale(16, peak)} >= 1 x {protect} - {scale(16, peak)} / abs ? ?'

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
        process = process.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']) \
            .f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        smth = core.std.MakeDiff(limitI, process)
    else:
        smth = core.std.Expr([limitI, process], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    level2 = core.std.Expr([limitI, diff], expr=[f'x y {neutral[1]} - {Mfactor} / -'])
    diff2 = core.std.Expr([level2, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process2 = RemoveGrain(diff2)
    if useDB:
        process2 = process2.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']) \
            .f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
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
        limitO = core.std.ShufflePlanes([limitO, input_orig], planes=[0, 1, 2],
                                        colorfamily=input_orig.format.color_family)
    return limitO


def DitherLumaRebuild(src: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, chroma: bool = True) -> vs.VideoNode:
    """
    Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks.
    (for the clip to be fed to motion search only)
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    is_gray = src.format.color_family == vs.GRAY
    is_integer = src.format.sample_type == vs.INTEGER

    bits = get_depth(src)
    neutral = 1 << (bits - 1)

    k = (s0 - 1) * c
    t = f'x {scale_value(16, 8, bits)} - {scale_value(219, 8, bits)} / 0 max 1 min' if is_integer else 'x 0 max 1 min'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + ' \
        + (f'{scale_value(256, 8, bits)} *' if is_integer else '')
    return src.std.Expr(expr=e if is_gray
                        else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma and is_integer else ''])
