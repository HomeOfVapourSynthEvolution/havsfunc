"""Blurring functions."""

import math
from typing import Optional, Sequence, Union

import vapoursynth as vs
from vsutil import Dither, depth, get_depth

core = vs.core


def Gauss(clip: vs.VideoNode,
          p: Optional[float] = None,
          sigma: Optional[float] = None,
          planes: Optional[Union[int, Sequence[int]]] = None
          ) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Gauss: this is not a clip')

    if p is None and sigma is None:
        raise vs.Error('Gauss: must have p or sigma')

    if p is not None and not 0.385 <= p <= 64.921:
        raise vs.Error('Gauss: p must be between 0.385 and 64.921 (inclusive)')

    if sigma is not None and not 0.334 <= sigma <= 4.333:
        raise vs.Error('Gauss: sigma must be between 0.334 and 4.333 (inclusive)')

    if sigma is None and p is not None:
        # Translate AviSynth parameter to standard parameter.
        sigma = math.sqrt(1.0 / (2.0 * (p / 10.0) * math.log(2)))

    # 6 * sigma + 1 rule-of-thumb.
    taps = int(math.ceil(sigma * 6 + 1))
    if not taps % 2:
        taps += 1

    # Gaussian kernel.
    kernel = []
    for x in range(int(math.floor(taps / 2))):
        kernel.append(1.0 / (math.sqrt(2.0 * math.pi) * sigma) * math.exp(-(x * x) / (2 * sigma * sigma)))

    # Renormalize to -1023...1023.
    for i in range(1, len(kernel)):
        kernel[i] *= 1023 / kernel[0]
    kernel[0] = 1023

    # Symmetry.
    kernel = kernel[::-1] + kernel[1:]

    return clip.std.Convolution(matrix=kernel, planes=planes, mode='hv')


def MinBlur(clp: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    """Nifty Gauss/Median combination"""
    from mvsfunc import LimitFilter

    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: this is not a clip')

    plane_range = range(clp.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1] * 9

    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes) \
            .std.Convolution(matrix=matrix2, planes=planes)
        if get_depth(clp) == 16:
            s16 = clp
            RG4 = depth(clp, 12, dither_type=Dither.NONE).ctmf.CTMF(radius=3, planes=planes)
            RG4 = LimitFilter(s16, depth(RG4, 16), thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    return core.std.Expr([clp, RG11, RG4], expr=['x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
                                                 if i in planes else '' for i in plane_range])


def sbr(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    """make a highpass on a blur's difference (well, kind of that)"""
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbr: this is not a clip')

    neutral = 1 << (get_depth(c) - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1] * 9

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)

    RG11DD = core.std.Expr(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?'
              if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)


def sbrV(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbrV: this is not a clip')

    neutral = 1 << (get_depth(c) - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1]
    matrix2 = [1, 4, 6, 4, 1]

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes, mode='v')
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes, mode='v')
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes, mode='v')

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes, mode='v')
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes, mode='v')
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes, mode='v')

    RG11DD = core.std.Expr(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?'
              if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)
