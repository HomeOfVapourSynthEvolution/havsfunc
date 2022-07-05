from __future__ import annotations

from typing import Optional, Sequence, Union

import vapoursynth as vs

core = vs.core


def AvsPrewitt(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    return core.std.Expr(
        [
            clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False),
        ],
        expr=['x y max z max a max' if i in planes else '' for i in plane_range],
    )


def mt_clamp(
    clip: vs.VideoNode,
    bright_limit: vs.VideoNode,
    dark_limit: vs.VideoNode,
    overshoot: int = 0,
    undershoot: int = 0,
    planes: Optional[Union[int, Sequence[int]]] = None,
) -> vs.VideoNode:
    if not (isinstance(clip, vs.VideoNode) and isinstance(bright_limit, vs.VideoNode)
            and isinstance(dark_limit, vs.VideoNode)):
        raise vs.Error('mt_clamp: this is not a clip')

    if bright_limit.format.id != clip.format.id or dark_limit.format.id != clip.format.id:
        raise vs.Error('mt_clamp: clips must have the same format')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    return core.std.Expr([clip, bright_limit, dark_limit], expr=[f'x y {overshoot} + min z {undershoot} - max'
                                                                 if i in planes else '' for i in plane_range])


def mt_expand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None,
                    sw: int = 1, sh: int = 1) -> vs.VideoNode:
    """
    Calls std.Maximum multiple times in order to grow the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles and
            losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Growing shape width. 0 is allowed.

        sh: Growing shape height. 0 is allowed.
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_expand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1] * 8
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_expand_multi(src.std.Maximum(planes=planes, coordinates=mode_m),
                              mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src


def mt_inpand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None,
                    sw: int = 1, sh: int = 1) -> vs.VideoNode:
    """
    Calls std.Minimum multiple times in order to shrink the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles
            and losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Shrinking shape width. 0 is allowed.

        sh: Shrinking shape height. 0 is allowed.
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inpand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1] * 8
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_inpand_multi(src.std.Minimum(planes=planes, coordinates=mode_m),
                              mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src


def mt_inflate_multi(src: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None,
                     radius: int = 1) -> vs.VideoNode:
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inflate_multi: this is not a clip')

    for _ in range(radius):
        src = src.std.Inflate(planes=planes)
    return src


def mt_deflate_multi(src: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None,
                     radius: int = 1) -> vs.VideoNode:
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_deflate_multi: this is not a clip')

    for _ in range(radius):
        src = src.std.Deflate(planes=planes)
    return src
