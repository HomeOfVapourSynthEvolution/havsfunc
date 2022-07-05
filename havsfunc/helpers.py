"""Useful helper functions."""
import math
from typing import Union, Sequence, Optional
import vapoursynth as vs


def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)


def m4(x: Union[float, int]) -> int:
    return 16 if x < 16 else cround(x / 4) * 4


def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255


# sin(pi x / 2) for -1 < x < 1 using Taylor series
def sine_expr(var):
    return f'{-3.5988432352121e-6} {var} * {var} * {0.00016044118478736} + {var} * {var} ' \
           + f'* {-0.0046817541353187} + {var} * {var} * {0.079692626246167} + {var} * {var} * ' \
           + f'{-0.64596409750625} + {var} * {var} * {1.5707963267949} + {var} *'


def AverageFrames(
    clip: vs.VideoNode,
    weights: Union[float, Sequence[float]],
    scenechange: Optional[float] = None,
    planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AverageFrames: this is not a clip')

    if scenechange:
        clip = SCDetect(clip, threshold=scenechange)
    return clip.std.AverageFrames(weights=weights, scenechange=scenechange, planes=planes)


def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)


def SCDetect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    def copy_property(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
        fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
        return fout

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')

    sc = sc.misc.SCDetect(threshold=threshold)
    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame(clips=[clip, sc], selector=copy_property)

    return sc


def Weave(clip: vs.VideoNode, tff: Optional[bool] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Weave: this is not a clip')

    if tff is None:
        with clip.get_frame(0) as f:
            if f.props.get('_Field') not in [1, 2]:
                raise vs.Error('Weave: tff was not specified and field order '
                               'could not be determined from frame properties')

    return clip.std.DoubleWeave(tff=tff)[::2]
