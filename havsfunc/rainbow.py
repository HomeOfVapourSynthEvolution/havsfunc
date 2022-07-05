from __future__ import annotations

from functools import partial

import vapoursynth as vs
from vsutil import plane, split

from .helpers import SCDetect
from .misc import scale

core = vs.core


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

    ythresh = scale(ythresh, peak)
    cthresh = scale(cthresh, peak)
    maxdiff = scale(maxdiff, peak)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_y = plane(input, 0)
    input_minus_y, input_minus_u, input_minus_v = split(input_minus)
    input_plus_y, input_plus_u, input_plus_v = split(input_plus)

    average_y = core.std.Expr([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < x y + 2 / 0 ?'])
    average_u = core.std.Expr([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])
    average_v = core.std.Expr([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])

    ymask = average_y.std.Binarize(threshold=1 << shift)
    if usemaxdiff:
        diffplus_y = core.std.Expr([input_plus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffminus_y = core.std.Expr([input_minus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffs_y = core.std.Lut2(diffplus_y, diffminus_y, function=lambda x, y: x & y)
        ymask = core.std.Lut2(ymask, diffs_y, function=lambda x, y: x & y)
    cmask = core.std.Lut2(average_u.std.Binarize(threshold=129 << shift),
                          average_v.std.Binarize(threshold=129 << shift), function=lambda x, y: x & y)
    cmask = cmask.resize.Point(input.width, input.height)

    themask = core.std.Lut2(ymask, cmask, function=lambda x, y: x & y)

    fixed_y = core.std.Merge(average_y, input_y)

    output = core.std.ShufflePlanes([core.std.MaskedMerge(input_y, fixed_y, themask), input], planes=[0, 1, 2],
                                    colorfamily=input.format.color_family)

    input = SCDetect(input, threshold=scnchg / 255)
    output = output.std.FrameEval(eval=partial(YDifferenceFromPrevious, clips=[input, output]), prop_src=input)
    output = output.std.FrameEval(eval=partial(YDifferenceToNext, clips=[input, output]), prop_src=input)

    if mask:
        return themask
    else:
        return output


#####################################################
#                                                   #
# LUTDeRainbow, a derainbowing script by Scintilla  #
# Last updated 10/3/08                              #
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
    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or input.format.bits_per_sample > 10:
        raise vs.Error('LUTDeRainbow: This is not an 8-10 bit YUV clip')

    shift = input.format.bits_per_sample - 8
    peak = (1 << input.format.bits_per_sample) - 1

    cthresh = scale(cthresh, peak)
    ythresh = scale(ythresh, peak)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_u = plane(input, 1)
    input_v = plane(input, 2)
    input_minus_y, input_minus_u, input_minus_v = split(input_minus)
    input_plus_y, input_plus_u, input_plus_v = split(input_plus)

    average_y = core.std.Expr([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < {peak} 0 ?']) \
        .resize.Bilinear(input_u.width, input_u.height)
    average_u = core.std.Expr([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])
    average_v = core.std.Expr([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])

    umask = average_u.std.Binarize(threshold=21 << shift)
    vmask = average_v.std.Binarize(threshold=21 << shift)
    themask = core.std.Lut2(umask, vmask, function=lambda x, y: x & y)
    if y:
        umask = core.std.Lut2(umask, average_y, function=lambda x, y: x & y)
        vmask = core.std.Lut2(vmask, average_y, function=lambda x, y: x & y)
        themask = core.std.Lut2(themask, average_y, function=lambda x, y: x & y)

    fixed_u = core.std.Merge(average_u, input_u)
    fixed_v = core.std.Merge(average_v, input_v)

    output_u = core.std.MaskedMerge(input_u, fixed_u, themask if linkUV else umask)
    output_v = core.std.MaskedMerge(input_v, fixed_v, themask if linkUV else vmask)

    output = core.std.ShufflePlanes([input, output_u, output_v], planes=[0, 0, 0],
                                    colorfamily=input.format.color_family)

    if mask:
        return themask.resize.Point(input.width, input.height)
    else:
        return output
