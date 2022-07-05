from __future__ import annotations

import math
from functools import partial
from typing import Optional, Sequence

import vapoursynth as vs
from vsutil import plane

from .aa import daa
from .blur import sbrV
from .helpers import cround, scale
from .levels import DitherLumaRebuild
from .qtgmc import QTGMC

core = vs.core


def smartfademod(clip: vs.VideoNode,
                 threshold: float = 0.4,
                 show: bool = False,
                 tff: Optional[bool] = None
                 ) -> vs.VideoNode:
    """
    Aimed at removing interlaced fades in anime. Uses luma difference between two fields as activation threshold.

    Parameters:
        clip: Clip to process.

        threshold: Threshold for fade detection.

        show: Displays luma difference between fields without processing anything.

        tff: Since VapourSynth only has a weak notion of field order internally, tff may have to be set.
            Setting tff to true means top field first and false
            means bottom field first. Note that the _FieldBased frame property, if present, takes precedence over tff.
    """

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
                raise vs.Error('smartfademod: tff was not specified and field order could not be determined '
                               'from frame properties')

    sep = clip.std.SeparateFields(tff=tff)
    even = sep[::2].std.PlaneStats()
    odd = sep[1::2].std.PlaneStats()
    defade = daa(clip)
    return clip.std.FrameEval(eval=partial(frame_eval, orig=clip, defade=defade),
                              prop_src=[even, odd], clip_src=[clip, defade])


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
    if frate is not None and abs(irate * numr / cround(numr / frfac) - frate) > abs(irate * cround(frate * 100) / cround(irate * 100) - frate):  # noqa
        numr = cround(frate * 100)
    denm = cround(numr / frfac)

    ###### source preparation & lut ######
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(core.std.Merge(source, source.std.Trim(first=1), weight=[0, 0.5]),
                             source.std.Trim(first=1), weight=[0.5, 0])

    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)
    dclip = dclip.resize.Point(dclip.width if srad == 4 else int(dclip.width / 2 / srad + 4) * 4,
                               dclip.height if srad == 4 else int(dclip.height / 2 / srad + 4) * 4).std.Trim(first=2)
    if mode < 0:
        dclip = core.std.StackVertical([core.std.StackHorizontal([plane(dclip, 1), plane(dclip, 2)]), plane(dclip, 0)])
    else:
        dclip = plane(dclip, 0)
    if bom:
        dclip = dclip.std.Expr(expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - ' \
            + '* ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = core.std.Expr([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = core.std.Expr([diff.std.Trim(first=1), core.std.MergeDiff(diff, diff.std.Trim(first=2))], expr=[expr2]) \
            .resize.Bilinear(bsize, bsize)
    dclp = diff.std.Trim(first=1).std.Lut(function=lambda x: max(cround(abs(x - 128) ** 1.1 - 1), 0)) \
        .resize.Bilinear(bsize, bsize)

    ###### postprocessing ######
    if bom:
        sourceDuplicate = source.std.DuplicateFrames(frames=[0])
        sourceTrim1 = source.std.Trim(first=1)
        sourceTrim2 = source.std.Trim(first=2)

        unblend1 = core.std.Expr([sourceDuplicate, source], expr=['y 2 * x -'])
        unblend2 = core.std.Expr([sourceTrim1, sourceTrim2], expr=['x 2 * y -'])

        qmask1 = core.std.MakeDiff(unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
                                   unblend1, planes=[0])
        qmask2 = core.std.MakeDiff(unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
                                   unblend2, planes=[0])
        diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
        bmask = core.std.Expr([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = 'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?' \
            .format(i=scale(4, peak), peak=peak, j=scale(200, peak), k=scale(28, peak))
        dmask = core.std.Expr([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = core.std.Expr([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = core.std.Expr([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['y x 2 / - z a 2 / - +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(unblend1, unblend2, dmask.std.Convolution(matrix=matrix, planes=[0])
                                       .std.Expr(expr=['', repr(neutral)]))
        elif omode == 'pp2':
            fin = core.std.MaskedMerge(unblend1, unblend2, bmask.std.Convolution(matrix=matrix, planes=[0]),
                                       first_plane=True)
        elif omode == 'pp3':
            fin = core.std.MaskedMerge(unblend1, unblend2, pmask.std.Convolution(matrix=matrix, planes=[0]),
                                       first_plane=True).std.Convolution(matrix=matrix, planes=[1, 2])
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
        nonlocal lfr, offs, ldet, lpos, d32, d21, d10, d01, d12, d23, d34, m42, m31, m20, m11, m02, m13, m24, bp2
        nonlocal bp1, bn0, bn1, bn2, bn3, cp2, cp1, cn0, cn1, cn2, cn3

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
        blend = bbool and bc * 5 > cc and dbc + dcn > 1.5 * thr and (dbb < 7 * dbc or dbb < 8 * dcn) and \
                (dnn < 8 * dcn or dnn < 7 * dbc) and (mb < mb1 and mb < mc or mn < mn1 and mn < mc or (dbb + dnn) \
                * 4 < dbc + dcn or (bb * cc * 5 < bc * cb or mb > thr) and (bn * cc * 5 < bc * cn or mn > thr) and bc > thr)  # noqa
        clear = dbb + dbc > thr and dcn + dnn > thr and (bc < 2 * bb or bc < 2 * bn) and (dbb + dnn) * 2 > dbc + dcn \
                and (mc < 0.96 * mb and mc < 0.96 * mn and (bb * 2 > cb or bn * 2 > cn) and cc > cb and cc > cn or \
                frfac > 0.45 and frfac < 0.55 and 0.8 * mc > mb1 and 0.8 * mc > mn1 and mb > 0.8 * mn and mn > 0.8 * mb)  # noqa
        highd = dcn > 5 * dbc and dcn > 5 * dnn and dcn > thr and dbc < thr and dnn < thr
        lowd = dcn * 5 < dbc and dcn * 5 < dnn and dbc > thr and dnn > thr and dcn < thr and frfac > 0.35 and \
            (frfac < 0.51 or dcn * 5 < dbb)
        res = d43 < thr and d32 < thr and d21 < thr and d10 < thr and d01 < thr and d12 < thr and d23 < thr and \
              d34 < thr or dbc * 4 < dbb and dcn * 4 < dbb and dnn * 4 < dbb and dn2 * 4 < dbb or dcn * 4 < dbc and \
              dnn * 4 < dbc and dn2 * 4 < dbc  # noqa

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
        elif dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == lpos) or (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == lpos and dnn < 0.9 * dbc or dnn * 9 < dbc and dcn * 3 < dbc:  # noqa
            dup = 1
        elif (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == lpos and dbb < 0.9 * dcn or dbb * 9 < dcn and dbc * 3 < dcn or dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == lpos):  # noqa
            dup = -1
        else:
            dup = 0
        mer = not bom and opos == pos and dup == 0 and abs(mode) > 2 and (dbc * 8 < dcn or dbc * 8 < dbb or dcn * 8 < dbc or dcn * 8 < dnn or dbc * 2 < thr or dcn * 2 < thr or dnn * 9 < dbc and dcn * 3 < dbc or dbb * 9 < dcn and dbc * 3 < dcn)  # noqa

        ### deblend - doubleblend removal - postprocessing ###
        add = bp1 * cn2 > bn2 * cp1 * (1 + thr * 0.01) and bn0 * cn2 > bn2 * cn0 * (1 + thr * 0.01) and cn2 * \
            bn1 > cn1 * bn2 * (1 + thr * 0.01)
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
                if bp1 * cp2 > bp2 * cp1 * (1 + thr * 0.01) and bn0 * cp2 > bp2 * cn0 * (1 + thr * 0.01) and cp2 * bn1 > cn1 * bp2 * (1 + thr * 0.01) and (not add or cp2 * bn2 > cn2 * bp2):  # noqa
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
    return ChangeFPS(last.std.Cache(make_linear=True), source.fps_num * numr, source.fps_den * denm)


# frame_ref = start of AABCD pattern
def dec_txt60mc(src, frame_ref, srcbob=False, draft=False, tff=None, opencl=False, device=None):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('dec_txt60mc: this is not a clip')

    if not (srcbob or isinstance(tff, bool)):
        raise vs.Error("dec_txt60mc: 'tff' must be set when srcbob=False. Setting tff to true means top field first "
                       "and false means bottom field first")

    field_ref = frame_ref if srcbob else frame_ref * 2
    field_ref %= 5
    invpos = (5 - field_ref) % 5
    pel = 1 if draft else 2
    blksize = 16 if src.width > 1024 or src.height > 576 else 8
    overlap = blksize // 2

    if srcbob:
        last = src
    elif draft:
        last = src.resize.Bob(tff=tff, filter_param_a=1 / 3, filter_param_b=1 / 3)
    else:
        last = QTGMC(src, TR0=1, TR1=1, TR2=1, SourceMatch=3, Lossless=2, TFF=tff, opencl=opencl, device=device)

    clean = last.std.SelectEvery(cycle=5, offsets=[4 - invpos])
    if invpos > 2:
        jitter = core.std.AssumeFPS(
            last.std.Trim(length=1) * 2 + last.std.SelectEvery(cycle=5, offsets=[6 - invpos, 7 - invpos]),
            fpsnum=24000, fpsden=1001)
    elif invpos > 1:
        jitter = core.std.AssumeFPS(
            last.std.Trim(length=1) + last.std.SelectEvery(cycle=5, offsets=[2 - invpos, 6 - invpos]),
            fpsnum=24000, fpsden=1001)
    else:
        jitter = last.std.SelectEvery(cycle=5, offsets=[1 - invpos, 2 - invpos])
    jsup_pre = DitherLumaRebuild(jitter, s0=1).mv.Super(pel=pel)
    jsup = jitter.mv.Super(pel=pel, levels=1)
    vect_f = jsup_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
    vect_b = jsup_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
    comp = core.mv.FlowInter(jitter, jsup, vect_b, vect_f)
    fixed = comp[::2]
    last = core.std.Interleave([fixed, clean])
    return last.std.Trim(first=invpos // 3)


# 30pテロ部を24pに変換して返す
def ivtc_txt30mc(src, frame_ref, draft=False, tff=None, opencl=False, device=None):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('ivtc_txt30mc: this is not a clip')

    if not isinstance(tff, bool):
        raise vs.Error("ivtc_txt30mc: 'tff' must be set. Setting tff to true means top field first "
                       " and false means bottom field first")

    frame_ref %= 5
    offset = [0, 0, -1, 1, 1][frame_ref]
    pattern = [0, 1, 0, 0, 1][frame_ref]
    direction = [-1, -1, 1, 1, 1][frame_ref]
    pel = 1 if draft else 2
    blksize = 16 if src.width > 1024 or src.height > 576 else 8
    overlap = blksize // 2

    if draft:
        last = src.resize.Bob(tff=tff, filter_param_a=1 / 3, filter_param_b=1 / 3)
    else:
        last = QTGMC(src, TR0=1, TR1=1, TR2=1, SourceMatch=3, Lossless=2, TFF=tff, opencl=opencl, device=device)

    if pattern == 0:
        if offset == -1:
            c1 = core.std.AssumeFPS(
                last.std.Trim(length=1) + last.std.SelectEvery(
                    cycle=10, offsets=[2 + offset, 7 + offset, 5 + offset, 10 + offset]), fpsnum=24000, fpsden=1001)
        else:
            c1 = last.std.SelectEvery(cycle=10, offsets=[offset, 2 + offset, 7 + offset, 5 + offset])
        if offset == 1:
            part1 = last.std.SelectEvery(cycle=10, offsets=[4])
            part2 = last.std.SelectEvery(cycle=10, offsets=[5])
            part3 = last.std.Trim(first=10).std.SelectEvery(cycle=10, offsets=[0])
            part4 = last.std.SelectEvery(cycle=10, offsets=[9])
            c2 = core.std.Interleave([part1, part2, part3, part4])
        else:
            c2 = last.std.SelectEvery(cycle=10, offsets=[3 + offset, 4 + offset, 9 + offset, 8 + offset])
    else:
        if offset == 1:
            part1 = last.std.SelectEvery(cycle=10, offsets=[3])
            part2 = last.std.SelectEvery(cycle=10, offsets=[5])
            part3 = last.std.Trim(first=10).std.SelectEvery(cycle=10, offsets=[0])
            part4 = last.std.SelectEvery(cycle=10, offsets=[8])
            c1 = core.std.Interleave([part1, part2, part3, part4])
        else:
            c1 = last.std.SelectEvery(cycle=10, offsets=[2 + offset, 4 + offset, 9 + offset, 7 + offset])
        if offset == -1:
            c2 = core.std.AssumeFPS(
                last.std.Trim(length=1) + last.std.SelectEvery(
                    cycle=10, offsets=[1 + offset, 6 + offset, 5 + offset, 10 + offset]), fpsnum=24000, fpsden=1001)
        else:
            c2 = last.std.SelectEvery(cycle=10, offsets=[offset, 1 + offset, 6 + offset, 5 + offset])

    super1_pre = DitherLumaRebuild(c1, s0=1).mv.Super(pel=pel)
    super1 = c1.mv.Super(pel=pel, levels=1)
    vect_f1 = super1_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
    vect_b1 = super1_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
    fix1 = core.mv.FlowInter(c1, super1, vect_b1, vect_f1, time=50 + direction * 25) \
        .std.SelectEvery(cycle=4, offsets=[0, 2])

    super2_pre = DitherLumaRebuild(c2, s0=1).mv.Super(pel=pel)
    super2 = c2.mv.Super(pel=pel, levels=1)
    vect_f2 = super2_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
    vect_b2 = super2_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
    fix2 = core.mv.FlowInter(c2, super2, vect_b2, vect_f2).std.SelectEvery(cycle=4, offsets=[0, 2])

    if pattern == 0:
        return core.std.Interleave([fix1, fix2])
    else:
        return core.std.Interleave([fix2, fix1])


# Version 1.1
# frame_ref = start of clean-combed-combed-clean-clean pattern
def ivtc_txt60mc(src, frame_ref, srcbob=False, draft=False, tff=None, opencl=False, device=None):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('ivtc_txt60mc: this is not a clip')

    if not (srcbob or isinstance(tff, bool)):
        raise vs.Error("ivtc_txt60mc: 'tff' must be set when srcbob=False. Setting tff to true means top field first "
                       "and false means bottom field first")

    field_ref = frame_ref if srcbob else frame_ref * 2
    field_ref %= 5
    invpos = (5 - field_ref) % 5
    pel = 1 if draft else 2
    blksize = 16 if src.width > 1024 or src.height > 576 else 8
    overlap = blksize // 2

    if srcbob:
        last = src
    elif draft:
        last = src.resize.Bob(tff=tff, filter_param_a=1 / 3, filter_param_b=1 / 3)
    else:
        last = QTGMC(src, TR0=1, TR1=1, TR2=1, SourceMatch=3, Lossless=2, TFF=tff, opencl=opencl, device=device)

    if invpos > 1:
        clean = core.std.AssumeFPS(
            last.std.Trim(length=1) + last.std.SelectEvery(cycle=5, offsets=[6 - invpos]), fpsnum=12000, fpsden=1001)
    else:
        clean = last.std.SelectEvery(cycle=5, offsets=[1 - invpos])
    if invpos > 3:
        jitter = core.std.AssumeFPS(
            last.std.Trim(length=1) + last.std.SelectEvery(cycle=5, offsets=[4 - invpos, 8 - invpos]),
            fpsnum=24000, fpsden=1001)
    else:
        jitter = last.std.SelectEvery(cycle=5, offsets=[3 - invpos, 4 - invpos])
    jsup_pre = DitherLumaRebuild(jitter, s0=1).mv.Super(pel=pel)
    jsup = jitter.mv.Super(pel=pel, levels=1)
    vect_f = jsup_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
    vect_b = jsup_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
    comp = core.mv.FlowInter(jitter, jsup, vect_b, vect_f)
    fixed = comp[::2]
    last = core.std.Interleave([clean, fixed])
    return last.std.Trim(first=invpos // 2)


# Vinverse: a small, but effective function against (residual) combing, by Didée
# sstr: strength of contra sharpening
# amnt: change no pixel by more than this (default=255: unrestricted)
# chroma: chroma mode, True=process chroma, False=pass chroma through
# scl: scale factor for vshrpD*vblurD < 0
def Vinverse(clp, sstr=2.7, amnt=255, chroma=True, scl=0.25):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Vinverse: this is not a clip')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if not chroma and clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = plane(clp, 0)
    else:
        clp_orig = None

    vblur = clp.std.Convolution(matrix=[50, 99, 50], mode='v')
    vblurD = core.std.MakeDiff(clp, vblur)
    vshrp = core.std.Expr(
        [vblur, vblur.std.Convolution(matrix=[1, 4, 6, 4, 1], mode='v')], expr=[f'x x y - {sstr} * +'])
    vshrpD = core.std.MakeDiff(vshrp, vblur)
    expr = f'x {neutral} - y {neutral} - * 0 < x {neutral} - abs y {neutral} - abs < x y ? {neutral} - {scl} * ' \
           + f'{neutral} + x {neutral} - abs y {neutral} - abs < x y ? ?'
    vlimD = core.std.Expr([vshrpD, vblurD], expr=[expr])
    last = core.std.MergeDiff(vblur, vlimD)
    if amnt <= 0:
        return clp
    elif amnt < 255:
        last = core.std.Expr([clp, last], expr=['x {AMN} + y < x {AMN} + x {AMN} - y > x {AMN} - y ? ?'
                             .format(AMN=scale(amnt, peak))])

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last


def Vinverse2(clp, sstr=2.7, amnt=255, chroma=True, scl=0.25):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Vinverse2: this is not a clip')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if not chroma and clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = plane(clp, 0)
    else:
        clp_orig = None

    vblur = sbrV(clp)
    vblurD = core.std.MakeDiff(clp, vblur)
    vshrp = core.std.Expr([vblur, vblur.std.Convolution(matrix=[1, 2, 1], mode='v')], expr=[f'x x y - {sstr} * +'])
    vshrpD = core.std.MakeDiff(vshrp, vblur)
    expr = f'x {neutral} - y {neutral} - * 0 < x {neutral} - abs y {neutral} - abs < x y ? {neutral} - {scl} * ' \
           + f'{neutral} + x {neutral} - abs y {neutral} - abs < x y ? ?'
    vlimD = core.std.Expr([vshrpD, vblurD], expr=[expr])
    last = core.std.MergeDiff(vblur, vlimD)
    if amnt <= 0:
        return clp
    elif amnt < 255:
        last = core.std.Expr([clp, last], expr=['x {AMN} + y < x {AMN} + x {AMN} - y > x {AMN} - y ? ?'
                             .format(AMN=scale(amnt, peak))])

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last


#------------------------------------------------------------------------------#
#                                                                              #
#                         InterFrame 2.8.2 by SubJunk                          #
#                                                                              #
#         A frame interpolation script that makes accurate estimations         #
#                   about the content of non-existent frames                   #
#      Its main use is to give videos higher framerates like newer TVs do      #
#------------------------------------------------------------------------------#
def InterFrame(Input, Preset='Medium', Tuning='Film', NewNum=None, NewDen=1, GPU=False, InputType='2D',
               OverrideAlgo=None, OverrideArea=None, FrameDouble=False):
    if not isinstance(Input, vs.VideoNode):
        raise vs.Error('InterFrame: this is not a clip')

    # Validate inputs
    Preset = Preset.lower()
    Tuning = Tuning.lower()
    InputType = InputType.upper()

    if Preset not in ['medium', 'fast', 'faster', 'fastest']:
        raise vs.Error(f"InterFrame: '{Preset}' is not a valid preset")

    if Tuning not in ['film', 'smooth', 'animation', 'weak']:
        raise vs.Error(f"InterFrame: '{Tuning}' is not a valid tuning")

    if InputType not in ['2D', 'SBS', 'OU', 'HSBS', 'HOU']:
        raise vs.Error(f"InterFrame: '{InputType}' is not a valid InputType")

    def InterFrameProcess(clip):
        # Create SuperString
        if Preset in ['fast', 'faster', 'fastest']:
            SuperString = '{pel:1,'
        else:
            SuperString = '{'

        SuperString += 'gpu:1}' if GPU else 'gpu:0}'

        # Create VectorsString
        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString = '{block:{w:32,'
        elif Preset in ['fast', 'faster'] or not GPU:
            VectorsString = '{block:{w:16,'
        else:
            VectorsString = '{block:{w:8,'

        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString += 'overlap:0'
        elif Preset == 'faster' and GPU:
            VectorsString += 'overlap:1'
        else:
            VectorsString += 'overlap:2'

        if Tuning == 'animation':
            VectorsString += '},main:{search:{coarse:{type:2,'
        elif Preset == 'faster':
            VectorsString += '},main:{search:{coarse:{'
        else:
            VectorsString += '},main:{search:{distance:0,coarse:{'

        if Tuning == 'animation':
            VectorsString += 'distance:-6,satd:false},distance:0,'
        elif Tuning == 'weak':
            VectorsString += 'distance:-1,trymany:true,'
        else:
            VectorsString += 'distance:-10,'

        if Tuning == 'animation' or Preset in ['faster', 'fastest']:
            VectorsString += 'bad:{sad:2000}}}}}'
        elif Tuning == 'weak':
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250,search:{distance:-1,satd:true}}]}'
        else:
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250}]}'

        # Create SmoothString
        if NewNum is not None:
            SmoothString = '{rate:{num:' + repr(NewNum) + ',den:' + repr(NewDen) + ',abs:true},'
        elif clip.fps_num / clip.fps_den in [15, 25, 30] or FrameDouble:
            SmoothString = '{rate:{num:2,den:1,abs:false},'
        else:
            SmoothString = '{rate:{num:60000,den:1001,abs:true},'

        if OverrideAlgo is not None:
            SmoothString += 'algo:' + repr(OverrideAlgo) + ',mask:{cover:80,'
        elif Tuning == 'animation':
            SmoothString += 'algo:2,mask:{'
        elif Tuning == 'smooth':
            SmoothString += 'algo:23,mask:{'
        else:
            SmoothString += 'algo:13,mask:{cover:80,'

        if OverrideArea is not None:
            SmoothString += f'area:{OverrideArea}'
        elif Tuning == 'smooth':
            SmoothString += 'area:150'
        else:
            SmoothString += 'area:0'

        if Tuning == 'weak':
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0,limits:{blocks:50}}}'
        else:
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0}}'

        # Make interpolation vector clip
        Super = clip.svp1.Super(SuperString)
        Vectors = core.svp1.Analyse(Super['clip'], Super['data'], clip, VectorsString)

        # Put it together
        return core.svp2.SmoothFps(clip, Super['clip'], Super['data'], Vectors['clip'], Vectors['data'], SmoothString)

    # Get either 1 or 2 clips depending on InputType
    if InputType == 'SBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2))
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2))
        return core.std.StackHorizontal([FirstEye, SecondEye])
    elif InputType == 'OU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2))
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2))
        return core.std.StackVertical([FirstEye, SecondEye])
    elif InputType == 'HSBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2).resize.Spline36(Input.width, Input.height))
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2).resize.Spline36(Input.width, Input.height))
        return core.std.StackHorizontal([FirstEye.resize.Spline36(Input.width // 2, Input.height),
                                         SecondEye.resize.Spline36(Input.width // 2, Input.height)])
    elif InputType == 'HOU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2)
                                     .resize.Spline36(Input.width, Input.height))
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2).resize.Spline36(Input.width, Input.height))
        return core.std.StackVertical([FirstEye.resize.Spline36(Input.width, Input.height // 2),
                                       SecondEye.resize.Spline36(Input.width, Input.height // 2)])
    else:
        return InterFrameProcess(Input)


def ChangeFPS(clip: vs.VideoNode, fpsnum: int, fpsden: int = 1) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('ChangeFPS: this is not a clip')

    factor = (fpsnum / fpsden) * (clip.fps_den / clip.fps_num)

    def frame_adjuster(n: int) -> vs.VideoNode:
        real_n = math.floor(n / factor)
        one_frame_clip = clip[real_n] * (len(clip) + 100)
        return one_frame_clip

    attribute_clip = clip.std.BlankClip(length=math.floor(len(clip) * factor), fpsnum=fpsnum, fpsden=fpsden)
    return attribute_clip.std.FrameEval(eval=frame_adjuster)
