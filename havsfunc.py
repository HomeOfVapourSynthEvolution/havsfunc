import vapoursynth as vs
import functools
import math

class HAvsFunc():
    
    """Holy's ported AviSynth functions for VapourSynth."""
    
    
    def __init__(self):
        self.core = vs.get_core()
    
    
    # Suggested by Mystery Keeper in "Denoise of tv-anime" thread
    def ediaa(self, a):
        if not isinstance(a, vs.VideoNode):
            raise ValueError('ediaa: This is not a clip !')
        
        return self.Resize(self.core.std.Transpose(self.core.eedi2.EEDI2(self.core.std.Transpose(self.core.eedi2.EEDI2(a, field=1)), field=1)),
                           a.width, a.height, -.5, -.5)
    
    
    # Anti-aliasing with contra-sharpening by Didée
    def daa(self, c):
        if not isinstance(c, vs.VideoNode):
            raise ValueError('daa: This is not a clip !')
        
        nn = self.core.nnedi3.nnedi3(c, field=3)
        dbl = self.core.std.Merge(self.core.std.SelectEvery(nn, 2, 0), self.core.std.SelectEvery(nn, 2, 1))
        dblD = self.core.std.MakeDiff(c, dbl)
        shrpD = self.core.std.MakeDiff(dbl, self.core.rgvs.RemoveGrain(dbl, 20 if c.width>1100 else 11))
        DD = self.core.rgvs.Repair(shrpD, dblD, 13)
        return self.core.std.MergeDiff(dbl, DD)
    
    
    # Anti-aliasing with edge masking by martino, mask using "sobel" taken from Kintaro's useless filterscripts and modded by thetoof for spline36
    def maa(self, input, noring=False):
        if not isinstance(input, vs.VideoNode) or input.format.id != vs.YUV420P8:
            raise ValueError('maa: This is not a YUV420P8 clip !')
        if not isinstance(noring, bool):
            raise ValueError("maa: 'noring' must be bool")
        
        mask = self.core.generic.Inflate(self.core.generic.Sobel(self.core.std.ShufflePlanes(input, planes=[0], colorfamily=vs.GRAY), 8, 8, rshift=3))
        aa_clip = self.Resize(
          self.core.avs.SangNom2(
            self.core.std.Transpose(
              self.core.avs.SangNom2(
                self.core.std.Transpose(
                  self.Resize(input, input.width*2, input.height*2, kernel='spline64' if noring else 'spline36', planes=[3, 1, 1], noring=noring))))),
          input.width, input.height, planes=[3, 1, 1])
        return self.core.std.MaskedMerge(input, aa_clip, mask, planes=[0])
    
    
    # Developped in the "fine anime antialiasing thread"
    #
    # Parameters:
    #  dark [int]       - strokes darkening strength. Default is 38
    #  thin [int]       - Presharpening. Default is 10
    #  sharp [int]      - Postsharpening. Default is 150
    #  smooth [int]     - Postsmoothing. Default is -1
    #  stabilize [bool] - Use post stabilization with Motion Compensation. Default is false
    #  tradius [int]    - 1 = MDegrain1 / 2 = MDegrain2 / 3 = MDegrain3. Default is 2
    #  aapel [int]      - accuracy of the motion estimation. Default is 2
    #  aaov [int]       - block overlap value. Default is 8 for HD / 4 for SD
    #  aablk [int]      - Size of a block. Default is 16 for HD / 8 for SD
    #  aatype [string]  - Use SangNom2() or EEDI2() for anti-aliasing. Default is 'SangNom2'
    #  noring [bool]    - In case of supersampling, indicates that a non-ringing algorithm must be used for aatype='SangNom2'. Default is false
    def SharpAAMCmod(self, orig, dark=38, thin=10, sharp=150, smooth=-1,
                     stabilize=False, tradius=2, aapel=2, aaov=None, aablk=None, aatype='SangNom2', noring=False):
        if not isinstance(orig, vs.VideoNode) or orig.format.id != vs.YUV420P8:
            raise ValueError('SharpAAMCmod: This is not a YUV420P8 clip !')
        
        w = orig.width
        h = orig.height
        if aaov is None:
            aaov = 8 if w > 1100 else 4
        if aablk is None:
            aablk = 16 if w > 1100 else 8
        
        if not isinstance(dark, int) or dark < 0 or dark > 256:
            raise ValueError("SharpAAMCmod: 'dark' have not a correct value! [0...256]")
        if not isinstance(thin, int) or thin < -128 or thin > 127:
            raise ValueError("SharpAAMCmod: 'thin' have not a correct value! [-128...127]")
        if not isinstance(sharp, int) or sharp < 0:
            raise ValueError("SharpAAMCmod: 'sharp' have not a correct value! [>=0]")
        if not isinstance(smooth, int) or smooth < -2 or smooth > 100:
            raise ValueError("SharpAAMCmod: 'smooth' have not a correct value! [-2,-1,0,1...100]")
        if not isinstance(stabilize, bool):
            raise ValueError("SharpAAMCmod: 'stabilize' must be bool")
        if not isinstance(tradius, int) or tradius < 1 or tradius > 3:
            raise ValueError("SharpAAMCmod: 'tradius' have not a correct value! [1,2,3]")
        if not isinstance(aapel, int) or aapel not in (1, 2, 4):
            raise ValueError("SharpAAMCmod: 'aapel' have not a correct value! [1,2,4]")
        if not isinstance(aablk, int) or aablk not in (4, 8, 16, 32):
            raise ValueError("SharpAAMCmod: 'aablk' have not a correct value! [4,8,16,32]")
        if not isinstance(aaov, int) or aaov > int(aablk / 2) or aaov % 2 != 0:
            raise ValueError("SharpAAMCmod: 'aaov' must be at least half aablk or less and must be an even figure")
        if not isinstance(aatype, str):
            raise ValueError("SharpAAMCmod: 'aatype' must be string")
        aatype = aatype.lower()
        if aatype not in ('sangnom2', 'eedi2'):
            raise ValueError("SharpAAMCmod: Please use SangNom2 or EEDI2 for 'aatype'")
        if not isinstance(noring, bool):
            raise ValueError("SharpAAMCmod: 'noring' must be bool")
        
        # x 128 / 0.86 ^ 255 *
        def get_lut(x):
            return min(round((x / 128) ** .86 * 255), 255)
        
        orig_y = self.core.std.ShufflePlanes(orig, planes=[0], colorfamily=vs.GRAY)
        m = self.core.std.Lut(self.Logic(self.core.generic.Convolution(orig_y, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
                                         self.core.generic.Convolution(orig_y, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False),
                                         'max'),
                              function=get_lut)
        if thin == 0 and dark == 0:
            preaa = orig
        elif thin == 0:
            preaa = self.FastLineDarkenMOD(orig, strength=dark)
        elif dark == 0:
            preaa = self.core.avs.aWarpSharp2(orig, depth=thin)
        else:
            preaa = self.core.avs.aWarpSharp2(self.FastLineDarkenMOD(orig, strength=dark), depth=thin)
        if aatype == 'sangnom2':
            aa = self.core.std.Merge(
              self.Resize(
                self.core.avs.SangNom2(
                  self.core.std.Transpose(
                    self.core.avs.SangNom2(
                      self.core.std.Transpose(self.Resize(preaa, w*2, h*2, kernel='spline64' if noring else 'spline36', planes=[3, 1, 1], noring=noring))))),
                w, h, planes=[3, 1, 1]),
              preaa,
              weight=[0, 1])
        else:
            aa = self.ediaa(preaa)
        if sharp == 0 and smooth == 0:
            postsh = aa
        else:
            postsh = self.LSFmod(aa, strength=sharp, overshoot=1, soft=smooth)
        merged = self.core.std.MaskedMerge(orig, postsh, m)
        
        if stabilize:
            sD = self.core.std.MakeDiff(orig, merged)
            
            origsuper = self.core.mv.Super(orig, pel=aapel)
            sDsuper = self.core.mv.Super(sD, pel=aapel, levels=1)
            
            fv1 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=False, delta=1, overlap=aaov)
            bv1 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=True, delta=1, overlap=aaov)
            if tradius > 1:
                fv2 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=False, delta=2, overlap=aaov)
                bv2 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=True, delta=2, overlap=aaov)
            if tradius > 2:
                fv3 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=False, delta=3, overlap=aaov)
                bv3 = self.core.mv.Analyse(origsuper, blksize=aablk, isb=True, delta=3, overlap=aaov)
            if tradius == 1:
                sDD = self.core.mv.Degrain1(sD, sDsuper, bv1, fv1)
            elif tradius == 2:
                sDD = self.core.mv.Degrain2(sD, sDsuper, bv1, fv1, bv2, fv2)
            else:
                sDD = self.core.mv.Degrain3(sD, sDsuper, bv1, fv1, bv2, fv2, bv3, fv3)
            
            reduc = .4
            sDD = self.core.std.Merge(self.core.std.Expr([sD, sDD], 'x 128 - abs y 128 - abs < x y ?'), sDD, weight=[1-reduc, 0])
            
            return self.core.std.MakeDiff(orig, sDD)
        else:
            return merged
    
    
    # Changes 2008-08-18: (Didée)
    # - Replaced the ugly stackXXX cascade with mt_LutSpa() (requires MaskTools v2.0a35)
    # - Changed Quant and Offset defaults to 24,28,2,4,4,8
    #
    # Changes 2010-05-25:
    # - Explicitly specified parameters of mt_LutSpa()
    #   (required due to position of new 'biased' parameter, starting from MaskTools 2.0a43)
    # - Non mod 16 input is now padded with borders internally
    #
    # Changes 2010-08-18:
    # - Replaced AddBorders with PointResize
    # - Changed Quant and Offset defaults to 18,19,3,4,1,1 to reduce blurring
    #
    # Changes 2010-10-16:
    # - Replaced 'relative' with the new 'mode' parameter in mt_LutSpa(), starting from MaskTools 2.0a45
    # - Changed Quant and Offset defaults to 24,26,1,1,2,2 to increase effectiveness, but still within sensible limits.
    #   (see for details: http://forum.doom9.org/showthread.php?p=810932#post810932)
    #
    # Changes 2011-11-29: (06_taro)
    # - Replaced (chroma=uv>2?"process":"ignore") by (chroma=uv>2?"process":"copy") to avoid garbage clip when uv=2.
    #   The formal parameter is not used by MaskTools2 any more, if ever used.
    #   Foxyshadis once mentioned chroma="ignore" but I had never found a document containing it.
    #
    # Parameters:
    #  quant1 [int] - Strength of block edge deblocking. Default is 24
    #  quant2 [int] - Strength of block internal deblocking. Default is 26
    #  aOff1 [int]  - halfway "sensitivity" and halfway a strength modifier for borders. Default is 1
    #  aOff2 [int]  - halfway "sensitivity" and halfway a strength modifier for block interiors. Default is 1
    #  bOff1 [int]  - "sensitivity to detect blocking" for borders. Default is 2
    #  bOff2 [int]  - "sensitivity to detect blocking" for block interiors. Default is 2
    #  uv [int]     - 3: use proposed method for chroma deblocking, 2: no chroma deblocking at all(fastest method), 1|-1: directly use chroma debl. from the normal|strong Deblock()
    def Deblock_QED(self, clp, quant1=24, quant2=26, aOff1=1, bOff1=2, aOff2=1, bOff2=2, uv=3):
        if not isinstance(clp, vs.VideoNode):
            raise ValueError('Deblock_QED: This is not a clip !')
        if not isinstance(quant1, int) or quant1 < 0 or quant1 > 60:
            raise ValueError("Deblock_QED: 'quant1' have not a correct value! [0...60]")
        if not isinstance(quant2, int) or quant2 < 0 or quant2 > 60:
            raise ValueError("Deblock_QED: 'quant2' have not a correct value! [0...60]")
        if not isinstance(aOff1, int):
            raise ValueError("Deblock_QED: 'aOff1' must be integer")
        if not isinstance(bOff1, int):
            raise ValueError("Deblock_QED: 'bOff1' must be integer")
        if not isinstance(aOff2, int):
            raise ValueError("Deblock_QED: 'aOff2' must be integer")
        if not isinstance(bOff2, int):
            raise ValueError("Deblock_QED: 'bOff2' must be integer")
        if not isinstance(uv, int) or uv not in (-1, 1, 2, 3):
            raise ValueError("Deblock_QED: 'uv' have not a correct value! [-1,1,2,3]")
        
        neutral = repr(1 << (clp.format.bits_per_sample - 1))
        peak = repr((1 << clp.format.bits_per_sample) - 1)
        
        if clp.format.num_planes == 1:
            uv = 2
        planes = [0, 1, 2] if uv == 3 else [0]
        
        # add borders if clp is not mod 8
        w = clp.width
        h = clp.height
        padX = 0 if w % 8 == 0 else 8 - w % 8
        padY = 0 if h % 8 == 0 else 8 - h % 8
        clp = self.Resize(clp, w+padX, h+padY, 0, 0, w+padX, h+padY, kernel='point')
        
        # block
        block = self.core.std.BlankClip(clp, width=6, height=6, format=vs.GRAY8, length=1, color=[0])
        block = self.core.std.AddBorders(block, 1, 1, 1, 1, color=[255])
        horizontal = []
        vertical = []
        for i in range(int(clp.width / 8)):
            horizontal += [block]
        block = self.core.std.StackHorizontal(horizontal)
        for i in range(int(clp.height / 8)):
            vertical += [block]
        block = self.core.std.StackVertical(vertical)
        if clp.format.num_planes == 3:
            blockc = self.core.std.CropAbs(block, width=clp.width>>clp.format.subsampling_w, height=clp.height>>clp.format.subsampling_h)
            block = self.core.std.ShufflePlanes([block, blockc], planes=[0, 0, 0], colorfamily=clp.format.color_family)
        block = self.core.fmtc.bitdepth(block, bits=clp.format.bits_per_sample, fulls=False, fulld=True)
        block = self.core.std.Loop(block, clp.num_frames)
        
        # create normal deblocking (for block borders) and strong deblocking (for block interiour)
        normal = self.core.deblock.Deblock(clp, quant=quant1, aoffset=aOff1, boffset=bOff1, planes=[0, 1, 2] if uv!=2 else [0])
        strong = self.core.deblock.Deblock(clp, quant=quant2, aoffset=aOff2, boffset=bOff2, planes=[0, 1, 2] if uv!=2 else [0])
        
        # build difference maps of both
        normalD = self.core.std.MakeDiff(clp, normal, planes=planes)
        strongD = self.core.std.MakeDiff(clp, strong, planes=planes)
        
        # separate border values of the difference maps, and set the interiours to '128'
        expr = 'y '+peak+' = x '+neutral+' ?'
        normalD2 = self.core.std.Expr([normalD, block], expr if uv==3 or clp.format.num_planes==1 else [expr, ''])
        strongD2 = self.core.std.Expr([strongD, block], expr if uv==3 or clp.format.num_planes==1 else [expr, ''])
        
        # interpolate the border values over the whole block: DCTFilter can do it. (Kiss to Tom Barry!)
        # (Note: this is not fully accurate, but a reasonable approximation.)
        # add borders if clp is not mod 16
        sw = strongD2.width
        sh = strongD2.height
        remX = 0 if sw % 16 == 0 else 16 - sw % 16
        remY = 0 if sh % 16 == 0 else 16 - sh % 16
        expr = 'x '+neutral+' - 1.01 * '+neutral+' +'
        strongD3 = self.core.std.CropRel(
          self.core.dct.Filter(
            self.core.std.Expr(self.Resize(strongD2, sw+remX, sh+remY, 0, 0, sw+remX, sh+remY, kernel='point'), expr if uv==3 or clp.format.num_planes==1 else [expr, '']),
            [1, 1, 0, 0, 0, 0, 0, 0]),
          left=0, top=0, right=remX, bottom=remY)
        
        # apply compensation from "normal" deblocking to the borders of the full-block-compensations calculated from "strong" deblocking ...
        expr = 'y '+neutral+' = x y ?'
        strongD4 = self.core.std.Expr([strongD3, normalD2], expr if uv==3 or clp.format.num_planes==1 else [expr, ''])
        
        # ... and apply it.
        deblocked = self.core.std.MakeDiff(clp, strongD4, planes=planes) 
        
        # simple decisions how to treat chroma
        if clp.format.num_planes == 3:
            if uv == -1:
                deblocked = self.core.std.Merge(deblocked, strong, weight=[0, 1])
            elif uv == 1:
                deblocked = self.core.std.Merge(deblocked, normal, weight=[0, 1])
        
        # remove mod 8 borders
        return self.core.std.CropRel(deblocked, left=0, top=0, right=padX, bottom=padY)
    
    
    # rx, ry [float, 1.0 ... 2.0 ... ~3.0]
    # As usual, the radii for halo removal.
    # Note: this function is rather sensitive to the radius settings. Set it as low as possible! If radius is set too high, it will start missing small spots.
    #
    # darkkstr, brightstr [float, 0.0 ... 1.0] [<0.0 and >1.0 possible]
    # The strength factors for processing dark and bright halos. Default 1.0 both for symmetrical processing.
    # On Comic/Anime, darkstr=0.4~0.8 sometimes might be better ... sometimes. In General, the function seems to preserve dark lines rather good.
    #
    # lowsens, highsens [int, 0 ... 50 ... 100]
    # Sensitivity settings, not that easy to describe them exactly ...
    # In a sense, they define a window between how weak an achieved effect has to be to get fully accepted, and how strong an achieved effect has to be to get fully discarded.
    # Defaults are 50 and 50 ... try and see for yourself.
    #
    # ss [float, 1.0 ... 1.5 ...]
    # Supersampling factor, to avoid creation of aliasing.
    #
    # noring [bool]
    # In case of supersampling, indicates that a non-ringing algorithm must be used.
    def DeHalo_alpha(self, clp, rx=2., ry=2., darkstr=1., brightstr=1., lowsens=50, highsens=50, ss=1.5, noring=False):
        if not isinstance(clp, vs.VideoNode) or clp.format.color_family != vs.YUV:
            raise ValueError('DeHalo_alpha: This is not a YUV clip !')
        if not (isinstance(rx, float) or isinstance(rx, int)) or rx < 1 or rx > 3:
            raise ValueError("DeHalo_alpha: 'rx' have not a correct value! [1.0...3.0]")
        if not (isinstance(ry, float) or isinstance(ry, int)) or ry < 1 or ry > 3:
            raise ValueError("DeHalo_alpha: 'ry' have not a correct value! [1.0...3.0]")
        if not (isinstance(darkstr, float) or isinstance(darkstr, int)):
            raise ValueError("DeHalo_alpha: 'darkstr' must be float or integer")
        if not (isinstance(brightstr, float) or isinstance(brightstr, int)):
            raise ValueError("DeHalo_alpha: 'brightstr' must be float or integer")
        if not isinstance(lowsens, int) or lowsens < 0 or lowsens > 100:
            raise ValueError("DeHalo_alpha: 'lowsens' have not a correct value! [0...100]")
        if not isinstance(highsens, int) or highsens < 0 or highsens > 100:
            raise ValueError("DeHalo_alpha: 'highsens' have not a correct value! [0...100]")
        if not (isinstance(ss, float) or isinstance(ss, int)) or ss < 1:
            raise ValueError("DeHalo_alpha: 'ss' have not a correct value! [>=1.0]")
        if not isinstance(noring, bool):
            raise ValueError("DeHalo_alpha: 'noring' must be bool")
        
        shift = clp.format.bits_per_sample - 8
        
        LOS = repr(lowsens << shift)
        HIS = repr(highsens / 100)
        DRK = repr(darkstr)
        BRT = repr(brightstr)
        ox = clp.width
        oy = clp.height
        
        clp_src = clp
        clp = self.core.std.ShufflePlanes(clp, planes=[0], colorfamily=vs.GRAY)
        halos = self.Resize(self.Resize(clp, self.m4(ox/rx), self.m4(oy/ry), kernel='bicubic'), ox, oy, kernel='bicubic', a1=1, a2=0)
        are = self.core.std.Expr([self.core.generic.Maximum(clp), self.core.generic.Minimum(clp)], 'x y -')
        ugly = self.core.std.Expr([self.core.generic.Maximum(halos), self.core.generic.Minimum(halos)], 'x y -')
        so = self.core.std.Expr([ugly, are], 'y x - y '+repr(0.001*2**shift)+' + / '+repr((1<<clp.format.bits_per_sample)-1)+' * '+LOS+' - y '+repr(256<<shift)+' + '+repr(512<<shift)+' / '+HIS+' + *')
        lets = self.core.std.MaskedMerge(halos, clp, so)
        if ss == 1:
            remove = self.core.rgvs.Repair(clp, lets, 1)
        else:
            remove = self.Resize(
              self.Logic(
                self.Logic(self.Resize(clp, self.m4(ox*ss), self.m4(oy*ss), kernel='spline64' if noring else 'spline36', noring=noring),
                           self.Resize(self.core.generic.Maximum(lets), self.m4(ox*ss), self.m4(oy*ss), kernel='bicubic'),
                           'min'),
                self.Resize(self.core.generic.Minimum(lets), self.m4(ox*ss), self.m4(oy*ss), kernel='bicubic'),
                'max'),
              ox, oy)
        them = self.core.std.Expr([clp, remove], 'x y < x x y - '+DRK+' * - x x y - '+BRT+' * - ?')
        
        return self.core.std.ShufflePlanes([them, clp_src], planes=[0, 1, 2], colorfamily=vs.YUV)
    
    
    # Y'et A'nother H'alo R'educing script
    def YAHR(self, clp):
        if not isinstance(clp, vs.VideoNode) or clp.format.id != vs.YUV420P8:
            raise ValueError('YAHR: This is not a YUV420P8 clip !')
        
        b1 = self.core.rgvs.RemoveGrain(self.MinBlur(clp, 2, planes=[0]), [11, 0])
        b1D = self.core.std.MakeDiff(clp, b1, planes=[0])
        w1 = self.core.avs.aWarpSharp2(clp, depth=32, chroma=3)
        w1b1 = self.core.rgvs.RemoveGrain(self.MinBlur(w1, 2, planes=[0]), [11, 0])
        w1b1D = self.core.std.MakeDiff(w1, w1b1, planes=[0])
        DD = self.core.rgvs.Repair(b1D, w1b1D, [13, 0])
        DD2 = self.core.std.MakeDiff(b1D, DD, planes=[0])
        return self.core.std.MakeDiff(clp, DD2, planes=[0])
    
    
    ######
    ###
    ### HQDering mod v1.8      by mawen1250      2014.03.22
    ###
    ### Requirements: GenericFilters, RemoveGrain/Repair, dither v1.25.1, dfttest v1.9.4, RemoveGrainHD v0.5
    ###
    ### Applies deringing by using a smart smoother near edges (where ringing occurs) only
    ###
    ### Parameters:
    ###  Y,U,V [bool]    - If true, the corresponding plane is processed. Otherwise, it is copied through to the output image as is. Default is true,false,false
    ###  lsb_in [bool]   - Input clip is 16-bit stacked or not. If true and smoothed/ringmask clips are provided, they must be 16-bit stacked as well. Default is false
    ###  lsb [bool]      - Output clip is 16-bit stacked or not. Default is false
    ###  dmode [int]     - Dither mode for 16-bit to 8-bit conversion. Default is 6
    ###  show [bool]     - Whether to output mask clip instead of filtered clip. Default is false
    ###  mthr [int]      - Threshold of prewitt edge mask, lower value means more aggressive processing. But for strong ringing, lower value will treat some ringing as edge, which protects this ringing from being processed. Default is 60
    ###  minp [int]      - Inpanding of prewitt edge mask, higher value means more aggressive processing. Default is 1
    ###  mrad [int]      - Expanding of edge mask, higher value means more aggressive processing. Default is 1
    ###  msmooth [int]   - Inflate of edge mask, smooth boundaries of mask. Default is 1
    ###  incedge [bool]  - Whether to include edge in ring mask, by default ring mask only include area near edges. Default is false
    ###  nrmode [int]    - Kernel of dering - 0: dfttest, 1: MinBlur(radius=1), 2: MinBlur(radius=2), 3: MinBlur(radius=3). Default is 2 for HD / 1 for SD
    ###  nrmodec [int]   - Kernel of dering for chroma. Default is nrmode
    ###  sigma [float]   - dfttest: sigma for medium frequecies. Default is 128.0
    ###  sigma2 [float]  - dfttest: sigma for low&high frequecies. Default is sigma/16
    ###  sbsize [int]    - dfttest: length of the sides of the spatial window. Default is 8 for HD / 6 for SD
    ###  sosize [int]    - dfttest: spatial overlap amount. Default is 6 for HD / 4 for SD
    ###  sharp [bool]    - Whether to use contra-sharpening to resharp deringed clip(luma only). Default is true
    ###  drrep [int]     - Use repair for details retention, recommended values are 13/12/1. Default is 13 for nrmode>0 / 0 for nrmode=0
    ###  thr [float]     - The same meaning with "thr" in Dither_limit_dif16, valid value range is [0, 128.0]. Default is 12.0
    ###  elast [float]   - The same meaning with "elast" in Dither_limit_dif16, valid value range is [1, inf). Default is 2.0
    ###                    Larger "thr" will result in more pixels being taken from processed clip
    ###                    Larger "thr" will result in less pixels being taken from input clip
    ###                    Larger "elast" will result in more pixels being blended from processed&input clip, for smoother merging
    ###  darkthr [float] - Threshold for darker area near edges, set it lower if you think deringing destroys too much lines, etc. Default is thr/4
    ###                    When "darkthr" is not equal to "thr", "thr" limits darkening while "darkthr" limits brightening
    ###
    ######
    def HQDeringmod(self, input, smoothed=None, ringmask=None, mrad=1, msmooth=1, incedge=False, mthr=60, minp=1,
                    nrmode=None, nrmodec=None, sigma=128., sigma2=None, sbsize=None, sosize=None, sharp=True, drrep=None, thr=12., elast=2., darkthr=None,
                    Y=True, U=False, V=False, lsb_in=False, lsb=False, dmode=6, show=False):
        if not isinstance(input, vs.VideoNode) or input.format.id != vs.YUV420P8:
            raise ValueError('HQDeringmod: This is not a YUV420P8 clip !')
        
        HD = input.width >= 1280 or (lsb_in and input.height >= 1440) or (not lsb_in and input.height >= 720)
        if nrmode is None:
            nrmode = 2 if HD else 1
        if nrmodec is None:
            nrmodec = nrmode
        if sigma2 is None:
            sigma2 = sigma / 16
        if sbsize is None:
            sbsize = 8 if HD else 6
        if sosize is None:
            sosize = 6 if HD else 4
        if drrep is None:
            drrep = 0 if nrmode == 0 else 13
        if darkthr is None:
            darkthr = thr / 4
        
        if smoothed is not None and (not isinstance(smoothed, vs.VideoNode) or smoothed.format.id != vs.YUV420P8):
            raise ValueError("HQDeringmod: 'smoothed' is not a YUV420P8 clip !")
        if ringmask is not None and (not isinstance(ringmask, vs.VideoNode) or ringmask.format.id != vs.YUV420P8):
            raise ValueError("HQDeringmod: 'ringmask' is not a YUV420P8 clip !")
        if not isinstance(mrad, int) or mrad < 0:
            raise ValueError("HQDeringmod: 'mrad' have not a correct value! [>=0]")
        if not isinstance(msmooth, int) or msmooth < 0:
            raise ValueError("HQDeringmod: 'msmooth' have not a correct value! [>=0]")
        if not isinstance(incedge, bool):
            raise ValueError("HQDeringmod: 'incedge' must be bool")
        if not isinstance(mthr, int) or mthr < 0:
            raise ValueError("HQDeringmod: 'mthr' have not a correct value! [>=0]")
        if not isinstance(minp, int) or minp < 0 or minp > 4:
            raise ValueError("HQDeringmod: 'minp' have not a correct value! [0...4]")
        if not isinstance(nrmode, int) or nrmode < 0 or nrmode > 3:
            raise ValueError("HQDeringmod: 'nrmode' have not a correct value! [0,1,2,3]")
        if not isinstance(nrmodec, int) or nrmodec < 0 or nrmodec > 3:
            raise ValueError("HQDeringmod: 'nrmodec' have not a correct value! [0,1,2,3]")
        if not (isinstance(sigma, float) or isinstance(sigma, int)) or sigma < 0:
            raise ValueError("HQDeringmod: 'sigma' have not a correct value! [>=0.0]")
        if not (isinstance(sigma2, float) or isinstance(sigma2, int)) or sigma2 < 0:
            raise ValueError("HQDeringmod: 'sigma2' have not a correct value! [>=0.0]")
        if not isinstance(sbsize, int) or sbsize < 1:
            raise ValueError("HQDeringmod: 'sbsize' have not a correct value! [>=1]")
        if not isinstance(sosize, int) or sosize < 0 or sosize >= sbsize:
            raise ValueError("HQDeringmod: 'sosize' have not a correct value! [0...sbsize-1]")
        if not isinstance(sharp, bool):
            raise ValueError("HQDeringmod: 'sharp' must be bool")
        if not isinstance(drrep, int) or drrep not in (0, 1, 2, 3, 4, 11, 12, 13, 14):
            raise ValueError("HQDeringmod: 'drrep' have not a correct value! [0,1,2,3,4,11,12,13,14]")
        if not (isinstance(thr, float) or isinstance(thr, int)) or thr < 0 or thr > 128:
            raise ValueError("HQDeringmod: 'thr' have not a correct value! [0.0...128.0]")
        if not (isinstance(elast, float) or isinstance(elast, int)) or elast < 1:
            raise ValueError("HQDeringmod: 'elast' have not a correct value! [>=1.0]")
        if not (isinstance(darkthr, float) or isinstance(darkthr, int)) or darkthr < 0 or darkthr > 128:
            raise ValueError("HQDeringmod: 'darkthr' have not a correct value! [0.0...128.0]")
        if not isinstance(Y, bool):
            raise ValueError("HQDeringmod: 'Y' must be bool")
        if not isinstance(U, bool):
            raise ValueError("HQDeringmod: 'U' must be bool")
        if not isinstance(V, bool):
            raise ValueError("HQDeringmod: 'V' must be bool")
        if not isinstance(lsb_in, bool):
            raise ValueError("HQDeringmod: 'lsb_in' must be bool")
        if not isinstance(lsb, bool):
            raise ValueError("HQDeringmod: 'lsb' must be bool")
        if not isinstance(dmode, int) or dmode < 0 or dmode > 7:
            raise ValueError("HQDeringmod: 'dmode' have not a correct value! [0,1,2,3,4,5,6,7]")
        if not isinstance(show, bool):
            raise ValueError("HQDeringmod: 'show' must be bool")
        
        sigma = repr(sigma)
        sigma2 = repr(sigma2)
        
        if Y and U and V:
            planes = [0, 1, 2]
        elif Y and U:
            planes = [0, 1]
        elif Y and V:
            planes = [0, 2]
        elif U and V:
            planes = [1, 2]
        elif Y:
            planes = [0]
        elif U:
            planes = [1]
        elif V:
            planes = [2]
        else:
            return input
        
        # Pre-Process: Bit Depth Conversion
        if lsb_in:
            if ringmask:
                ringmask = self.core.fmtc.stack16tonative(ringmask)
        elif lsb:
            input = self.Dither_convert_8_to_16(input)
            if smoothed:
                smoothed = self.Dither_convert_8_to_16(smoothed)
            if ringmask:
                ringmask = self.core.fmtc.bitdepth(ringmask, bits=16)
        
        # Kernel: Smoothing
        if not smoothed:
            if nrmode == nrmodec:
                if nrmode == 0:
                    if lsb or lsb_in:
                        smoothed = self.core.avs.dfttest(input, Y=Y, U=U, V=V, sbsize=sbsize, sosize=sosize, tbsize=1, lsb=True, lsb_in=True,
                                                         sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                    else:
                        smoothed = self.core.avs.dfttest(input, Y=Y, U=U, V=V, sbsize=sbsize, sosize=sosize, tbsize=1,
                                                         sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                else:
                    if lsb or lsb_in:
                        smoothed = self.MinBlur(input, nrmode, planes=planes, lsb=True, lsb_in=True)
                    else:
                        smoothed = self.MinBlur(input, nrmode, planes=planes)
            else:
                if not Y:
                    smoothl = input
                elif nrmode == 0:
                    if lsb or lsb_in:
                        smoothl = self.core.avs.dfttest(input, U=False, V=False, sbsize=sbsize, sosize=sosize, tbsize=1, lsb=True, lsb_in=True,
                                                        sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                    else:
                        smoothl = self.core.avs.dfttest(input, U=False, V=False, sbsize=sbsize, sosize=sosize, tbsize=1,
                                                        sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                else:
                    if lsb or lsb_in:
                        smoothl = self.MinBlur(input, nrmode, planes=[0], lsb=True, lsb_in=True)
                    else:
                        smoothl = self.MinBlur(input, nrmode, planes=[0])
                if not (U or V):
                    smoothc = input
                elif nrmodec == 0:
                    if lsb or lsb_in:
                        smoothc = self.core.avs.dfttest(input, Y=False, U=U, V=V, sbsize=sbsize, sosize=sosize, tbsize=1, lsb=True, lsb_in=True,
                                                        sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                    else:
                        smoothc = self.core.avs.dfttest(input, Y=False, U=U, V=V, sbsize=sbsize, sosize=sosize, tbsize=1,
                                                        sstring='0.0:'+sigma2+' 0.05:'+sigma+' 0.5:'+sigma+' 0.75:'+sigma2+' 1.0:0.0')
                else:
                    if lsb or lsb_in:
                        smoothc = self.MinBlur(input, nrmodec, planes=[1, 2] if U and V else [1] if U else [2], lsb=True, lsb_in=True)
                    else:
                        smoothc = self.MinBlur(input, nrmodec, planes=[1, 2] if U and V else [1] if U else [2])
                if lsb or lsb_in:
                    smoothed = self.core.std.Merge(self.core.fmtc.stack16tonative(smoothl), self.core.fmtc.stack16tonative(smoothc), weight=[0, 1])
                    smoothed = self.core.fmtc.nativetostack16(smoothed)
                else:
                    smoothed = self.core.std.Merge(smoothl, smoothc, weight=[0, 1])
        
        # Post-Process: Contra-Sharpening
        if not (sharp and Y):
            sclp = smoothed
        elif lsb or lsb_in:
            sclp = self.ContraSharpeningHD(smoothed, input, lsb=True, lsb_in=True)
        else:
            sclp = self.ContraSharpeningHD(smoothed, input)
        
        # Post-Process: Repairing
        if drrep == 0:
            repclp = sclp
        elif lsb or lsb_in:
            repclp = self.core.avs.Dither_repair16(input, sclp, drrep if Y else -1, drrep if U else -1, drrep if V else -1)
        else:
            repclp = self.core.rgvs.Repair(input, sclp, [drrep if Y else 0, drrep if U else 0, drrep if V else 0])
        
        # Post-Process: Limiting
        if (thr == 0 and darkthr == 0) or (thr == 128 and darkthr == 128):
            limitclp = repclp
        elif lsb or lsb_in:
            limitclp = self.core.fmtc.nativetostack16(self.LimitDiff(self.core.fmtc.stack16tonative(repclp), self.core.fmtc.stack16tonative(input),
                                                                     thr=thr, elast=elast, darkthr=darkthr, planes=planes))
        else:
            limitclp = self.LimitDiff(repclp, input, thr=thr, elast=elast, darkthr=darkthr, planes=planes)
        
        # Post-Process: Ringing Mask Generating
        if not ringmask:
            if lsb or lsb_in:
                prewittm = self.core.generic.Prewitt(self.core.std.ShufflePlanes(self.core.fmtc.stack16tonative(input), planes=[0], colorfamily=vs.GRAY), mthr<<8)
            else:
                prewittm = self.core.generic.Prewitt(self.core.std.ShufflePlanes(input, planes=[0], colorfamily=vs.GRAY), mthr)
            fmask = self.core.generic.Hysteresis(self.core.rgvs.RemoveGrain(prewittm, 4), prewittm)
            if mrad > 0:
                omask = self.mt_expand_multi(fmask, sw=mrad, sh=mrad)
            else:
                omask = fmask
            if msmooth > 0:
                omask = self.mt_inflate_multi(omask, radius=msmooth)
            if incedge:
                ringmask = omask
            else:
                if minp > 3:
                    imask = self.core.generic.Minimum(self.core.generic.Minimum(fmask))
                elif minp > 2:
                    imask = self.core.generic.Minimum(self.core.generic.Minimum(self.core.generic.Inflate(fmask)))
                elif minp > 1:
                    imask = self.core.generic.Minimum(fmask)
                elif minp > 0:
                    imask = self.core.generic.Minimum(self.core.generic.Inflate(fmask))
                else:
                    imask = fmask
                ringmask = self.core.std.Expr([omask, imask], 'x 65535 y - * 65535 /' if lsb or lsb_in else 'x 255 y - * 255 /')
        
        # Mask Merging & Output
        if show:
            last = ringmask
        elif lsb or lsb_in:
            last = self.core.std.MaskedMerge(self.core.fmtc.stack16tonative(input), self.core.fmtc.stack16tonative(limitclp), ringmask,
                                             planes=planes, first_plane=True)
        else:
            last = self.core.std.MaskedMerge(input, limitclp, ringmask, planes=planes, first_plane=True)
        
        if lsb:
            return self.core.fmtc.nativetostack16(last)
        elif lsb_in:
            return self.core.fmtc.bitdepth(last, bits=8, dmode=dmode)
        else:
            return last
    
    
    #-------------------------------------------------------------------#
    #                                                                   #
    #                    QTGMC 3.33, by Vit, 2012                       #
    #                                                                   #
    #   A high quality deinterlacer using motion-compensated temporal   #
    #  smoothing, with a range of features for quality and convenience  #
    #          Originally based on TempGaussMC_beta2 by Didée           #
    #                                                                   #
    #-------------------------------------------------------------------#
    #
    # Full documentation is in the 'QTGMC' html file that comes with this script
    #
    # --- LATEST CHANGES ---
    #
    # v3.33
    # - Increased maximum value for Rep0, Rep1 and Rep2 to 7 (from 5). Higher values help with flicker on static detail, potential for minor motion blur
    # - Bug fix for the fact that Bob always outputs a BFF clip regardless of field order of input (thanks ajp_anton)
    # - Improved generation of noise (NoiseDeint="Generate") for noise bypass / EZKeepGrain
    # - Minor change to denoising
    #
    # --- REQUIREMENTS ---
    #
    # Input colorspaces: YV12
    #
    # Core plugins:
    #	MVTools2 (2.5.11.2 or above)
    #	GenericFilters
    #	NNEDI3
    #	RemoveGrain/Repair
    #
    # Additional plugins:
    #	NNEDI2, NNEDI, EEDI3, EEDI2, TDeInt - if selected directly or via a source-match preset
    #	VerticalCleaner - for SVThin or Lossless modes
    #	FFT3DFilter - if selected for noise processing
    #	dfttest - if selected for noise processing
    #	FFT3dGPU - if selected for noise processing
    #		For FFT3DFilter & ddftest you also need the FFTW3 library (FFTW.org). On Windows the file needed for both is libfftw3f-3.dll. However, for FFT3DFilter
    #		the file needs to be called FFTW3.dll, so you will need two copies and rename one. On Windows put the files in your System32 or SysWow64 folder
    #	AddGrain - if NoiseDeint="Generate" selected for noise bypass
    #
    # --- GETTING STARTED ---
    #
    # Install AviSynth and ensure you have at least the core plugins listed in the requirements section above. Put them in the plugins autoload folder.
    # To use QTGMC write a script like this:
    #	YourSource("yourfile")   # DGDecode_mpeg2source, FFVideoSource, AviSource, whatever your source requires
    #	QTGMC( Preset="Slow" )
    #	SelectEven()             # Add this line to keep original frame rate, leave it out for smoother doubled frame rate
    #
    # Save this script with an ".avs" extension. You can now use it as an AVI source for encoding.
    #
    # The "Preset" used selects sensible settings for a given encoding speed. Choose a preset from:
    #	"Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast" & "Draft"
    # The default preset is "Slower"
    # Don't be obsessed with using slower settings as the differences can be small. HD material benefits little from extreme settings (and will be very slow)
    # For much faster speeds read the full documentation, the section on 'Multi-threading'
    #
    # There are many settings for tweaking the script, full details in the main documentation. You can display settings currently being used with "ShowSettings":
    #	QTGMC( Preset="Slow", ShowSettings=True )
    def QTGMC(self, Input, Preset='Slower', TR0=None, TR1=None, TR2=None, Rep0=None, Rep1=0, Rep2=None, EdiMode=None, RepChroma=True, NNSize=None,
              NNeurons=None, EdiQual=1, EdiMaxD=None, ChromaEdi='', EdiThreads=0, EdiExt=None, Sharpness=None, SMode=None, SLMode=None, SLRad=None, SOvs=0,
              SVThin=.0, Sbb=None, SrchClipPP=None, SubPel=None, SubPelInterp=2, BlockSize=None, Overlap=None, Search=None, SearchParam=None, PelSearch=None,
              ChromaMotion=None, TrueMotion=False, Lambda=None, LSAD=None, PNew=None, PLevel=None, GlobalMotion=True, DCT=0, ThSAD1=640, ThSAD2=256, ThSCD1=180,
              ThSCD2=98, SourceMatch=0, MatchPreset=None, MatchEdi=None, MatchPreset2=None, MatchEdi2=None, MatchTR2=1, MatchEnhance=.5, Lossless=0,
              NoiseProcess=None, EZDenoise=None, EZKeepGrain=None, NoisePreset='Fast', Denoiser=None, DenoiseThreads=None, DenoiseMC=None, NoiseTR=None,
              Sigma=None, ChromaNoise=False, ShowNoise=.0, GrainRestore=None, NoiseRestore=None, NoiseDeint=None, StabilizeNoise=None, InputType=0,
              ProgSADMask=None, FPSDivisor=1, ShutterBlur=0, ShutterAngleSrc=180, ShutterAngleOut=180, SBlurLimit=4, Border=False, Precise=None, Tuning='None',
              ShowSettings=False, ForceTR=0, TFF=None):
        #---------------------------------------
        # Presets
        
        # Select presets / tuning
        if not isinstance(Preset, str):
            raise ValueError("QTGMC: 'Preset' must be string")
        Preset = Preset.lower()
        presets = ('placebo', 'very slow', 'slower', 'slow', 'medium', 'fast', 'faster', 'very fast', 'super fast', 'ultra fast', 'draft')
        try:
            pNum = presets.index(Preset)
        except:
            raise ValueError("QTGMC: 'Preset' choice is invalid")
        
        if MatchPreset is None:
            mpNum1 = pNum + 3 if pNum + 3 <= 9 else 9
            MatchPreset = presets[mpNum1]
        else:
            if not isinstance(MatchPreset, str):
                raise ValueError("QTGMC: 'MatchPreset' must be string")
            try:
                mpNum1 = presets[:10].index(MatchPreset.lower())
            except:
                raise ValueError("QTGMC: 'MatchPreset' choice is invalid/unsupported")
        
        if MatchPreset2 is None:
            mpNum2 = mpNum1 + 2 if mpNum1 + 2 <= 9 else 9
            MatchPreset2 = presets[mpNum2]
        else:
            if not isinstance(MatchPreset2, str):
                raise ValueError("QTGMC: 'MatchPreset2' must be string")
            try:
                mpNum2 = presets[:10].index(MatchPreset2.lower())
            except:
                raise ValueError("QTGMC: 'MatchPreset2' choice is invalid/unsupported")
        
        if not isinstance(NoisePreset, str):
            raise ValueError("QTGMC: 'NoisePreset' must be string")
        try:
            npNum = presets[2:7].index(NoisePreset.lower())
        except:
            raise ValueError("QTGMC: 'NoisePreset' choice is invalid")
        
        if not isinstance(Tuning, str):
            raise ValueError("QTGMC: 'Tuning' must be string")
        try:
            tNum = ('none', 'dv-sd', 'dv-hd').index(Tuning.lower())
        except:
            raise ValueError("QTGMC: 'Tuning' choice is invalid")
        
        # Tunings only affect blocksize in this version
        bs = (16, 16, 32)[tNum]
        bs2 = 32 if bs >= 16 else bs * 2
        
        #                                                   Very                                                              Very       Super      Ultra
        # Preset groups:                          Placebo   Slow       Slower     Slow       Medium     Fast       Faster     Fast       Fast       Fast       Draft
        if TR0          is None: TR0          = ( 2,        2,         2,         2,         2,         2,         1,         1,         1,         1,         0        )[pNum]
        if TR1          is None: TR1          = ( 2,        2,         2,         1,         1,         1,         1,         1,         1,         1,         1        )[pNum]
        if TR2 is not None:
            TR2X = TR2
        else:
            TR2X                              = ( 3,        2,         1,         1,         1,         0,         0,         0,         0,         0,         0        )[pNum]
        if Rep0         is None: Rep0         = ( 4,        4,         4,         4,         3,         3,         0,         0,         0,         0,         0        )[pNum]
        if Rep2         is None: Rep2         = ( 4,        4,         4,         4,         4,         4,         4,         4,         3,         3,         0        )[pNum]
        if EdiMode is not None:
            if not isinstance(EdiMode, str):
                raise ValueError("QTGMC: 'EdiMode' must be string")
            EdiMode = EdiMode.lower()
        else:
            EdiMode                           = ('nnedi3', 'nnedi3',  'nnedi3',  'nnedi3',  'nnedi3',  'nnedi3',  'nnedi3',  'nnedi3',  'nnedi3',  'tdeint',  'bob'     )[pNum]
        if NNSize       is None: NNSize       = ( 1,        1,         1,         1,         5,         5,         4,         4,         4,         4,         4        )[pNum]
        if NNeurons     is None: NNeurons     = ( 2,        2,         1,         1,         1,         0,         0,         0,         0,         0,         0        )[pNum]
        if EdiMaxD      is None: EdiMaxD      = ( 12,       10,        8,         7,         7,         6,         6,         5,         4,         4,         4        )[pNum]
        if not isinstance(ChromaEdi, str):
            raise ValueError("QTGMC: 'ChromaEdi' must be string")
        ChromaEdi = ChromaEdi.lower()
        if SMode        is None: SMode        = ( 2,        2,         2,         2,         2,         2,         2,         2,         2,         2,         0        )[pNum]
        if SLMode is not None:
            SLModeX = SLMode
        else:
            SLModeX                           = ( 2,        2,         2,         2,         2,         2,         2,         2,         0,         0,         0        )[pNum]
        if SLRad        is None: SLRad        = ( 3,        1,         1,         1,         1,         1,         1,         1,         1,         1,         1        )[pNum]
        if Sbb          is None: Sbb          = ( 3,        1,         1,         0,         0,         0,         0,         0,         0,         0,         0        )[pNum]
        if SrchClipPP   is None: SrchClipPP   = ( 3,        3,         3,         3,         3,         2,         2,         2,         1,         1,         0        )[pNum]
        if SubPel       is None: SubPel       = ( 2,        2,         2,         2,         1,         1,         1,         1,         1,         1,         1        )[pNum]
        if BlockSize    is None: BlockSize    = ( bs,       bs,        bs,        bs,        bs,        bs,        bs2,       bs2,       bs2,       bs2,       bs2      )[pNum]
        bs = BlockSize
        if Overlap      is None: Overlap      = (int(bs/2), int(bs/2), int(bs/2), int(bs/2), int(bs/2), int(bs/2), int(bs/2), int(bs/4), int(bs/4), int(bs/4), int(bs/4))[pNum]
        if Search       is None: Search       = (5,         4,         4,         4,         4,         4,         4,         4,         0,         0,         0        )[pNum]
        if SearchParam  is None: SearchParam  = (2,         2,         2,         2,         2,         2,         2,         1,         1,         1,         1        )[pNum]
        if PelSearch    is None: PelSearch    = (2,         2,         2,         2,         1,         1,         1,         1,         1,         1,         1        )[pNum]
        if ChromaMotion is None: ChromaMotion = (True,      True,      True,      False,     False,     False,     False,     False,     False,     False,     False    )[pNum]
        if Precise      is None: Precise      = (True,      True,      False,     False,     False,     False,     False,     False,     False,     False,     False    )[pNum]
        if ProgSADMask  is None: ProgSADMask  = (10.,       10.,       10.,       10.,       10.,       .0,        .0,        .0,        .0,        .0,        .0       )[pNum]
        
        # Noise presets                               Slower      Slow       Medium     Fast       Faster
        if Denoiser is not None:
            if not isinstance(Denoiser, str):
                raise ValueError("QTGMC: 'Denoiser' must be string")
            Denoiser = Denoiser.lower()
        else:
            Denoiser                              = ('dfttest',  'dfttest', 'dfttest', 'fft3df',  'fft3df')[npNum]
        if DenoiseMC      is None: DenoiseMC      = ( True,       True,      False,     False,     False  )[npNum]
        if NoiseTR        is None: NoiseTR        = ( 2,          1,         1,         1,         0      )[npNum]
        if Denoiser == 'fft3dgpu' and NoiseTR == 2:
            NoiseTR = 1     # FFT3dGPU doesn't support bt=5, so restrict the maximum of NoiseTR to 1
        if NoiseDeint is not None:
            if not isinstance(NoiseDeint, str):
                raise ValueError("QTGMC: 'NoiseDeint' must be string")
            NoiseDeint = NoiseDeint.lower()
        else:
            NoiseDeint                            = ('generate', 'bob',      '',        '',        ''     )[npNum]
        if StabilizeNoise is None: StabilizeNoise = ( True,       True,      True,      False,     False  )[npNum]
        
        # The basic source-match step corrects and re-runs the interpolation of the input clip. So it initialy uses same interpolation settings as the main preset
        MatchNNSize = NNSize
        MatchNNeurons = NNeurons
        MatchEdiMaxD = EdiMaxD
        MatchEdiQual = EdiQual
        
        # However, can use a faster initial interpolation when using source-match allowing the basic source-match step to "correct" it with higher quality settings
        if SourceMatch != 0 and mpNum1 < pNum:
            raise ValueError("QTGMC: 'MatchPreset' cannot use a slower setting than 'Preset'")
        # Basic source-match presets
        if SourceMatch != 0:
            #                     Very                                            Very   Super  Ultra
            #           Placebo   Slow   Slower   Slow   Medium   Fast   Faster   Fast   Fast   Fast
            NNSize   = (1,        1,     1,       1,     5,       5,     4,       4,     4,     4)[mpNum1]
            NNeurons = (2,        2,     1,       1,     1,       0,     0,       0,     0,     0)[mpNum1]
            EdiMaxD  = (12,       10,    8,       7,     7,       6,     6,       5,     4,     4)[mpNum1]
            EdiQual  = (1,        1,     1,       1,     1,       1,     1,       1,     1,     1)[mpNum1]
        TempEdi = EdiMode   # Main interpolation is actually done by basic-source match step when enabled, so a little swap and wriggle is needed
        if SourceMatch != 0 and MatchEdi is not None:
            if not isinstance(MatchEdi, str):
                raise ValueError("QTGMC: 'MatchEdi' must be string")
            EdiMode = MatchEdi.lower()
        MatchEdi = TempEdi
        
        #                                             Very                                                        Very      Super     Ultra
        # Refined source-match presets      Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast      Fast
        if MatchEdi2 is not None:
            if not isinstance(MatchEdi2, str):
                raise ValueError("QTGMC: 'MatchEdi2' must be string")
            MatchEdi2 = MatchEdi2.lower()
        else:
            MatchEdi2                   = ('nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'tdeint',  '')[mpNum2]
        MatchNNSize2                    = ( 1,        1,        1,        1,        5,        5,        4,        4,        4,        4 )[mpNum2]
        MatchNNeurons2                  = ( 2,        2,        1,        1,        1,        0,        0,        0,        0,        0 )[mpNum2]
        MatchEdiMaxD2                   = ( 12,       10,       8,        7,        7,        6,        6,        5,        4,        4 )[mpNum2]
        MatchEdiQual2                   = ( 1,        1,        1,        1,        1,        1,        1,        1,        1,        1 )[mpNum2]
        
        #---------------------------------------
        # Settings
        
        if not isinstance(Input, vs.VideoNode) or Input.format.id != vs.YUV420P8:
            raise ValueError('QTGMC: This is not a YUV420P8 clip !')
        if not isinstance(TR0, int) or TR0 < 0 or TR0 > 2:
            raise ValueError("QTGMC: 'TR0' have not a correct value! [0,1,2]")
        if not isinstance(TR1, int) or TR1 < 0 or TR1 > 2:
            raise ValueError("QTGMC: 'TR1' have not a correct value! [0,1,2]")
        if not isinstance(Rep0, int) or Rep0 < 0:
            raise ValueError("QTGMC: 'Rep0' have not a correct value! [>=0]")
        if not isinstance(Rep1, int) or Rep1 < 0:
            raise ValueError("QTGMC: 'Rep1' have not a correct value! [>=0]")
        if not isinstance(Rep2, int) or Rep2 < 0:
            raise ValueError("QTGMC: 'Rep2' have not a correct value! [>=0]")
        if not isinstance(RepChroma, bool):
            raise ValueError("QTGMC: 'RepChroma' must be bool")
        if not isinstance(NNSize, int) or NNSize < 0 or NNSize > 6:
            raise ValueError("QTGMC: 'NNSize' have not a correct value! [0,1,2,3,4,5,6]")
        if not isinstance(NNeurons, int) or NNeurons < 0 or NNeurons > 4:
            raise ValueError("QTGMC: 'NNeurons' have not a correct value! nnedi2 [0,1,2], nnedi3 [0,1,2,3,4]")
        if not isinstance(EdiQual, int) or EdiQual < 1 or EdiQual > 3:
            raise ValueError("QTGMC: 'EdiQual' have not a correct value! nnedi2 [1,2,3], nnedi3 [1,2]")
        if not isinstance(EdiMaxD, int) or EdiMaxD < 1:
            raise ValueError("QTGMC: 'EdiMaxD' have not a correct value! [>=1]")
        if not isinstance(EdiThreads, int) or EdiThreads < 0:
            raise ValueError("QTGMC: 'EdiThreads' have not a correct value! [>=0]")
        if EdiExt is not None and (not isinstance(EdiExt, vs.VideoNode) or EdiExt.format.id != vs.YUV420P8):
            raise ValueError("QTGMC: 'EdiExt' is not a YUV420P8 clip !")
        if not isinstance(SMode, int) or SMode < 0 or SMode > 2:
            raise ValueError("QTGMC: 'SMode' have not a correct value! [0,1,2]")
        if not isinstance(SLRad, int) or SLRad < 0:
            raise ValueError("QTGMC: 'SLRad' have not a correct value! [>=0]")
        if not isinstance(SOvs, int) or SOvs < 0 or SOvs > 255:
            raise ValueError("QTGMC: 'SOvs' have not a correct value! [0...255]")
        if not (isinstance(SVThin, float) or isinstance(SVThin, int)) or SVThin < 0:
            raise ValueError("QTGMC: 'SVThin' have not a correct value! [>=0.0]")
        if not isinstance(Sbb, int) or Sbb < 0 or Sbb > 3:
            raise ValueError("QTGMC: 'Sbb' have not a correct value! [0,1,2,3]")
        if not isinstance(SrchClipPP, int) or SrchClipPP < 0 or SrchClipPP > 3:
            raise ValueError("QTGMC: 'SrchClipPP' have not a correct value! [0,1,2,3]")
        if not isinstance(SubPel, int) or SubPel not in (1, 2, 4):
            raise ValueError("QTGMC: 'SubPel' have not a correct value! [1,2,4]")
        if not isinstance(SubPelInterp, int) or SubPelInterp < 0 or SubPelInterp > 2:
            raise ValueError("QTGMC: 'SubPelInterp' have not a correct value! [0,1,2]")
        if not isinstance(BlockSize, int) or BlockSize not in (4, 8, 16, 32):
            raise ValueError("QTGMC: 'BlockSize' have not a correct value! [4,8,16,32]")
        if not isinstance(Overlap, int) or Overlap > int(BlockSize / 2) or Overlap % 2 != 0:
            raise ValueError("QTGMC: 'Overlap' must be at least half BlockSize or less and must be an even figure")
        if not isinstance(Search, int) or Search < 0 or Search > 5:
            raise ValueError("QTGMC: 'Search' have not a correct value! [0,1,2,3,4,5]")
        if not isinstance(SearchParam, int) or SearchParam < 0:
            raise ValueError("QTGMC: 'SearchParam' have not a correct value! [>=0]")
        if not isinstance(PelSearch, int) or PelSearch < 0:
            raise ValueError("QTGMC: 'PelSearch' have not a correct value! [>=0]")
        if not isinstance(ChromaMotion, bool):
            raise ValueError("QTGMC: 'ChromaMotion' must be bool")
        if not isinstance(TrueMotion, bool):
            raise ValueError("QTGMC: 'TrueMotion' must be bool")
        if not isinstance(GlobalMotion, bool):
            raise ValueError("QTGMC: 'GlobalMotion' must be bool")
        if not isinstance(DCT, int) or DCT < 0 or DCT > 10:
            raise ValueError("QTGMC: 'DCT' have not a correct value! [0,1,2,3,4,5,6,7,8,9,10]")
        if not isinstance(ThSAD1, int) or ThSAD1 < 0:
            raise ValueError("QTGMC: 'ThSAD1' have not a correct value! [>=0]")
        if not isinstance(ThSAD2, int) or ThSAD2 < 0:
            raise ValueError("QTGMC: 'ThSAD2' have not a correct value! [>=0]")
        if not isinstance(ThSCD1, int) or ThSCD1 < 0:
            raise ValueError("QTGMC: 'ThSCD1' have not a correct value! [>=0]")
        if not isinstance(ThSCD2, int) or ThSCD2 < 0 or ThSCD2 > 255:
            raise ValueError("QTGMC: 'ThSCD2' have not a correct value! [0...255]")
        if not isinstance(SourceMatch, int) or SourceMatch < 0 or SourceMatch > 3:
            raise ValueError("QTGMC: 'SourceMatch' have not a correct value! [0,1,2,3]")
        if not isinstance(MatchTR2, int) or MatchTR2 < 0 or MatchTR2 > 2:
            raise ValueError("QTGMC: 'MatchTR2' have not a correct value! [0,1,2]")
        if not (isinstance(MatchEnhance, float) or isinstance(MatchEnhance, int)) or MatchEnhance < 0:
            raise ValueError("QTGMC: 'MatchEnhance' have not a correct value! [>=0.0]")
        if not isinstance(Lossless, int) or Lossless < 0 or Lossless > 2:
            raise ValueError("QTGMC: 'Lossless' have not a correct value! [0,1,2]")
        if EZDenoise is not None and (not (isinstance(EZDenoise, float) or isinstance(EZDenoise, int)) or EZDenoise < 0):
            raise ValueError("QTGMC: 'EZDenoise' have not a correct value! [>=0.0]")
        if EZKeepGrain is not None and (not (isinstance(EZKeepGrain, float) or isinstance(EZKeepGrain, int)) or EZKeepGrain < 0):
            raise ValueError("QTGMC: 'EZKeepGrain' have not a correct value! [>=0.0]")
        if EZDenoise is not None and EZDenoise > 0 and EZKeepGrain is not None and EZKeepGrain > 0:
            raise ValueError("QTGMC: EZDenoise and EZKeepGrain cannot be used together")
        if not isinstance(DenoiseMC, bool):
            raise ValueError("QTGMC: 'DenoiseMC' must be bool")
        if not isinstance(NoiseTR, int) or NoiseTR < 0 or NoiseTR > 2:
            raise ValueError("QTGMC: 'NoiseTR' have not a correct value! [0,1,2]")
        if not isinstance(ChromaNoise, bool):
            raise ValueError("QTGMC: 'ChromaNoise' must be bool")
        if not (isinstance(ShowNoise, bool) or isinstance(ShowNoise, float) or isinstance(ShowNoise, int)):
            raise ValueError("QTGMC: 'ShowNoise' only accepts bool and float inputs")
        if not isinstance(StabilizeNoise, bool):
            raise ValueError("QTGMC: 'StabilizeNoise' must be bool")
        if not isinstance(InputType, int) or InputType < 0 or InputType > 3:
            raise ValueError("QTGMC: 'InputType' have not a correct value! [0,1,2,3]")
        if not (isinstance(ProgSADMask, float) or isinstance(ProgSADMask, int)) or ProgSADMask < 0:
            raise ValueError("QTGMC: 'ProgSADMask' have not a correct value! [>=0.0]")
        if not isinstance(FPSDivisor, int) or FPSDivisor < 1:
            raise ValueError("QTGMC: 'FPSDivisor' have not a correct value! [>=1]")
        if not isinstance(ShutterBlur, int) or ShutterBlur < 0 or ShutterBlur > 3:
            raise ValueError("QTGMC: 'ShutterBlur' have not a correct value! [0,1,2,3]")
        if not isinstance(ShutterAngleSrc, int) or ShutterAngleSrc < 0 or ShutterAngleSrc > 360:
            raise ValueError("QTGMC: 'ShutterAngleSrc' have not a correct value! [0...360]")
        if not isinstance(ShutterAngleOut, int) or ShutterAngleOut < 0 or ShutterAngleOut > 360:
            raise ValueError("QTGMC: 'ShutterAngleOut' have not a correct value! [0...360]")
        if not isinstance(SBlurLimit, int) or SBlurLimit < 0:
            raise ValueError("QTGMC: 'SBlurLimit' have not a correct value! [>=0]")
        if not isinstance(Border, bool):
            raise ValueError("QTGMC: 'Border' must be bool")
        if not isinstance(Precise, bool):
            raise ValueError("QTGMC: 'Precise' must be bool")
        if not isinstance(ShowSettings, bool):
            raise ValueError("QTGMC: 'ShowSettings' must be bool")
        if not isinstance(ForceTR, int) or ForceTR < 0 or ForceTR > 3:
            raise ValueError("QTGMC: 'ForceTR' have not a correct value! [0,1,2,3]")
        if not isinstance(TFF, bool):
            raise ValueError("QTGMC: 'TFF' must be set. Setting TFF to true means top field first and false means bottom field first")
        
        # Core and Interpolation defaults
        if SourceMatch > 0 and TR2 is None:
            TR2 = 1 if TR2X == 0 else TR2X  # ***TR2 defaults always at least 1 when using source-match***
        else:
            TR2 = TR2X
        if EdiMode == 'nnedi2' and NNeurons > 2:
            NNeurons = 2    # Smaller range for NNeurons in NNEDI2 (which calls it nsize)
        if EdiMode == 'nnedi3' and EdiQual > 2 :
            EdiQual = 2     # Smaller range for EdiQual in NNEDI3
        
        # Source-match defaults
        MatchTR1 = TR1
        
        # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match), but adjusted to give roughly same sharpness for all settings
        if Sharpness == 0:
            SMode = 0
        SLMode = 0 if SourceMatch > 0 and SLMode is None else SLModeX   # ***Sharpness limiting disabled by default for source-match***
        if SLRad == 0:
            SLMode = 0
        spatialSL = SLMode in (1, 3)
        temporalSL = SLMode in (2, 4)
        if Sharpness is None:
            Sharpness = 0 if SMode == 0 else .2 if SourceMatch > 0 else 1                               # Default sharpness is 1.0, or 0.2 if using source-match
        sharpMul = 2 if temporalSL else 1.5 if spatialSL else 1                                         # Adjust sharpness based on other settings
        sharpAdj = Sharpness * (sharpMul * (.2 + TR1 * .15 + TR2 * .25) + (.1 if SMode == 1 else 0))    # [This needs a bit more refinement]
        if SMode == 0:
            Sbb = 0
        
        # Noise processing settings
        if NoiseProcess is None:
            if EZDenoise and EZDenoise > 0:
                NoiseProcess = 1
            elif (EZKeepGrain and EZKeepGrain > 0) or Preset in ('placebo', 'very slow'):
                NoiseProcess = 2
            else:
                NoiseProcess = 0
        if not isinstance(NoiseProcess, int) or NoiseProcess < 0 or NoiseProcess > 2:
            raise ValueError("QTGMC: 'NoiseProcess' have not a correct value! [0,1,2]")
        if GrainRestore is None:
            if EZDenoise and EZDenoise > 0:
                GrainRestore = 0
            elif EZKeepGrain and EZKeepGrain > 0:
                GrainRestore = .3 * math.sqrt(EZKeepGrain)
            else:
                GrainRestore = (0, .7, .3)[NoiseProcess]
        if not (isinstance(GrainRestore, float) or isinstance(GrainRestore, int)) or GrainRestore < 0:
            raise ValueError("QTGMC: 'GrainRestore' have not a correct value! [>=0.0]")
        if NoiseRestore is None:
            if EZDenoise and EZDenoise > 0:
                NoiseRestore = 0
            elif EZKeepGrain and EZKeepGrain > 0:
                NoiseRestore = .1 * math.sqrt(EZKeepGrain)
            else:
                NoiseRestore = (0, .3, .1)[NoiseProcess]
        if not (isinstance(NoiseRestore, float) or isinstance(NoiseRestore, int)) or NoiseRestore < 0:
            raise ValueError("QTGMC: 'NoiseRestore' have not a correct value! [>=0.0]")
        if Sigma is None:
            if EZDenoise and EZDenoise > 0:
                Sigma = EZDenoise
            elif EZKeepGrain and EZKeepGrain > 0:
                Sigma = 4 * EZKeepGrain
            else:
                Sigma = 2
        if DenoiseThreads is None:
            DenoiseThreads = EdiThreads
        if Denoiser in ('fft3df', 'fft3dfilter') and DenoiseThreads == 0:
            DenoiseThreads = 1  # FFT3DFilter doesn't support autodetection of number of logical processors on system
        if isinstance(ShowNoise, bool):
            ShowNoise = 10 if ShowNoise else 0
        if ShowNoise > 0:
            NoiseProcess = 2
            NoiseRestore = 1
        if NoiseProcess == 0:
            NoiseTR = 0
            GrainRestore = 0
            NoiseRestore = 0
        totalRestore = GrainRestore + NoiseRestore
        if totalRestore == 0:
            StabilizeNoise = False
        noiseTD = (1, 3, 5)[NoiseTR]
        noiseCentre = '128' if Denoiser == 'dfttest' else '128.5'
        
        # MVTools settings
        if Lambda is None:
            Lambda = int((1000 if TrueMotion else 100) * BlockSize ** 2 / 64)
        if LSAD is None:
            LSAD = 1200 if TrueMotion else 400
        if PNew is None:
            PNew = 50 if TrueMotion else 25
        if PLevel is None:
            PLevel = 1 if TrueMotion else 0
        
        # Motion blur settings
        if ShutterAngleOut * FPSDivisor == ShutterAngleSrc:     # If motion blur output is same as input
            ShutterBlur = 0
        
        # Miscellaneous
        if InputType not in (2, 3):
            ProgSADMask = 0
        rgBlur = 11 if Precise else 12
        
        # Get maximum temporal radius needed
        maxTR = SLRad if temporalSL else 0
        if MatchTR2 > maxTR:
            maxTR = MatchTR2
        if TR1 > maxTR:
            maxTR = TR1
        if TR2 > maxTR:
            maxTR = TR2
        if NoiseTR > maxTR:
            maxTR = NoiseTR
        if (ProgSADMask > 0 or StabilizeNoise or ShutterBlur > 0) and maxTR < 1:
            maxTR = 1
        if ForceTR > maxTR:
            maxTR = ForceTR
        
        if not isinstance(TR2, int) or TR2 < 0 or TR2 > 3:
            raise ValueError("QTGMC: 'TR2' have not a correct value! [0,1,2,3]")
        if not (isinstance(Sharpness, float) or isinstance(Sharpness, int)) or Sharpness < 0:
            raise ValueError("QTGMC: 'Sharpness' have not a correct value! [>=0.0]")
        if not isinstance(SLMode, int) or SLMode < 0 or SLMode > 4:
            raise ValueError("QTGMC: 'SLMode' have not a correct value! [0,1,2,3,4]")
        if not isinstance(Lambda, int) or Lambda < 0:
            raise ValueError("QTGMC: 'Lambda' have not a correct value! [>=0]")
        if not isinstance(LSAD, int) or LSAD < 0:
            raise ValueError("QTGMC: 'LSAD' have not a correct value! [>=0]")
        if not isinstance(PNew, int) or PNew < 0:
            raise ValueError("QTGMC: 'PNew' have not a correct value! [>=0]")
        if not isinstance(PLevel, int) or PLevel < 0 or PLevel > 2:
            raise ValueError("QTGMC: 'PLevel' have not a correct value! [0,1,2]")
        if not isinstance(DenoiseThreads, int) or DenoiseThreads < 0:
            raise ValueError("QTGMC: 'DenoiseThreads' have not a correct value! [>=0]")
        if not (isinstance(Sigma, float) or isinstance(Sigma, int)) or Sigma < 0:
            raise ValueError("QTGMC: 'Sigma' have not a correct value! [>=0.0]")
        
        #---------------------------------------
        # Pre-Processing
        
        w = Input.width
        h = Input.height
        epsilon = .0001
        
        # Reverse "field" dominance for progressive repair mode 3 (only difference from mode 2)
        if InputType == 3:
            TFF = not TFF
        
        # Pad vertically during processing (to prevent artefacts at top & bottom edges)
        if Border:
            clip = self.Resize(Input, w, h+8, 0, -4, 0, h+8+epsilon, kernel='point')
            h += 8
        else:
            clip = Input
        
        # Calculate padding needed for MVTools super clips to avoid crashes [fixed in latest MVTools, but keeping this code for a while]
        hpad = w - (int((w - Overlap) / (BlockSize - Overlap)) * (BlockSize - Overlap) + Overlap)
        vpad = h - (int((h - Overlap) / (BlockSize - Overlap)) * (BlockSize - Overlap) + Overlap)
        if hpad < 8:
            hpad = 8    # But match default padding if possible
        if vpad < 8:
            vpad = 8
        
        #---------------------------------------
        # Motion Analysis
        
        # Bob the input as a starting point for motion search clip
        if InputType == 0:
            bobbed = self.Bob(clip, 0, .5, TFF)
        elif InputType == 1:
            bobbed = clip
        else:
            bobbed = self.core.generic.Blur(clip, 0, .5)
        
        CMts = 255 if ChromaMotion else 0
        CMrg = 12 if ChromaMotion else 0
        
        # The bobbed clip will shimmer due to being derived from alternating fields. Temporally smooth over the neighboring frames using a binomial kernel. Binomial
        # kernels give equal weight to even and odd frames and hence average away the shimmer. The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
        # These kernels are approximately Gaussian kernels, which work well as a prefilter before motion analysis (hence the original name for this script)
        # Create linear weightings of neighbors first                         -2    -1    0     1     2
        if TR0 > 0: ts1 = self.TemporalSoften(bobbed, 1, 255, CMts, 28, 2)  # 0.00  0.33  0.33  0.33  0.00
        if TR0 > 1: ts2 = self.TemporalSoften(bobbed, 2, 255, CMts, 28, 2)  # 0.20  0.20  0.20  0.20  0.20
        
        # Combine linear weightings to give binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
        if TR0 == 0:
            binomial0 = bobbed
        elif TR0 == 1:
            binomial0 = self.core.std.Merge(ts1, bobbed, weight=.25 if ChromaMotion else [.25, 0])
        else:
            binomial0 = self.core.std.Merge(self.core.std.Merge(ts1, ts2, weight=.357 if ChromaMotion else [.357, 0]), bobbed,
                                            weight=.125 if ChromaMotion else [.125, 0])
        
        # Remove areas of difference between temporal blurred motion search clip and bob that are not due to bob-shimmer - removes general motion blur
        if Rep0 == 0:
            repair0 = binomial0
        else:
            repair0 = self.QTGMC_KeepOnlyBobShimmerFixes(binomial0, bobbed, Rep0, RepChroma and ChromaMotion)
        
        # x 7 + y < x 2 + x 7 - y > x 2 - x 51 * y 49 * + 100 / ? ?
        def get_lut(x, y):
            if x + 7 < y:
                return min(x + 2, 255)
            elif x - 7 > y:
                return max(x - 2, 0)
            else:
                return round((x * 51 + y * 49) / 100)
        
        # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
        # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
        if SrchClipPP == 1:
            spatialBlur = self.Resize(self.core.rgvs.RemoveGrain(self.Resize(repair0, int(w/2), int(h/2), kernel='bilinear'), [12, CMrg]), w, h, kernel='bilinear')
        elif SrchClipPP >= 2:
            spatialBlur = self.Resize(self.core.rgvs.RemoveGrain(repair0, [12, CMrg]), w, h, 0, 0, w+epsilon, h+epsilon, kernel='gauss', a1=2)
        if SrchClipPP > 1:
            spatialBlur = self.core.std.Merge(spatialBlur, repair0, weight=.1 if ChromaMotion else [.1, 0])
            expr = 'x 3 + y < x 3 + x 3 - y > x 3 - y ? ?'
            tweaked = self.core.std.Expr([repair0, bobbed], expr if ChromaMotion else [expr, ''])
        if SrchClipPP == 0:
            srchClip = repair0
        elif SrchClipPP < 3:
            srchClip = spatialBlur
        else:
            srchClip = self.core.std.Lut2(spatialBlur, tweaked, planes=[0, 1, 2] if ChromaMotion else [0], function=get_lut)
        
        # Calculate forward and backward motion vectors from motion search clip
        if maxTR > 0:
            srchSuper = self.core.mv.Super(srchClip, pel=SubPel, sharp=SubPelInterp, hpad=hpad, vpad=vpad, chroma=ChromaMotion)
            bVec1 = self.core.mv.Analyse(
              srchSuper, isb=True, delta=1, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
            fVec1 = self.core.mv.Analyse(
              srchSuper, isb=False, delta=1, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
        if maxTR > 1:
            bVec2 = self.core.mv.Analyse(
              srchSuper, isb=True, delta=2, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
            fVec2 = self.core.mv.Analyse(
              srchSuper, isb=False, delta=2, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
        if maxTR > 2:
            bVec3 = self.core.mv.Analyse(
              srchSuper, isb=True, delta=3, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
            fVec3 = self.core.mv.Analyse(
              srchSuper, isb=False, delta=3, blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch,
              truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
        
        #---------------------------------------
        # Noise Processing
        
        # Expand fields to full frame size before extracting noise (allows use of motion vectors which are frame-sized)
        if NoiseProcess > 0:
            if InputType > 0:
                fullClip = clip
            else:
                fullClip = self.Bob(clip, 0, 1, TFF)
        if NoiseTR > 0:
            fullSuper = self.core.mv.Super(fullClip, pel=SubPel, levels=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)    #TEST chroma OK?
        
        # Create a motion compensated temporal window around current frame and use to guide denoisers
        if NoiseProcess > 0:
            if not DenoiseMC or NoiseTR == 0:
                noiseWindow = fullClip
            elif NoiseTR == 1:
                noiseWindow = self.core.std.Interleave([self.core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                                                        fullClip,
                                                        self.core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)])
            else:
                noiseWindow = self.core.std.Interleave([self.core.mv.Compensate(fullClip, fullSuper, fVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                                                        self.core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                                                        fullClip,
                                                        self.core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                                                        self.core.mv.Compensate(fullClip, fullSuper, bVec2, thscd1=ThSCD1, thscd2=ThSCD2)])
            if Denoiser == 'dfttest':
                dnWindow = self.core.avs.dfttest(noiseWindow, U=ChromaNoise, V=ChromaNoise, sigma=Sigma*4, tbsize=noiseTD, threads=DenoiseThreads)
            elif Denoiser == 'fft3dgpu':
                dnWindow = self.core.avs.fft3dGPU(noiseWindow, plane=4 if ChromaNoise else 0, sigma=Sigma, bt=noiseTD, precision=2)
            else:
                dnWindow = self.core.avs.FFT3DFilter(noiseWindow, plane=4 if ChromaNoise else 0, sigma=Sigma, bt=noiseTD, ncpu=DenoiseThreads)
        
            # Rework denoised clip to match source format - various code paths here: discard the motion compensation window, discard doubled lines (from point resize)
            # Also reweave to get interlaced noise if source was interlaced (could keep the full frame of noise, but it will be poor quality from the point resize)
            if not DenoiseMC:
                if InputType > 0:
                    denoised = dnWindow
                else:
                    denoised = self.Weave(self.core.std.SelectEvery(self.core.std.SeparateFields(dnWindow, TFF), 4, [0, 3]), TFF)
            elif InputType > 0:
                if NoiseTR == 0:
                    denoised = dnWindow
                else:
                    denoised = self.core.std.SelectEvery(dnWindow, noiseTD, NoiseTR)
            else:
                denoised = self.Weave(self.core.std.SelectEvery(self.core.std.SeparateFields(dnWindow, TFF), noiseTD*4, [NoiseTR*2, NoiseTR*6+3]), TFF)
        
        CNplanes = [0, 1, 2] if ChromaNoise else [0]
        
        # Get actual noise from difference. Then 'deinterlace' where we have weaved noise - create the missing lines of noise in various ways
        if NoiseProcess > 0 and totalRestore > 0:
            noise = self.core.std.MakeDiff(clip, denoised, planes=CNplanes)
            if InputType > 0:
                deintNoise = noise
            elif NoiseDeint == 'bob':
                deintNoise = self.Bob(noise, 0, .5, TFF)
            elif NoiseDeint == 'generate':
                deintNoise = self.QTGMC_Generate2ndFieldNoise(noise, denoised, ChromaNoise, TFF)
            else:
                deintNoise = self.core.std.DoubleWeave(self.core.std.SeparateFields(noise, TFF), TFF)
            
            # Motion-compensated stabilization of generated noise
            if StabilizeNoise:
                noiseSuper = self.core.mv.Super(deintNoise, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)
                mcNoise = self.core.mv.Compensate(deintNoise, noiseSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
                expr = 'x 128 - abs y 128 - abs > x y ? 0.6 * x y + 0.2 * +'
                finalNoise = self.core.std.Expr([deintNoise, mcNoise], expr if ChromaNoise else [expr, ''])
            else:
                finalNoise = deintNoise
        
        # If NoiseProcess=1 denoise input clip. If NoiseProcess=2 leave noise in the clip and let the temporal blurs "denoise" it for a stronger effect
        innerClip = denoised if NoiseProcess == 1 else clip
        
        #---------------------------------------
        # Interpolation
        
        # Support badly deinterlaced progressive content - drop half the fields and reweave to get 1/2fps interlaced stream appropriate for QTGMC processing
        if InputType in (2, 3):
            ediInput = self.Weave(self.core.std.SelectEvery(self.core.std.SeparateFields(innerClip, TFF), 4, [0, 3]), TFF)
        else:
            ediInput = innerClip
        
        # Create interpolated image as starting point for output
        if EdiExt:
            edi1 = self.Resize(EdiExt, w, h, 0, (EdiExt.height-h)/2, 0, h+epsilon, kernel='point')
        else:
            edi1 = self.QTGMC_Interpolate(ediInput, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, EdiThreads, bobbed, ChromaEdi, TFF)
        
        # InputType=2,3: use motion mask to blend luma between original clip & reweaved clip based on ProgSADMask setting. Use chroma from original clip in any case
        if ProgSADMask > 0:
            inputTypeBlend = self.core.avs.MMask(srchClip, bVec1, kind=1, ml=ProgSADMask)
        if InputType not in (2, 3):
            edi = edi1
        elif ProgSADMask == 0:
            edi = self.core.std.Merge(edi1, innerClip, weight=[0, 1])
        else:
            edi = self.core.std.MaskedMerge(innerClip, edi1, inputTypeBlend, planes=[0])
        
        # Get the max/min value for each pixel over neighboring motion-compensated frames - used for temporal sharpness limiting
        if TR1 > 0 or temporalSL:
            ediSuper = self.core.mv.Super(edi, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
        if temporalSL:
            bComp1 = self.core.mv.Compensate(edi, ediSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
            fComp1 = self.core.mv.Compensate(edi, ediSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2)
            tMax = self.Logic(self.Logic(edi, fComp1, 'max'), bComp1, 'max')
            tMin = self.Logic(self.Logic(edi, fComp1, 'min'), bComp1, 'min')
            if SLRad > 1:
                bComp3 = self.core.mv.Compensate(edi, ediSuper, bVec3, thscd1=ThSCD1, thscd2=ThSCD2)
                fComp3 = self.core.mv.Compensate(edi, ediSuper, fVec3, thscd1=ThSCD1, thscd2=ThSCD2)
                tMax = self.Logic(self.Logic(tMax, fComp3, 'max'), bComp3, 'max')
                tMin = self.Logic(self.Logic(tMin, fComp3, 'min'), bComp3, 'min')
        
        #---------------------------------------
        # Create basic output
        
        # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames. As above, this is done to remove shimmer from
        # alternate frames so the same binomial kernels are used. However, by using motion-compensated smoothing this time we avoid motion blur. The use of
        # MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different, but they evaluate to the same values
        # Create linear weightings of neighbors first                                                                               -2    -1    0     1     2
        if TR1 > 0: degrain1 = self.core.mv.Degrain1(edi, ediSuper, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)   # 0.00  0.33  0.33  0.33  0.00
        if TR1 > 1: degrain2 = self.core.mv.Degrain1(edi, ediSuper, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)   # 0.33  0.00  0.33  0.00  0.33
        
        # Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
        if TR1 == 0:
            binomial1 = edi
        elif TR1 == 1:
            binomial1 = self.core.std.Merge(degrain1, edi, weight=.25)
        else:
            binomial1 = self.core.std.Merge(self.core.std.Merge(degrain1, degrain2, weight=.2), edi, weight=.0625)
        
        # Remove areas of difference between smoothed image and interpolated image that are not bob-shimmer fixes: repairs residual motion blur from temporal smooth
        if Rep1 == 0:
            repair1 = binomial1
        else:
            repair1 = self.QTGMC_KeepOnlyBobShimmerFixes(binomial1, edi, Rep1, RepChroma)
        
        # Apply source match - use difference between output and source to succesively refine output [extracted to function to clarify main code path]
        if SourceMatch == 0:
            match = repair1
        else:
            match = self.QTGMC_ApplySourceMatch(repair1, InputType, ediInput, bVec1 if maxTR>0 else None, fVec1 if maxTR>0 else None,
                                                bVec2 if maxTR>1 else None, fVec2 if maxTR>1 else None, SubPel, SubPelInterp, hpad, vpad, ThSAD1, ThSCD1,
                                                ThSCD2, SourceMatch, MatchTR1, MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD, MatchTR2,
                                                MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2, MatchEdiMaxD2, MatchEnhance, EdiThreads, TFF)
        
        # Lossless=2 - after preparing an interpolated, de-shimmered clip, restore the original source fields into it and clean up any artefacts.
        # This mode will not give a true lossless result because the resharpening and final temporal smooth are still to come, but it will add further detail.
        # However, it can introduce minor combing. This setting is best used together with source-match (it's effectively the final source-match stage).
        if Lossless == 2:
            lossed1 = self.QTGMC_MakeLossless(match, innerClip, InputType, TFF)
        else:
            lossed1 = match
        
        #---------------------------------------
        # Resharpen / retouch output
        
        # Resharpen to counteract temporal blurs. Little sharpening needed for source-match mode since it has already recovered sharpness from source
        if SMode == 2:
            vresharp1 = self.core.std.Merge(self.core.generic.Maximum(lossed1, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]),
                                            self.core.generic.Minimum(lossed1, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]))
            if Precise:
                vresharp = self.core.std.Expr([vresharp1, lossed1], 'x y < x 1 + x y > x 1 - x ? ?')    # Precise mode: reduce tiny overshoot
            else:
                vresharp = vresharp1
        if SMode == 0:
            resharp = lossed1
        elif SMode == 1:
            resharp = self.core.std.Expr([lossed1, self.core.rgvs.RemoveGrain(lossed1, rgBlur)], 'x x y - '+repr(sharpAdj)+' * +')
        else:
            resharp = self.core.std.Expr([lossed1, self.core.rgvs.RemoveGrain(vresharp, rgBlur)], 'x x y - '+repr(sharpAdj)+' * +')
        
        # Slightly thin down 1-pixel high horizontal edges that have been widened into neigboring field lines by the interpolator
        SVThinSc = SVThin * 6
        if SVThin > 0:
            vertMedD = self.core.std.Expr([lossed1, self.core.avs.VerticalCleaner(lossed1, mode=1, modeU=-1, modeV=-1)], ['y x - '+repr(SVThinSc)+' * 128 +', ''])
            vertMedD = self.core.generic.Blur(vertMedD, .5, 0)
            neighborD = self.core.std.Expr([vertMedD, self.core.rgvs.RemoveGrain(vertMedD, [rgBlur, 0])], ['y 128 - abs x 128 - abs > y 128 ?', ''])
            thin = self.core.std.MergeDiff(resharp, neighborD, planes=[0])
        else:
            thin = resharp
        
        # Back blend the blurred difference between sharpened & unsharpened clip, before (1st) sharpness limiting (Sbb == 1,3). A small fidelity improvement
        if Sbb not in (1, 3):
            backBlend1 = thin
        else:
            backBlend1 = self.core.std.MakeDiff(thin,
                                                self.Resize(self.core.rgvs.RemoveGrain(self.core.std.MakeDiff(thin, lossed1, planes=[0]), [12, 0]),
                                                            w, h, 0, 0, w+epsilon, h+epsilon, kernel='gauss', a1=5),
                                                planes=[0])
        
        # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
        # Occurs here (before final temporal smooth) if SLMode == 1,2. This location will restrict sharpness more, but any artefacts introduced will be smoothed
        if SLMode == 1:
            if SLRad <= 1:
                sharpLimit1 = self.core.rgvs.Repair(backBlend1, edi, 1)
            else:
                sharpLimit1 = self.core.rgvs.Repair(backBlend1, self.core.rgvs.Repair(backBlend1, edi, 12), 1)
        elif SLMode == 2:
            sharpLimit1 = self.Clamp(backBlend1, tMax, tMin, SOvs, SOvs)
        else:
            sharpLimit1 = backBlend1
        
        # Back blend the blurred difference between sharpened & unsharpened clip, after (1st) sharpness limiting (Sbb == 2,3). A small fidelity improvement
        if Sbb < 2:
            backBlend2 = sharpLimit1
        else:
            backBlend2 = self.core.std.MakeDiff(sharpLimit1,
                                                self.Resize(self.core.rgvs.RemoveGrain(self.core.std.MakeDiff(sharpLimit1, lossed1, planes=[0]), [12, 0]),
                                                            w, h, 0, 0, w+epsilon, h+epsilon, kernel='gauss', a1=5),
                                                planes=[0])
        
        # Add back any extracted noise, prior to final temporal smooth - this will restore detail that was removed as "noise" without restoring the noise itself
        # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
        if GrainRestore == 0:
            addNoise1 = backBlend2
        else:
            expr = 'x '+noiseCentre+' - '+repr(GrainRestore)+' * 128 +'
            addNoise1 = self.core.std.MergeDiff(backBlend2, self.core.std.Expr(finalNoise, expr if ChromaNoise else [expr, '']), planes=CNplanes)
        
        # Final light linear temporal smooth for denoising
        if TR2 > 0:
            stableSuper = self.core.mv.Super(addNoise1, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
        if TR2 == 0:
            stable = addNoise1
        elif TR2 == 1:
            stable = self.core.mv.Degrain1(addNoise1, stableSuper, bVec1, fVec1, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
        elif TR2 == 2:
            stable = self.core.mv.Degrain2(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
        else:
            stable = self.core.mv.Degrain3(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
        
        # Remove areas of difference between final output & basic interpolated image that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
        if Rep2 == 0:
            repair2 = stable
        else:
            repair2 = self.QTGMC_KeepOnlyBobShimmerFixes(stable, edi, Rep2, RepChroma)
        
        # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
        # Occurs here (after final temporal smooth) if SLMode == 3,4. Allows more sharpening here, but more prone to introducing minor artefacts
        if SLMode == 3:
            if SLRad <= 1:
                sharpLimit2 = self.core.rgvs.Repair(repair2, edi, 1)
            else:
                sharpLimit2 = self.core.rgvs.Repair(repair2, self.core.rgvs.Repair(repair2, edi, 12), 1)
        elif SLMode == 4:
            sharpLimit2 = self.Clamp(repair2, tMax, tMin, SOvs, SOvs)
        else:
            sharpLimit2 = repair2
        
        # Lossless=1 - inject source fields into result and clean up inevitable artefacts. Provided NoiseRestore=0.0 or 1.0, this mode will make the script result
        # properly lossless, but this will retain source artefacts and cause some combing (where the smoothed deinterlace doesn't quite match the source)
        if Lossless == 1:
            lossed2 = self.QTGMC_MakeLossless(sharpLimit2, innerClip, InputType, TFF)
        else:
            lossed2 = sharpLimit2
        
        # Add back any extracted noise, after final temporal smooth. This will appear as noise/grain in the output
        # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
        if NoiseRestore == 0:
            addNoise2 = lossed2
        else:
            expr = 'x '+noiseCentre+' - '+repr(NoiseRestore)+' * 128 +'
            addNoise2 = self.core.std.MergeDiff(lossed2, self.core.std.Expr(finalNoise, expr if ChromaNoise else [expr, '']), planes=CNplanes)
        
        #---------------------------------------
        # Post-Processing
        
        # Shutter motion blur - get level of blur depending on output framerate and blur already in source
        blurLevel = (ShutterAngleOut * FPSDivisor - ShutterAngleSrc) * 100 / 360
        if blurLevel < 0:
            raise ValueError('QTGMC: Cannot reduce motion blur already in source: increase ShutterAngleOut or FPSDivisor')
        elif blurLevel > 200:
            raise ValueError('QTGMC: Exceeded maximum motion blur level: decrease ShutterAngleOut or FPSDivisor')
        
        # ShutterBlur mode 2,3 - get finer resolution motion vectors to reduce blur "bleeding" into static areas
        rBlockDivide = (1, 1, 2, 4)[ShutterBlur]
        rBlockSize = int(BlockSize / rBlockDivide)
        rOverlap = int(Overlap / rBlockDivide)
        if rBlockSize < 4:
            rBlockSize = 4
        if rOverlap < 2:
            rOverlap = 2
        rBlockDivide = int(BlockSize / rBlockSize)
        rLambda = int(Lambda / rBlockDivide ** 2)
        if ShutterBlur > 1:
            sbBVec1 = self.core.mv.Recalculate(srchSuper, bVec1, thsad=ThSAD1, blksize=rBlockSize, overlap=rOverlap, search=Search, searchparam=SearchParam,
                                               truemotion=TrueMotion, _lambda=rLambda, pnew=PNew, dct=DCT, chroma=ChromaMotion)
            sbFVec1 = self.core.mv.Recalculate(srchSuper, fVec1, thsad=ThSAD1, blksize=rBlockSize, overlap=rOverlap, search=Search, searchparam=SearchParam,
                                               truemotion=TrueMotion, _lambda=rLambda, pnew=PNew, dct=DCT, chroma=ChromaMotion)
        elif ShutterBlur > 0:
            sbBVec1 = bVec1
            sbFVec1 = fVec1
        
        # Shutter motion blur - use MFlowBlur to blur along motion vectors
        if ShutterBlur > 0:
            sblurSuper = self.core.mv.Super(addNoise2, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            sblur = self.core.avs.MFlowBlur(addNoise2, sblurSuper, sbBVec1, sbFVec1, blur=blurLevel, thSCD1=ThSCD1, thSCD2=ThSCD2)
        
        # Shutter motion blur - use motion mask to reduce blurring in areas of low motion - also helps reduce blur "bleeding" into static areas, then select blur type
        if ShutterBlur > 0 and SBlurLimit > 0:
            sbMotionMask = self.core.avs.MMask(srchClip, bVec1, kind=0, ml=SBlurLimit)
        if ShutterBlur == 0:
            sblurred = addNoise2
        elif SBlurLimit == 0:
            sblurred = sblur
        else:
            sblurred = self.core.std.MaskedMerge(addNoise2, sblur, sbMotionMask)
        
        # Reduce frame rate
        if FPSDivisor > 1:
            decimated = self.core.std.SelectEvery(sblurred, FPSDivisor, 0)
        else:
            decimated = sblurred
        
        # Crop off temporary vertical padding
        if Border:
            cropped = self.core.std.CropRel(decimated, left=0, top=4, right=0, bottom=4)
            h -= 8
        else:
            cropped = decimated
        
        # Show output of choice + settings
        if ShowNoise == 0:
            output = cropped
        else:
            expr = 'x 128 - '+repr(ShowNoise)+' * 128 +'
            output = self.core.std.Expr(finalNoise, expr if ChromaNoise else [expr, '128'])
        if not ShowSettings:
            return output
        else:
            return self.core.avs.Subtitle(
              output, "TR0=" + repr(TR0) + " | TR1=" + repr(TR1) + " | TR2=" + repr(TR2) + " | Rep0=" + repr(Rep0) + " | Rep1=" + repr(Rep1) + " | Rep2=" +
              repr(Rep2) + " | RepChroma=" + repr(RepChroma) + "\\nEdiMode='" + EdiMode + "' | NNSize=" + repr(NNSize) + " | NNeurons=" + repr(NNeurons) +
              " | EdiQual=" + repr(EdiQual) + " | EdiMaxD=" + repr(EdiMaxD) + " | ChromaEdi='" + ChromaEdi + "' | EdiThreads=" + repr(EdiThreads) +
              "\\nSharpness=" + repr(Sharpness) + " | SMode=" + repr(SMode) + " | SLMode=" + repr(SLMode) + " | SLRad=" + repr(SLRad) + " | SOvs=" +
              repr(SOvs) + " | SVThin=" + repr(SVThin) + " | Sbb=" + repr(Sbb) + "\\nSrchClipPP=" + repr(SrchClipPP) + " | SubPel=" + repr(SubPel) +
              " | SubPelInterp=" + repr(SubPelInterp) + " | BlockSize=" + repr(BlockSize) + " | Overlap=" + repr(Overlap) + "\\nSearch=" + repr(Search) +
              " | SearchParam=" + repr(SearchParam) + " | PelSearch=" + repr(PelSearch) + " | ChromaMotion=" + repr(ChromaMotion) + " | TrueMotion=" +
              repr(TrueMotion) + "\\nLambda=" + repr(Lambda) + " | LSAD=" + repr(LSAD) + " | PNew=" + repr(PNew) + " | PLevel=" + repr(PLevel) +
              " | GlobalMotion=" + repr(GlobalMotion) + " | DCT=" + repr(DCT) + "\\nThSAD1=" + repr(ThSAD1) + " | ThSAD2=" + repr(ThSAD2) + " | ThSCD1=" +
              repr(ThSCD1) + " | ThSCD2=" + repr(ThSCD2) + "\\nSourceMatch=" + repr(SourceMatch) + " | MatchPreset='" + MatchPreset + "' | MatchEdi='" +
              MatchEdi + "'\\nMatchPreset2='" + MatchPreset2 + "' | MatchEdi2='" + MatchEdi2 + "' | MatchTR2=" + repr(MatchTR2) + " | MatchEnhance=" +
              repr(MatchEnhance) + " | Lossless=" + repr(Lossless) + "\\nNoiseProcess=" + repr(NoiseProcess) + " | Denoiser='" + Denoiser +
              "' | DenoiseThreads=" + repr(DenoiseThreads) + " | DenoiseMC=" + repr(DenoiseMC) + " | NoiseTR=" + repr(NoiseTR) + " | Sigma=" + repr(Sigma) +
              "\\nChromaNoise=" + repr(ChromaNoise) + " | ShowNoise=" + repr(ShowNoise) + " | GrainRestore=" + repr(GrainRestore) + " | NoiseRestore=" +
              repr(NoiseRestore) + "\\nNoiseDeint='" + NoiseDeint + "' | StabilizeNoise=" + repr(StabilizeNoise) + " | InputType=" + repr(InputType) +
              " | ProgSADMask=" + repr(ProgSADMask) + "\\nFPSDivisor=" + repr(FPSDivisor) + " | ShutterBlur=" + repr(ShutterBlur) + " | ShutterAngleSrc=" +
              repr(ShutterAngleSrc) + " | ShutterAngleOut=" + repr(ShutterAngleOut) + " | SBlurLimit=" + repr(SBlurLimit) + "\\nBorder=" + repr(Border) +
              " | Precise=" + repr(Precise) + "\\nPreset='" + Preset + "' | Tuning='" + Tuning + "' | ForceTR=" + repr(ForceTR),
              font="Lucida Console", size=11, lsp=12)
    
    #---------------------------------------
    # Helpers
    
    # Interpolate input clip using method given in EdiMode. Use Fallback or Bob as result if mode not in list. If ChromaEdi string if set then interpolate chroma
    # separately with that method (only really useful for EEDIx). The function is used as main algorithm starting point and for first two source-match stages
    def QTGMC_Interpolate(self, Input, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, EdiThreads, Fallback=None, ChromaEdi='', TFF=None):
        CEed = ChromaEdi == ''
        planes = [0, 1, 2] if CEed else [0]
        field = 3 if TFF else 2
        
        if InputType == 1:
            return Input
        elif EdiMode == 'nnedi3':
            interp = self.core.nnedi3.nnedi3(Input, field=field, U=CEed, V=CEed, nsize=NNSize, nns=NNeurons, qual=EdiQual)
        elif EdiMode == 'nnedi2':
            interp = self.core.avs.nnedi2(Input, field=field, U=CEed, V=CEed, nsize=NNeurons, qual=EdiQual, threads=EdiThreads)
        elif EdiMode == 'nnedi':
            interp = self.core.avs.nnedi(Input, field=field, U=CEed, V=CEed, threads=EdiThreads)
        elif EdiMode == 'eedi3+nnedi3':
            interp = self.core.eedi3.eedi3(Input, field=field, planes=planes, mdis=EdiMaxD,
                                           sclip=self.core.nnedi3.nnedi3(Input, field=field, U=CEed, V=CEed, nsize=NNSize, nns=NNeurons, qual=EdiQual))
        elif EdiMode == 'eedi3':
            interp = self.core.eedi3.eedi3(Input, field=field, planes=planes, mdis=EdiMaxD)
        elif EdiMode == 'eedi2':
            interp = self.core.eedi2.EEDI2(self.core.std.SeparateFields(Input, TFF), field=field, maxd=EdiMaxD)
        elif EdiMode == 'tdeint':
            interp = self.core.avs.TDeint(Input, mode=1, order=1 if TFF else 0)
        else:
            if isinstance(Fallback, vs.VideoNode):
                interp = Fallback
            else:
                interp = self.Bob(Input, 0, .5, TFF)
        
        if ChromaEdi == 'nnedi3':
            interpuv = self.core.nnedi3.nnedi3(Input, field=field, Y=False, nsize=4, nns=0, qual=1)
        elif ChromaEdi == 'bob':
            interpuv = self.Bob(Input, 0, .5, TFF)
        else:
            return interp
        
        return self.core.std.Merge(interp, interpuv, weight=[0, 1])
    
    # Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference, i.e. bob shimmer fixes
    # Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed
    #                  by this process. Restore remaining areas of difference back to as they were in reference clip.
    def QTGMC_KeepOnlyBobShimmerFixes(self, Input, Ref, Rep=1, Chroma=True):
        # ed is the erosion distance - how much to deflate then reflate to remove thin areas of interest: 0 = minimum to 6 = maximum
        # od is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:  0 = none to 3 = one full pixel
        # If Rep < 10, then ed = Rep and od = 0, otherwise ed = 10s digit and od = 1s digit (nasty method, but kept for compatibility with original TGMC)
        ed = Rep if Rep < 10 else int(Rep / 10)
        od = 0 if Rep < 10 else Rep % 10
        planes = [0, 1, 2] if Chroma else [0]
        
        diff = self.core.std.MakeDiff(Ref, Input)
        
        # Areas of positive difference                                                                              # ed = 0 1 2 3 4 5 6 7
        choke1 = self.core.generic.Minimum(diff, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])               #      x x x x x x x x    1 pixel   \
        if ed > 2: choke1 = self.core.generic.Minimum(choke1, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . x x x x x    1 pixel    |  Deflate to remove thin areas
        if ed > 5: choke1 = self.core.generic.Minimum(choke1, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . . . . x x    1 pixel   /
        if ed % 3 != 0: choke1 = self.core.generic.Deflate(choke1, planes=planes)                                   #      . x x . x x . x    A bit more deflate & some horizonal effect
        if ed in (2, 5): choke1 = self.core.rgvs.RemoveGrain(choke1, 4)                                             #      . . x . . x . .    Local median
        choke1 = self.core.generic.Maximum(choke1, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])             #      x x x x x x x x    1 pixel  \
        if ed > 1: choke1 = self.core.generic.Maximum(choke1, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . x x x x x x    1 pixel   | Reflate again
        if ed > 4: choke1 = self.core.generic.Maximum(choke1, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . . . x x x    1 pixel  /
        
        # Over-dilation - extra reflation up to about 1 pixel
        if od == 1:
            choke1 = self.core.generic.Inflate(choke1, planes=planes)
        elif od == 2:
            choke1 = self.core.generic.Inflate(self.core.generic.Inflate(choke1, planes=planes), planes=planes)
        elif od >= 3:
            choke1 = self.core.generic.Maximum(choke1, planes=planes)
        
        # Areas of negative difference (similar to above)
        choke2 = self.core.generic.Maximum(diff, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if ed > 2:
            choke2 = self.core.generic.Maximum(choke2, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if ed > 5:
            choke2 = self.core.generic.Maximum(choke2, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if ed % 3 != 0:
            choke2 = self.core.generic.Inflate(choke2, planes=planes)
        if ed in (2, 5):
            choke2 = self.core.rgvs.RemoveGrain(choke2, 4)
        choke2 = self.core.generic.Minimum(choke2, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if ed > 1:
            choke2 = self.core.generic.Minimum(choke2, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if ed > 4:
            choke2 = self.core.generic.Minimum(choke2, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
        if od == 1:
            choke2 = self.core.generic.Deflate(choke2, planes=planes)
        elif od == 2:
            choke2 = self.core.generic.Deflate(self.core.generic.Deflate(choke2, planes=planes), planes=planes)
        elif od >= 3:
            choke2 = self.core.generic.Minimum(choke2, planes=planes)
        
        # Combine above areas to find those areas of difference to restore
        expr1 = 'x 129 < x y 128 < 128 y ? ?'
        expr2 = 'x 127 > x y 128 > 128 y ? ?'
        restore = self.core.std.Expr([self.core.std.Expr([diff, choke1], expr1 if Chroma else [expr1, '']), choke2], expr2 if Chroma else [expr2, ''])
        
        return self.core.std.MergeDiff(Input, restore, planes=planes)
    
    # Given noise extracted from an interlaced source (i.e. the noise is interlaced), generate "progressive" noise with a new "field" of noise injected. The new
    # noise is centered on a weighted local average and uses the difference between local min & max as an estimate of local variance
    def QTGMC_Generate2ndFieldNoise(self, Input, InterleavedClip, ChromaNoise=False, TFF=None):
        planes = [0, 1, 2] if ChromaNoise else [0]
        origNoise = self.core.std.SeparateFields(Input, TFF)
        noiseMax = self.core.generic.Maximum(self.core.generic.Maximum(origNoise, planes=planes), planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
        noiseMin = self.core.generic.Minimum(self.core.generic.Minimum(origNoise, planes=planes), planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
        random = self.core.grain.Add(self.core.std.BlankClip(self.core.std.SeparateFields(InterleavedClip, TFF), color=[128, 128, 128]),
                                     var=1800, uvar=1800 if ChromaNoise else 0)
        expr = 'x 128 - y * 256 / 128 +'
        varRandom = self.core.std.Expr([self.core.std.MakeDiff(noiseMax, noiseMin, planes=planes), random], expr if ChromaNoise else [expr, ''])
        newNoise = self.core.std.MergeDiff(noiseMin, varRandom, planes=planes)
        return self.Weave(self.core.std.Interleave([origNoise, newNoise]), TFF)
    
    # Insert the source lines into the result to create a true lossless output. However, the other lines in the result have had considerable processing and won't
    # exactly match source lines. There will be some slight residual combing. Use vertical medians to clean a little of this away
    def QTGMC_MakeLossless(self, Input, Source, InputType, TFF):
        if InputType == 1:
            raise ValueError('QTGMC: Lossless modes are incompatible with InputType=1')
        
        # Weave the source fields and the "new" fields that have generated in the input
        if InputType == 0:
            srcFields = self.core.std.SeparateFields(Source, TFF)
        else:
            srcFields = self.core.std.SelectEvery(self.core.std.SeparateFields(Source, TFF), 4, [0, 3])
        newFields = self.core.std.SelectEvery(self.core.std.SeparateFields(Input, TFF), 4, [1, 2])
        processed = self.Weave(self.core.std.SelectEvery(self.core.std.Interleave([srcFields, newFields]), 4, [0, 1, 3, 2]), TFF)
        
        # x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?
        def get_lut(x, y):
            if (x - 128) * (y - 128) < 0:
                return 128
            elif abs(x - 128) < abs(y - 128):
                return x
            else:
                return y
        
        # Clean some of the artefacts caused by the above - creating a second version of the "new" fields
        vertMedian = self.core.avs.VerticalCleaner(processed, mode=1)
        vertMedDiff = self.core.std.MakeDiff(processed, vertMedian)
        vmNewDiff1 = self.core.std.SelectEvery(self.core.std.SeparateFields(vertMedDiff, TFF), 4, [1, 2])
        vmNewDiff2 = self.core.std.Lut2(self.core.avs.VerticalCleaner(vmNewDiff1, mode=1), vmNewDiff1, function=get_lut)
        vmNewDiff3 = self.core.rgvs.Repair(vmNewDiff2, self.core.rgvs.RemoveGrain(vmNewDiff2, 2), 1)
        
        # Reweave final result
        return self.Weave(self.core.std.SelectEvery(self.core.std.Interleave([srcFields, self.core.std.MakeDiff(newFields, vmNewDiff3)]),
                                                    4, [0, 1, 3, 2]), TFF)
    
    # Source-match, a three stage process that takes the difference between deinterlaced input and the original interlaced source, to shift the input more towards
    # the source without introducing shimmer. All other arguments defined in main script
    def QTGMC_ApplySourceMatch(self, Deinterlace, InputType, Source, bVec1, fVec1, bVec2, fVec2, SubPel, SubPelInterp, hpad, vpad, ThSAD1, ThSCD1, ThSCD2,
                               SourceMatch, MatchTR1, MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD,
                               MatchTR2, MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2, MatchEdiMaxD2, MatchEnhance, EdiThreads, TFF):
        # Basic source-match. Find difference between source clip & equivalent fields in interpolated/smoothed clip (called the "error" in formula below). Ideally
        # there should be no difference, we want the fields in the output to be as close as possible to the source whilst remaining shimmer-free. So adjust the
        # *source* in such a way that smoothing it will give a result closer to the unadjusted source. Then rerun the interpolation (edi) and binomial smooth with
        # this new source. Result will still be shimmer-free and closer to the original source.
        # Formula used for correction is P0' = P0 + (P0-P1)/(k+S(1-k)), where P0 is original image, P1 is the 1st attempt at interpolation/smoothing , P0' is the
        # revised image to use as new source for interpolation/smoothing, k is the weighting given to the current frame in the smooth, and S is a factor indicating
        # "temporal similarity" of the error from frame to frame, i.e. S = average over all pixels of [neighbor frame error / current frame error] . Decreasing
        # S will make the result sharper, sensible range is about -0.25 to 1.0. Empirically, S=0.5 is effective [will do deeper analysis later]
        errorTemporalSimilarity = .5    # S in formula described above
        errorAdjust1 = (1, 2/(1+errorTemporalSimilarity), 8/(3+5*errorTemporalSimilarity))[MatchTR1]
        if SourceMatch < 1 or InputType == 1:
            match1Clip = Deinterlace
        else:
            match1Clip = self.Weave(self.core.std.SelectEvery(self.core.std.SeparateFields(Deinterlace, TFF), 4, [0, 3]), TFF)
        if SourceMatch < 1 or MatchTR1 == 0:
            match1Update = Source
        else:
            match1Update = self.core.std.Expr([Source, match1Clip], 'x '+repr(errorAdjust1+1)+' * y '+repr(errorAdjust1)+' * -')
        if SourceMatch > 0:
            match1Edi = self.QTGMC_Interpolate(match1Update, InputType, MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD, EdiThreads, TFF=TFF)
            if MatchTR1 > 0:
                match1Super = self.core.mv.Super(match1Edi, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
                match1Degrain1 = self.core.mv.Degrain1(match1Edi, match1Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
            if MatchTR1 > 1:
                match1Degrain2 = self.core.mv.Degrain1(match1Edi, match1Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if SourceMatch < 1:
            match1 = Deinterlace
        elif MatchTR1 == 0:
            match1 = match1Edi
        elif MatchTR1 == 1:
            match1 = self.core.std.Merge(match1Degrain1, match1Edi, weight=.25)
        else:
            match1 = self.core.std.Merge(self.core.std.Merge(match1Degrain1, match1Degrain2, weight=.2), match1Edi, weight=.0625)
        if SourceMatch < 2:
            return match1
        
        # Enhance effect of source-match stages 2 & 3 by sharpening clip prior to refinement (source-match tends to underestimate so this will leave result sharper)
        if SourceMatch > 1 and MatchEnhance > 0:
            match1Shp = self.core.std.Expr([match1, self.core.rgvs.RemoveGrain(match1, 12)], 'x x y - '+repr(MatchEnhance)+' * +')
        else:
            match1Shp = match1
        
        # Source-match refinement. Find difference between source clip & equivalent fields in (updated) interpolated/smoothed clip. Interpolate & binomially smooth
        # this difference then add it back to output. Helps restore differences that the basic match missed. However, as this pass works on a difference rather than
        # the source image it can be prone to occasional artefacts (difference images are not ideal for interpolation). In fact a lower quality interpolation such
        # as a simple bob often performs nearly as well as advanced, slower methods (e.g. NNEDI3)
        if SourceMatch < 2 or InputType == 1:
            match2Clip = match1Shp
        else:
            match2Clip = self.Weave(self.core.std.SelectEvery(self.core.std.SeparateFields(match1Shp, TFF), 4, [0, 3]), TFF)
        if SourceMatch > 1:
            match2Diff = self.core.std.MakeDiff(Source, match2Clip)
            match2Edi = self.QTGMC_Interpolate(match2Diff, InputType, MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2, MatchEdiMaxD2, EdiThreads, TFF=TFF)
            if MatchTR2 > 0:
                match2Super = self.core.mv.Super(match2Edi, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
                match2Degrain1 = self.core.mv.Degrain1(match2Edi, match2Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
            if MatchTR2 > 1:
                match2Degrain2 = self.core.mv.Degrain1(match2Edi, match2Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if SourceMatch < 2:
            match2 = match1
        elif MatchTR2 == 0:
            match2 = match2Edi
        elif MatchTR2 == 1:
            match2 = self.core.std.Merge(match2Degrain1, match2Edi, weight=.25)
        else:
            match2 = self.core.std.Merge(self.core.std.Merge(match2Degrain1, match2Degrain2, weight=.2), match2Edi, weight=.0625)
        
        # Source-match second refinement - correct error introduced in the refined difference by temporal smoothing. Similar to error correction from basic step
        errorAdjust2 = (1, 2/(1+errorTemporalSimilarity), 8/(3+5*errorTemporalSimilarity))[MatchTR2]
        if SourceMatch < 3 or MatchTR2 == 0:
            match3Update = match2Edi
        else:
            match3Update = self.core.std.Expr([match2Edi, match2], 'x '+repr(errorAdjust2+1)+' * y '+repr(errorAdjust2)+' * -')
        if SourceMatch > 2:
            if MatchTR2 > 0:
                match3Super = self.core.mv.Super(match3Update, pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
                match3Degrain1 = self.core.mv.Degrain1(match3Update, match3Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
            if MatchTR2 > 1:
                match3Degrain2 = self.core.mv.Degrain1(match3Update, match3Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if SourceMatch < 3:
            match3 = match2
        elif MatchTR2 == 0:
            match3 = match3Update
        elif MatchTR2 == 1:
            match3 = self.core.std.Merge(match3Degrain1, match3Update, weight=.25)
        else:
            match3 = self.core.std.Merge(self.core.std.Merge(match3Degrain1, match3Degrain2, weight=.2), match3Update, weight=.0625)
        
        # Apply difference calculated in source-match refinement
        return self.core.std.MergeDiff(match1Shp, match3)
    
    
    # ivtc_txt60mc v1.1 by cretindesalpes (http://forum.doom9.org/showthread.php?p=1466105#post1466105)
    def ivtc_txt60mc(self, src, frame_ref, srcbob=False, draft=False, tff=None):
        if not isinstance(src, vs.VideoNode) or src.format.id != vs.YUV420P8:
            raise ValueError('ivtc_txt60mc: This is not a YUV420P8 clip !')
        if not isinstance(frame_ref, int) or frame_ref < 0:
            raise ValueError("ivtc_txt60mc: 'frame_ref' have not a correct value! [>=0]")
        if not isinstance(srcbob, bool):
            raise ValueError("ivtc_txt60mc: 'srcbob' must be bool")
        if not isinstance(draft, bool):
            raise ValueError("ivtc_txt60mc: 'draft' must be bool")
        if not srcbob and not isinstance(tff, bool):
            raise ValueError("ivtc_txt60mc: 'tff' must be set if srcbob is not true. Setting tff to true means top field first and false means bottom field first")
        
        field_ref = frame_ref if srcbob else frame_ref * 2
        field_ref %= 5
        invpos = (5 - field_ref) % 5
        pel = 1 if draft else 2
        blksize = 16 if src.width >= 1280 or src.height >= 720 else 8
        overlap = int(blksize / 2)
        
        if srcbob:
            last = src
        elif draft:
            last = self.Bob(src, tff=tff)
        else:
            last = self.QTGMC(src, TR0=1, TR1=1, TR2=1, SourceMatch=3, Lossless=2, TFF=tff)
        
        if invpos > 1:
            clean = self.core.std.AssumeFPS(self.core.std.Trim(last, 0, 0)+self.core.std.SelectEvery(last, 5, 6-invpos), fpsnum=12000, fpsden=1001)
        else:
            clean = self.core.std.SelectEvery(last, 5, 1-invpos)
        if invpos > 3:
            jitter = self.core.std.AssumeFPS(self.core.std.Trim(last, 0, 0)+self.core.std.SelectEvery(last, 5, [4-invpos, 8-invpos]), fpsnum=24000, fpsden=1001)
        else:
            jitter = self.core.std.SelectEvery(last, 5, [3-invpos, 4-invpos])
        jsup = self.core.avs.MSuper(jitter, pel=pel)
        vect_f = self.core.avs.MAnalyse(jsup, blksize=blksize, isb=False, delta=1, overlap=overlap)
        vect_b = self.core.avs.MAnalyse(jsup, blksize=blksize, isb=True, delta=1, overlap=overlap)
        comp = self.core.avs.MFlowInter(jitter, jsup, vect_b, vect_f)
        fixed = self.core.std.SelectEvery(comp, 2, 0)
        last = self.core.std.Interleave([clean, fixed])
        return self.core.std.AssumeFPS(self.core.std.Trim(last, int(invpos/2)), fpsnum=24000, fpsden=1001)
    
    
    # Vinverse: a small, but effective Function against (residual) combing, by Didée
    # sstr  : strength of contra sharpening
    # amnt  : change no pixel by more than this (default=255: unrestricted)
    # chroma: chroma mode, True=process chroma, False=pass chroma through
    def Vinverse(self, clp, sstr=2.7, amnt=255, chroma=True):
        if not isinstance(clp, vs.VideoNode) or clp.format.color_family != vs.YUV:
            raise ValueError('Vinverse: This is not a YUV clip !')
        if not (isinstance(sstr, float) or isinstance(sstr, int)) or sstr < 0:
            raise ValueError("Vinverse: 'sstr' have not a correct value! [>=0.0]")
        if not isinstance(amnt, int) or amnt < 1 or amnt > 255:
            raise ValueError("Vinverse: 'amnt' have not a correct value! [1...255]")
        if not isinstance(chroma, bool):
            raise ValueError("Vinverse: 'chroma' must be bool")
        
        shift = clp.format.bits_per_sample - 8
        NEU = repr(128 << shift)
        
        # x 128 - y 128 - * 0 < x 128 - abs y 128 - abs < x y ? 128 - 0.25 * 128 + x 128 - abs y 128 - abs < x y ? ?
        def get_lut1(x, y):
            tmp = x if abs(x - 128) < abs(y - 128) else y
            if (x - 128) * (y - 128) < 0:
                return round((tmp - 128) * .25 + 128)
            else:
                return tmp
        # x AMN + y < x AMN + x AMN - y > x AMN - y ? ?
        def get_lut2(x, y):
            if x + amnt < y:
                return x + amnt
            elif x - amnt > y:
                return x - amnt
            else:
                return y
        
        planes = [0, 1, 2] if chroma else [0]
        STR = repr(sstr)
        AMN = repr(amnt << shift)
        if not chroma:
            clp_src = clp
            clp = self.core.std.ShufflePlanes(clp, planes=[0], colorfamily=vs.GRAY)
        vblur = self.core.generic.Convolution(clp, [50, 99, 50], mode='v', planes=planes)
        vblurD = self.core.std.MakeDiff(clp, vblur, planes=planes)
        vshrp = self.core.std.Expr([vblur, self.core.generic.Convolution(vblur, [1, 4, 6, 4, 1], mode='v', planes=planes)], 'x x y - '+STR+' * +')
        vshrpD = self.core.std.MakeDiff(vshrp, vblur, planes=planes)
        if shift > 0:
            vlimD = self.core.std.Expr([vshrpD, vblurD], 'x '+NEU+' - y '+NEU+' - * 0 < x '+NEU+' - abs y '+NEU+' - abs < x y ? '+NEU+' - 0.25 * '+NEU+' + x '+NEU+' - abs y '+NEU+' - abs < x y ? ?')
        else:
            vlimD = self.core.std.Lut2(vshrpD, vblurD, planes=planes, function=get_lut1)
        last = self.core.std.MergeDiff(vblur, vlimD, planes=planes)
        if amnt < 255:
            if shift > 0:
                last = self.core.std.Expr([clp, last], 'x '+AMN+' + y < x '+AMN+' + x '+AMN+' - y > x '+AMN+' - y ? ?')
            else:
                last = self.core.std.Lut2(clp, last, planes=planes, function=get_lut2)
        if chroma:
            return last
        else:
            return self.core.std.ShufflePlanes([last, clp_src], planes=[0, 1, 2], colorfamily=vs.YUV)
    
    
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
    # ythresh (default=10) - This determines how close the luma values of the
    #	pixel in the previous and next frames have to be for the pixel to
    #	be hit.  Higher values (within reason) should catch more dot crawl,
    #	but may introduce unwanted artifacts.  Probably shouldn't be set
    #	above 20 or so.
    #
    # cthresh (default=10) - This determines how close the chroma values of the
    #	pixel in the previous and next frames have to be for the pixel to
    #	be hit.  Just as with ythresh.
    #
    # maxdiff (default=50) - This is the maximum difference allowed between the
    #	luma values of the pixel in the CURRENT frame and in each of its
    #	neighbour frames (so, the upper limit to what fluctuations are
    #	considered dot crawl).  Lower values will reduce artifacts but may
    #	cause the filter to miss some dot crawl.  Obviously, this should
    #	never be lower than ythresh.  Meaningless if usemaxdiff = false.
    #
    # scnchg (default=25) - Scene change detection threshold.  Any frame with
    #	total luma difference between it and the previous/next frame greater
    #	than this value will not be processed.
    #
    # usemaxdiff (default=true) - Whether or not to reject luma fluctuations
    #	higher than maxdiff.  Setting this to false is not recommended, as
    #	it may introduce artifacts; but on the other hand, it produces a
    #	30% speed boost.  Test on your particular source.
    #
    # mask (default=false) - When set true, the function will return the mask
    #	instead of the image.  Use to find the best values of cthresh,
    #	ythresh, and maxdiff.
    #	(The scene change threshold, scnchg, is not reflected in the mask.)
    #
    ###################
    #
    # Changelog:
    #
    # 10/3/08: Is this thing on?
    #
    ###################
    def LUTDeCrawl(self, input, ythresh=10, cthresh=15, maxdiff=50, scnchg=25, usemaxdiff=True, mask=False):
        if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV:
            raise ValueError('LUTDeCrawl: This is not a YUV clip !')
        if not isinstance(ythresh, int) or ythresh < 0:
            raise ValueError("LUTDeCrawl: 'ythresh' have not a correct value! [>=0]")
        if not isinstance(cthresh, int) or cthresh < 0:
            raise ValueError("LUTDeCrawl: 'cthresh' have not a correct value! [>=0]")
        if not isinstance(maxdiff, int) or maxdiff < 0:
            raise ValueError("LUTDeCrawl: 'maxdiff' have not a correct value! [>=0]")
        if not isinstance(scnchg, int) or scnchg < 1 or scnchg > 254:
            raise ValueError("LUTDeCrawl: 'scnchg' have not a correct value! [1...254]")
        if not isinstance(usemaxdiff, bool):
            raise ValueError("LUTDeCrawl: 'usemaxdiff' must be bool")
        if not isinstance(mask, bool):
            raise ValueError("LUTDeCrawl: 'mask' must be bool")
        
        shift = input.format.bits_per_sample - 8
        peak = repr((1 << input.format.bits_per_sample) - 1)
        
        ythr = repr(ythresh << shift)
        cthr = repr(cthresh << shift)
        md = repr(maxdiff << shift)
        
        input_minus = self.core.std.Trim(input, 0, 0) + input
        input_plus = self.core.std.Trim(input, 1) + self.core.std.Trim(input, input.num_frames-1)
        
        input_y = self.core.std.ShufflePlanes(input, planes=[0], colorfamily=vs.GRAY)
        input_minus_y = self.core.std.ShufflePlanes(input_minus, planes=[0], colorfamily=vs.GRAY)
        input_minus_u = self.core.std.ShufflePlanes(input_minus, planes=[1], colorfamily=vs.GRAY)
        input_minus_v = self.core.std.ShufflePlanes(input_minus, planes=[2], colorfamily=vs.GRAY)
        input_plus_y = self.core.std.ShufflePlanes(input_plus, planes=[0], colorfamily=vs.GRAY)
        input_plus_u = self.core.std.ShufflePlanes(input_plus, planes=[1], colorfamily=vs.GRAY)
        input_plus_v = self.core.std.ShufflePlanes(input_plus, planes=[2], colorfamily=vs.GRAY)
        
        average_y = self.core.std.Expr([input_minus_y, input_plus_y], 'x y - abs '+ythr+' < x y + 2 / 0 ?')
        average_u = self.core.std.Expr([input_minus_u, input_plus_u], 'x y - abs '+cthr+' < '+peak+' 0 ?')
        average_v = self.core.std.Expr([input_minus_v, input_plus_v], 'x y - abs '+cthr+' < '+peak+' 0 ?')
        
        ymask = self.core.generic.Binarize(average_y, threshold=1<<shift)
        if usemaxdiff:
            diffplus_y = self.core.std.Expr([input_plus_y, input_y], 'x y - abs '+md+' < '+peak+' 0 ?')
            diffminus_y = self.core.std.Expr([input_minus_y, input_y], 'x y - abs '+md+' < '+peak+' 0 ?')
            diffs_y = self.Logic(diffplus_y, diffminus_y, 'and')
            ymask = self.Logic(ymask, diffs_y, 'and')
        cmask = self.Logic(self.core.generic.Binarize(average_u, threshold=129<<shift), self.core.generic.Binarize(average_v, threshold=129<<shift), 'and')
        cmask = self.Resize(cmask, input.width, input.height, kernel='point')
        
        themask = self.Logic(ymask, cmask, 'and')
        
        fixed_y = self.core.std.Merge(average_y, input_y)
        
        output = self.core.std.ShufflePlanes([self.core.std.MaskedMerge(input_y, fixed_y, themask), input], planes=[0, 1, 2], colorfamily=vs.YUV)
        
        def YDifferenceFromPrevious(n, f, clips):
            if f.props._SceneChangePrev:
                return clips[0]
            else:
                return clips[1]
        def YDifferenceToNext(n, f, clips):
            if f.props._SceneChangeNext:
                return clips[0]
            else:
                return clips[1]
        
        input = self.core.std.DuplicateFrames(input, [0, input.num_frames-1])
        input = self.set_scenechange(input, scnchg<<shift)
        input = self.core.std.DeleteFrames(input, [0, input.num_frames-1])
        output = self.core.std.FrameEval(output, eval=functools.partial(YDifferenceFromPrevious, clips=[input, output]), prop_src=input)
        output = self.core.std.FrameEval(output, eval=functools.partial(YDifferenceToNext, clips=[input, output]), prop_src=input)
        
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
    # cthresh (default=10) - This determines how close the chroma values of the
    #	pixel in the previous and next frames have to be for the pixel to
    #	be hit.  Higher values (within reason) should catch more rainbows,
    #	but may introduce unwanted artifacts.  Probably shouldn't be set
    #	above 20 or so.
    #
    # ythresh (default=10) - If the y parameter is set true, then this
    #	determines how close the luma values of the pixel in the previous
    #	and next frames have to be for the pixel to be hit.  Just as with
    #	cthresh.
    #
    # y (default=true) - Determines whether luma difference will be considered
    #	in determining which pixels to hit and which to leave alone.
    #
    # linkUV (default=true) - Determines whether both chroma channels are
    #	considered in determining which pixels in each channel to hit.
    #	When set true, only pixels that meet the thresholds for both U and
    #	V will be hit; when set false, the U and V channels are masked
    #	separately (so a pixel could have its U hit but not its V, or vice
    #	versa).
    #
    # mask (default=false) - When set true, the function will return the mask
    #	(for combined U/V) instead of the image.  Formerly used to find the
    #	best values of cthresh and ythresh.  If linkUV=false, then this
    #	mask won't actually be used anyway (because each chroma channel
    #	will have its own mask).
    #
    ###################
    #
    # Changelog:
    #
    # 6/23/05: Is this thing on?
    # 6/24/05: Replaced whole mask mechanism; new mask checks to see that BOTH channels
    # 	of the chroma are within the threshold from previous frame to next
    # 7/1/05: Added Y option, to take luma into account when deciding whether to use the
    #	averaged chroma; added ythresh and cthresh parameters, to determine how close
    #	the chroma/luma values of a pixel have to be to be considered the same
    #	(y=true is meant to cut down on artifacts)
    # 9/2/05: Suddenly realized this wouldn't work for YUY2 and made it YV12 only;
    #	added linkUV option, to decide whether to use a separate mask for each chroma
    #	channel or use the same one for both.
    # 10/3/08: Fixed "cthresh" typos in documentation; killed repmode since I realized I
    #	wasn't using Repair anymore; finally upgraded to MaskTools 2.
    #
    ###################
    def LUTDeRainbow(self, input, cthresh=10, ythresh=10, y=True, linkUV=True, mask=False):
        if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV:
            raise ValueError('LUTDeRainbow: This is not a YUV clip !')
        if not isinstance(cthresh, int) or cthresh < 0:
            raise ValueError("LUTDeRainbow: 'cthresh' have not a correct value! [>=0]")
        if not isinstance(ythresh, int) or ythresh < 0:
            raise ValueError("LUTDeRainbow: 'ythresh' have not a correct value! [>=0]")
        if not isinstance(y, bool):
            raise ValueError("LUTDeRainbow: 'y' must be bool")
        if not isinstance(linkUV, bool):
            raise ValueError("LUTDeRainbow: 'linkUV' must be bool")
        if not isinstance(mask, bool):
            raise ValueError("LUTDeRainbow: 'mask' must be bool")
        
        shift = input.format.bits_per_sample - 8
        peak = repr((1 << input.format.bits_per_sample) - 1)
        
        cthr = repr(cthresh << shift)
        ythr = repr(ythresh << shift)
        
        input_minus = self.core.std.Trim(input, 0, 0) + input
        input_plus = self.core.std.Trim(input, 1) + self.core.std.Trim(input, input.num_frames-1)
        
        input_u = self.core.std.ShufflePlanes(input, planes=[1], colorfamily=vs.GRAY)
        input_v = self.core.std.ShufflePlanes(input, planes=[2], colorfamily=vs.GRAY)
        input_minus_y = self.core.std.ShufflePlanes(input_minus, planes=[0], colorfamily=vs.GRAY)
        input_minus_u = self.core.std.ShufflePlanes(input_minus, planes=[1], colorfamily=vs.GRAY)
        input_minus_v = self.core.std.ShufflePlanes(input_minus, planes=[2], colorfamily=vs.GRAY)
        input_plus_y = self.core.std.ShufflePlanes(input_plus, planes=[0], colorfamily=vs.GRAY)
        input_plus_u = self.core.std.ShufflePlanes(input_plus, planes=[1], colorfamily=vs.GRAY)
        input_plus_v = self.core.std.ShufflePlanes(input_plus, planes=[2], colorfamily=vs.GRAY)
        
        average_y = self.Resize(self.core.std.Expr([input_minus_y, input_plus_y], 'x y - abs '+ythr+' < '+peak+' 0 ?'),
                                int(input.width/2), int(input.height/2), kernel='bilinear')
        average_u = self.core.std.Expr([input_minus_u, input_plus_u], 'x y - abs '+cthr+' < x y + 2 / 0 ?')
        average_v = self.core.std.Expr([input_minus_v, input_plus_v], 'x y - abs '+cthr+' < x y + 2 / 0 ?')
        
        umask = self.core.generic.Binarize(average_u, threshold=21<<shift)
        vmask = self.core.generic.Binarize(average_v, threshold=21<<shift)
        if y:
            umask = self.Logic(umask, average_y, 'and')
            vmask = self.Logic(vmask, average_y, 'and')
        
        themask = self.Logic(umask, vmask, 'and')
        if y:
            themask = self.Logic(themask, average_y, 'and')
        
        fixed_u = self.core.std.Merge(average_u, input_u)
        fixed_v = self.core.std.Merge(average_v, input_v)
        
        output_u = self.core.std.MaskedMerge(input_u, fixed_u, themask if linkUV else umask)
        output_v = self.core.std.MaskedMerge(input_v, fixed_v, themask if linkUV else vmask)
        
        output = self.core.std.ShufflePlanes([input, output_u, output_v], planes=[0, 0, 0], colorfamily=vs.YUV)
        
        if mask:
            return self.Resize(themask, input.width, input.height, kernel='point')
        else:
            return output
    
    
    ######
    ###
    ### GrainStabilizeMC v1.0      by mawen1250      2014.03.22
    ###
    ### Requirements: GenericFilters, RemoveGrain/Repair, mvtools v2.6.0.5
    ###
    ### Temporal-only on-top grain stabilizer
    ### Only stabilize the difference ( on-top grain ) between source clip and spatial-degrained clip
    ###
    ### Parameters:
    ###  Y,U,V [bool] - If true, the corresponding plane is processed. Otherwise, it is copied through to the output image as is. Default is true,false,false
    ###  lsb [bool]   - MDegrain generates 16-bit data when set to true, output clip will be 16-bit native. Default is false
    ###  nrmode [int] - Mode to get grain/noise from input clip. 0: 3x3 Average Blur, 1: 3x3 SBR, 2: 5x5 SBR, 3: 7x7 SBR. Or define your own denoised clip "p". Default is 2 for HD / 1 for SD
    ###  radius [int] - Temporal radius of MDegrain for grain stabilize(1-3). Default is 1
    ###  adapt [int]  - Threshold for luma-adaptative mask. -1: off, 0: source, 255: invert. Or define your own luma mask clip "Lmask". Default is -1
    ###  rep [int]    - Mode of repair to avoid artifacts, set 0 to turn off this operation. Default is 13
    ###
    ######
    def GSMC(self, input, p=None, Lmask=None, nrmode=None, radius=1, adapt=-1, rep=13, Y=True, U=False, V=False, lsb=False):
        if not isinstance(input, vs.VideoNode) or input.format.id not in (vs.YUV420P8, vs.YUV420P9, vs.YUV420P10, vs.YUV420P16):
            raise ValueError('GSMC: This is not a YUV420P8, YUV420P9, YUV420P10 or YUV420P16 clip !')
        
        HD = input.width > 1024 or input.height > 576
        if nrmode is None:
            nrmode = 2 if HD else 1
        
        if p is not None and (not isinstance(p, vs.VideoNode) or input.format.id != p.format.id):
            raise ValueError("GSMC: 'p' must be the same format as input !")
        if Lmask is not None and (not isinstance(Lmask, vs.VideoNode) or Lmask.format.color_family not in (vs.YUV, vs.GRAY) or input.format.bits_per_sample != Lmask.format.bits_per_sample):
            raise ValueError("GSMC: 'Lmask' must be the same format as input or the grayscale equivalent !")
        if not isinstance(nrmode, int) or nrmode < 0 or nrmode > 3:
            raise ValueError("GSMC: 'nrmode' have not a correct value! [0,1,2,3]")
        if not isinstance(radius, int) or radius < 1 or radius > 3:
            raise ValueError("GSMC: 'radius' have not a correct value! [1...3]")
        if not isinstance(adapt, int) or adapt < -1 or adapt > 255:
            raise ValueError("GSMC: 'adapt' have not a correct value! [-1,0...255]")
        if not isinstance(rep, int) or rep not in (0, 1, 2, 3, 4, 11, 12, 13, 14):
            raise ValueError("GSMC: 'rep' have not a correct value! [0,1,2,3,4,11,12,13,14]")
        if not isinstance(Y, bool):
            raise ValueError("GSMC: 'Y' must be bool")
        if not isinstance(U, bool):
            raise ValueError("GSMC: 'U' must be bool")
        if not isinstance(V, bool):
            raise ValueError("GSMC: 'V' must be bool")
        if not isinstance(lsb, bool):
            raise ValueError("GSMC: 'lsb' must be bool")
        
        shift = input.format.bits_per_sample - 8
        
        if Y and U and V:
            planes = [0, 1, 2]
        elif Y and U:
            planes = [0, 1]
        elif Y and V:
            planes = [0, 2]
        elif U and V:
            planes = [1, 2]
        elif Y:
            planes = [0]
        elif U:
            planes = [1]
        elif V:
            planes = [2]
        else:
            return input
        
        chromamv = U or V
        blksize = 32 if HD else 16
        overlap = int(blksize / 4)
        if not Y:
            if not U:
                plane = 2
            elif not V:
                plane = 1
            else:
                plane = 3
        elif not U and not V:
            plane = 0
        else:
            plane = 4
        
        # Kernel: Spatial Noise Dumping
        if p:
            pre_nr = p
        elif nrmode == 0:
            pre_nr = self.core.rgvs.RemoveGrain(input, [20 if Y else 0, 20 if U else 0, 20 if V else 0])
        else:
            pre_nr = self.sbr(input, nrmode, planes=planes)
        dif_nr = self.core.std.MakeDiff(input, pre_nr, planes=planes)
        
        # Kernel: MC Grain Stabilize
        if shift > 0:
            pre_nr8 = self.core.fmtc.bitdepth(pre_nr, bits=8, planes=[0, 1, 2] if chromamv else [0], dmode=1)
            dif_nr8 = self.core.fmtc.bitdepth(dif_nr, bits=8, planes=[0, 1, 2] if chromamv else [0], dmode=1)
        else:
            pre_nr8 = pre_nr
            dif_nr8 = dif_nr
        
        psuper = self.core.avs.MSuper(pre_nr8, chroma=chromamv)
        difsuper = self.core.avs.MSuper(dif_nr8, levels=1, chroma=chromamv)
        
        fv1 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=False, chroma=chromamv, delta=1, truemotion=False, _global=True, overlap=overlap)
        bv1 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=True, chroma=chromamv, delta=1, truemotion=False, _global=True, overlap=overlap)
        if radius > 1:
            fv2 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=False, chroma=chromamv, delta=2, truemotion=False, _global=True, overlap=overlap)
            bv2 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=True, chroma=chromamv, delta=2, truemotion=False, _global=True, overlap=overlap)
        if radius > 2:
            fv3 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=False, chroma=chromamv, delta=3, truemotion=False, _global=True, overlap=overlap)
            bv3 = self.core.avs.MAnalyse(psuper, blksize=blksize, isb=True, chroma=chromamv, delta=3, truemotion=False, _global=True, overlap=overlap)
        
        if radius == 1:
            dif_sb = self.core.avs.MDegrain1(dif_nr8, difsuper, bv1, fv1, thSAD=300, thSADC=150, plane=plane, thSCD1=300, thSCD2=100, lsb=lsb)
        elif radius == 2:
            dif_sb = self.core.avs.MDegrain2(dif_nr8, difsuper, bv1, fv1, bv2, fv2, thSAD=300, thSADC=150, plane=plane, thSCD1=300, thSCD2=100, lsb=lsb)
        else:
            dif_sb = self.core.avs.MDegrain3(dif_nr8, difsuper, bv1, fv1, bv2, fv2, bv3, fv3, thSAD=300, thSADC=150, plane=plane, thSCD1=300, thSCD2=100, lsb=lsb)
        
        # Post-Process: Luma-Adaptive Mask Merging & Repairing
        if lsb:
            dif_sb = self.core.fmtc.stack16tonative(dif_sb)
            if shift < 8:
                input = self.core.fmtc.bitdepth(input, bits=16)
                pre_nr = self.core.fmtc.bitdepth(pre_nr, bits=16)
        elif shift > 0:
            dif_sb = self.LimitDiff(dif_nr, self.core.fmtc.bitdepth(dif_sb, bits=8+shift, planes=[0, 1, 2] if chromamv else [0]), thr=1, elast=2, planes=planes)
        stable = self.core.std.MergeDiff(pre_nr, dif_sb, planes=planes)
        if rep > 0:
            stable = self.core.rgvs.Repair(stable, input, [rep if Y else 0, rep if U else 0, rep if V else 0])
        
        # x adapt - abs 255 * adapt 128 - abs 128 + /
        def get_lut(x):
            return round(abs(x - (adapt << shift)) * ((255 << shift) + 2 ** shift - 1) / (abs((adapt << shift) - (128 << shift)) + (128 << shift)))
        
        if Lmask:
            return self.core.std.MaskedMerge(input, stable, Lmask, planes=planes, first_plane=True)
        elif adapt == -1:
            return stable
        else:
            input_y = self.core.std.ShufflePlanes(input, planes=[0], colorfamily=vs.GRAY)
            if adapt == 0:
                Lmask = self.core.rgvs.RemoveGrain(input_y, 19)
            elif adapt == 255:
                Lmask = self.core.rgvs.RemoveGrain(self.core.generic.Invert(input_y), 19)
            else:
                Lmask = self.core.rgvs.RemoveGrain(self.core.std.Lut(input_y, function=get_lut), 19)
            return self.core.std.MaskedMerge(input, stable, Lmask, planes=planes)
    
    
    ################################################################################################
    ###                                                                                          ###
    ###                           Simple MDegrain Mod - SMDegrain()                              ###
    ###                                                                                          ###
    ###                       Mod by Dogway - Original idea by Caroliano                         ###
    ###                                                                                          ###
    ###          Special Thanks: Sagekilla, Didée, cretindesalpes, Gavino and MVtools people     ###
    ###                                                                                          ###
    ###                       v2.2d (Dogway's mod) - 05 March 2013                               ###
    ###                                                                                          ###
    ################################################################################################
    ###
    ### General purpose simple degrain function. Pure temporal denoiser. Basically a wrapper(function)/frontend of mvtools2+mdegrain
    ### with some common related options added. Aim is at accessibility and quality but not targeted to any specific kind of source.
    ### The reason behind is to keep it simple so aside masktools2 you will only need MVTools2.
    ###
    ### Check documentation for deep explanation on settings and defaults.
    ### Doom10 thread: (http://doom10.org/index.php?topic=2178.0)
    ###
    ################################################################################################
    def SMDegrain(self, input, tr=3, thSAD=400, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False, tff=None, plane=4, Globals=0,
                  pel=None, subpixel=2, prefilter=0, blksize=None, overlap=None, search=4, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None,
                  thSCD1=None, thSCD2=130, chroma=True, hpad=None, vpad=None, lsb=None, lsb_in=False, lsb_out=False, mode=0, Str=1., Amp=.0625):
        if not isinstance(input, vs.VideoNode) or input.format.id != vs.YUV420P8:
            raise ValueError('SMDegrain: This is not a YUV420P8 clip !')
        
        # Defaults & Conditionals
        thSAD2 = int(thSAD / 2)
        if thSADC is None:
            thSADC = thSAD2
        
        if lsb is None:
            lsb = lsb_in or lsb_out
        
        GlobalR = Globals == 1
        GlobalO = Globals == 3
        if1 = CClip is not None
        
        if contrasharp is None:
            contrasharp = not GlobalO and if1
        
        w = input.width
        h = input.height
        preclip = isinstance(prefilter, vs.VideoNode)
        ifC = isinstance(contrasharp, bool)
        if0 = contrasharp if ifC else contrasharp > 0
        if2 = if0 and lsb and not GlobalO
        if3 = tr > 3
        if4 = w > 1279 or (lsb_in and h > 1439) or (not lsb_in and h > 719)
        
        if pel is None:
            pel = 1 if if4 else 2
        pelclip = pel > 1 and subpixel == 3
        
        if blksize is None:
            blksize = 16 if if4 else 8
        blk2 = int(blksize / 2)
        if overlap is None:
            overlap = blk2
        ovl2 = int(overlap / 2)
        if truemotion is None:
            truemotion = not if4
        if MVglobal is None:
            MVglobal = truemotion
        if thSCD1 is None:
            thSCD1 = int((blksize * 2.5) ** 2)
        
        planes = [0, 1, 2] if chroma else [0]
        Chr = 3 if chroma else 1
        plane0 = plane != 0
        U = 3 if plane0 and plane != 2 else 2
        V = 3 if plane0 and plane != 1 else 2
        Uin = 3 if lsb_in else U
        Vin = 3 if lsb_in else V
        
        if hpad is None:
            hpad = 0 if if4 else blksize
        if vpad is None:
            vpad = 0 if if4 else blksize
        if limitc is None:
            limitc = limit
        
        # Error Report
        if not isinstance(tr, int) or tr < 1:
            raise ValueError("SMDegrain: 'tr' have not a correct value! [>=1]")
        if not isinstance(thSAD, int):
            raise ValueError("SMDegrain: 'thSAD' must be integer")
        if not isinstance(thSADC, int):
            raise ValueError("SMDegrain: 'thSADC' must be integer")
        if not isinstance(RefineMotion, bool):
            raise ValueError("SMDegrain: 'RefineMotion' must be bool")
        if not (ifC or isinstance(contrasharp, int)):
            raise ValueError("SMDegrain: 'contrasharp' only accepts bool and integer inputs")
        if if1 and (not isinstance(CClip, vs.VideoNode) or CClip.format.id != vs.YUV420P8):
            raise ValueError("SMDegrain: 'CClip' is not a YUV420P8 clip !")
        if not isinstance(interlaced, bool):
            raise ValueError("SMDegrain: 'interlaced' must be bool")
        if interlaced and not isinstance(tff, bool):
            raise ValueError("SMDegrain: 'tff' must be set if source is interlaced. Setting tff to true means top field first and false means bottom field first")
        if not isinstance(plane, int) or plane < 0 or plane > 4:
            raise ValueError("SMDegrain: 'plane' have not a correct value! [0,1,2,3,4]")
        if not isinstance(Globals, int) or Globals < 0 or Globals > 3:
            raise ValueError("SMDegrain: 'Globals' have not a correct value! [0,1,2,3]")
        if not isinstance(pel, int) or pel not in (1, 2, 4):
            raise ValueError("SMDegrain: 'pel' have not a correct value! [1,2,4]")
        if not isinstance(subpixel, int) or subpixel < 0 or subpixel > 3:
            raise ValueError("SMDegrain: 'subpixel' have not a correct value! [0,1,2,3]")
        if not (isinstance(prefilter, int) or (preclip and prefilter.format.id == vs.YUV420P8)):
            raise ValueError("SMDegrain: 'prefilter' only accepts integer and YUV420P8 clip inputs")
        if isinstance(prefilter, int) and (prefilter < 0 or prefilter > 3):
            raise ValueError("SMDegrain: 'prefilter' have not a correct value! [0,1,2,3]")
        if not isinstance(blksize, int) or blksize not in (4, 8, 16, 32):
            raise ValueError("SMDegrain: 'blksize' have not a correct value! [4,8,16,32]")
        if not isinstance(overlap, int) or overlap > blk2 or overlap % 2 != 0:
            raise ValueError("SMDegrain: 'overlap' must be at least half blksize or less and must be an even figure")
        if not isinstance(search, int) or search < 0 or search > 7:
            raise ValueError("SMDegrain: 'search' have not a correct value! [0,1,2,3,4,5,6,7]")
        if not isinstance(truemotion, bool):
            raise ValueError("SMDegrain: 'truemotion' must be bool")
        if not isinstance(MVglobal, bool):
            raise ValueError("SMDegrain: 'MVglobal' must be bool")
        if not isinstance(dct, int) or dct < 0 or dct > 10:
            raise ValueError("SMDegrain: 'dct' have not a correct value! [0,1,2,3,4,5,6,7,8,9,10]")
        if not isinstance(limit, int) or limit < 1 or limit > 255:
            raise ValueError("SMDegrain: 'limit' have not a correct value! [1...255]")
        if not isinstance(limitc, int) or limitc < 1 or limitc > 255:
            raise ValueError("SMDegrain: 'limitc' have not a correct value! [1...255]")
        if not isinstance(thSCD1, int):
            raise ValueError("SMDegrain: 'thSCD1' must be integer")
        if not isinstance(thSCD2, int) or thSCD2 < 0 or thSCD2 > 255:
            raise ValueError("SMDegrain: 'thSCD2' have not a correct value! [0...255]")
        if not isinstance(chroma, bool):
            raise ValueError("SMDegrain: 'chroma' must be bool")
        if not isinstance(hpad, int) or hpad < 0:
            raise ValueError("SMDegrain: 'hpad' have not a correct value! [>=0]")
        if not isinstance(vpad, int) or vpad < 0:
            raise ValueError("SMDegrain: 'vpad' have not a correct value! [>=0]")
        if not isinstance(lsb, bool):
            raise ValueError("SMDegrain: 'lsb' must be bool")
        if not isinstance(lsb_in, bool):
            raise ValueError("SMDegrain: 'lsb_in' must be bool")
        if not isinstance(lsb_out, bool):
            raise ValueError("SMDegrain: 'lsb_out' must be bool")
        if not isinstance(mode, int) or mode < -1 or mode > 8:
            raise ValueError("SMDegrain: 'mode' have not a correct value! [-1,0,1,2,3,4,5,6,7,8]")
        if not (isinstance(Str, float) or isinstance(Str, int)) or Str < 0 or Str > 8:
            raise ValueError("SMDegrain: 'Str' have not a correct value! [0.0...8.0]")
        if not (isinstance(Amp, float) or isinstance(Amp, int)) or Amp < 0 or Amp > 1:
            raise ValueError("SMDegrain: 'Amp' have not a correct value! [0.0...1.0]")
        if interlaced and h % 4 != 0:
            raise ValueError('SMDegrain: Interlaced sources require mod 4 height sizes')
        if lsb_in and interlaced:
            raise ValueError('SMDegrain: Interlaced 16 bit stacked clip is not a recognized format')
        if RefineMotion and blksize < 8:
            raise ValueError('SMDegrain: For RefineMotion you need a blksize of at least 8')
        if lsb_in and not lsb:
            raise ValueError('SMDegrain: lsb_in requires lsb=True')
        if lsb_out and not lsb:
            raise ValueError('SMDegrain: lsb_out requires lsb=True')
        
        # RefineMotion Variables
        if RefineMotion:
            halfblksize = blk2                                          # MRecalculate works with half block size
            halfoverlap = overlap if overlap == 2 else ovl2 + ovl2 % 2  # Halve the overlap to suit the halved block size
        if RefineMotion or if3:
            halfthSAD = thSAD2                                          # MRecalculate uses a more strict thSAD, which defaults to 150 (half of function's default of 300)
        if if3:
            halfthSADC = int(thSADC / 2)                                # For MDegrainN()
        
        # Input preparation for: LSB_IN and Interlacing
        if not interlaced:
            inputP = input
        else:
            inputP = self.core.std.SeparateFields(input, tff)
        
        if lsb_in:
            input8h = self.core.avs.DitherPost(inputP, mode=6)
        input8 = input8h if lsb_in else inputP
        
        # x 16 < 255 x 75 > 0 255 x 16 - 255 75 16 - / * - ? ?
        def get_lut(x):
            if lsb:
                if x < 4096:
                    return 65535
                elif x > 19200:
                    return 0
                else:
                    return round(65535 - (x - 4096) * (65535 / 15104))
            else:
                if x < 16:
                    return 255
                elif x > 75:
                    return 0
                else:
                    return round(255 - (x - 16) * (255 / 59))
        
        # Prefilter
        if not GlobalR:
            if preclip:
                pref = prefilter
            elif prefilter == 0:
                pref = inputP
            elif prefilter == 1:
                if lsb:
                    pref = self.MinBlur(inputP, 1, planes=planes, lsb=True, lsb_in=lsb_in)
                else:
                    pref = self.MinBlur(inputP, 1, planes=planes)
            elif prefilter == 2:
                if lsb:
                    pref = self.MinBlur(inputP, 2, planes=planes, lsb=True, lsb_in=lsb_in)
                else:
                    pref = self.MinBlur(inputP, 2, planes=planes)
            else:
                if lsb:
                    clip1 = self.core.avs.dfttest(inputP, U=chroma, V=chroma, tbsize=1, sstring='0.0:4.0 0.2:9.0 1.0:15.0', lsb=True, lsb_in=lsb_in)
                    clip1 = self.core.fmtc.stack16tonative(clip1)
                    if lsb_in:
                        clip2 = self.core.fmtc.stack16tonative(inputP)
                    else:
                        clip2 = self.core.fmtc.bitdepth(inputP, bits=16)
                    mask = self.core.std.Lut(clip2, planes=[0], function=get_lut)
                    pref = self.core.fmtc.nativetostack16(self.core.std.MaskedMerge(clip1, clip2, mask, planes=planes, first_plane=chroma))
                else:
                    pref = self.core.std.MaskedMerge(
                      self.core.avs.dfttest(inputP, U=chroma, V=chroma, tbsize=1, sstring='0.0:4.0 0.2:9.0 1.0:15.0', dither=1),
                      inputP,
                      self.core.std.Lut(input8, planes=[0], function=get_lut),
                      planes=planes, first_plane=chroma)
        else:
            pref = input8
        
        # Default Auto-Prefilter - Luma expansion TV->PC (up to 16% more values for motion estimation)
        if not GlobalR:
            if lsb:
                if not preclip and ((lsb_in and prefilter == 0) or prefilter > 0):
                    pref = self.Dither_Luma_Rebuild(pref, s0=Str, c=Amp, chroma=chroma, lsb_in=True)
                else:
                    pref = self.Dither_Luma_Rebuild(self.Dither_convert_8_to_16(pref), s0=Str, c=Amp, chroma=chroma, lsb_in=True)
            else:
                pref = self.Dither_Luma_Rebuild(pref, s0=Str, c=Amp, chroma=chroma)
        
        # Subpixel 3
        if pelclip:
            pclip = self.core.nnedi3.nnedi3_rpow2(pref, rfactor=pel, nns=4, qual=2)
            if not GlobalR:
                pclip2 = self.core.nnedi3.nnedi3_rpow2(input8, rfactor=pel, nns=4, qual=2)
        
        # Motion vectors search
        if pelclip:
            super_search = self.core.avs.MSuper(pref, pel=pel, chroma=chroma, hpad=hpad, vpad=vpad, pelclip=pclip, rfilter=4)
        else:
            super_search = self.core.avs.MSuper(pref, pel=pel, sharp=subpixel, chroma=chroma, hpad=hpad, vpad=vpad, rfilter=4)
        if not GlobalR:
            if pelclip:
                super_render = self.core.avs.MSuper(input8, pel=pel, chroma=plane0, hpad=hpad, vpad=vpad, levels=1, pelclip=pclip2)
                if RefineMotion:
                    Recalculate = self.core.avs.MSuper(pref, pel=pel, chroma=chroma, hpad=hpad, vpad=vpad, levels=1, pelclip=pclip)
            else:
                super_render = self.core.avs.MSuper(input8, pel=pel, sharp=subpixel, chroma=plane0, hpad=hpad, vpad=vpad, levels=1)
                if RefineMotion:
                    Recalculate = self.core.avs.MSuper(pref, pel=pel, sharp=subpixel, chroma=chroma, hpad=hpad, vpad=vpad, levels=1)
        else:
            super_render = super_search
        if not if3:
            if interlaced:
                if tr > 2:
                    if not GlobalR:
                        bv6 = self.core.avs.MAnalyse(super_search, isb=True, delta=6, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        fv6 = self.core.avs.MAnalyse(super_search, isb=False, delta=6, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        if RefineMotion:
                            bv6 = self.core.avs.MRecalculate(Recalculate, bv6, overlap=halfoverlap, blksize=halfblksize,
                                                             thSAD=halfthSAD, chroma=chroma, truemotion=truemotion)
                            fv6 = self.core.avs.MRecalculate(Recalculate, fv6, overlap=halfoverlap, blksize=halfblksize,
                                                             thSAD=halfthSAD, chroma=chroma, truemotion=truemotion)
                    else:
                        bv6 = self.bv6
                        fv6 = self.fv6
                if tr > 1:
                    if not GlobalR:
                        bv4 = self.core.avs.MAnalyse(super_search, isb=True, delta=4, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        fv4 = self.core.avs.MAnalyse(super_search, isb=False, delta=4, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        if RefineMotion:
                            bv4 = self.core.avs.MRecalculate(Recalculate, bv4, overlap=halfoverlap, blksize=halfblksize,
                                                             thSAD=halfthSAD, chroma=chroma, truemotion=truemotion)
                            fv4 = self.core.avs.MRecalculate(Recalculate, fv4, overlap=halfoverlap, blksize=halfblksize,
                                                             thSAD=halfthSAD, chroma=chroma, truemotion=truemotion)
                    else:
                        bv4 = self.bv4
                        fv4 = self.fv4
            else:
                if tr > 2:
                    if not GlobalR:
                        bv3 = self.core.avs.MAnalyse(super_search, isb=True, delta=3, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        fv3 = self.core.avs.MAnalyse(super_search, isb=False, delta=3, overlap=overlap, blksize=blksize,
                                                     search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                        if RefineMotion:
                            bv3 = self.core.avs.MRecalculate(Recalculate, bv3, overlap=halfoverlap, blksize=halfblksize,
                                                             thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                            fv3 = self.core.avs.MRecalculate(Recalculate, fv3, overlap=halfoverlap, blksize=halfblksize,
                                                             thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                    else:
                        bv3 = self.bv3
                        fv3 = self.fv3
                if not GlobalR:
                    bv1 = self.core.avs.MAnalyse(super_search, isb=True, delta=1, overlap=overlap, blksize=blksize,
                                                 search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                    fv1 = self.core.avs.MAnalyse(super_search, isb=False, delta=1, overlap=overlap, blksize=blksize,
                                                 search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                    if RefineMotion:
                        bv1 = self.core.avs.MRecalculate(Recalculate, bv1, overlap=halfoverlap, blksize=halfblksize,
                                                         thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                        fv1 = self.core.avs.MRecalculate(Recalculate, fv1, overlap=halfoverlap, blksize=halfblksize,
                                                         thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                else:
                    bv1 = self.bv1
                    fv1 = self.fv1
            if interlaced or tr > 1:
                if not GlobalR:
                    bv2 = self.core.avs.MAnalyse(super_search, isb=True, delta=2, overlap=overlap, blksize=blksize,
                                                 search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                    fv2 = self.core.avs.MAnalyse(super_search, isb=False, delta=2, overlap=overlap, blksize=blksize,
                                                 search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct)
                    if RefineMotion:
                        bv2 = self.core.avs.MRecalculate(Recalculate, bv2, overlap=halfoverlap, blksize=halfblksize,
                                                         thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                        fv2 = self.core.avs.MRecalculate(Recalculate, fv2, overlap=halfoverlap, blksize=halfblksize,
                                                         thsad=halfthSAD, chroma=chroma, truemotion=truemotion)
                else:
                    bv2 = self.bv2
                    fv2 = self.fv2
        else:
            tr2 = tr * 2
            if not GlobalR:
                vmulti = self.core.avs.MAnalyse(super_search, multi=True, overlap=overlap, blksize=blksize,
                                                search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, dct=dct, delta=tr2 if interlaced else tr)
                if RefineMotion:
                    vmulti = self.core.avs.MRecalculate(Recalculate, vmulti, overlap=halfoverlap, blksize=halfblksize,
                                                        thsad=halfthSAD, chroma=chroma, truemotion=truemotion, tr=tr2 if interlaced else tr)
                if interlaced:
                    vmulti = self.core.std.SelectEvery(vmulti, 4, [2, 3])
            else:
                vmulti = self.vmulti
                vmulti = self.core.std.SelectEvery(vmulti, self.Rtr*2, list(range(tr2)))
        
        # Finally, MDegrain
        if not GlobalO:
            if interlaced:
                if lsb:
                    if if3:
                        output = self.core.avs.MDegrainN(input8, super_render, vmulti, tr, thsad2=halfthSAD, thsadc2=halfthSADC, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    elif tr == 3:
                        output = self.core.avs.MDegrain3(input8, super_render, bv2, fv2, bv4, fv4, bv6, fv6, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    elif tr == 2:
                        output = self.core.avs.MDegrain2(input8, super_render, bv2, fv2, bv4, fv4, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    else:
                        output = self.core.avs.MDegrain1(input8, super_render, bv2, fv2, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                else:
                    if if3:
                        output = self.core.avs.MDegrainN(input8, super_render, vmulti, tr, thsad2=halfthSAD, thsadc2=halfthSADC, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=False)
                    elif tr == 3:
                        output = self.core.avs.MDegrain3(input8, super_render, bv2, fv2, bv4, fv4, bv6, fv6, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
                    elif tr == 2:
                        output = self.core.avs.MDegrain2(input8, super_render, bv2, fv2, bv4, fv4, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
                    else:
                        output = self.core.avs.MDegrain1(input8, super_render, bv2, fv2, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
            else:
                if lsb:
                    if if3:
                        output = self.core.avs.MDegrainN(input8, super_render, vmulti, tr, thsad2=halfthSAD, thsadc2=halfthSADC, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    elif tr == 3:
                        output = self.core.avs.MDegrain3(input8, super_render, bv1, fv1, bv2, fv2, bv3, fv3, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    elif tr == 2:
                        output = self.core.avs.MDegrain2(input8, super_render, bv1, fv1, bv2, fv2, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                    else:
                        output = self.core.avs.MDegrain1(input8, super_render, bv1, fv1, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=True)
                else:
                    if if3:
                        output = self.core.avs.MDegrainN(input8, super_render, vmulti, tr, thsad2=halfthSAD, thsadc2=halfthSADC, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane, lsb=False)
                    elif tr == 3:
                        output = self.core.avs.MDegrain3(input8, super_render, bv1, fv1, bv2, fv2, bv3, fv3, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
                    elif tr == 2:
                        output = self.core.avs.MDegrain2(input8, super_render, bv1, fv1, bv2, fv2, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
                    else:
                        output = self.core.avs.MDegrain1(input8, super_render, bv1, fv1, thSAD=thSAD, thSADC=thSADC,
                                                         thSCD1=thSCD1, thSCD2=thSCD2, limit=limit, limitC=limitc, plane=plane)
        
        # LSB_IN merging
        if not GlobalO and lsb:
            if lsb_in:
                blnkclp = self.core.std.BlankClip(input8h, color=[0, 0, 0])
                if not if4:
                    output = self.Dither_merge16_8(input, output, self.DitherBuildMask(input8h, self.core.avs.DitherPost(output, mode=6, u=U, v=V)), u=U, v=V)
                else:
                    expr = 'x y = 0 255 ?'
                    output = self.Dither_merge16_8(
                      input, output,
                      self.core.generic.Maximum(
                        self.core.std.Expr([input8h, self.core.avs.DitherPost(output, mode=6, u=U, v=V)], [expr, expr if U==3 else '', expr if V==3 else '']),
                        planes=[0, 1, 2] if U==3 and V==3 else [0, 1] if U==3 else [0, 2] if V==3 else [0]),
                      u=U, v=V)
            elif if2:
                blnkclp = self.core.std.BlankClip(input8, color=[0, 0, 0])
        
        # Contrasharp (only sharpens luma)
        if ifC and if0 and if4:
            self.Super = super_render
            if not if3:
                self.bv1 = bv2 if interlaced else bv1
                self.fv1 = fv2 if interlaced else fv1
            else:
                self.bv1 = self.core.std.SelectEvery(vmulti, tr2, 0)
                self.fv1 = self.core.std.SelectEvery(vmulti, tr2, 1)
        
        if not GlobalO and if0:
            if if1:
                CCh = CClip.height
                ref8 = h == CCh * 2
                ref16 = h == int(CCh / 2)
                
                if interlaced and ref16:
                    raise ValueError('SMDegrain: Interlaced 16 bit stacked CClip is not a recognized format')
                
                if lsb_in:
                    if not ref8:
                        CClip = self.core.avs.DitherPost(CClip, mode=-1, u=1, v=1)
                elif interlaced:
                    CClip = self.core.std.SeparateFields(CClip, tff)
                elif ref16:
                    CClip = self.core.avs.DitherPost(CClip, mode=-1, u=1, v=1)
            elif lsb_in:
                CClip = self.core.avs.DitherPost(input, mode=-1, u=1, v=1)
            else:
                CClip = inputP
        
        if not GlobalO and if2:
            OutTO8 = self.core.avs.DitherPost(output, mode=-1, u=1, v=1)
            if ifC:
                ctr16 = self.ContraSharpeningHD(OutTO8, CClip, HD=if4)
            else:
                ctr16 = self.LSFmod(OutTO8, strength=contrasharp, source=CClip, Lmode=0, soft=0, soothe=False, defaults='slow')
            
            ctr16P = ctr16
            ctr16  = self.core.std.StackVertical([ctr16P, blnkclp])
            ctr16  = self.Dither_merge16_8(output, ctr16, self.DitherBuildMask(ctr16P, OutTO8, planes=[0]))
            ctr16  = self.core.std.Merge(ctr16, output, weight=[0, 1])
        
        # Globals Output
        if GlobalO or Globals == 2:
            self.Super = super_render
            if not if3:
                self.bv1 = bv1
                self.fv1 = fv1
                if interlaced or tr > 1:
                    self.bv2 = bv2
                    self.fv2 = fv2
                if not interlaced and tr > 2:
                    self.bv3 = bv3
                    self.fv3 = fv3
                if interlaced and tr > 1:
                    self.bv4 = bv4
                    self.fv4 = fv4
                if interlaced and tr > 2:
                    self.bv6 = bv6
                    self.fv6 = fv6
            else:
                self.bv1 = self.core.std.SelectEvery(vmulti, tr2, 0)
                self.fv1 = self.core.std.SelectEvery(vmulti, tr2, 1)
                self.bv3 = self.core.std.SelectEvery(vmulti, tr2, 4)
                self.fv3 = self.core.std.SelectEvery(vmulti, tr2, 5)
                if interlaced:
                    self.bv2 = bv1
                    self.fv2 = fv1
                    self.bv4 = self.core.std.SelectEvery(vmulti, tr2, 2)
                    self.fv4 = self.core.std.SelectEvery(vmulti, tr2, 3)
                    self.bv6 = bv3
                    self.fv6 = fv3
                else:
                    self.bv2 = self.core.std.SelectEvery(vmulti, tr2, 2)
                    self.fv2 = self.core.std.SelectEvery(vmulti, tr2, 3)
                self.vmulti = vmulti
                self.Rtr = tr
        
        # Output
        if not GlobalO:
            if lsb_out:
                if if0:
                    return ctr16
                else:
                    return output
            elif if0:
                if lsb:
                    if interlaced:
                        return self.core.avs.DitherPost(self.Weave(ctr16, tff), mode=6, interlaced=True, u=U, v=V)
                    else:
                        return self.core.avs.DitherPost(ctr16, mode=mode, u=Uin, v=Vin)
                elif interlaced:
                    if ifC:
                        return self.Weave(self.ContraSharpeningHD(output, CClip, HD=if4), tff)
                    else:
                        return self.Weave(self.LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soft=0, soothe=False, defaults='slow'), tff)
                elif ifC:
                    return self.ContraSharpeningHD(output, CClip, HD=if4)
                else:
                    return self.LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soft=0, soothe=False, defaults='slow')
            elif lsb:
                if interlaced:
                    return self.core.avs.DitherPost(self.Weave(output, tff), mode=6, interlaced=True, u=U, v=V)
                else:
                    return self.core.avs.DitherPost(output, mode=mode, u=Uin, v=Vin)
            elif interlaced:
                return self.Weave(output, tff)
            else:
                return output
        else:
            return input
    
    
    #------------------------------------------------------------------------------#
    #                                                                              #
    #                         InterFrame 2.5.0 by SubJunk                          #
    #                                                                              #
    #         A frame interpolation script that makes accurate estimations         #
    #                   about the content of non-existent frames                   #
    #      Its main use is to give videos higher framerates like newer TVs do      #
    #------------------------------------------------------------------------------#
    #
    # For instructions and further information see the included InterFrame.html
    # For news go to spirton.com
    def InterFrame(self, Input, Preset='Medium', Tuning='Film', NewNum=0, NewDen=0, GPU=False, InputType='2D', OverrideAlgo=0, OverrideArea=0):
        # Validate inputs
        if not isinstance(Input, vs.VideoNode) or Input.format.id != vs.YUV420P8:
            raise ValueError('InterFrame: This is not a YUV420P8 clip !')
        if not isinstance(Preset, str):
            raise ValueError("InterFrame: 'Preset' must be string")
        Preset = Preset.lower()
        if Preset not in ('medium', 'fast', 'faster'):
            raise ValueError("InterFrame: '"+Preset+"' is not a valid preset. Please check the documentation for a list of the valid presets.")
        if not isinstance(Tuning, str):
            raise ValueError("InterFrame: 'Tuning' must be string")
        Tuning = Tuning.lower()
        if Tuning not in ('film', 'smooth', 'animation', 'weak'):
            raise ValueError("InterFrame: '"+Tuning+"' is not a valid tuning. Please check the documentation for a list of the valid tunings.")
        if not isinstance(NewNum, int) or NewNum < 0:
            raise ValueError("InterFrame: 'NewNum' have not a correct value! [>=0]")
        if not isinstance(NewDen, int) or NewDen < 0:
            raise ValueError("InterFrame: 'NewDen' have not a correct value! [>=0]")
        if not isinstance(GPU, bool):
            raise ValueError("InterFrame: 'GPU' must be bool")
        if not isinstance(InputType, str):
            raise ValueError("InterFrame: 'InputType' must be string")
        InputType = InputType.lower()
        if InputType not in ('2d', 'sbs', 'ou', 'hsbs', 'hou'):
            raise ValueError("InterFrame: '"+InputType+"' is not a valid InputType. Please check the documentation for a list of the valid InputTypes.")
        if not isinstance(OverrideAlgo, int) or OverrideAlgo < 0:
            raise ValueError("InterFrame: 'OverrideAlgo' have not a correct value! [>=0]")
        if not isinstance(OverrideArea, int) or OverrideArea < 0:
            raise ValueError("InterFrame: 'OverrideArea' have not a correct value! [>=0]")
        
        # Convert integers to strings
        NewNum = repr(NewNum)
        NewDen = repr(NewDen)
        OverrideAlgo = repr(OverrideAlgo)
        OverrideArea = repr(OverrideArea)
        
        # Create SuperString
        SuperString = '{pel:1,' if Preset in ('fast', 'faster') else '{'
        
        SuperString += 'scale:{up:0,down:4},gpu:1,rc:false}' if GPU else 'scale:{up:2,down:4},gpu:0,rc:false}'
        
        # Create VectorsString
        if Tuning == 'animation':
            VectorsString = '{block:{w:32,'
        elif Preset in ('fast', 'faster') or not GPU:
            VectorsString = '{block:{w:16,'
        else:
            VectorsString = '{block:{w:8,'
        
        if Tuning == 'animation':
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
        
        if Tuning == 'animation' or Preset == 'faster':
            VectorsString += 'bad:{sad:2000}}}}}'
        elif Tuning == 'weak':
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250,search:{distance:-1,satd:true}}]}'
        else:
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250}]}'
        
        # Create SmoothString
        if NewNum != '0':
            SmoothString = '{rate:{num:'+NewNum+',den:'+NewDen+',abs:true},'
        elif Input.fps_num / Input.fps_den in (15, 25, 30):
            SmoothString = '{rate:{num:2,den:1,abs:false},'
        else:
            SmoothString = '{rate:{num:60000,den:1001,abs:true},'
        
        if OverrideAlgo != '0':
            SmoothString += 'algo:'+OverrideAlgo+',mask:{cover:80,'
        elif Tuning == 'animation':
            SmoothString += 'algo:2,mask:{'
        elif Tuning == 'smooth':
            SmoothString += 'algo:23,mask:{'
        else:
            SmoothString += 'algo:13,mask:{cover:80,'
        
        if OverrideArea != '0':
            SmoothString += 'area:'+OverrideArea
        elif Tuning == 'smooth':
            SmoothString += 'area:150'
        else:
            SmoothString += 'area:0'
        
        if Tuning == 'weak':
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0,limits:{blocks:50}}}'
        else:
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0}}'
        
        def InterFrameProcess(Input):
            # Make interpolation vector clip
            Super = self.core.avs.SVSuper(Input, SuperString)
            Vectors = self.core.avs.SVAnalyse(Super, VectorsString)
            
            # Put it together
            return self.core.avs.SVSmoothFps(Input, Super, Vectors, SmoothString, url='www.svp-team.com', mt=1)
        
        # Get either 1 or 2 clips depending on InputType
        w = Input.width
        h = Input.height
        w2 = int(w / 2)
        h2 = int(h / 2)
        if InputType == 'sbs':
            FirstEye = InterFrameProcess(self.core.std.CropRel(Input, left=0, top=0, right=w2, bottom=0))
            SecondEye = InterFrameProcess(self.core.std.CropRel(Input, left=w2, top=0, right=0, bottom=0))
            return self.core.std.StackHorizontal([FirstEye, SecondEye])
        elif InputType == 'ou':
            FirstEye = InterFrameProcess(self.core.std.CropRel(Input, left=0, top=0, right=0, bottom=h2))
            SecondEye = InterFrameProcess(self.core.std.CropRel(Input, left=0, top=h2, right=0, bottom=0))
            return self.core.std.StackHorizontal([FirstEye, SecondEye])
        elif InputType == 'hsbs':
            FirstEye = InterFrameProcess(self.Resize(self.core.std.CropRel(Input, left=0, top=0, right=w2, bottom=0), w, h))
            SecondEye = InterFrameProcess(self.Resize(self.core.std.CropRel(Input, left=w2, top=0, right=0, bottom=0), w, h))
            return self.core.std.StackHorizontal([self.Resize(FirstEye, w2, h), self.Resize(SecondEye, w2, h)])
        elif InputType == 'hou':
            FirstEye = InterFrameProcess(self.Resize(self.core.std.CropRel(Input, left=0, top=0, right=0, bottom=h2), w, h))
            SecondEye = InterFrameProcess(self.Resize(self.core.std.CropRel(Input, left=0, top=h2, right=0, bottom=0), w, h))
            return self.core.std.StackHorizontal([self.Resize(FirstEye, w, h2), self.Resize(SecondEye, w, h2)])
        else:
            return InterFrameProcess(Input)
    
    
    ##############################
    # FastLineDarken 1.4x MT MOD #
    ##############################
    #
    # Written by Vectrangle    (http://forum.doom9.org/showthread.php?t=82125)
    # Didée: - Speed Boost, Updated: 11th May 2007
    # Dogway - added protection option. 12-May-2011
    #
    #  * requires YUV input
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
    #
    # Changelog:
    #  1.4  - added protection option. Prevents darkest lines to be over darkened thus creating artifacts (i.e. aliasing, clipping...)
    #       - Optmized the code as suggested by Didée for possible faster processing. It also deals with the green screen bug.
    #  1.3  - added ability to thin lines, now runs much slower unless thinning=0. Changed the defaults (again)
    #  1.2  - huge speed increase using yv12lutxy =)
    #       - weird darkening issues gone (they were caused by yv12layer)
    #       - show option no longer available due to optimizations. Use subtract() instead
    #  1.1  - added luma_cap option
    #  1.0  - initial release
    def FastLineDarkenMOD(self, c, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0):
        if not isinstance(c, vs.VideoNode) or c.format.color_family != vs.YUV:
            raise ValueError('FastLineDarkenMOD: This is not a YUV clip !')
        if not isinstance(strength, int) or strength < 0 or strength > 256:
            raise ValueError("FastLineDarkenMOD: 'strength' have not a correct value! [0...256]")
        if not isinstance(protection, int) or protection < 0 or protection > 50:
            raise ValueError("FastLineDarkenMOD: 'protection' have not a correct value! [0...50]")
        if not isinstance(luma_cap, int) or luma_cap < 0 or luma_cap > 255:
            raise ValueError("FastLineDarkenMOD: 'luma_cap' have not a correct value! [0...255]")
        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError("FastLineDarkenMOD: 'threshold' have not a correct value! [>=0]")
        if not isinstance(thinning, int) or thinning < 0 or thinning > 256:
            raise ValueError("FastLineDarkenMOD: 'thinning' have not a correct value! [0...256]")
        
        shift = c.format.bits_per_sample - 8
        peak = (1 << c.format.bits_per_sample) - 1
        
        Str = repr(strength / 128)
        lum = repr(peak if luma_cap == 255 else luma_cap << shift)
        thr = repr(threshold << shift)
        thn = repr(thinning / 16)
        
        c_src = c
        c = self.core.std.ShufflePlanes(c, planes=[0], colorfamily=vs.GRAY)
        exin = self.core.generic.Minimum(self.core.generic.Maximum(c, threshold=int(peak/(protection+1))))
        thick = self.core.std.Expr([c, exin], 'y '+lum+' < y '+lum+' ? x '+thr+' + > x y '+lum+' < y '+lum+' ? - 0 ? '+Str+' * x +')
        
        if thinning == 0:
            last = thick
        else:
            diff = self.core.std.Expr([c, exin], 'y '+lum+' < y '+lum+' ? x '+thr+' + > x y '+lum+' < y '+lum+' ? - 0 ? '+repr(127<<shift)+' +')
            linemask = self.core.rgvs.RemoveGrain(
              self.core.std.Expr(self.core.generic.Minimum(diff), 'x '+repr(127<<shift)+' - '+thn+' * '+repr(255<<shift)+' +'), 20)
            thin = self.core.std.Expr([self.core.generic.Maximum(c), diff], 'x y '+repr(127<<shift)+' - '+Str+' 1 + * +')
            last = self.core.std.MaskedMerge(thin, thick, linemask)
        return self.core.std.ShufflePlanes([last, c_src], planes=[0, 1, 2], colorfamily=vs.YUV)
    
    
    def Resize(self, src, w, h, sx=.0, sy=.0, sw=.0, sh=.0, kernel='spline36', taps=4, a1=None, a2=None, css=None, planes=[3, 3, 3], cplace='mpeg2',
               cplaces=None, cplaced=None, interlaced=2, interlacedd=2, flt=False, noring=False, bits=None, fulls=None, fulld=None, dmode=3, ampo=1., ampn=.0):
        if not isinstance(src, vs.VideoNode):
            raise ValueError('Resize: This is not a clip !')
        if not isinstance(w, int) or w <= 0:
            raise ValueError('Resize: width must be > 0')
        if not isinstance(h, int) or h <= 0:
            raise ValueError('Resize: height must be > 0')
        if not isinstance(kernel, str):
            raise ValueError("Resize: 'kernel' must be string")
        kernel = kernel.lower()
        if bits is None:
            bits = src.format.bits_per_sample
        
        sr_h = w / src.width
        sr_v = h / src.height
        sr_up = max(sr_h, sr_v)
        sr_dw = 1 / min(sr_h, sr_v)
        sr = max(sr_up, sr_dw)
        assert(sr >= 1)
        
        # Depending on the scale ratio, we may blend or totally disable the ringing cancellation
        thr = 2.5
        nrb = sr > thr
        nrf = sr < thr + 1 and noring
        if nrb:
            nrr = min(sr-thr, 1)
            nrv = round((1 - nrr) * 255)
            nrv = nrv * 256 + nrv
        
        if kernel in ('cubic', 'bicubic'):
            if a1 is None:
                a1 = 1 / 3
            if a2 is None:
                a2 = 1 / 3
        elif kernel in ('gauss', 'gaussian'):
            if a1 is None:
                a1 = 30
        main = self.core.fmtc.resample(src, w, h, sx, sy, sw, sh, kernel=kernel, taps=taps, a1=a1, a2=a2, css=css, planes=planes, cplace=cplace,
                                       cplaces=cplaces, cplaced=cplaced, interlaced=interlaced, interlacedd=interlacedd, flt=flt)
        
        if nrf:
            nrng = self.core.fmtc.resample(src, w, h, sx, sy, sw, sh, kernel='gauss', taps=taps, a1=100, css=css, planes=planes, cplace=cplace,
                                           cplaces=cplaces, cplaced=cplaced, interlaced=interlaced, interlacedd=interlacedd, flt=flt)
            
            # To do: use a simple frame blending instead of Merge
            last = self.core.rgvs.Repair(main, nrng, 1)
            if nrb:
                nrm = self.core.std.BlankClip(main, color=[nrv, nrv, nrv] if main.format.num_planes==3 else nrv)
                last = self.core.std.MaskedMerge(main, last, nrm)
        else:
            last = main
        
        if bits < 16:
            return self.core.fmtc.bitdepth(last, bits=bits, fulls=fulls, fulld=fulld, dmode=dmode, ampo=ampo, ampn=ampn)
        else:
            return last
    
    
    # Converts the luma channel to linear light
    def GammaToLinear(self, src, fulls=True, fulld=True, curve='709', planes=[0, 1, 2], gcor=1., sigmoid=False, thr=.5, cont=6.5):
        if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
            raise ValueError('GammaToLinear: This is not a 16-bit clip !')
        if not isinstance(fulls, bool):
            raise ValueError("GammaToLinear: 'fulls' must be bool")
        if not isinstance(fulld, bool):
            raise ValueError("GammaToLinear: 'fulld' must be bool")
        if not isinstance(curve, str):
            raise ValueError("GammaToLinear: 'curve' must be string")
        if not (isinstance(planes, list) or isinstance(planes, int)):
            raise ValueError("GammaToLinear: 'planes' must be list or integer")
        if not (isinstance(gcor, float) or isinstance(gcor, int)):
            raise ValueError("GammaToLinear: 'gcor' must be float or integer")
        if not isinstance(sigmoid, bool):
            raise ValueError("GammaToLinear: 'sigmoid' must be bool")
        if not (isinstance(thr, float) or isinstance(thr, int)) or thr < 0 or thr > 1:
            raise ValueError("GammaToLinear: 'thr' have not a correct value! [0.0...1.0]")
        if not (isinstance(cont, float) or isinstance(cont, int)) or cont <= 0:
            raise ValueError("GammaToLinear: 'cont' have not a correct value! [>0.0]")
        
        return self.LinearAndGamma(src, False, fulls, fulld, curve.lower(), planes, gcor, sigmoid, thr, cont)
    
    # Converts back a clip to gamma-corrected luma
    def LinearToGamma(self, src, fulls=True, fulld=True, curve='709', planes=[0, 1, 2], gcor=1., sigmoid=False, thr=.5, cont=6.5):
        if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
            raise ValueError('LinearToGamma: This is not a 16-bit clip !')
        if not isinstance(fulls, bool):
            raise ValueError("LinearToGamma: 'fulls' must be bool")
        if not isinstance(fulld, bool):
            raise ValueError("LinearToGamma: 'fulld' must be bool")
        if not isinstance(curve, str):
            raise ValueError("LinearToGamma: 'curve' must be string")
        if not (isinstance(planes, list) or isinstance(planes, int)):
            raise ValueError("LinearToGamma: 'planes' must be list or integer")
        if not (isinstance(gcor, float) or isinstance(gcor, int)):
            raise ValueError("LinearToGamma: 'gcor' must be float or integer")
        if not isinstance(sigmoid, bool):
            raise ValueError("LinearToGamma: 'sigmoid' must be bool")
        if not (isinstance(thr, float) or isinstance(thr, int)) or thr < 0 or thr > 1:
            raise ValueError("LinearToGamma: 'thr' have not a correct value! [0.0...1.0]")
        if not (isinstance(cont, float) or isinstance(cont, int)) or cont <= 0:
            raise ValueError("LinearToGamma: 'cont' have not a correct value! [>0.0]")
        
        return self.LinearAndGamma(src, True, fulls, fulld, curve.lower(), planes, gcor, sigmoid, thr, cont)
    
    def LinearAndGamma(self, src, l2g_flag, fulls, fulld, curve, planes, gcor, sigmoid, thr, cont):
        if curve == 'srgb':
            c_num = 0
        elif curve in ('709', '601', '170'):
            c_num = 1
        elif curve == '240':
            c_num = 2
        elif curve == '2020':
            c_num = 3
        else:
            raise ValueError('LinearAndGamma: wrong curve value.')
        if src.format.num_planes == 1:
            planes = [0]
        
        #                 BT-709/601
        #        sRGB     SMPTE 170M   SMPTE 240M   BT-2020
        k0    = (0.04045, 0.081,       0.0912,      0.08145)[c_num]
        phi   = (12.92,   4.5,         4.0,         4.5)[c_num]
        alpha = (0.055,   0.099,       0.1115,      0.0993)[c_num]
        gamma = (2.4,     2.22222,     2.22222,     2.22222)[c_num]
        
        def g2l(x):
            expr = x / 65536 if fulls else (x - 4096) / 56064
            if expr <= k0:
                expr /= phi
            else:
                expr = ((expr + alpha) / (1 + alpha)) ** gamma
            if gcor != 1 and expr >= 0:
                expr **= gcor
            if sigmoid:
                x0 = 1 / (1 + math.exp(cont * thr))
                x1 = 1 / (1 + math.exp(cont * (thr - 1)))
                x1m0 = x1 - x0
                expr = thr - math.log(max(1 / max(expr * x1m0 + x0, 0.000001) - 1, 0.000001)) / cont
            if fulld:
                return max(min(round(expr * 65536), 65535), 0)
            else:
                return max(min(round(expr * 56064 + 4096), 65535), 0)
        
        # E' = (E <= k0 / phi)   ?   E * phi   :   (E ^ (1 / gamma)) * (alpha + 1) - alpha
        def l2g(x):
            expr = x / 65536 if fulls else (x - 4096) / 56064
            if sigmoid:
                x0 = 1 / (1 + math.exp(cont * thr))
                x1 = 1 / (1 + math.exp(cont * (thr - 1)))
                x1m0 = x1 - x0
                expr = (1 / (1 + math.exp(cont * (thr - expr))) - x0) / x1m0
            if gcor != 1 and expr >= 0:
                expr **= gcor
            if expr <= k0 / phi:
                expr *= phi
            else:
                expr = expr ** (1 / gamma) * (alpha + 1) - alpha
            if fulld:
                return max(min(round(expr * 65536), 65535), 0)
            else:
                return max(min(round(expr * 56064 + 4096), 65535), 0)
        
        return self.core.std.Lut(src, planes=planes, function=l2g if l2g_flag else g2l)
    
    
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
    ### +-----------+
    ### | CHANGELOG |
    ### +-----------+
    ###
    ### v1.9 : - tweaked settings
    ###        - default preset is now defaults="fast" /!\
    ###
    ### v1.8 : - changed preblur to allow more tweaking (bool->string)
    ###        - tweaked settings
    ###        - cleaned the code
    ###        - updated documentation
    ###
    ### v1.7 : - changed Smethod=4 to "source"
    ###
    ### v1.6 : - added preblur option
    ###        - added new Smethod=4
    ###
    ### v1.5 : - fixed LUT expression (thanks to Didée)
    ###        - changed Smethod to Smethod+secure
    ###
    ### v1.4 : - changed defaults="new" to defaults="slow" & defaults="fast"
    ###        - added show parameter
    ###        - cleaned a little the code
    ###
    ### v1.3 : - changed a little Smethod=3&5 (same effect, but more precise)
    ###        - added new calculation for soft (soft=-2) [default on]
    ###        - added warning about bad settings (no more silent)
    ###        - updated the documentation
    ###
    ### v1.2 : - added new Lmode<0 (limit with repair)
    ###        - added 2 new Smode (unsharp masking)
    ###        - changed Smode order: now old Smode3-4 is new Smode3-4 to avoid mistake
    ###
    ### v1.1 : - fixed a bug with dest_x!=ox or dest_y!=oy
    ###        - replaced Lfactor by over/undershoot2
    ###
    ### v1.0 : - deleted old Smode(1-4), added new Smode(1-3) & Smethod(1-5)
    ###        - added parameters for nonlinear sharpening (S2zp,S2pwr,S2dmpLo,S2dmpHi)
    ###        - corrected the nonlinear formula
    ###        - added new Lmode 2 & 4 + fixed Lmode 0
    ###        - added faster edgemask
    ###        - added soothe temporal stabilization, 2 parameters: soothe & keep
    ###        - replaced lanczosresize by spline36resize
    ###        - moved "strength" parameter (first place)
    ###        - deleted wide, special and exborder
    ###        - changed some code (cosmetic)
    ###        - added "defaults" parameter (to switch between original and modded version)
    ###        - added documentation
    ###
    ###
    ###
    ### +--------------+
    ### | DEPENDENCIES |
    ### +--------------+
    ###
    ### -> GenericFilters
    ### -> RemoveGrain/Repair
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
    ### Smode [int: 1,2]
    ### ----------------------
    ### Sharpen mode:
    ###    =1 : Range sharpening
    ###    =2 : Nonlinear sharpening (corrected version)
    ###
    ### Smethod [int: 1,2,3]
    ### --------------------
    ### Sharpen method:
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
    ### preblur [string: "ON","OFF",...]
    ### --------------------------------
    ### Mode to avoid noise sharpening & ringing
    ### "ON" is sufficient to prevent ringing, but to prevent noise sharpening you should set your own denoiser (the first argument of the denoiser must be 'tmp')
    ###    Usage:   LSFmod(preblur="self.core.namespace.YourFavoriteDenoiser()")
    ###    Example: LSFmod(preblur="self.core.avs.FFT3DFilter(tmp,sigma=4.0,plane=0)")
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
    ###    Example: last.FFT3Dfilter().LSFmod(source=last,Lmode=0,soft=0)
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
    ### noring [bool]
    ### -------------
    ### In case of supersampling, indicates that a non-ringing algorithm must be used.
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
    ###                   - preblur     = "OFF"
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
    ###                   - edgemaskHQ  = True
    ###
    ###                   - ss_x        = Smode==1?1.50:1.25
    ###                   - ss_y        = ss_x
    ###                   - noring      = false
    ###                   - dest_x      = ox
    ###                   - dest_y      = oy
    ###
    ###
    ### defaults="slow" : - strength    = 100
    ### ----------------- - Smode       = 2
    ###                   - Smethod     = 3
    ###                   - kernel      = 11
    ###
    ###                   - preblur     = "OFF"
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
    ###                   - ss_x        = 1.50
    ###                   - ss_y        = ss_x
    ###                   - noring      = false
    ###                   - dest_x      = ox
    ###                   - dest_y      = oy
    ###
    ###
    ### defaults="fast" : - strength    = 100
    ### ----------------- - Smode       = 1
    ###                   - Smethod     = 2
    ###                   - kernel      = 11
    ###
    ###                   - preblur     = "OFF"
    ###                   - secure      = true
    ###                   - source      = undefined
    ###
    ###                   - Szrp        = 16
    ###                   - Spwr        = 4
    ###                   - SdmpLo      = 4
    ###                   - SdmpHi      = 48
    ###
    ###                   - Lmode       = 1
    ###                   - overshoot   = strength/100
    ###                   - undershoot  = overshoot
    ###                   - overshoot2  = overshoot*2
    ###                   - undershoot2 = overshoot2
    ###
    ###                   - soft        = 0
    ###                   - soothe      = true
    ###                   - keep        = 20
    ###
    ###                   - edgemode    = 0
    ###                   - edgemaskHQ  = false
    ###
    ###                   - ss_x        = 1.25
    ###                   - ss_y        = ss_x
    ###                   - noring      = false
    ###                   - dest_x      = ox
    ###                   - dest_y      = oy
    ###
    ################################################################################################
    def LSFmod(self, input, strength=100, Smode=None, Smethod=None, kernel=11, preblur='OFF', secure=None, source=None,
               Szrp=16, Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None, overshoot2=None, undershoot2=None,
               soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ss_x=None, ss_y=None, noring=False, dest_x=None, dest_y=None, defaults='fast'):
        if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV:
            raise ValueError('LSFmod: This is not a YUV clip !')
        
        ### DEFAULTS
        version = 'v1.9'
        if not isinstance(defaults, str):
            raise ValueError("LSFmod: 'defaults' must be string")
        try:
            num = ('old', 'slow', 'fast').index(defaults.lower())
        except:
            raise ValueError('LSFmod: Defaults must be "old" or "slow" or "fast" !')
        
        ox = input.width
        oy = input.height
        
        if Smode is None:
            Smode = (1, 2, 1)[num]
        if Smethod is None:
            Smethod = (2 if Smode==1 else 1, 3, 2)[num]
        if secure is None:
            secure = (False, True, True)[num]
        if Spwr is None:
            Spwr = (2, 4, 4)[num]
        if SdmpLo is None:
            SdmpLo = (int(strength/25), 4, 4)[num]
        if SdmpHi is None:
            SdmpHi = (0, 48, 48)[num]
        if Lmode is None:
            Lmode = (1, 4, 1)[num]
        if overshoot is None:
            overshoot = (1, int(strength/100), int(strength/100))[num]
        if undershoot is None:
            undershoot = overshoot
        if overshoot2 is None:
            overshoot2 = overshoot * 2
        if undershoot2 is None:
            undershoot2 = overshoot2
        if soft is None:
            soft = (0, -2, 0)[num]
        if soothe is None:
            soothe = (False, True, True)[num]
        if keep is None:
            keep = (25, 20, 20)[num]
        if edgemaskHQ is None:
            edgemaskHQ = (True, True, False)[num]
        if ss_x is None:
            ss_x = (1.5 if Smode==1 else 1.25, 1.5, 1.25)[num]
        if ss_y is None:
            ss_y = ss_x
        if dest_x is None:
            dest_x = ox
        if dest_y is None:
            dest_y = oy
        
        if not isinstance(strength, int) or strength < 0:
            raise ValueError("LSFmod: 'strength' have not a correct value! [>=0]")
        if not isinstance(Smode, int) or Smode < 1 or Smode > 2:
            raise ValueError("LSFmod: 'Smode' have not a correct value! [1,2]")
        if not isinstance(Smethod, int) or Smethod < 1 or Smethod > 3:
            raise ValueError("LSFmod: 'Smethod' have not a correct value! [1,2,3]")
        if not isinstance(kernel, int) or kernel not in (11, 12, 19, 20):
            raise ValueError("LSFmod: 'kernel' have not a correct value! [11,12,19,20]")
        if not isinstance(preblur, str):
            raise ValueError("LSFmod: 'preblur' must be string")
        preblur = preblur.lower()
        if not isinstance(secure, bool):
            raise ValueError("LSFmod: 'secure' must be bool")
        if source is not None and (not isinstance(source, vs.VideoNode) or input.format.id != source.format.id):
            raise ValueError("LSFmod: 'source' must be the same format as input !")
        if not isinstance(Szrp, int) or Szrp < 1 or Szrp > 255:
            raise ValueError("LSFmod: 'Szrp' have not a correct value! [1...255]")
        if not isinstance(Spwr, int) or Spwr < 1:
            raise ValueError("LSFmod: 'Spwr' have not a correct value! [>=1]")
        if not isinstance(SdmpLo, int) or SdmpLo < 0 or SdmpLo > 255:
            raise ValueError("LSFmod: 'SdmpLo' have not a correct value! [0...255]")
        if not isinstance(SdmpHi, int) or SdmpHi < 0 or SdmpHi > 255:
            raise ValueError("LSFmod: 'SdmpHi' have not a correct value! [0...255]")
        if not isinstance(Lmode, int) or Lmode > 4:
            raise ValueError("LSFmod: 'Lmode' have not a correct value! [...-1,0,1,2,3,4]")
        if not isinstance(overshoot, int) or overshoot < 0 or overshoot > 255:
            raise ValueError("LSFmod: 'overshoot' have not a correct value! [0...255]")
        if not isinstance(undershoot, int) or undershoot < 0 or undershoot > 255:
            raise ValueError("LSFmod: 'undershoot' have not a correct value! [0...255]")
        if not isinstance(overshoot2, int) or overshoot2 < 0 or overshoot2 > 255:
            raise ValueError("LSFmod: 'overshoot2' have not a correct value! [0...255]")
        if not isinstance(undershoot2, int) or undershoot2 < 0 or undershoot2 > 255:
            raise ValueError("LSFmod: 'undershoot2' have not a correct value! [0...255]")
        if not isinstance(soft, int) or soft < -2 or soft > 100:
            raise ValueError("LSFmod: 'soft' have not a correct value! [-2,-1,0,1...100]")
        if not isinstance(soothe, bool):
            raise ValueError("LSFmod: 'soothe' must be bool")
        if not isinstance(keep, int) or keep < 0 or keep > 100:
            raise ValueError("LSFmod: 'keep' have not a correct value! [0...100]")
        if not isinstance(edgemode, int) or edgemode < -1 or edgemode > 2:
            raise ValueError("LSFmod: 'edgemode' have not a correct value! [-1,0,1,2]")
        if not isinstance(edgemaskHQ, bool):
            raise ValueError("LSFmod: 'edgemaskHQ' must be bool")
        if not (isinstance(ss_x, float) or isinstance(ss_x, int)) or ss_x < 1:
            raise ValueError("LSFmod: 'ss_x' have not a correct value! [>=1.0]")
        if not (isinstance(ss_y, float) or isinstance(ss_y, int)) or ss_y < 1:
            raise ValueError("LSFmod: 'ss_y' have not a correct value! [>=1.0]")
        if not isinstance(noring, bool):
            raise ValueError("LSFmod: 'noring' must be bool")
        if not isinstance(dest_x, int):
            raise ValueError("LSFmod: 'dest_x' must be integer")
        if not isinstance(dest_y, int):
            raise ValueError("LSFmod: 'dest_y' must be integer")
        
        shift = input.format.bits_per_sample - 8
        neutral = repr(128 << shift)
        peak = (1 << input.format.bits_per_sample) - 1
        
        if soft == -1:
            soft = min(math.sqrt(((ss_x + ss_y) / 2 - 1) * 100) * 10, 100)
        elif soft == -2:
            soft = min(int((1 + (2 / (ss_x + ss_y))) * math.sqrt(strength)), 100)
        
        xxs = round(ox * ss_x / 8) * 8
        yys = round(oy * ss_y / 8) * 8
        
        Str = strength / 100
        
        # x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?
        def get_lut1(x, y):
            if (x - 128) * (y - 128) < 0:
                return 128
            elif abs(x - 128) < abs(y - 128):
                return x
            else:
                return y
        # x y == x x x y - abs Szrp / 1 Spwr / ^ Szrp * str * x y - x y - abs / * x y - 2 ^ Szrp 2 ^ SdmpLo + * x y - 2 ^ SdmpLo + Szrp 2 ^ * / * 1 SdmpHi 0 == 0 Szrp SdmpHi / 4 ^ ? + 1 SdmpHi 0 == 0 x y - abs SdmpHi / 4 ^ ? + / * + ?
        def get_lut2(x):
            neu = 128 << shift
            mul = 2 ** shift
            if x == neu:
                return x
            else:
                tmp1 = (x - neu) / mul
                tmp2 = tmp1 ** 2
                tmp3 = Szrp ** 2
                return max(min(round(x + (abs(tmp1) / Szrp) ** (1 / Spwr) * Szrp * (Str * mul) * (1 if x > neu else -1) * (tmp2 * (tmp3 + SdmpLo) / ((tmp2 + SdmpLo) * tmp3)) * ((1 + (0 if SdmpHi == 0 else (Szrp / SdmpHi) ** 4)) / (1 + (0 if SdmpHi == 0 else (abs(tmp1) / SdmpHi) ** 4)))), peak), 0)
        # x 128 / 0.86 ^ 255 *
        def get_lut3(x):
            return min(round((x / (128 << shift)) ** .86 * peak), peak)
        # x 32 / 0.86 ^ 255 *
        def get_lut4(x):
            return min(round((x / (32 << shift)) ** .86 * peak), peak)
        # x 128 - abs y 128 - abs > y soft * x (100-soft) * + 100 / x ?
        def get_lut5(x, y):
            if abs(x - 128) > abs(y - 128):
                return round((y * soft + x * (100 - soft)) / 100)
            else:
                return x
        # x 128 - y 128 - * 0 < x 128 - 100 / keep * 128 + x 128 - abs y 128 - abs > x keep * y 100 keep - * + 100 / x ? ?
        def get_lut6(x, y):
            if (x - 128) * (y - 128) < 0:
                return round((x - 128) / 100 * keep + 128)
            elif abs(x - 128) > abs(y - 128):
                return round((x * keep + y * (100 - keep)) / 100)
            else:
                return x
        
        ### SHARP
        if ss_x > 1 or ss_y > 1:
            tmp = self.Resize(input, xxs, yys, kernel='spline64' if noring else 'spline36', noring=noring)
        else:
            tmp = input
        tmp_src = tmp
        tmp = self.core.std.ShufflePlanes(tmp, planes=[0], colorfamily=vs.GRAY)
        if preblur == 'off':
            pre = tmp
        elif preblur == 'on':
            diff1 = self.core.std.MakeDiff(tmp, self.core.rgvs.RemoveGrain(tmp, 11))
            diff2 = self.core.std.MakeDiff(tmp, self.core.rgvs.RemoveGrain(tmp, 4))
            if shift > 0:
                clip2 = self.core.std.Expr([diff1, diff2], 'x '+neutral+' - y '+neutral+' - * 0 < '+neutral+' x '+neutral+' - abs y '+neutral+' - abs < x y ? ?')
            else:
                clip2 = self.core.std.Lut2(diff1, diff2, function=get_lut1)
            pre = self.core.std.MakeDiff(tmp, clip2)
        else:
            pre = eval(preblur)
        
        dark_limit = self.core.generic.Minimum(pre)
        bright_limit = self.core.generic.Maximum(pre)
        minmaxavg = self.core.std.Merge(dark_limit, bright_limit)
        
        if Smethod == 1:
            method = self.core.rgvs.RemoveGrain(pre, kernel)
        elif Smethod == 2:
            method = minmaxavg
        else:
            method = self.core.rgvs.RemoveGrain(minmaxavg, kernel)
        
        if secure:
            method = self.core.std.Expr([method, pre], 'x y < x '+repr(1<<shift)+' + x y > x '+repr(1<<shift)+' - x ? ?')
        
        if preblur != 'off':
            method = self.core.std.MakeDiff(tmp, self.core.std.MakeDiff(pre, method))
        
        if Smode == 1:
            normsharp = self.core.std.Expr([tmp, method], 'x x y - '+repr(Str)+' * +')
        else:
            sharpdiff = self.core.std.MakeDiff(tmp, method)
            sharpdiff = self.core.std.Lut(sharpdiff, function=get_lut2)
            normsharp = self.core.std.MergeDiff(method, sharpdiff)
        
        ### LIMIT
        normal = self.Clamp(normsharp, bright_limit, dark_limit, overshoot, undershoot)
        second = self.Clamp(normsharp, bright_limit, dark_limit, overshoot2, undershoot2)
        zero = self.Clamp(normsharp, bright_limit, dark_limit, 0, 0)
        
        if edgemaskHQ:
            edge = self.core.std.Lut(self.Logic(self.core.generic.Convolution(tmp, [8, 16, 8, 0, 0, 0, -8, -16, -8], divisor=4, saturate=False),
                                                self.core.generic.Convolution(tmp, [8, 0, -8, 16, 0, -16, 8, 0, -8], divisor=4, saturate=False),
                                                'max'),
                                     function=get_lut3)
        else:
            edge = self.core.std.Lut(self.core.generic.Sobel(tmp, rshift=2), function=get_lut4)
        
        if Lmode < 0:
            limit1 = self.core.rgvs.Repair(normsharp, tmp, abs(Lmode))
        elif Lmode == 0:
            limit1 = normsharp
        elif Lmode == 1:
            limit1 = normal
        elif Lmode == 2:
            limit1 = self.core.std.MaskedMerge(normsharp, normal, self.core.generic.Inflate(edge))
        elif Lmode == 3:
            limit1 = self.core.std.MaskedMerge(normal, zero, self.core.generic.Inflate(edge))
        else:
            limit1 = self.core.std.MaskedMerge(second, normal, self.core.generic.Inflate(edge))
        
        if edgemode == 0:
            limit2 = limit1
        elif edgemode == 1:
            limit2 = self.core.std.MaskedMerge(tmp, limit1, self.core.rgvs.RemoveGrain(self.core.generic.Inflate(self.core.generic.Inflate(edge)), 11))
        else:
            limit2 = self.core.std.MaskedMerge(limit1, tmp, self.core.rgvs.RemoveGrain(self.core.generic.Inflate(self.core.generic.Inflate(edge)), 11))
        
        ### SOFT
        if soft == 0:
            PP1 = limit2
        else:
            sharpdiff = self.core.std.MakeDiff(tmp, limit2)
            sharpdiff2 = self.core.rgvs.RemoveGrain(sharpdiff, 19)
            if shift > 0:
                sharpdiff3 = self.core.std.Expr([sharpdiff, sharpdiff2],
                                                'x '+neutral+' - abs y '+neutral+' - abs > y '+repr(soft)+' * x '+repr(100-soft)+' * + 100 / x ?')
            else:
                sharpdiff3 = self.core.std.Lut2(sharpdiff, sharpdiff2, function=get_lut5)
            PP1 = self.core.std.MakeDiff(tmp, sharpdiff3)
        
        ### SOOTHE
        if soothe:
            diff = self.core.std.MakeDiff(tmp, PP1)
            diff2 = self.TemporalSoften(diff, 1, 255<<shift, 0, 32<<shift, 2)
            if shift > 0:
                diff3 = self.core.std.Expr([diff, diff2], 'x '+neutral+' - y '+neutral+' - * 0 < x '+neutral+' - 100 / '+repr(keep)+' * '+neutral+' + x '+neutral+' - abs y '+neutral+' - abs > x '+repr(keep)+' * y 100 '+repr(keep)+' - * + 100 / x ? ?')
            else:
                diff3 = self.core.std.Lut2(diff, diff2, function=get_lut6)
            PP2 = self.core.std.MakeDiff(tmp, diff3)
        else:
            PP2 = PP1
        
        ### OUTPUT
        if dest_x != ox or dest_y != oy:
            out = self.Resize(self.core.std.ShufflePlanes([PP2, tmp_src], planes=[0, 1, 2], colorfamily=vs.YUV), dest_x, dest_y)
        elif ss_x > 1 or ss_y > 1:
            out = self.core.std.ShufflePlanes([self.Resize(PP2, dest_x, dest_y), input], planes=[0, 1, 2], colorfamily=vs.YUV)
        else:
            out = self.core.std.ShufflePlanes([PP2, input], planes=[0, 1, 2], colorfamily=vs.YUV)
        
        if edgemode == -1:
            return self.Resize(edge, dest_x, dest_y)
        elif source:
            if dest_x != ox or dest_y != oy:
                src = self.Resize(source, dest_x, dest_y)
                In = self.Resize(input, dest_x, dest_y)
            else:
                src = source
                In = input
            
            shrpD = self.core.std.MakeDiff(In, out, planes=[0])
            shrpL = self.core.std.Expr([self.core.rgvs.Repair(shrpD, self.core.std.MakeDiff(In, src, planes=[0]), [1, 0]), shrpD],
                                       ['x '+neutral+' - abs y '+neutral+' - abs < x y ?', ''])
            return self.core.std.MakeDiff(In, shrpL, planes=[0])
        else:
            return out
    
    
    # Parameters:
    #  g1str [float]       - strength of grain / for dark areas. Default is 7.0
    #  g2str [float]       - strength of grain / for midtone areas. Default is 5.0
    #  g3str [float]       - strength of grain / for bright areas. Default is 3.0
    #  g1shrp [int]        - sharpness of grain / for dark areas (NO EFFECT when g1size=1.0 !!). Default is 60
    #  g2shrp [int]        - sharpness of grain / for midtone areas (NO EFFECT when g2size=1.0 !!). Default is 66
    #  g3shrp [int]        - sharpness of grain / for bright areas (NO EFFECT when g3size=1.0 !!). Default is 80
    #  g1size [float]      - size of grain / for dark areas. Default is 1.5
    #  g2size [float]      - size of grain / for midtone areas. Default is 1.2
    #  g3size [float]      - size of grain / for bright areas. Default is 0.9
    #  temp_avg [int]      - percentage of noise's temporal averaging. Default is 0
    #  ontop_grain [float] - additional grain to put on top of prev. generated grain. Default is 0.0
    #  th1 [int]           - start of dark->midtone mixing zone. Default is 24
    #  th2 [int]           - end of dark->midtone mixing zone. Default is 56
    #  th3 [int]           - start of midtone->bright mixing zone. Default is 128
    #  th4 [int]           - end of midtone->bright mixing zone. Default is 160
    def GrainFactory3(self, clp, g1str=7., g2str=5., g3str=3., g1shrp=60, g2shrp=66, g3shrp=80, g1size=1.5, g2size=1.2, g3size=.9, temp_avg=0, ontop_grain=0.,
                      th1=24, th2=56, th3=128, th4=160):
        if not isinstance(clp, vs.VideoNode):
            raise ValueError('GrainFactory3: This is not a clip !')
        if not (isinstance(g1str, float) or isinstance(g1str, int)) or g1str < 0:
            raise ValueError("GrainFactory3: 'g1str' have not a correct value! [>=0.0]")
        if not (isinstance(g2str, float) or isinstance(g2str, int)) or g2str < 0:
            raise ValueError("GrainFactory3: 'g2str' have not a correct value! [>=0.0]")
        if not (isinstance(g3str, float) or isinstance(g3str, int)) or g3str < 0:
            raise ValueError("GrainFactory3: 'g3str' have not a correct value! [>=0.0]")
        if not isinstance(g1shrp, int) or g1shrp < 0 or g1shrp > 100:
            raise ValueError("GrainFactory3: 'g1shrp' have not a correct value! [0...100]")
        if not isinstance(g2shrp, int) or g2shrp < 0 or g2shrp > 100:
            raise ValueError("GrainFactory3: 'g2shrp' have not a correct value! [0...100]")
        if not isinstance(g3shrp, int) or g3shrp < 0 or g3shrp > 100:
            raise ValueError("GrainFactory3: 'g3shrp' have not a correct value! [0...100]")
        if not (isinstance(g1size, float) or isinstance(g1size, int)) or g1size < .5 or g1size > 4:
            raise ValueError("GrainFactory3: 'g1size' have not a correct value! [0.5...4.0]")
        if not (isinstance(g2size, float) or isinstance(g2size, int)) or g2size < .5 or g2size > 4:
            raise ValueError("GrainFactory3: 'g2size' have not a correct value! [0.5...4.0]")
        if not (isinstance(g3size, float) or isinstance(g3size, int)) or g3size < .5 or g3size > 4:
            raise ValueError("GrainFactory3: 'g3size' have not a correct value! [0.5...4.0]")
        if not isinstance(temp_avg, int) or temp_avg < 0 or temp_avg > 100:
            raise ValueError("GrainFactory3: 'temp_avg' have not a correct value! [0...100]")
        if not (isinstance(ontop_grain, float) or isinstance(ontop_grain, int)) or ontop_grain < 0:
            raise ValueError("GrainFactory3: 'ontop_grain' have not a correct value! [>=0.0]")
        if not isinstance(th1, int):
            raise ValueError("GrainFactory3: 'th1' must be integer")
        if not isinstance(th2, int):
            raise ValueError("GrainFactory3: 'th2' must be integer")
        if not isinstance(th3, int):
            raise ValueError("GrainFactory3: 'th3' must be integer")
        if not isinstance(th4, int):
            raise ValueError("GrainFactory3: 'th4' must be integer")
        
        shift = clp.format.bits_per_sample - 8
        peak = (1 << clp.format.bits_per_sample) - 1
        
        ox = clp.width
        oy = clp.height
        sx1 = self.m4(ox / g1size)
        sy1 = self.m4(oy / g1size)
        sx1a = self.m4((ox + sx1) / 2)
        sy1a = self.m4((oy + sy1) / 2)
        sx2 = self.m4(ox / g2size)
        sy2 = self.m4(oy / g2size)
        sx2a = self.m4((ox + sx2) / 2)
        sy2a = self.m4((oy + sy2) / 2)
        sx3 = self.m4(ox / g3size)
        sy3 = self.m4(oy / g3size)
        sx3a = self.m4((ox + sx3) / 2)
        sy3a = self.m4((oy + sy3) / 2)
        
        b1 = g1shrp / -50 + 1
        b2 = g2shrp / -50 + 1
        b3 = g3shrp / -50 + 1
        b1a = b1 / 2
        b2a = b2 / 2
        b3a = b3 / 2
        c1 = (1 - b1) / 2
        c2 = (1 - b2) / 2
        c3 = (1 - b3) / 2
        c1a = (1 - b1a) / 2
        c2a = (1 - b2a) / 2
        c3a = (1 - b3a) / 2
        tmpavg = temp_avg / 100
        th1 <<= shift
        th2 <<= shift
        th3 <<= shift
        th4 <<= shift
        
        color = [128 << shift, 128 << shift, 128 << shift] if clp.format.num_planes == 3 else 128 << shift
        grainlayer1 = self.core.grain.Add(self.core.std.BlankClip(clp, width=sx1, height=sy1, color=color), g1str)
        if sx1 != ox or sy1 != oy:
            if g1size > 1.5:
                grainlayer1 = self.Resize(self.core.fmtc.resample(grainlayer1, sx1a, sy1a, kernel='bicubic', a1=b1a, a2=c1a),
                                          ox, oy, kernel='bicubic', a1=b1a, a2=c1a, bits=clp.format.bits_per_sample)
            else:
                grainlayer1 = self.Resize(grainlayer1, ox, oy, kernel='bicubic', a1=b1, a2=c1)
        grainlayer2 = self.core.grain.Add(self.core.std.BlankClip(clp, width=sx2, height=sy2, color=color), g2str)
        if sx2 != ox or sy2 != oy:
            if g2size > 1.5:
                grainlayer2 = self.Resize(self.core.fmtc.resample(grainlayer2, sx2a, sy2a, kernel='bicubic', a1=b2a, a2=c2a),
                                          ox, oy, kernel='bicubic', a1=b2a, a2=c2a, bits=clp.format.bits_per_sample)
            else:
                grainlayer2 = self.Resize(grainlayer2, ox, oy, kernel='bicubic', a1=b2, a2=c2)
        grainlayer3 = self.core.grain.Add(self.core.std.BlankClip(clp, width=sx3, height=sy3, color=color), g3str)
        if sx3 != ox or sy3 != oy:
            if g3size > 1.5:
                grainlayer3 = self.Resize(self.core.fmtc.resample(grainlayer3, sx3a, sy3a, kernel='bicubic', a1=b3a, a2=c3a),
                                          ox, oy, kernel='bicubic', a1=b3a, a2=c3a, bits=clp.format.bits_per_sample)
            else:
                grainlayer3 = self.Resize(grainlayer3, ox, oy, kernel='bicubic', a1=b3, a2=c3)
        
        # x th1 < 0 x th2 > 255 255 th2 th1 - / x th1 - * ? ?
        def get_lut1(x):
            if x < th1:
                return 0
            elif x > th2:
                return peak
            else:
                return max(min(round(peak / (th2 - th1) * (x - th1)), peak), 0)
        # x th3 < 0 x th4 > 255 255 th4 th3 - / x th3 - * ? ?
        def get_lut2(x):
            if x < th3:
                return 0
            elif x > th4:
                return peak
            else:
                return max(min(round(peak / (th4 - th3) * (x - th3)), peak), 0)
        
        grainlayer = self.core.std.MaskedMerge(self.core.std.MaskedMerge(grainlayer1, grainlayer2, self.core.std.Lut(clp, planes=[0], function=get_lut1), planes=[0]),
                                               grainlayer3, self.core.std.Lut(clp, planes=[0], function=get_lut2), planes=[0])
        if temp_avg > 0:
            grainlayer = self.core.std.Merge(grainlayer, self.TemporalSoften(grainlayer, 1, 255<<shift, 0, 0, 2),
                                             weight=[tmpavg, 0] if clp.format.num_planes==3 else tmpavg)
        if ontop_grain > 0:
            grainlayer = self.core.grain.Add(grainlayer, ontop_grain)
        
        return self.core.std.MakeDiff(clp, grainlayer, planes=[0])
    
    
    ####################
    #                  #
    # HELPER FUNCTIONS #
    #                  #
    ####################
    def Clamp(self, clip, bright_limit, dark_limit, overshoot=0, undershoot=0, planes=[0, 1, 2]):
        if clip.format.id != bright_limit.format.id or clip.format.id != dark_limit.format.id:
            raise ValueError('Clamp: clips must have the same format !')
        if isinstance(planes, int):
            planes = [planes]
        bexpr = 'x y '+repr(overshoot)+' + > y '+repr(overshoot)+' + x ?'
        dexpr = 'x y '+repr(undershoot)+' - < y '+repr(undershoot)+' - x ?'
        if clip.format.num_planes == 3:
            bexpr = [bexpr if 0 in planes else '', bexpr if 1 in planes else '', bexpr if 2 in planes else '']
            dexpr = [dexpr if 0 in planes else '', dexpr if 1 in planes else '', dexpr if 2 in planes else '']
        clip = self.core.std.Expr([clip, bright_limit], bexpr)
        return self.core.std.Expr([clip, dark_limit], dexpr)
    
    def LimitDiff(self, filtered, original, smooth=True, thr=1., elast=None, darkthr=None, planes=[0, 1, 2]):
        if filtered.format.id != original.format.id:
            raise ValueError('LimitDiff: clips must have the same format !')
        if elast is None:
            elast = 3 if smooth else 128 / thr
        if darkthr is None:
            darkthr = thr
        if filtered.format.num_planes == 1:
            planes = [0]
        if isinstance(planes, int):
            planes = [planes]
        
        shift = filtered.format.bits_per_sample - 8
        neutral = 128 << shift
        
        thr = max(min(thr, 128), 0)
        darkthr = max(min(darkthr, 128), 0)
        if thr == 0 and darkthr == 0:
            return original
        elif thr == 128 and darkthr == 128:
            return filtered
        elast = max(elast, 1)
        if elast == 1:
            smooth = False
        thr *= 2 ** shift
        darkthr *= 2 ** shift
        
        # diff   = filtered - original
        # alpha  = 1 / (thr * (elast - 1))
        # beta   = thr * elast
        # When smooth=True  :
        # output = diff <= thr  ? filtered : \
        #          diff >= beta ? original : \
        #                         original + alpha * diff * (beta - abs(diff))
        # When smooth=False :
        # output = diff <= thr  ? filtered : \
        #          diff >= beta ? original : \
        #                         original + thr * (diff / abs(diff))
        def get_lut1(x):
            _diff = x - neutral
            _absdiff = abs(_diff)
            _thr = darkthr if _diff > 0 else thr
            _beta = _thr * elast
            _smooth = (1 / (_thr * (elast - 1))) * _diff * (_beta - _absdiff) if smooth else _thr * (_diff / _absdiff)
            if _absdiff <= _thr:
                return x
            elif _absdiff >= _beta:
                return neutral
            else:
                return round(neutral + _smooth)
        def get_lut2(x):
            _diff = x - neutral
            _absdiff = abs(_diff)
            _beta = thr * elast
            _smooth = (1 / (thr * (elast - 1))) * _diff * (_beta - _absdiff) if smooth else thr * (_diff / _absdiff)
            if _absdiff <= thr:
                return x
            elif _absdiff >= _beta:
                return neutral
            else:
                return round(neutral + _smooth)
        
        diff = self.core.std.MakeDiff(filtered, original, planes=planes)
        if 0 in planes:
            diff = self.core.std.Lut(diff, planes=[0], function=get_lut1)
        if 1 in planes or 2 in planes:
            diff = self.core.std.Lut(diff, planes=[1, 2] if 1 in planes and 2 in planes else [1] if 1 in planes else [2], function=get_lut2)
        return self.core.std.MergeDiff(original, diff, planes=planes)
    
    def Logic(self, clip1, clip2, mode='and', th1=0, th2=0, planes=[0, 1, 2]):
        if clip1.format.id != clip2.format.id:
            raise ValueError('Logic: clips must have the same format !')
        mode = mode.lower()
        if clip1.format.num_planes == 1:
            planes = [0]
        if isinstance(planes, int):
            planes = [planes]
        if mode == 'and':
            return self.core.std.Lut2(clip1, clip2, planes=planes, function=lambda x, y: x & y)
        elif mode == 'or':
            return self.core.std.Lut2(clip1, clip2, planes=planes, function=lambda x, y: x | y)
        elif mode == 'xor':
            return self.core.std.Lut2(clip1, clip2, planes=planes, function=lambda x, y: x ^ y)
        elif mode == 'andn':
            return self.core.std.Lut2(clip1, clip2, planes=planes, function=lambda x, y: x & ~y)
        elif mode == 'min':
            expr = 'x '+repr(th1)+' + y '+repr(th2)+' + min'
            if clip1.format.num_planes == 3:
                expr = [expr if 0 in planes else '', expr if 1 in planes else '', expr if 2 in planes else '']
            return self.core.std.Expr([clip1, clip2], expr)
        elif mode == 'max':
            expr = 'x '+repr(th1)+' + y '+repr(th2)+' + max'
            if clip1.format.num_planes == 3:
                expr = [expr if 0 in planes else '', expr if 1 in planes else '', expr if 2 in planes else '']
            return self.core.std.Expr([clip1, clip2], expr)
        else:
            raise ValueError('Logic: \'mode\' must be either "and", "or", "xor", "andn", "min" or "max" !')
    
    def Bob(self, clip, b=1/3, c=1/3, tff=None):
        if not isinstance(tff, bool):
            raise ValueError("Bob: 'tff' must be set. Setting tff to true means top field first and false means bottom field first")
        bits = clip.format.bits_per_sample
        clip = self.core.std.SeparateFields(clip, tff)
        clip = self.core.fmtc.resample(clip, scalev=2, kernel='bicubic', a1=b, a2=c, interlaced=1, interlacedd=0, tff=tff)
        if bits < 16:
            return self.core.fmtc.bitdepth(clip, bits=bits)
        else:
            return clip
    
    def TemporalSoften(self, clip, radius=4, luma_threshold=4, chroma_threshold=8, scenechange=15, mode=2):
        if scenechange:
            clip = self.set_scenechange(clip, scenechange)
        return self.core.focus2.TemporalSoften2(clip, radius, luma_threshold, chroma_threshold, scenechange)
    
    def Weave(self, clip, tff):
        return self.core.std.SelectEvery(self.core.std.DoubleWeave(clip, tff), 2, 0)
    
    def m4(self, x):
        return 16 if x < 16 else int(round(x / 4) * 4)
    
    def set_scenechange(self, clip, thresh=15):
        def set_props(n, f):
            fout = f[0].copy()
            fout.props._SceneChangePrev = f[1].props._SceneChangePrev
            fout.props._SceneChangeNext = f[1].props._SceneChangeNext
            return fout
        
        sc = clip
        if clip.format.color_family == vs.RGB:
            sc = self.core.std.ShufflePlanes(clip, planes=[0], colorfamily=vs.GRAY)
        sc = self.core.scd.Detect(sc, thresh)
        if clip.format.color_family == vs.RGB:
            sc = self.core.std.ModifyFrame(clip, clips=[clip, sc], selector=set_props)
        return sc
    
    ########################################
    ## Didée's functions:
    #
    # contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.
    # script function from Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)
    #
    # In final version 2.0d ContraHD() was merged, to allow proper HD sharpening.
    # In this case global variables of (before denoising) source MSuper and forward and backward
    # compensated motion vectors are necessary as: Super, bv1 and fv1, if used as a stand alone function.
    # Don't know who made (mod) it, so I can't give proper credits, sorry.
    def ContraSharpeningHD(self, denoised, original, HD=False, overshoot=0, lsb=False, lsb_in=False, dmode=3):
        if HD:
            if lsb_in:
                original8 = self.core.avs.DitherPost(original, mode=-1)
            else:
                original8 = original
            
            try:
                Super = self.Super
                bv1 = self.bv1
                fv1 = self.fv1
            except:
                Super = self.core.mv.Super(original8, rfilter=4)
                bv1 = self.core.mv.Analyse(Super, blksize=16, isb=True, delta=1, overlap=8)
                fv1 = self.core.mv.Analyse(Super, blksize=16, isb=False, delta=1, overlap=8)
            
            cb1 = self.core.mv.Compensate(original8, Super, bv1)
            cf1 = self.core.mv.Compensate(original8, Super, fv1)
            if lsb_in:
                cb1 = self.core.fmtc.stack16tonative(self.core.avs.Dither_limit_dif16(original, self.Dither_convert_8_to_16(cb1), thr=1, elast=2))
                cf1 = self.core.fmtc.stack16tonative(self.core.avs.Dither_limit_dif16(original, self.Dither_convert_8_to_16(cf1), thr=1, elast=2))
                pmax = self.Logic(self.Logic(self.core.fmtc.stack16tonative(original), cb1, 'max', planes=[0]), cf1, 'max', planes=[0])
                pmin = self.Logic(self.Logic(self.core.fmtc.stack16tonative(original), cb1, 'min', planes=[0]), cf1, 'min', planes=[0])
            else:
                pmax = self.Logic(self.Logic(original, cb1, 'max', planes=[0]), cf1, 'max', planes=[0])
                pmin = self.Logic(self.Logic(original, cb1, 'min', planes=[0]), cf1, 'min', planes=[0])
                if lsb:
                    pmax = self.core.fmtc.bitdepth(pmax, bits=16)
                    pmin = self.core.fmtc.bitdepth(pmin, bits=16)
        
        if lsb and not lsb_in:
            denoised = self.Dither_convert_8_to_16(denoised)
            original = self.Dither_convert_8_to_16(original)
        
        if lsb or lsb_in:
            s = self.MinBlur(denoised, 2 if HD else 1, planes=[0], lsb=True, lsb_in=True)
            allD = self.core.avs.Dither_sub16(original, denoised, u=1, v=1, dif=True)
            if HD:
                clip2 = self.core.avs.Dither_removegrain16(self.core.avs.Dither_removegrain16(s, 20, -1), 20, -1)
            else:
                clip2 = self.core.avs.Dither_removegrain16(s, 11, -1)
            ssD = self.core.avs.Dither_sub16(s, clip2, u=1, v=1, dif=True)
            if HD:
                clip2 = self.core.avs.Dither_repair16(ssD, allD, 1, -1)
            else:
                clip2 = allD
            ssDD = self.core.avs.Dither_repair16(ssD, clip2, 12 if HD else 1, -1)
            ssDD = self.core.std.Expr([self.core.fmtc.stack16tonative(ssDD), self.core.fmtc.stack16tonative(ssD)], ['x 32768 - abs y 32768 - abs < x y ?', ''])
            
            last = self.core.std.MergeDiff(self.core.fmtc.stack16tonative(denoised), ssDD, planes=[0])
            if HD:
                last = self.Clamp(last, pmax, pmin, overshoot<<8, overshoot<<8, planes=[0])
        else:
            s = self.MinBlur(denoised, 2 if HD else 1, planes=[0])                              # Damp down remaining spots of the denoised clip.
            allD = self.core.std.MakeDiff(original, denoised, planes=[0])                       # The difference achieved by the denoising.
            if HD:
                clip2 = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(s, [20, 0]), [20, 0])
            else:
                clip2 = self.core.rgvs.RemoveGrain(s, [11, 0])
            ssD = self.core.std.MakeDiff(s, clip2, planes=[0])                                  # The difference of a simple kernel blur.
            if HD:
                clip2 = self.core.rgvs.Repair(ssD, allD, [1, 0])
            else:
                clip2 = allD
            ssDD = self.core.rgvs.Repair(ssD, clip2, [12 if HD else 1, 0])                      # Limit the difference to the max of what the denoising removed locally.
            ssDD = self.core.std.Expr([ssDD, ssD], ['x 128 - abs y 128 - abs < x y ?', ''])     # abs(diff) after limiting may not be bigger than before.
            
            last = self.core.std.MergeDiff(denoised, ssDD, planes=[0])                          # Apply the limited difference. (Sharpening is just inverse blurring)
            if HD:
                last = self.Clamp(last, pmax, pmin, overshoot, overshoot, planes=[0])
        
        if lsb:
            return self.core.fmtc.nativetostack16(last)
        elif lsb_in:
            return self.core.fmtc.bitdepth(last, bits=8, dmode=dmode)
        else:
            return last
    
    # MinBlur   by Didée (http://avisynth.org/mediawiki/MinBlur)
    # Nifty Gauss/Median combination
    def MinBlur(self, clp, r=1, planes=[0, 1, 2], lsb=False, lsb_in=False, dmode=3):
        if clp.format.num_planes == 1:
            planes = [0]
        if isinstance(planes, int):
            planes = [planes]
        if 0 in planes:
            Y4 = 4
            Y11 = 11
            Y20 = 20
            Yexpr = 'x 32768 - y 32768 - * 0 < 32768 x 32768 - abs y 32768 - abs < x y ? ?'
        else:
            Y4 = Y11 = Y20 = 0
            Yexpr = ''
        if 1 in planes:
            U4 = Y4
            U11 = Y11
            U20 = Y20
            Uexpr = Yexpr
        else:
            U4 = U11 = U20 = Y4
            Uexpr = Yexpr
        if 2 in planes:
            V4 = Y4
            V11 = Y11
            V20 = Y20
            Vexpr = Yexpr
        else:
            V4 = V11 = V20 = Y4
            Vexpr = Yexpr
        M4 = [Y4] if clp.format.num_planes == 1 else [Y4, U4, V4]
        M11 = [Y11] if clp.format.num_planes == 1 else [Y11, U11, V11]
        M20 = [Y20] if clp.format.num_planes == 1 else [Y20, U20, V20]
        
        # x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?
        def get_lut(x, y):
            if (x - 128) * (y - 128) < 0:
                return 128
            elif abs(x - 128) < abs(y - 128):
                return x
            else:
                return y
        
        if lsb_in:
            clp = self.core.fmtc.stack16tonative(clp)
            clp8 = self.core.fmtc.bitdepth(clp, bits=8, dmode=1)
        elif lsb:
            clp8 = clp
            clp = self.core.fmtc.bitdepth(clp, bits=16)
        
        if r <= 1:
            RG11 = self.core.rgvs.RemoveGrain(clp, M11)
            RG4 = self.core.rgvs.RemoveGrain(clp, M4)
        elif r == 2:
            RG11 = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(clp, M11), M20)
            RG4 = self.core.ctmf.CTMF(clp8 if lsb or lsb_in else clp, radius=2, planes=planes)
            if lsb_in:
                RG4 = self.LimitDiff(clp, self.core.fmtc.bitdepth(RG4, bits=16), thr=1, elast=2, planes=planes)
            elif lsb:
                RG4 = self.core.fmtc.bitdepth(RG4, bits=16)
        else:
            RG11 = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(clp, M11), M20), M20)
            RG4 = self.core.ctmf.CTMF(clp8 if lsb or lsb_in else clp, radius=3, planes=planes)
            if lsb_in:
                RG4 = self.LimitDiff(clp, self.core.fmtc.bitdepth(RG4, bits=16), thr=1, elast=2, planes=planes)
            elif lsb:
                RG4 = self.core.fmtc.bitdepth(RG4, bits=16)
        
        RG11D = self.core.std.MakeDiff(clp, RG11, planes=planes)
        RG4D = self.core.std.MakeDiff(clp, RG4, planes=planes)
        if lsb or lsb_in:
            DD = self.core.std.Expr([RG11D, RG4D], [Yexpr] if clp.format.num_planes==1 else [Yexpr, Uexpr, Vexpr])
        else:
            DD = self.core.std.Lut2(RG11D, RG4D, planes=planes, function=get_lut)
        last = self.core.std.MakeDiff(clp, DD, planes=planes)
        
        if lsb:
            return self.core.fmtc.nativetostack16(last)
        elif lsb_in:
            return self.core.fmtc.bitdepth(last, bits=8, dmode=dmode)
        else:
            return last
    
    # make a highpass on a blur's difference (http://forum.doom9.org/showthread.php?p=1323257#post1323257)
    def sbr(self, clp, r=1, planes=[0, 1, 2]):
        if clp.format.num_planes == 1:
            planes = [0]
        if isinstance(planes, int):
            planes = [planes]
        shift = clp.format.bits_per_sample - 8
        neutral = repr(128 << shift)
        if 0 in planes:
            Y11 = 11
            Y20 = 20
            Yexpr = 'x y - x '+neutral+' - * 0 < '+neutral+' x y - abs x '+neutral+' - abs < x y - '+neutral+' + x ? ?'
        else:
            Y11 = Y20 = 0
            Yexpr = ''
        if 1 in planes:
            U11 = Y11
            U20 = Y20
            Uexpr = Yexpr
        else:
            U11 = U20 = Y11
            Uexpr = Yexpr
        if 2 in planes:
            V11 = Y11
            V20 = Y20
            Vexpr = Yexpr
        else:
            V11 = V20 = Y11
            Vexpr = Yexpr
        M11 = [Y11] if clp.format.num_planes == 1 else [Y11, U11, V11]
        M20 = [Y20] if clp.format.num_planes == 1 else [Y20, U20, V20]
        
        # x y - x 128 - * 0 < 128 x y - abs x 128 - abs < x y - 128 + x ? ?
        def get_lut(x, y):
            if (x - y) * (x - 128) < 0:
                return 128
            elif abs(x - y) < abs(x - 128):
                return x - y + 128
            else:
                return x
        
        if r <= 1:
            RG11 = self.core.rgvs.RemoveGrain(clp, M11)
        elif r == 2:
            RG11 = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(clp, M11), M20)
        else:
            RG11 = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(clp, M11), M20), M20)
        RG11D = self.core.std.MakeDiff(clp, RG11, planes=planes)
        if r <= 1:
            RG11DS = self.core.rgvs.RemoveGrain(RG11D, M11)
        elif r == 2:
            RG11DS = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(RG11D, M11), M20)
        else:
            RG11DS = self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(self.core.rgvs.RemoveGrain(RG11D, M11), M20), M20)
        if shift > 0:
            RG11DD = self.core.std.Expr([RG11D, RG11DS], [Yexpr] if clp.format.num_planes==1 else [Yexpr, Uexpr, Vexpr])
        else:
            RG11DD = self.core.std.Lut2(RG11D, RG11DS, planes=planes, function=get_lut)
        return self.core.std.MakeDiff(clp, RG11DD, planes=planes)
    
    ########################################
    ## cretindesalpes' functions:
    #
    # Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)
    # By courtesy of cretindesalpes. (http://forum.doom9.org/showthread.php?p=1548318#post1548318)
    def Dither_Luma_Rebuild(self, src, s0=2., c=.0625, chroma=True, lsb=False, lsb_in=False, dmode=3):
        def get_lut(x):
            k = (s0 - 1) * c
            t = max(min((x - 4096) / 56064 if lsb or lsb_in else (x - 16) / 219, 1), 0)
            return min(round((k * (1 + c - (1 + c) * c / (t + c)) + t * (1 - k)) * (65536 if lsb or lsb_in else 255)), 65535 if lsb or lsb_in else 255)
        
        if lsb_in:
            src = self.core.fmtc.stack16tonative(src)
        elif lsb:
            src = self.core.fmtc.bitdepth(src, bits=16)
        
        last = self.core.std.Lut(src, planes=[0], function=get_lut)
        if chroma:
            last = self.core.std.Expr(last, ['', 'x 32768 - 32768 * 28672 / 32768 +' if lsb or lsb_in else 'x 128 - 128 * 112 / 128 +'])
        
        if lsb:
            return self.core.fmtc.nativetostack16(last)
        elif lsb_in:
            return self.core.fmtc.bitdepth(last, bits=8, dmode=dmode)
        else:
            return last
    
    def DitherBuildMask(self, cnew, cold=None, edgelvl=8, radius=2, planes=[0, 1, 2]):
        if isinstance(cold, vs.VideoNode):
            if cnew.format.id != cold.format.id:
                raise ValueError('DitherBuildMask: clips must have the same format !')
            cold_flag = True
        else:
            cold_flag = False
        if cnew.format.num_planes == 1:
            planes = [0]
        if isinstance(planes, int):
            planes = [planes]
        
        shift = cnew.format.bits_per_sample - 8
        edgelvl <<= shift
        
        cedgn = self.core.generic.Prewitt(cnew, edgelvl, edgelvl, planes=planes)
        
        m = cedgn
        if cold_flag:
            cedgo = self.core.generic.Prewitt(cold, edgelvl, edgelvl, planes=planes)
            expr = 'x y = 0 '+repr((255 << shift) + 2 ** shift - 1)+' ?'
            if cnew.format.num_planes == 3:
                expr = [expr if 0 in planes else '', expr if 1 in planes else '', expr if 2 in planes else '']
            cdif = self.core.std.Expr([cold, cnew], expr)
            m = self.Logic(m, cedgo, 'or', planes=planes)
            m = self.Logic(m, cdif, 'or', planes=planes)
        
        if radius > 1:
            return self.mt_expand_multi(m, planes=planes, sw=radius-1, sh=radius-1)
        else:
            return m
    
    def Dither_get_msb(self, src):
        w = src.width
        h = int(src.height / 2)
        return self.core.std.CropAbs(src, width=w, height=h, x=0, y=0)
    
    def Dither_get_lsb(self, src):
        w = src.width
        h = int(src.height / 2)
        return self.core.std.CropAbs(src, width=w, height=h, x=0, y=h)
    
    def Dither_gen_null_lsb(self, src):
        return self.core.std.BlankClip(src, color=[0, 0, 0])
    
    def Dither_convert_8_to_16(self, src):
        return self.core.std.StackVertical([src, self.Dither_gen_null_lsb(src)])
    
    def Dither_merge16_8(self, src1, src2, mask, luma=False, y=3, u=2, v=2):
        mask16 = self.core.std.StackVertical([mask, mask])
        return self.core.avs.Dither_merge16(src1, src2, mask16, luma=luma, y=y, u=u, v=v)
    
    #=============================================================================
    #	mt_expand_multi
    #	mt_inpand_multi
    #
    #	Calls mt_expand or mt_inpand multiple times in order to grow or shrink
    #	the mask from the desired width and height.
    #
    #	Parameters:
    #	- sw   : Growing/shrinking shape width. 0 is allowed. Default: 1
    #	- sh   : Growing/shrinking shape height. 0 is allowed. Default: 1
    #	- mode : "rectangle" (default), "ellipse" or "losange". Replaces the
    #		mt_xxpand mode. Ellipses are actually combinations of
    #		rectangles and losanges and look more like octogons.
    #		Losanges are truncated (not scaled) when sw and sh are not
    #		equal.
    #	Other parameters are the same as mt_xxpand.
    #=============================================================================
    def mt_expand_multi(self, src, mode='rectangle', planes=[0, 1, 2], sw=1, sh=1):
        if src.format.num_planes == 1:
            planes = [0]
        
        if sw > 0 and sh > 0:
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
        elif sw > 0:
            mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
        elif sh > 0:
            mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            mode_m = None
        
        if mode_m:
            return self.mt_expand_multi(self.core.generic.Maximum(src, planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw-1, sh=sh-1)
        else:
            return src
    
    def mt_inpand_multi(self, src, mode='rectangle', planes=[0, 1, 2], sw=1, sh=1):
        if src.format.num_planes == 1:
            planes = [0]
        
        if sw > 0 and sh > 0:
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
        elif sw > 0:
            mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
        elif sh > 0:
            mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            mode_m = None
        
        if mode_m:
            return self.mt_inpand_multi(self.core.generic.Minimum(src, planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw-1, sh=sh-1)
        else:
            return src
    
    def mt_inflate_multi(self, src, planes=[0, 1, 2], radius=1):
        if src.format.num_planes == 1:
            planes = [0]
        
        if radius > 0:
            return self.mt_inflate_multi(self.core.generic.Inflate(src, planes=planes), planes=planes, radius=radius-1)
        else:
            return src
    
    def mt_deflate_multi(self, src, planes=[0, 1, 2], radius=1):
        if src.format.num_planes == 1:
            planes = [0]
        
        if radius > 0:
            return self.mt_deflate_multi(self.core.generic.Deflate(src, planes=planes), planes=planes, radius=radius-1)
        else:
            return src
