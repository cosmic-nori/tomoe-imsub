#!/usr/bin/env python3
# coding: utf-8

# created on 2022/11/28

import os
import subprocess
# import numpy as np
# from astropy.io import fits
from argparse import ArgumentParser as ap
import datetime
import time
import tomoeutils
from astropy.io import fits
import numpy as np
# import sep

#print(sys.version)
#print(sys.path)

def main(args):
    start_date = datetime.datetime.now()
    start_time = time.time()
    #print(args.sci)
    runhotpants(args.sci, args.ref, args.outdir, args.debug)

    end_date = datetime.datetime.now()
    elapsed_time = time.time() - start_time
    print('Start date: {}'.format(start_date))
    print('End date: {}'.format(end_date))
    print('-------------------------------------------------------')
    print('| Elapsed time of running the main func:', str(datetime.timedelta(seconds=elapsed_time)))
    print('-------------------------------------------------------')

def runhotpants(sci_img, ref_img, out_dir, frame_num=11, debug=False):
    """
    Run the hotpants image subtraction software as a Python script

    Parameters
    ----------
    sci_img : filename (full path)
        science image to be differenced
    ref_img : filename (full path)
        reference image
    out_dir : directory
        directory where output FITS files will be stored
    
    #####################################################
    # HOTPANTS software
    # https://github.com/acbecker/hotpants
    # Version 5.1.11
    #####################################################
    Required options:
    [-inim fitsfile]  : comparison image to be differenced
    [-tmplim fitsfile]: template image
    [-outim fitsfile] : output difference image

    Additional options:
    [-tu tuthresh]    : upper valid data count, template (25000)
    [-tuk tucthresh]  : upper valid data count for kernel, template (tuthresh)
    [-tl tlthresh]    : lower valid data count, template (0)
    [-tg tgain]       : gain in template (1)
    [-tr trdnoise]    : e- readnoise in template (0)
    [-tp tpedestal]   : ADU pedestal in template (0)
    [-tni fitsfile]   : input template noise array (undef)
    [-tmi fitsfile]   : input template mask image (undef)
    [-iu iuthresh]    : upper valid data count, image (25000)
    [-iuk iucthresh]  : upper valid data count for kernel, image (iuthresh)
    [-il ilthresh]    : lower valid data count, image (0)
    [-ig igain]       : gain in image (1)
    [-ir irdnoise]    : e- readnoise in image (0)
    [-ip ipedestal]   : ADU pedestal in image (0)
    [-ini fitsfile]   : input image noise array (undef)
    [-imi fitsfile]   : input image mask image (undef)

    [-ki fitsfile]    : use kernel table in image header (undef)
    [-r rkernel]      : convolution kernel half width (10)
    [-kcs step]       : size of step for spatial convolution (2 * rkernel + 1)
    [-ft fitthresh]   : RMS threshold for good centroid in kernel fit (20.0)
    [-sft scale]      : scale fitthresh by this fraction if... (0.5)
    [-nft fraction]   : this fraction of stamps are not filled (0.1)
    [-mins spread]    : Fraction of kernel half width to spread input mask (1.0)
    [-mous spread]    : Ditto output mask, negative = no diffim masking (1.0)
    [-omi  fitsfile]  : Output bad pixel mask (undef)
    [-gd xmin xmax ymin ymax]
                        : only use subsection of full image (full image)

    [-nrx xregion]    : number of image regions in x dimension (1)
    [-nry yregion]    : number of image regions in y dimension (1)
    -- OR --
    [-rf regionfile]  : ascii file with image regions 'xmin:xmax,ymin:ymax'
    -- OR --
    [-rkw keyword num]: header 'keyword[0->(num-1)]' indicates valid regions

    [-nsx xstamp]     : number of each region's stamps in x dimension (10)
    [-nsy ystamp]     : number of each region's stamps in y dimension (10)
    -- OR --
    [-ssf stampfile]  : ascii file indicating substamp centers 'x y'
    -- OR --
    [-cmp cmpfile]    : .cmp file indicating substamp centers 'x y'

    [-afssc find]     : autofind stamp centers so #=-nss when -ssf,-cmp (1)
    [-nss substamps]  : number of centroids to use for each stamp (3)
    [-rss radius]     : half width substamp to extract around each centroid (15)

    [-savexy file]    : save positions of stamps for convolution kernel (undef)
    [-c  toconvolve]  : force convolution on (t)emplate or (i)mage (undef)
    [-n  normalize]   : normalize to (t)emplate, (i)mage, or (u)nconvolved (t)
    [-fom figmerit]   : (v)ariance, (s)igma or (h)istogram convolution merit (v)
    [-sconv]          : all regions convolved in same direction (0)
    [-ko kernelorder] : spatial order of kernel variation within region (2)
    [-bgo bgorder]    : spatial order of background variation within region (1)
    [-ssig statsig]   : threshold for sigma clipping statistics  (3.0)
    [-ks badkernelsig]: high sigma rejection for bad stamps in kernel fit (2.0)
    [-kfm kerfracmask]: fraction of abs(kernel) sum for ok pixel (0.990)
    [-okn]            : rescale noise for 'ok' pixels (0)
    [-fi fill]        : value for invalid (bad) pixels (1.0e-30)
    [-fin fill]       : noise image only fillvalue (0.0e+00)
    [-convvar]        : convolve variance not noise (0)

    [-oni fitsfile]   : output noise image (undef)
    [-ond fitsfile]   : output noise scaled difference image (undef)
    [-nim]            : add noise image as layer to sub image (0)
    [-ndm]            : add noise-scaled sub image as layer to sub image (0)

    [-oci fitsfile]   : output convolved image (undef)
    [-cim]            : add convolved image as layer to sub image (0)

    [-allm]           : output all possible image layers

    [-nc]             : do not clobber output image (0)
    [-hki]            : print extensive kernel info to output image header (0)

    [-oki fitsfile]   : new fitsfile with kernel info (under)

    [-sht]            : output images 16 bitpix int, vs -32 bitpix float (0)
    [-obs bscale]     : if -sht, output image BSCALE, overrides -inim (1.0)
    [-obz bzero]      : if -sht, output image BZERO , overrides -inim (0.0)
    [-nsht]           : output noise image 16 bitpix int, vs -32 bitpix float (0)
    [-nbs bscale]     : noise image only BSCALE, overrides -obs (1.0)
    [-nbz bzero]      : noise image only BZERO,  overrides -obz (0.0)

    [-ng  ngauss degree0 sigma0 .. degreeN sigmaN]
                        : ngauss = number of gaussians which compose kernel (3)
                        : degree = degree of polynomial associated with gaussian #
                                    (6 4 2)
                        : sigma  = width of gaussian #
                                    (0.70 1.50 3.00)
                        : N = 0 .. ngauss - 1

                        : (3 6 0.70 4 1.50 2 3.00
    [-pca nk k0.fits ... n(k-1).fits]
                        : nk      = number of input basis functions
                        : k?.fits = name of fitsfile holding basis function
                        : Since this uses input basis functions, it will fix :
                        :    hwKernel 
                        :    
    [-v] verbosity    : level of verbosity, 0-2 (1) 
    NOTE: Fits header params will be added to the difference image
        COMMAND             (what was called on the command line)
        NREGION             (number of regions in image)
        PHOTNORM            (to which system the difference image is normalized)
        TARGET              (image which was differenced)
        TEMPLATE            (template for the difference imaging)
        DIFFIM              (output difference image)
        MASKVAL             (value for masked pixels)
        REGION??            (IRAF-format limits for each region in the image)
        CONVOL??            (which image was convolved for each region)
        KSUM??              (sum of the convolution kernel for each region)
    
    Returns
    -------

    """
    # sci_img = '/home/arima/tomoe/tomoeflash/imsub/test/fits/frame_fits/rTMQ2202003220028809433_frame001.fits'
    # ref_img = '/home/arima/tomoe/tomoeflash/imsub/test/fits/stack_fits/rTMQ2202003220028809433_median_stack.fits'
    # out_dir = '/home/arima/tomoe/tomoeflash/imsub/test/diff_fits'

    # Output FITS name
    sci_fits_basename = os.path.splitext(os.path.basename(sci_img))[0]
    out_fitsname = out_dir + sci_fits_basename + '_hp_diff.fits'

    sci_header, sci_data = tomoeutils.openfits(sci_img)
    ref_header, ref_data = tomoeutils.openfits(ref_img)

    try:
        frame_num = round(sci_header['EXPTIME'] / sci_header['TFRAME']) - 1
        # frame_num = round(sci_header['EXPTIME'] / sci_header['T_FRAME'])  original : but not found in testfits (TFRAME)
    except:
        frame_num = frame_num - 1

    # Get header info.
    print('# of frames = {}'.format(frame_num))

    sci_gain = float(sci_header['GAIN']) * frame_num
    sci_rnoise = float(sci_header['RNOISE'])
    ref_renorm = ref_data*np.max(sci_data)/np.max(ref_data)

    hdu = fits.PrimaryHDU(ref_renorm, header=None )
    hdu.writeto('ref_renorm.fits' ,overwrite=True)

    ### User-defined thresholds ###
    saturate = 15000.0
    # for science image
    iu = saturate
    il = -50
    ig = sci_gain
    ir = sci_rnoise
    # for reference image
    tu = 15000.0
    tl = -50
    tg = 2.53*100
    tr = ir

    # mous = 1.0

    # Command to run hotpants software
    cmd_hotpants_base = "hotpants -inim " + sci_img + " -tmplim " + 'ref_renorm.fits' + " -outim " + out_fitsname + " -iu " + str(iu) + " -il " + str(il) + " -ig " + str(ig) + " -ir " + str(ir)  + " -tu " + str(tu) + " -tl " +  str(tl) + " -tg " + str(tg) + " -tr " + str(tr)  #+ " -mous " +str(mous)
    #cmd = ["hotpants", "-inim " + sci_img ," -tmplim" + ref_img, "-outim " + out_fitsname]
    if debug:
        cmd_hotpants = cmd_hotpants_base
    if not debug:
        cmd_hotpants = cmd_hotpants_base + " >/dev/null 2>&1"
    
    #os.system(command_hotpants)
    subprocess.run(cmd_hotpants, shell=True)
    # subprocess.call(['rm','ref_renorm.fits'])
    return out_fitsname



if __name__ == '__main__':
    sci_default = '/home/arima/tomoe/tomoeflash/imsub/test/fits/frame_fits/rTMQ2202003220028809433_frame001.fits'
    ref_default = '/home/arima/tomoe/tomoeflash/imsub/test/fits/stack_fits/rTMQ2202003220028809433_median_stack.fits'
    out_dir_default = '/home/arima/tomoe/tomoeflash/imsub/test/diff_fits'
    # make parser
    parser = ap(description='Execute hotpants image subtraction anaysis')
    parser.add_argument('-s', '--sci', type=str, default=sci_default, help='Input science FITS file')
    parser.add_argument('-r', '--ref', type=str, default=ref_default, help='Input reference FITS file')
    parser.add_argument('-o', '--outdir', type=str,default=out_dir_default, help='Output directory')
    parser.add_argument('-d', '--debug', action='store_true', help='Print the hotpants stdout messages')
    # analyse arguments
    args = parser.parse_args()
    main(args)
    