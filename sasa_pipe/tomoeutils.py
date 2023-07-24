#!/usr/bin/env python3
# coding: utf-8

# created: 2022/11/28
# updated: 2022/12/13

import os,sys
import time
from functools import wraps 
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import sep


def stop_watch(func):
    """
    Measure the execution time of a function
    # The original source code can be found here
    # https://www.st-hakky-blog.com/entry/2018/01/26/214255
    """
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        print(f"### Elapsed time of {func.__name__}" + ": {0:.2f} [sec] ###".format(elapsed_time))
        return result
    return wrapper


class FitsHandler:
    def __init__(self, fits_path, out_dir):
        self.fits_path = fits_path
        self.fits_basename = os.path.splitext(os.path.basename(fits_path))[0]
        self.out_dir = out_dir
        self.header = None
        self.data = None
        
    @stop_watch
    def open_fits(self, verbose=False):
        with fits.open(self.fits_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
        if header['NAXIS'] != 3 and header['NAXIS'] != 2:
            raise Exception('Unexpected data was given: header NAXIS must be 2 or 3')
    
        if header['NAXIS'] == 2:
            if verbose:
                print('Image with NAXIS = 2 was given.') #SKIP STACKING PROCESS

        self.header = header
        self.data = data
        return header, data #####  not comment out sasa1228
    
    @staticmethod
    @stop_watch
    def static_open(fits_path):
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
        if header['NAXIS'] != 3 and header['NAXIS'] != 2:
            raise Exception('Unexpected data was given: header NAXIS must be 2 or 3')
        return header, data


class SourceFinder(FitsHandler):
    MAX_DETECTNUM = 3000
    def __init__(self, fits_path, thresh, minarea, filter_kernel=None, mask=None, maskthresh=0.0, aper_rad1=5.0, aper_rad2=7.0):
        super().__init__(fits_path)
        self.thresh = thresh
        self.minarea = minarea
        self.filter_kernel = filter_kernel
        self.mask = mask
        self.maskthresh = maskthresh
        self.aper_rad1 = aper_rad1
        self.aper_rad2 = aper_rad2
        self.df_obj = None
        self.header, self.data = FitsHandler.static_open(fits_path)
        self.gain = self.header['GAIN'] ### Need to check!!!
        self.bkg_mean = None
        self.bkg_rms = None
        self.data_bkgsub = None
        self.segmap = None

    @staticmethod
    @stop_watch
    def subtract_bkg(data, bw=64, bh=64, fw=3, fh=3, verbose=False, bkg_raw=False):
        # SEP parameters for estimating background level
        sep_bkg_params = dict(
        bw = bw, # block width
        bh = bh, # block height
        fw = fw,  # filter width
        fh = fh,  # filter height
        )
        if data.dtype.byteorder == '>':
            # Please learn about "byte order" (see the SEP official tutorial below)
            # https://sep.readthedocs.io/en/latest/tutorial.html
            data = data.byteswap().newbyteorder()

        # Measure a spatially varying background on the image
        bkg = sep.Background(data, **sep_bkg_params)

        # get a "global" mean and noise (rms) of the image background
        bkg_mean = bkg.globalback
        bkg_rms  = bkg.globalrms
        if verbose:
            print('### Background mean = {0:.2f}, rms = {1:.2f} ###'.format(bkg_mean, bkg_rms))
        # subtract the background from an image
        data_bkgsub = data - bkg
    
        if bkg_raw:
            return bkg, bkg_mean, bkg_rms
        else:
            return data_bkgsub, bkg_mean, bkg_rms
    
    @stop_watch
    def run_sep(self, exec_bkgsub=True, use_segmap=True, radius1=5.0, radius2=7.0):
        """
        Perform object detection and aperture photometry on 2D images by SEP
        """
        data_bkgsub, bkg_mean, bkg_rms = SourceFinder.subtract_bkg(self.data)
        self.data_bkgsub, self.bkg_mean, self.bkg_rms = data_bkgsub, bkg_mean, bkg_rms
        print('bkg mean, rms = {:.2f}, {:.2f}'.format(self.bkg_mean, self.bkg_rms))

        if exec_bkgsub:
            data = self.data_bkgsub
        if not exec_bkgsub:
            data = self.data
            if data.dtype.byteorder == '>':
                data = data.byteswap().newbyteorder()
        
        # SEP parameters for extracting sources
        sep_extract_params = dict(
            thresh = self.thresh,
            minarea = self.minarea,
            err = self.bkg_rms,
            filter_kernel = self.filter_kernel,
            filter_type = 'matched',
            segmentation_map = use_segmap,
            mask = self.mask,
            maskthresh = self.maskthresh
            )
        
        if use_segmap:
            obj, data_segmap = sep.extract(data, **sep_extract_params)
            self.segmap = data_segmap
        if not use_segmap:
            obj = sep.extract(data, **sep_extract_params)
            # obj = sep.extract(data, thresh=self.thresh, minarea=self.minarea, err=self.bkg_rms,\
            #      filter_kernel=self.filter_kernel, filter_type='matched', segmentation_map=use_segmap,\
            #          mask=self.mask, maskthresh=self.maskthresh)
        
        # Perform aperture photometry (with two different radii)
        ap1_flux, ap1_ferr, ap1_flag = sep.sum_circle(data, obj['x'], obj['y'], r=self.aper_rad1, err=self.bkg_rms, gain=self.gain)
        ap2_flux, ap2_ferr, ap2_flag = sep.sum_circle(data, obj['x'], obj['y'], r=self.aper_rad2, err=self.bkg_rms, gain=self.gain)

        # Make column names of Pandas.DataFrame for the aperture photometory 
        ap1_flux_col = 'flux_{}pix'.format(int(self.aper_rad1))
        ap1_ferr_col = 'ferr_{}pix'.format(int(self.aper_rad1))
        ap1_flag_col = 'aper1_flag'
        ap2_flux_col = 'flux_{}pix'.format(int(self.aper_rad2))
        ap2_ferr_col = 'ferr_{}pix'.format(int(self.aper_rad2))
        ap2_flag_col = 'aper2_flag'

        # calculate effective radius "R_e"
        reff, flag = sep.flux_radius(data, obj['x'], obj['y'], 6.*obj['a'], frac=0.5, normflux=ap1_flux) # , subpix=5
        reff_colname = 'reff'

        # Make a padas DataFrame
        col_names = np.array(obj.dtype.names) 
        df_obj = pd.DataFrame(obj, columns=col_names)

        # Add results of aperture photometory
        df_obj[ap1_flux_col] = ap1_flux
        df_obj[ap1_ferr_col] = ap1_ferr
        df_obj[ap1_flag_col] = ap1_flag
        df_obj[ap2_flux_col] = ap2_flux
        df_obj[ap2_ferr_col] = ap2_ferr
        df_obj[ap2_flag_col] = ap2_flag
        # effective radius
        df_obj[reff_colname] = reff

        # Make padas.DataFrame
        self.df_obj = df_obj


# Get FITS file (*.fits) names of two types: (1) full path and (2) basename only
def get_filename(dir_path, keyword='.fits', slash=True):
    """
    Get FITS file (*.fits) names of two types: (1) full path and (2) basename only
    """
    if slash:
        file_names = sorted(glob.glob(dir_path + '/*' + keyword))
    elif not slash:
        file_names = sorted(glob.glob(dir_path + '*' + keyword))

    file_basenames = [os.path.basename(name) for name in file_names]
    if len(file_names)==0:
        print('No file found. Please check your input directory')
    return file_names, file_basenames

def openfits(input_file, debug=False):
    """
    Open a 2D or 3D FITS file and return the (1) header and (2) data
    
    Parameters
    ----------
    input_file : str
        input file path

    Returns
    -------
    header & data
    """
    with fits.open(input_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if header['NAXIS'] != 3 and header['NAXIS'] != 2:
        raise Exception('Unexpected data was given: header NAXIS must be 2 or 3')
    
    if header['NAXIS'] == 2:
        if debug:
            print('Image with NAXIS = 2 was given.') #SKIP STACKING PROCESS
    
    return header, data


def subtractbkg(data, image_only=False, verbose=False):
    """
    Subtract sky background from an input image
    
    Returns
    -------
    data_bkgsub : numpy.ndarray
        background subtracted image 
    bkg_mean : float
        background mean
    bkg_rms : float
        background rms
    """

    # SEP parameters for estimating background level
    sep_bkg_option = dict(
    bw = 64, # block width
    bh = 64, # block height
    fw = 3,  # filter width
    fh = 3,  # filter height
    )
    if data.dtype.byteorder == '>':
        # Please note about byte order, see SEP official tutorial below
        # https://sep.readthedocs.io/en/latest/tutorial.html
        data = data.byteswap().newbyteorder()

    # Measure a spatially varying background on the image
    bkg = sep.Background(data, **sep_bkg_option)

    # get a "global" mean and noise (rms) of the image background
    bkg_mean = bkg.globalback
    bkg_rms  = bkg.globalrms
    if verbose:
        print('Background mean = {0:.2f}, rms = {1:.2f}'.format(bkg_mean, bkg_rms))
    # subtract the background from an image
    data_bkgsub = data - bkg
    
    if image_only:
        return data_bkgsub
    else:
        return data_bkgsub, bkg_mean, bkg_rms

# ################################################ SASAOKA 0117:

def mk_sourcecorr(newim):
    nhead, NS = openfits(newim)
    frame_num = 11 #default
    try:
        frame_num = round(nhead['EXPTIME'] / nhead['TFRAME']) - 1
    # frame_num = round(sci_header['EXPTIME'] / sci_header['T_FRAME'])  original : but not found in testfits (TFRAME)
    except:
        frame_num = frame_num - 1

    gain_base = float(nhead['GAIN'])
    Nrnoise_e = float(nhead['RNOISE'])
    Nfullwell = float(nhead['FULLWELL'])   #[/electron]

    r_N2 = (Nrnoise_e)**2*(1.253)**2*frame_num #[(electron/pix)^2]
    r_N = np.sqrt(r_N2)/frame_num  #[electron/pix]

    r_N = r_N/gain_base


    NbgS = float(nhead['BGMEDIAN'])   #[ADU]

    if Nfullwell/gain_base > 15000:
        Nsaturate = 15000
    else:
        Nsaturate = Nfullwell/gain_base


    NS = np.where(NS<0, 0 , NS)  #denom == 0 にならないように調整　あかんかも？
    numer = np.sqrt(NbgS**2 + r_N**2)
    denom = np.sqrt(NS + NbgS**2 + r_N**2)

    Dc_factor = numer/denom

    # D_corr = Dc_factor*D_hot
    return  Dc_factor

################################################ SASAOKA 0117:

def mk_badpixmask(newim):
    nhead, NS = openfits(newim)

#########################################################################
    maskim = np.zeros_like(NS)

    DET_ID=int(nhead['DET_ID'])
    pixlistbyARIMA=[[111, 528, 1902], [113, 885, 1030], [113, 149, 1413], [114, 945, 447], [114, 1077, 1692], [114, 1077, 1692], [114, 1081, 1692], [114, 945, 447], [114, 44, 1853], [114, 359, 1775], [114, 289, 40], [114, 528, 1444], [114, 1112, 822], [114, 186, 1332], [114, 186, 1332], [114, 525, 1692], [115, 914, 30], [115, 266, 461], [115, 299, 1139], [115, 299, 1139], [115, 299, 1139], [115, 266, 461], [116, 381, 1279], [122, 925, 704], [123, 224, 1480], [123, 224, 1480], [124, 369, 1474], [124, 1037, 371], [124, 133, 1422], [125, 893, 1071], [131, 946, 1374], [131, 12, 294], [133, 327, 1431], [135, 51, 608], [214, 1032, 1508], [215, 132, 77], [215, 121, 152], [224, 628, 1017], [225, 550, 1688], [225, 862, 1412], [225, 227, 1859], [225, 608, 1458], [225, 53, 1706], [225, 615, 1397], [225, 885, 1253], [225, 427, 738], [225, 74, 1581], [225, 1105, 710], [225, 701, 604], [225, 227, 1859], [225, 98, 56], [226, 945, 1559], [226, 880, 1343], [226, 483, 1085], [226, 945, 1559], [226, 945, 1559], [226, 1116, 1795], [233, 342, 1513], [233, 705, 1637], [233, 405, 89], [234, 658, 1318], [234, 713, 1070], [234, 280, 1441], [234, 999, 764], [234, 341, 1336], [234, 314, 1873], [234, 1073, 690], [234, 933, 408], [235, 209, 779], [241, 764, 52], [242, 101, 613], [243, 165, 298], [243, 932, 1532], [243, 946, 1832], [243, 435, 1869], [244, 842, 1847], [244, 1112, 113], [244, 148, 1359], [311, 713, 543], [316, 66, 786], [321, 962, 876], [322, 172, 1510], [324, 391, 1463], [334, 467, 1809], [341, 45, 973], [342, 1056, 1891], [343, 270, 862], [344, 362, 1471], [344, 721, 1970], [412, 311, 947], [412, 178, 1727], [416, 70, 1053], [422, 928, 1005], [423, 543, 617], [426, 741, 210], [433, 649, 678]]
    collistbyARIMA=[[111,1340],[111,1407],[116,1504],[116,1844],[213,1748],[214,832],[323,1055],[421,1240],[422,1260]]


    for i in range(len(pixlistbyARIMA)):
        if pixlistbyARIMA[i][0] == DET_ID:
            maskim[pixlistbyARIMA[i][2]-1:pixlistbyARIMA[i][2]+2,pixlistbyARIMA[i][1]-1:pixlistbyARIMA[i][1]+2] = 1
        elif pixlistbyARIMA[i][0] > DET_ID:
            break
    
    for i in range(len(collistbyARIMA)):
        if collistbyARIMA[i][0] == DET_ID:
            maskim[:,collistbyARIMA[i][1]-1:collistbyARIMA[i][1]+2] = 1
        elif collistbyARIMA[i][0] > DET_ID:
            break
    # hdu = fits.PrimaryHDU(maskim ,header=None )
    # hdu.writeto('maskim.fits' ,overwrite=True)

    return maskim.astype(np.float32)




def detectobj(data, thresh=5, minarea=5, filter_kernel=None, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23, mask=None, maskthresh=0.0):
    """
    Perform object detection and aperture photometry on 2D images by SEP
    """
    data_bkgsub, bkg_mean, bkg_rms = subtractbkg(data)
    if use_segmap:
        obj, data_segmap = sep.extract(data_bkgsub, thresh=thresh, minarea=minarea, err=bkg_rms, filter_kernel=filter_kernel, \
            filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
    elif not use_segmap:
        obj = sep.extract(data_bkgsub, thresh=thresh, minarea=minarea, err=bkg_rms, filter_kernel=filter_kernel, \
            filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
    
    # Perform aperture photometry (with two different radii)
    ap1_flux, ap1_ferr, ap1_flag = sep.sum_circle(data_bkgsub, obj['x'], obj['y'], r=radius1, err=bkg_rms, gain=gain)
    ap2_flux, ap2_ferr, ap2_flag = sep.sum_circle(data_bkgsub, obj['x'], obj['y'], r=radius2, err=bkg_rms, gain=gain)

    # Make column names of Pandas.DataFrame for the aperture photometory 
    ap1_flux_col = 'ap1flux_{}pix'.format(int(radius1))
    ap1_ferr_col = 'ap1ferr_{}pix'.format(int(radius1))
    ap1_flag_col = 'ap1_flag'
    ap2_flux_col = 'ap2flux_{}pix'.format(int(radius2))
    ap2_ferr_col = 'ap2ferr_{}pix'.format(int(radius2))
    ap2_flag_col = 'ap2_flag'

     # calculate effective radius "R_e"
    reff, flag = sep.flux_radius(data_bkgsub, obj['x'], obj['y'], 6.*obj['a'], frac=0.5, normflux=ap1_flux) # , subpix=5
    reff_colname = 'reff'

    # Make a padas DataFrame
    col_names = np.array(obj.dtype.names) 
    df_obj = pd.DataFrame(obj, columns=col_names)

    # Add results of aperture photometory
    df_obj[ap1_flux_col] = ap1_flux
    df_obj[ap1_ferr_col] = ap1_ferr
    df_obj[ap1_flag_col] = ap1_flag
    df_obj[ap2_flux_col] = ap2_flux
    df_obj[ap2_ferr_col] = ap2_ferr
    df_obj[ap2_flag_col] = ap2_flag
    # effective radius
    df_obj[reff_colname] = reff
    
    if use_segmap:
        return df_obj, data_bkgsub, data_segmap, bkg_mean, bkg_rms
    if not use_segmap:
        return df_obj, data_bkgsub, bkg_mean, bkg_rms




def detectobj2(newim, data,corr, thresh=5, minarea=5, filter_kernel=None, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23, mask=None, maskthresh=0.0):
    """
    Perform object detection and aperture photometry on 2D images by SEP
    """

    headnew,datnew = openfits(newim)
    #sasaoka1229
    if corr == True:
        Dc_factor = mk_sourcecorr(newim)
        Dcorr = Dc_factor*data
        hdu = fits.PrimaryHDU(Dcorr, header=headnew )
        hdu.writeto(newim.replace('.fits','_hp_diff_corr.fits') ,overwrite=True)
        data_bkgsub_x, bkg_mean_x, bkg_rms_x = subtractbkg(datnew)        
        data_bkgsub_c, bkg_mean_c, bkg_rms_c = subtractbkg(Dcorr)
        data_bkgsub_c = np.where(mask==1,0,data_bkgsub_c)
        # hdu = fits.PrimaryHDU(data_bkgsub_c, header=None )
        # hdu.writeto('data_bkgsub_c.fits' ,overwrite=True)
        print('Source Correction : True')
        if use_segmap:
            obj, data_segmap = sep.extract(data_bkgsub_c, thresh=thresh, minarea=minarea, err=bkg_rms_x, filter_kernel=filter_kernel, \
                filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
        elif not use_segmap:
            obj = sep.extract(data_bkgsub_c, thresh=thresh, minarea=minarea, err=bkg_rms_x, filter_kernel=filter_kernel, \
                filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)

    else:
        data_bkgsub, bkg_mean, bkg_rms = subtractbkg(data)
        data_bkgsub_x, bkg_mean_x, bkg_rms_x = subtractbkg(datnew)
        print('Source Correction : False')
        if use_segmap:
            obj, data_segmap = sep.extract(data_bkgsub, thresh=thresh, minarea=minarea, err=bkg_rms_x, filter_kernel=filter_kernel, \
                filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
        elif not use_segmap:
            obj = sep.extract(data_bkgsub, thresh=thresh, minarea=minarea, err=bkg_rms_x, filter_kernel=filter_kernel, \
                filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)

    data_bkgsub, bkg_mean, bkg_rms = subtractbkg(data)

    # Perform aperture photometry (with two different radii)
    ap1_flux, ap1_ferr, ap1_flag = sep.sum_circle(data_bkgsub, obj['x'], obj['y'], r=radius1, err=bkg_rms, gain=gain)
    ap2_flux, ap2_ferr, ap2_flag = sep.sum_circle(data_bkgsub, obj['x'], obj['y'], r=radius2, err=bkg_rms, gain=gain)


    # check_remain_photutil = circle_reg_photutil_listradius(data_bkgsub,bkg_rms_x,obj['xpeak'],obj['ypeak'],radius_check)

    # Make column names of Pandas.DataFrame for the aperture photometory 
    ap1_flux_col = 'ap1flux_{}pix'.format(int(radius1))
    ap1_ferr_col = 'ap1ferr_{}pix'.format(int(radius1))
    ap1_flag_col = 'ap1_flag'
    ap2_flux_col = 'ap2flux_{}pix'.format(int(radius2))
    ap2_ferr_col = 'ap2ferr_{}pix'.format(int(radius2))
    ap2_flag_col = 'ap2_flag'

    # calculate effective radius "R_e"
    reff, flag = sep.flux_radius(data_bkgsub, obj['x'], obj['y'], 6.*obj['a'], frac=0.5, normflux=ap1_flux) # , subpix=5
    reff_colname = 'reff'

    # Make a padas DataFrame
    col_names = np.array(obj.dtype.names) 
    df_obj = pd.DataFrame(obj, columns=col_names)

    # Add results of aperture photometory
    df_obj[ap1_flux_col] = ap1_flux
    df_obj[ap1_ferr_col] = ap1_ferr
    df_obj[ap1_flag_col] = ap1_flag
    df_obj[ap2_flux_col] = ap2_flux
    df_obj[ap2_ferr_col] = ap2_ferr
    df_obj[ap2_flag_col] = ap2_flag
    # df_obj[check_remain_photutil_col] = check_remain_photutil
    # df_obj[threetimes_longaxis_col] = check_remain_3bai


    # effective radius
    df_obj[reff_colname] = reff
   
    if use_segmap:
        return df_obj, data_bkgsub, data_segmap, bkg_mean_x, bkg_rms_x
    if not use_segmap:
        return df_obj, data_bkgsub, bkg_mean_x, bkg_rms_x


@stop_watch
def region_for_check(newim,refim, newlist_seg):
    nhead,new_data = openfits(newim)
    rhead,ref_data = openfits(refim)
    gain_base = float(nhead['GAIN'])
    Nfullwell = float(nhead['FULLWELL'])   #[/electron]

    if Nfullwell/gain_base > 15000:
        Nsaturate = 15000
    else:
        Nsaturate = Nfullwell/gain_base

    if new_data.dtype.byteorder == '>':
        new_data = new_data.byteswap().newbyteorder()

    if ref_data.dtype.byteorder == '>':
        ref_data = ref_data.byteswap().newbyteorder()

    obj = sep.extract(new_data, thresh=Nsaturate, minarea=1,  filter_kernel=None, \
            filter_type='matched', segmentation_map=False, mask=None)

    obj2 = sep.extract(ref_data, thresh=Nsaturate, minarea=1,  filter_kernel=None, \
            filter_type='matched', segmentation_map=False, mask=None)
    for i in range(len(obj['x'])):  #なんかもうちょいかっこいい列ないですかね。
        seg_val = newlist_seg[(obj['ypeak'][i])-1][(obj['xpeak'][i])-1]
        if seg_val > 0:
            newlist_seg= np.where(newlist_seg==seg_val,-1,newlist_seg).astype(np.int32)
    for i in range(len(obj2['x'])):  #なんかもうちょいかっこいい列ないですかね。
        seg_val = newlist_seg[(obj2['ypeak'][i])-1][(obj2['xpeak'][i])-1]
        if seg_val > 0:
            newlist_seg= np.where(newlist_seg==seg_val,-1,newlist_seg).astype(np.int32)
    tobemasked = np.where(newlist_seg==-1,1,0).astype(np.int32)
    # hdu = fits.PrimaryHDU(tobemasked, header=None )
    # hdu.writeto('sat_'+newim ,overwrite=True)
    return tobemasked


# def select_for_alart(dfnew,dfcorr,maskmatrix):
#     dfnew_cut = dfnew.copy().sort_values('cpeak')
#     dfnew_cut = dfnew_cut.query('8 < cpeak < 1000')
#     dfnew_cut.reset_index(inplace=True, drop=True)

#     dfcorr_cut = dfcorr.copy().sort_values('cpeak')
#     dfcorr_cut = dfcorr_cut.query('8 < cpeak < 1000') 
#     dfcorr_cut.reset_index(inplace=True, drop=True) 
#     def model_for_alart(dfnew_cut,dfcorr_cut):
#         from lmfit.models import PowerLawModel  
#         x_new_cut = dfnew_cut['cpeak']
#         x_corr_cut = dfcorr_cut['cpeak']
#         dfnew_cut['peak_sqrtnpix'] = dfnew_cut['peak'] / np.sqrt(dfnew_cut['npix'])
#         y_new_cut = dfnew_cut['peak_sqrtnpix']
#         model = PowerLawModel()
#         # model = PolynomialModel(3)
#         params = model.guess(y_new_cut, x=x_new_cut)
#         result = model.fit(y_new_cut, params, x=x_new_cut)
#         c=result.best_values.get('amplitude')
#         a=result.best_values.get('exponent')
#         # dely5_n = result.eval_uncertainty(x=x_new_cut,sigma=5)
#         dely5_d = result.eval_uncertainty(x=x_corr_cut,sigma=5)
#         return c,a,dely5_d
#     c,a, dely5_d = model_for_alart(dfnew_cut,dfcorr_cut)

#     candidate = []
#     for j in range(len(dfcorr_cut['x'])):
#         if maskmatrix[dfcorr_cut['ycpeak'][j]][dfcorr_cut['xcpeak'][j]] == 1:
#             continue
#         elif (c*(dfcorr_cut['cpeak'][j])**a - dely5_d[j] <= dfcorr_cut['peak'][j] / np.sqrt(dfcorr_cut['npix'][j]) <= c*(dfcorr_cut['cpeak'][j])**a + dely5_d[j]):
#             candidate.append(j)

#         else:
#             continue

#     if candidate != []:
#         dfcorr_fin = dfcorr_cut.copy()
#         dfcorr_fin = dfcorr_fin.iloc[candidate,:] 
#         return  dfcorr_fin, 1

#     else:
#         return dfcorr , 0
