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
        # return header, data
    
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

