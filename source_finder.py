#!/usr/bin/env python3
# coding: utf-8

import os,sys
import subprocess
import time
from functools import wraps
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

import sep
#---PythonPhot---#
from PythonPhot import rdpsf
#---my modules---#
from tomoeutils import stop_watch, FitsHandler


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


test_class = SourceFinder('/home/arima/tomoe/tomoeflash/imsub/test/fits/sn2022eyj/sTMQ1202203240073175631.fits', 3, 5)
test_class.run_sep()