#!/usr/bin/env python3
# coding: utf-8

# -----------------------------------
# Detect TNS object on a Tomo-e image
# autohr: N. Arima
# create: 2022/12/12
# update: 20yy/mm/dd
# -----------------------------------


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
from scipy import signal
#---PythonPhot---#
from PythonPhot import rdpsf
#---my modules---#
import mkpsf_func
import tomoeutils
from tomoeutils import stop_watch
from run_hotpants import runhotpants


class Subtractor:
    def __init__(self, sci_path, ref_path, outdir, name='hotpants'):
        self.path_sci = sci_path
        self.path_ref = ref_path
        self.pathbase_sci = os.path.splitext(os.path.basename(sci_path))[0]
        self.pathbase_ref = os.path.splitext(os.path.basename(ref_path))[0]
        self.outdir = outdir
        self.name = name
        self.header_sci = None
        self.header_ref = None
        self.data_sci = None
        self.data_ref = None
        self.path_out = self.outdir + '/' + self.pathbase_sci + '_diff.fits'
        self.header_diff = None
        self.data_diff = None

    def open_fits(self):
        for path in [self.path_sci, self.path_ref]:
            with fits.open(path) as hdul:
                header = hdul[0].header
                data = hdul[0].data
            if header['NAXIS'] != 3 and header['NAXIS'] != 2:
                raise Exception("Unexpected data was given: header 'NAXIS' value must be 2 or 3")
            if path == self.path_sci:
                self.header_sci = header
                self.data_sci = data
            if path == self.path_ref:
                self.header_ref = header
                self.data_ref = data
                
    def run_hotpants(self, verbose=False, frame_num=11):
        # Get header info.
        try:
            frame_num = round(self.header_sci['EXPTIME'] / self.header_sci['T_FRAME'])
        except:
            frame_num = frame_num
        print('# of frames of the science image = {}'.format(frame_num))
        sci_gain = float(self.header_sci['GAIN']) / frame_num
        sci_rnoise = float(self.header_sci['RNOISE'])

        ### User-defined thresholds ###
        # for science image
        saturate = 15000.0
        iu = saturate
        il = -50
        ig = sci_gain
        ir = sci_rnoise
        # for reference image
        tu = 30
        tl = il
        tg = ig
        tr = ir
        hotparam_dict = dict(
            # for science image
            iu = iu,    # saturation limit for science image
            il = il,    # lower limit for science image
            ig = ig, 
            ir = ir,
            # for reference image
            tu = tu,
            tl = il,
            tg = ig,
            tr = ir
            )
        # Command to run hotpants software
        #cmd = ["hotpants", "-inim " + sci_img ," -tmplim" + ref_img, "-outim " + out_fitsname]
        # cmd_hotpants_base = "hotpants -inim " +  self.path_sci + " -tmplim " +  self.path_ref + " -outim " + self.pathbase_sci\ 
        # + " -iu " + str(iu) + " -il " + str(il) + " -ig " + str(ig) + " -ir " + str(ir)  + " -tu " + str(tu)\ 
        # + " -tl " +  str(tl) + " -tg " + str(tg) + " -tr " + str(tr)

        cmd_hotpants_base = "hotpants -inim " +  self.path_sci + " -tmplim " +  self.path_ref + " -outim " +\
             self.path_out + " -iu " + str(iu) + " -il " + str(il) + " -ig " + str(ig) + " -ir " + str(ir) +\
                 " -tu " + str(tu) + " -tl " +  str(tl) + " -tg " + str(tg) + " -tr " + str(tr)
        if verbose:
            cmd_hotpants = cmd_hotpants_base
        if not verbose:
            cmd_hotpants = cmd_hotpants_base + " >/dev/null 2>&1"

        # Execute hotpants
        subprocess.run(cmd_hotpants, shell=True)

        # Get header & data of the difference image
        with fits.open(self.path_out) as hdul:
            header = hdul[0].header
            data = hdul[0].data
        self.header_diff = header
        self.data_diff = data


    
test_class = Subtractor('/home/arima/tomoe/tomoeflash/imsub/test/fits/sn2022eyj/sTMQ1202203240073175631.fits', '/home/arima/tomoe/tomoeflash/imsub/test/fits/sn2022eyj/sTMQ1202203240073175631_ref_tg.fits', '/home/arima')
test_class.open_fits()
test_class.run_hotpants()
print(test_class.path_sci)
print(test_class.path_ref)
print(test_class.header_sci['EXP_ID'])
print(test_class.header_ref['SATURATE'])