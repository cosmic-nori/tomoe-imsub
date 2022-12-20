#!/usr/bin/env python3
# coding: utf-8

import sys,os
import tomoeutils
from tomoeutils import FitsHandler
import numpy as np
from PythonPhot import aper
from PythonPhot import getpsf
from PythonPhot import rdpsf


class PSF(FitsHandler):
    def __init__(self, fits_path, out_dir, psfrad=7, fitrad=6, funcname='Gaussian', badpixmin=-500, badpixmax=15000):
        super().__init__(fits_path, out_dir)
        self.df_obj = None
        self.psfrad = psfrad
        self.fitrad = fitrad
        self.badpixmin = badpixmin
        self.badpixmax = badpixmax
        self.funcname = funcname
        self.header, self.data = FitsHandler.static_open(fits_path)
        self.gain = self.header['GAIN']

    @staticmethod
    def select_psfstars(df_obj, edge_pix=10, max_peak=15000.0):
        """
        Select bright stars from detected objects to make a PSF model of the image
        
        We adopt using this function 'select_psfstars' instead of 'make_df_psfstars'
        in order to manage images with poor seeing

        History
        -------
        2022/12/02 created 
        """
        
        # Remove objects detected at the edges of the sensor
        df_edgecut = df_obj.query('(x > @edge_pix  & x < 2000 - @edge_pix) & (y > @edge_pix  & y < 1128 - @edge_pix)')
        
        # Obtain quartile of log_reff
        Q1 = df_edgecut['log_reff'].quantile(0.25)
        Q3 = df_edgecut['log_reff'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_min = Q1 - (IQR) * 1.5
        outlier_max = Q3 + (IQR) * 1.5
        
        # Remove objects by the quartile (0.25 - 0.75) of log_reff and peak < max_peak
        df_psfstars = df_edgecut.query('(@outlier_min < log_reff < @outlier_max) & peak < @max_peak')
        # Sort by flux in descending order
        df_psfstars = df_psfstars.sort_values('flux', ascending=False).reset_index(drop=True)
        
        return df_psfstars


    def make_psfmodel(self, star_num=25, psfrad=5, fitrad=4):
        """
        Make a PSF model from a list of bright stars in the input image

        History
        -------
        2022/12/01 : created
        """

        sep_psfstar_params = dict(
        filter_kernel = None,
        thresh = 5, # (float)
        minarea = 10,  # (int) in pixels
        )
        
        gain = self.header['GAIN']
        # pix_scale = header[''] # arcsec / pixel
        pix_scale = 1.19 # arcsec / pixel

        df_psfstars, data_bkgsub, data_segmap, psfstar_bkg_mean, psfstar_bkg_rms = tomoeutils.detectobj(self.data, **sep_psfstar_params, gain=self.gain, use_segmap=True)
            
        # Add log(X2+Y2) and log(r_eff) columns o the dataframe
        df_psfstars.loc[:, 'log_x2y2'] = np.log10(df_psfstars['x2'] + df_psfstars['y2'])
        df_psfstars.loc[:, 'log_reff'] = np.log10(df_psfstars['reff'] * pix_scale) # in arcsec 

        # Select PSF stars
        df_psfstars_all = PSF.select_psfstars(df_psfstars)
        print('### {} PSF stars were found ###'.format(len(df_psfstars_all)))
        
        if len(df_psfstars_all) >= star_num:
            df_psfstars = df_psfstars_all[:star_num]
            df_psfstars = df_psfstars.reset_index(drop=True)
        else:
            df_psfstars = df_psfstars_all
            df_psfstars = df_psfstars.reset_index(drop=True)
            
        print('### {} PSF stars were used ###'.format(len(df_psfstars)))
        
        # for check params 2021/12/15
        # save PSF star catalog
        #psfstar_df_savename = self.cat_psfstar_dir + '/' + self.file_basename + '_df_psfstar.pkl'
        #df_psfstars.to_pickle(psfstar_df_savename)
        
        # Make a cutout images of PSF selected stars as a pdf file
        '''
        fig_titile = 'Objects used to make a PSF model ({}), seeing FWHM = {:.2f} arcsec'.format(len(df_psfstars),\
                                                                                                    fwhm_median)
        cutout.show_df_frameobj(df_psfstars, self.stack_filename, wcs=False, stack=True, ncol=5, nrow=5,\
                            figsize=(24, 24), title=fig_titile, output_path=self.cat_psfstar_dir,\
                            save=True, savename='_psfstar')
        '''
        #df_psfstars = df_psfstars.rename(columns={'index': 'segid'})
        #df_psfstars['segid'] += 1 # add 1 because the dataframe index starts at 0
        df_psfstars.insert(0, 'psfstar_id', np.arange(1, len(df_psfstars)+1, 1, dtype=int))
        
        star_xpos = df_psfstars['x'].values
        star_ypos = df_psfstars['y'].values

        dao_mag, dao_magerr, dao_flux, dao_fluxerr,\
        dao_sky, dao_skyerr, dao_badflag, dao_outstr =\
        aper.aper(self.data, star_xpos, star_ypos, phpadu=gain, apr=5, zeropoint=25, skyrad=[40,50], badpix=[self.badpixmin, self.badpixmax], exact=True)
        # use the stars at those coords to generate a PSF model
        # CALLING SEQUENCE:
        # gauss,psf,psfmag = getpsf.getpsf( image, xc, yc, apmag, sky, ronois, phpadu,\
        #                    idpsf, psfrad, fitrad, psfname)
        
        # Caution!: In the current script, you must set save=True (otherwise, the pipeline doesn't work)
        # savename of PSF model
        psf_model_savename = self.out_dir + '/' + self.fits_basename + '_psf.fits'
        # save PSF star catalog
        psfstar_df_savename = self.out_dir + '/' + self.fits_basename + '_df_psfstar.pkl'
        df_psfstars.to_pickle(psfstar_df_savename)

        #####################################
        # Here are getpsf optinal arguments #
        #####################################
        psfrad=psfrad; fitrad = fitrad; debug=False; verbose=False
        #psfrad=5; fitrad = 4; debug=True
        gauss, psf, psfmag = getpsf.getpsf(self.data, star_xpos, star_ypos, dao_mag, dao_sky, 1, 1, np.arange(len(star_xpos)), psfrad=psfrad,\
                                            fitrad=fitrad, psfname=psf_model_savename,\
                                            debug=debug, verbose=verbose) #, save=save
        # rdpsf.rdpsf needs the FITS file, psf_model_savename
        try:
            psf_data, psf_header = rdpsf.rdpsf(psf_model_savename)
            return psf_data, psf_header, df_psfstars
        
        
        except FileNotFoundError:
            print('### Sorry, getpsf could not make a PSF model FITS file ###')


class_test = PSF('/home/arima/tomoe/tomoeflash/imsub/test/fits/sn2022eyj/sTMQ1202203240073175631.fits', '/home/arima')
class_test.make_psfmodel()
