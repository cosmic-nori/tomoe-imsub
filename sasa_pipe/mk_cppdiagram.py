#!/usr/bin/env python

# detect_psf_comp_rev.py
"""
Perform object detection of embedded PSF models on an artificial image
and calculate the completeness to compare with the result of convolved detection
"""
# created    2021/11/17

# Importing packages
from argparse import ArgumentParser as ap
# from astropy.convolution import convolve
# from scipy import signal
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sys.path.append('/Users/arimanoriaki/Desktop/Desktop/Tomo-e_Gozen/PythonPhot/')
from PythonPhot import rdpsf
# sys.path.append('/Users/arimanoriaki/Desktop/Desktop/Tomo-e_Gozen/jupyter_files/my_modules/')
# from my_modules import pipeline_funcs as pipe

from astropy.io import fits
import sep

#2021/11/17 created
# def main(args):
def main(i,dfx,dfy,filename):
    """Perform detection of PSF models oembedded in an artificial image
    and calculate the completeness.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments defined as below.

    Returns
    -------
    bool
        Description of return value
    """
    # print('#=====Input base image name: {}=====#'.format(args.image))
    # print('#=====Input PSF name: {}=====#'.format(args.psf))

    thresh = 1.00

    # header, data = pipe.fits_open(args.image)
    # name = args.image
    # outdir = args.out
    name = filename
    outdir = filename.replace('.fits','_dir/')
    #if args.psf is not None:
    data_psf, header_hpsf = rdpsf.rdpsf(filename.replace('.fits','_psf.fits'))
    filter_kernel = data_psf
    filter_use = 'ON'
    sigmaX = header_hpsf['GAUSS4']  # Gaussian Sigma: X Direction
    sigmaY = header_hpsf['GAUSS5']  # Gaussian Sigma: Y Direction


    FWHM_X = 2.35482*sigmaX  #in pixel

    FWHM_Y = 2.35482*sigmaY  #in pixel

    separation = np.sqrt(FWHM_X**2+FWHM_Y**2)+ 0.5 #HWHM_rms

    # else:
    #     filter_kernel = None
    #     filter_use = 'OFF'

    #     separation = 3.5
    
    # Make a convolved image
    # data_psf_norm = data_psf / np.sum(data_psf) # normalization by dividing by the sum of the PSF data
    #data_conv = signal.convolve(data, data_psf_norm, mode='same')
    # astropy's convolution replaces the NaN pixels with a kernel-weighted
    # interpolation from their neighbors
    # data_conv = convolve(data, data_psf_norm)
    # data_corr_conv = convolve(data_corr, data_psf_norm)
    #data_conv_edgecut = data_conv[:, :]
    
    # Read the file which contains the position of embedded PSFs
    # df_psfpos = pd.read_csv((args.psfpos), index_col=0)
    
    # total_obj_lst_conv = []
    # true_num_lst_conv = []
    # false_num_lst_conv = []
    # #count_conv= 0
    
    # total_obj_lst_corr = []
    # true_num_lst_corr = []
    # false_num_lst_corr = []
    # #count_corr = 0
    
    # # SEP parameters
    #thresh = args.thresh
    # minarea=args.minarea; radius=args.radius; segmap=True
    minarea=1; radius=5; segmap=True
    # List of 'thresh' parameter
    # thresh = args.thresh # (start, stop, step)
    
    print('#=====SEP params: thresh={:.1f} , minarea={}, radius={}, segmap={}, filter_use={}=====#'.format(thresh, minarea, radius, segmap, filter_use))
    
    df_objnum_conv = pd.DataFrame(columns=['sn_in_new', 'minarea', 'obj_num', 'true_num', 'false_num'])

    header = fits.getheader(name)
    data = fits.getdata(name)
    diff_data = fits.getdata(name.replace('.fits','_hp_diff.fits'))
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
    if diff_data.dtype.byteorder == '>':
        # Please note about byte order, see SEP official tutorial below
        # https://sep.readthedocs.io/en/latest/tutorial.html
        diff_data = diff_data.byteswap().newbyteorder()
    # Measure a spatially varying background on the image
    bkg = sep.Background(data, **sep_bkg_option)

    # get a "global" mean and noise (rms) of the image background
    bkg_mean = bkg.globalback
    bkg_rms  = bkg.globalrms

    if data.dtype.byteorder == '>':
            # Please note about byte order, see SEP official tutorial below
            # https://sep.readthedocs.io/en/latest/tutorial.html
        data = data.byteswap().newbyteorder()



    data_yshape, data_xshape = data.shape
    # npsf = header['NPSF']
    # sn_ratio = header['SN_RATIO']

    df_new, segmap_new = detect_obj(data, thresh=thresh,\
                                            err=bkg_rms,minarea=minarea, filter_kernel=filter_kernel,\
                                            radius=radius, gain=0.23, use_segmap=segmap)

    df_conv, segmap_conv = detect_obj(diff_data, thresh=thresh,\
                                            err=bkg_rms,minarea=minarea, filter_kernel=filter_kernel,\
                                            radius=radius, gain=0.23, use_segmap=segmap)
                                            #SASA
    if i ==1: 
        df_new.to_csv('SN2021dn_new.csv')
        df_conv.to_csv('SN2021dn_conv.csv')
    # if i == 0:
    #         hdu = fits.PrimaryHDU(segmap_conv)
    #         hdu.writeto('segmap_SN6conv_mac04.fits', overwrite=True)

    # if i == 0:
    #         hdu = fits.PrimaryHDU(segmap_corr)
    #         hdu.writeto('segmap_SN6sourcecorr_mac04.fits', overwrite=True)
            
                                            #SASA
    # Exclude objects which are located near the edges of the image
    # edge = 10
    # df_conv = df_conv[(df_conv['x'] >= edge) & (df_conv['x'] <= (data_xshape-edge)) & (df_conv['y'] >= edge) & (df_conv['y'] <= (data_yshape-edge))]
    # df_conv.reset_index(drop=True)
    peak = []
    npix = []
    tnpix = []
    cpeak = []
    A = []
    B = []
    cf_f = []

    peaksatmasked = []
    npixsatmasked = []
    tnpixsatmasked = []
    cpeaksatmasked = []
    Asatmasked = []
    Bsatmasked = []
    cf_fsatmasked = []




    peakTNS = []
    npixTNS = []
    tnpixTNS = []
    cpeakTNS = []
    ATNS = []
    BTNS = []
    cf_fTNS = []
    satmask = fits.getdata('sat_'+name)

    f3=open('hoge_'+str(i)+'.reg','w')
    line = 'global color=red  dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n'
    f3.write(line)
    line = 'image \n'
    f3.write(line)
    for j in range(len(df_conv['x'])):
            if dist(df_conv['x'][j],df_conv['y'][j],dfx,dfy) <= 2.5:

                peakTNS.append(df_conv['peak'][j])
                npixTNS.append(df_conv['npix'][j])
                tnpixTNS.append(df_conv['tnpix'][j])
                cpeakTNS.append(df_conv['cpeak'][j])
                ATNS.append((df_conv['a'][j]))
                BTNS.append((df_conv['b'][j]))
                cf_fTNS.append((df_conv['flux'][j])/np.array(df_conv['cflux'][j]))
                print('FLUX : ',df_conv['ap1flux_7pix'][j])
                reg = "ellipse( " + str(df_conv['x'][j]) + " , " + str(df_conv['y'][j]) + " , "+ str(df_conv['a'][j]) +"," + str(df_conv['b'][j]) + str(df_conv['theta'][j] * 180. / np.pi) +  ") # color=green text={%s}\n" %('') #str(df_conv['ap2flux_7pix'][j])...
                f3.write(reg)
            
            elif satmask[df_conv['ycpeak'][j]][df_conv['xcpeak'][j]] == 1:
                peaksatmasked.append(df_conv['peak'][j])
                npixsatmasked.append(df_conv['npix'][j])
                tnpixsatmasked.append(df_conv['tnpix'][j])
                cpeaksatmasked.append(df_conv['cpeak'][j])
                Asatmasked.append((df_conv['a'][j]))
                Bsatmasked.append((df_conv['b'][j]))
                cf_fsatmasked.append((df_conv['flux'][j])/np.array(df_conv['cflux'][j]))
                reg = "ellipse( " + str(df_conv['x'][j]) + " , " + str(df_conv['y'][j]) + " , "+ str(df_conv['a'][j]) +"," + str(df_conv['b'][j]) + str(df_conv['theta'][j] * 180. / np.pi) +  ") # color=blue text={%s}\n" %('') #str(df_conv['ap2flux_7pix'][j])...
                f3.write(reg)

            
            else:
                peak.append(df_conv['peak'][j])
                npix.append(df_conv['npix'][j])
                tnpix.append(df_conv['tnpix'][j])
                cpeak.append(df_conv['cpeak'][j])
                A.append((df_conv['a'][j]))
                B.append((df_conv['b'][j]))
                cf_f.append((df_conv['flux'][j])/np.array(df_conv['cflux'][j]))
                reg = "ellipse( " + str(df_conv['x'][j]) + " , " + str(df_conv['y'][j]) + " , "+ str(df_conv['a'][j]) +"," + str(df_conv['b'][j]) + str(df_conv['theta'][j] * 180. / np.pi) +  ") # color=red text={%s}\n" %('') #str(df_conv['ap2flux_7pix'][j])...
                f3.write(reg)
            
            ellTNS = calc_ell(ATNS,BTNS)
            ell = calc_ell(A,B)
            ellsatmasked = calc_ell(Asatmasked,Bsatmasked)
    f3.close()


    X = df_new['cpeak']
    Y = df_new['peak']/np.sqrt(df_new['npix'])
    NPIX = df_new['npix']
    CF_F = (df_new['flux'])/np.array(df_new['cflux'])
    ELL = calc_ell(df_new['a'],df_new['b'])



    import matplotlib.cm as cm
    x1 = cpeak
    y1 = peak/np.sqrt(npix)


    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)

    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-ell')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20,marker='*',c=ellTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,marker='^',c=ellsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10,vmin=0, vmax=1, c=ell, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,alpha=0.15,c=ELL, cmap=cm.viridis)
    plt.legend()
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-ell_' +name.replace('.fits','')+ '_new-bkg*'+str(thresh)+"_diff_notcorr_wNdata.png", format="png", dpi=400)
    plt.figure()

#############################
    plt.xscale("log")
    plt.yscale("log")

    x1 = cpeak
    y1 = peak/np.sqrt(npix)


    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)
    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-ell')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20,marker='*',c=ellTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,marker='^',c=ellsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10,vmin=0, vmax=1, c=ell, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,alpha=0.15,c=ELL, cmap=cm.viridis)
    plt.legend()
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-ell_'+name.replace('.fits','') + '_new-bkg*'+str(thresh)+"_diff_notcorr_log_wNdata.png", format="png", dpi=400)
    plt.figure()


###########################
    x1 = cpeak
    y1 = peak/np.sqrt(npix)



    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)
    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-ell')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20,marker='*',c=ellTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,marker='^',c=ellsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10,vmin=0, vmax=1, c=ell, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,alpha=0.15,c=ELL, cmap=cm.viridis)
    plt.legend()
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-ell_'+name.replace('.fits','') + '_new-bkg*'+str(thresh)+"_diff_notcorr_wNdata.png", format="png", dpi=400)
    plt.figure()




    x1 = cpeak
    y1 = peak/np.sqrt(npix)


    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)

    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-cf/f')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20, vmin=1, vmax=1.5,marker='*',c=cf_fTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,vmin=1, vmax=1.5, marker='^',c=cf_fsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10, vmin=1, vmax=1.5,c=cf_f, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,vmin=1, vmax=1.5,alpha=0.15, c=CF_F, cmap=cm.viridis)
    plt.legend()
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-cf_f_' +name.replace('.fits','')+ '_new-bkg*'+str(thresh)+"_diff_notcorr_wNdata.png", format="png", dpi=400)
    plt.figure()

#############################
    plt.xscale("log")
    plt.yscale("log")

    x1 = cpeak
    y1 = peak/np.sqrt(npix)


    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)
    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-cf/f')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20, vmin=1, vmax=1.5,marker='*',c=cf_fTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,vmin=1, vmax=1.5, marker='^',c=cf_fsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10, vmin=1, vmax=1.5,c=cf_f, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,vmin=1, vmax=1.5,alpha=0.15, c=CF_F, cmap=cm.viridis)
    plt.legend()
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-cf_f_'+name.replace('.fits','') + '_new-bkg*'+str(thresh)+"_diff_notcorr_log_wNdata.png", format="png", dpi=400)
    plt.figure()


###########################
    x1 = cpeak
    y1 = peak/np.sqrt(npix)



    x2 = cpeakTNS
    y2 = peakTNS/np.sqrt(npixTNS)
    x3 = cpeaksatmasked
    y3 = peaksatmasked/np.sqrt(npixsatmasked)
    plt.title(name + ' new-bkg*'+str(thresh) +'σ ,  cpp-cf/f')
    plt.xlabel('cpeak')
    plt.ylabel(r'$peak/\sqrt{npix}$')
    plt.scatter(x2, y2,label='TNSobj',s=20, vmin=0, vmax=1.5,marker='*',c=cf_fTNS, cmap=cm.seismic)
    plt.scatter(x3, y3,label='sat_masked',s=10,vmin=0, vmax=1.5, marker='^',c=cf_fsatmasked, cmap=cm.seismic)
    sc = plt.scatter(x1, y1,label='else',s=10, vmin=0, vmax=1.5,c=cf_f, cmap=cm.seismic)
    sc2 = plt.scatter(X, Y,label='NEWIM',s=10,vmin=0, vmax=1.5,alpha=0.15, c=CF_F, cmap=cm.viridis)
    plt.legend()
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.colorbar(sc2)
    plt.colorbar(sc)
    plt.savefig(outdir+'cpp-cf_f_'+name.replace('.fits','') + '_new-bkg*'+str(thresh)+"_diff_notcorr_wNdata.png", format="png", dpi=400)
    plt.figure()


###########################
    if i== 1:
        # plt.figure(figsize=(5,6))
        # plt.rcParams["figure.subplot.left"] = 0.2


        frame_num=17
        dfnew_cut = df_new.copy().sort_values('cpeak')
        dfnew_cut['peak_sqrtnpix'] = dfnew_cut['peak'] / np.sqrt(dfnew_cut['npix'])
        dfnew_cut = dfnew_cut.query(' (10/(@frame_num)**0.5) -1 < peak_sqrtnpix < 0.33*cpeak+67/(@frame_num)**0.5')
        dfnew_cut = dfnew_cut.query('{} < cpeak < 1000'.format(8))
        dfnew_cut.reset_index(inplace=True, drop=True)

        def model_for_alart(dfnew_cut):
            from lmfit.models import PowerLawModel  
            x_new_cut = dfnew_cut['cpeak']
            #dfnew_cut['peak_sqrtnpix'] = dfnew_cut['peak'] / np.sqrt(dfnew_cut['npix'])
            y_new_cut = dfnew_cut['peak_sqrtnpix']
            model = PowerLawModel()
            # model = PolynomialModel(3)
            params = model.guess(y_new_cut, x=x_new_cut)
            result = model.fit(y_new_cut, params, x=x_new_cut)
            c=result.best_values.get('amplitude')
            a=result.best_values.get('exponent')
            # dely5_n = result.eval_uncertainty(x=x_new_cut,sigma=5)
            dely5_d = result.eval_uncertainty(x=x_new_cut,sigma=8)
            dely5_d = dely5_d  * np.sqrt(len(dfnew_cut['x'])/100)
            return c,a,dely5_d
        c,a, dely5_d = model_for_alart(dfnew_cut)

        fig, ax = plt.subplots()
        plt.xscale("log")
        plt.yscale("log")
        x1 = cpeak
        y1 = peak/np.sqrt(npix)
        xfit = np.arange(0,1e+3,1)

        x2 = cpeakTNS
        y2 = peakTNS/np.sqrt(npixTNS)
        x3 = cpeaksatmasked
        y3 = peaksatmasked/np.sqrt(npixsatmasked)
        plt.title('SN2021dn_New cpp-fit',fontsize=16)
        plt.xlabel('cpeak',fontsize=24)
        plt.ylabel(r'$peak/\sqrt{npix}$',fontsize=24)
        # plt.scatter(x2, y2,label='TNSobj',s=20, vmin=0, vmax=1.5,zorder=4,marker='*',c='red')
        # plt.scatter(x3, y3,label='sat_masked',s=10,vmin=0, vmax=1.5,zorder=3, marker='^',c='black')
        # plt.scatter(x1, y1,label='else',s=10, vmin=0, vmax=1.5,zorder=2, c='black')
        plt.scatter(X, Y,label='detections in New',s=10,vmin=0, vmax=1.5,alpha=0.7,zorder=1, c='blue')
        ax.fill_between(dfnew_cut['cpeak'], c*dfnew_cut['cpeak']**a-dely5_d, c*dfnew_cut['cpeak']**a+dely5_d,
                    color="C0",alpha=.3, label=r'star sequence', zorder=2)
        plt.vlines(8,0.1,500,colors='orange',label='cp = 8')
        plt.vlines(1000,0.1,500,colors='orange',label='cp = 1000')
        x = np.arange(5,1000,0.1)
        plt.plot(x,0.33*x+67/np.sqrt(17),c='black',alpha=0.3,label='boundary func')
        plt.ylim(1,100)
        plt.legend(loc='lower right',fontsize=16)
        plt.tick_params(labelsize=20)
        plt.savefig(outdir+"cpp-showfit.png", format="png", dpi=400 ,bbox_inches='tight')
        plt.figure()



def detect_obj(data, thresh=5, minarea=5, filter_kernel=None, use_segmap=True, err=5.0, radius=5.0, gain=0.23, mask=None, maskthresh=0.0):
    """
    Perform object detection and aperture photometry on 2D images by SEP
    """
    if use_segmap:
        obj, data_segmap = sep.extract(data, thresh=thresh, minarea=minarea, err=err, filter_kernel=filter_kernel, \
            filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
    elif not use_segmap:
        obj = sep.extract(data, thresh=thresh, minarea=minarea, err=err, filter_kernel=filter_kernel, \
            filter_type='matched', segmentation_map=use_segmap, mask=mask, maskthresh=maskthresh)
    
    # # Perform aperture photometry (with two different radii)
    ap1_flux, ap1_ferr, ap1_flag = sep.sum_circle(data, obj['x'], obj['y'], r=7.0, err=err, gain=gain)
    # ap2_flux, ap2_ferr, ap2_flag = sep.sum_circle(data_bkgsub, obj['x'], obj['y'], r=radius2, err=bkg_rms, gain=gain)

    # # Make column names of Pandas.DataFrame for the aperture photometory 
    ap1_flux_col = 'ap1flux_7pix'
    ap1_ferr_col = 'ap1ferr_7pix'
    ap1_flag_col = 'ap1_flag'
    # ap2_flux_col = 'ap2flux_{}pix'.format(int(radius2))
    # ap2_ferr_col = 'ap2ferr_{}pix'.format(int(radius2))
    # ap2_flag_col = 'ap2_flag'
    col_names = np.array(obj.dtype.names) 
    df_obj = pd.DataFrame(obj, columns=col_names)

    df_obj[ap1_flux_col] = ap1_flux
    df_obj[ap1_ferr_col] = ap1_ferr
    df_obj[ap1_flag_col] = ap1_flag

     # calculate effective radius "R_e"
    # reff, flag = sep.flux_radius(data_bkgsub, obj['x'], obj['y'], 6.*obj['a'], frac=0.5, normflux=ap1_flux) # , subpix=5
    # reff_colname = 'reff'

    # # Make a padas DataFrame


    
    if use_segmap:
        return df_obj, data_segmap
    if not use_segmap:
        return df_obj, err


def dist (x1, y1, x2, y2):
    d = np.sqrt((x1-x2)**2 + (y1 - y2)**2)
    return  d

def calc_ell(list1,list2):
    ell = []
    for i in range(len(list1)):
        ell.append(float(1-list2[i]/list1[i]))
    return np.array(ell)




if __name__ == "__main__":
    # parser = ap(description="Make a completeness plot of PSF detection rate")
    # parser.add_argument("image", type=str, help="Image (FITS) file path to be analyzed")
    # parser.add_argument("psf", type=str, help="PSF (FITS) file path for convolution kernel")
    # # parser.add_argument("psfpos", type=str,\
    # #                     help="PSF (csv) file path which contains the positions of PSFs in the input image")
    # # parser.add_argument("--thresh", type=float, default=1.0, help="SEP thresh parameter")
    # parser.add_argument("--minarea", type=int, default=1, help="SEP minarea parameter")
    # parser.add_argument("--radius", type=float, default=5.0, help="SEP radius parameter")
    # parser.add_argument("--out", type=str, default='./', help="output directory")
    # args = parser.parse_args()

    #main(args)


    df = pd.read_csv('choice_8fits.csv')

    for i in range(len(df['x'])):
        dfx=df['x'][i]
        dfy=df['y'][i]
        filename =df['filename'][i]
        main(i,dfx,dfy,filename)

  
