#modified on 2023-07-20


# Importing packages
from argparse import ArgumentParser as ap
import datetime
import time


import os, sys
import numpy as np
import pandas as pd
import subprocess
from PythonPhot import rdpsf

from astropy.io import fits
import sep
import tomoeutils

import run_hotpants
import mkpsf_func

# def main(args):
def main(i,dfx,dfy,filename,TNSname):
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

    thresh = 1.00
    fname = '../'+filename

    outdir = ('./result/')
    
    ref = fname.replace('.fits','_ref_tg.fits')



    start_date = datetime.datetime.now()
    print('# input science file = {}'.format(fname))

    # Run hotpants (image subtraction)
    start_time_hotpans = time.time()
    out_fitsname = run_hotpants.runhotpants(fname, ref, outdir, debug=False)
    elapsed_time_hotpants = time.time() - start_time_hotpans
    print('-------------------------------------------------------')
    print('| Elapsed time of hotpants image subtraction:', str(datetime.timedelta(seconds=elapsed_time_hotpants)))
    print('-------------------------------------------------------')
    out_fitsname = outdir + filename.replace('.fits','_hp_diff.fits')



    # Run PythonPhot.getpsf (make PSF model)
    start_time_mkpsf = time.time()
    psf_data, psf_header, seeing, df_psfstars ,seg_psf,ellips= mkpsf_func.make_psfmodel(fname, outdir, star_num=25, psfrad=5, fitrad=4)
    elapsed_time_mkpsf = time.time() - start_time_mkpsf
    print('-------------------------------------------------------')
    print('| Elapsed time of PythonPhot.getpsf:', str(datetime.timedelta(seconds=elapsed_time_mkpsf)))
    print('-------------------------------------------------------')

    # Open subtracted image
    diff_header, diff_data = tomoeutils.openfits(out_fitsname, debug=False)
    new_header, new_data = tomoeutils.openfits(fname, debug=False)



    filter_kernel = psf_data
    filter_use = 'ON'
    sigmaX = psf_header['GAUSS4']  # Gaussian Sigma: X Direction
    sigmaY = psf_header['GAUSS5']  # Gaussian Sigma: Y Direction



    sigmaR = np.sqrt((sigmaX**2+sigmaY**2)/2)


    # FWHM_X = 2.35482*sigmaX  #in pixel
    # FWHM_Y = 2.35482*sigmaY  #in pixel

    # seeing = np.sqrt(FWHM_X**2+FWHM_Y**2)




    # Run SEP.extract (object detection)
    start_time_sep = time.time()

    diff_header, diff_data = tomoeutils.openfits(out_fitsname, debug=False)
    new_header, data = tomoeutils.openfits(fname, debug=False)

    maskim = tomoeutils.mk_badpixmask(fname)       



    minarea=1; radius=5; segmap=True

    
    print('#=====SEP params: thresh={:.1f} , minarea={}, radius={}, segmap={}, filter_use={}=====#'.format(thresh, minarea, radius, segmap, filter_use))
    print(i, df['filename'][i])

    try:
        frame_num = round(new_header['EXPTIME'] /new_header['TFRAME']) - 1
        # frame_num = round(sci_header['EXPTIME'] / sci_header['T_FRAME'])  original : but not found in testfits (TFRAME)
    except:
        frame_num = 17
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

    data = data - bkg

    data_yshape, data_xshape = data.shape
    # npsf = header['NPSF']
    # sn_ratio = header['SN_RATIO']

    Dcorr = tomoeutils.mk_sourcecorr(fname)
    diff_data = diff_data*Dcorr
    bkgd = sep.Background(diff_data, **sep_bkg_option)

    diff_d = diff_data -bkgd

    try:
        df_new, segmap_new = detect_obj(data, thresh=thresh,\
                                            err=bkg_rms,minarea=minarea, filter_kernel=filter_kernel,\
                                            mask=maskim, radius=radius, gain=0.23, use_segmap=segmap)
    except:
        subprocess.call(['rm','ref_renorm.fits'])
        return i,filename, TNSname, seeing,sigmaX,sigmaY,-1,-1,-1,-1,elapsed_time_hotpants, elapsed_time_mkpsf, -1,-1
    try:
        df_conv, segmap_conv = detect_obj(diff_d, thresh=thresh,\
                                            err=bkg_rms,minarea=minarea, filter_kernel=filter_kernel,\
                                            mask=maskim,radius=radius, gain=0.23, use_segmap=segmap)
                                            #SASA

        print('### {0} objects were detected   ( {1} sigma, {2} pixel after conv) '.format(len(df_conv), thresh, minarea)  )
        elapsed_time_sep = time.time() - start_time_sep
        print('-------------------------------------------------------')
        print('| Elapsed time of SEP.extract:', str(datetime.timedelta(seconds=elapsed_time_sep)))
        print('-------------------------------------------------------')


    except:
        subprocess.call(['rm','ref_renorm.fits'])
        return i,filename, TNSname, seeing,sigmaX,sigmaY,-1,-1,-1,-1,elapsed_time_hotpants, elapsed_time_mkpsf, -1,-1



    start_time_classify = time.time()

    peakTNS = []
    npixTNS = []
    tnpixTNS = []
    cpeakTNS = []
    ATNS = []
    BTNS = []
    ap7FTNS = []
    
    satmask= tomoeutils.region_for_check(fname,'ref_renorm.fits',seg_psf)
    subprocess.call(['rm','ref_renorm.fits'])

    peakSAT = []
    npixSAT = []
    tnpixSAT = []
    cpeakSAT = []
    ASAT = []
    BSAT = []
    ap7FSAT = []


    df_new_cut = df_new.copy().sort_values('cpeak')
    df_new_cut['peak_sqrttnpix'] = df_new_cut['peak'] / np.sqrt(df_new_cut['tnpix'])###sasaoka
    df_new_cut.reset_index(inplace=True, drop=True)

    df_conv_cut = df_conv.copy().sort_values('cpeak')
    df_conv_cut.reset_index(inplace=True, drop=True) 

    for j in range(len(df_conv_cut['x'])):
            if dist(df_conv_cut['xcpeak'][j],df_conv_cut['ycpeak'][j],dfx,dfy) <= seeing:

                peakTNS.append(df_conv_cut['peak'][j])
                npixTNS.append(df_conv_cut['npix'][j])
                tnpixTNS.append(df_conv_cut['tnpix'][j])
                cpeakTNS.append(df_conv_cut['cpeak'][j])
                ATNS.append((df_conv_cut['a'][j]))
                BTNS.append((df_conv_cut['b'][j]))
                ap7FTNS.append((df_conv_cut['ap1flux_7pix'][j]))
    cand = 0
    true = 0

    #             continue

    x = df_conv_cut['cpeak']
    y = df_conv_cut['peak'] / np.sqrt(df_conv_cut['tnpix'])   

    ell_val = 1 - np.median(ellips)
    ub = moffat_bP08(x,seeing,bkg_rms,thresh)
    lb = moffat_bM08(x,seeing,bkg_rms,thresh)
    for j in range(len(x)):
        if satmask[df_conv_cut['ycpeak'][j]][df_conv_cut['xcpeak'][j]] == 1:
            continue
        elif 1 - df_conv_cut['b'][j]/df_conv_cut['a'][j] > 0.9 - 0.6*ell_val:
            continue

        elif (lb[j] <= y[j] <= ub[j]):
            if dist(df_conv_cut['x'][j],df_conv_cut['y'][j],dfx,dfy) <= seeing:
                true  = true + 1
            cand = cand + 1
            continue


    elapsed_time_classify = time.time() - start_time_classify
    # print(i,filename, TNSname, seeing,sigmaX,sigmaY,len(peakTNS),len(df_conv_cut['cpeak']),cand,true,elapsed_time_hotpants, elapsed_time_mkpsf, elapsed_time_sep, elapsed_time_classify)
    return i,filename, TNSname, seeing,sigmaX,sigmaY,len(peakTNS),len(df_conv_cut['cpeak']),cand,true,elapsed_time_hotpants, elapsed_time_mkpsf, elapsed_time_sep, elapsed_time_classify






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


    col_names = np.array(obj.dtype.names) 
    df_obj = pd.DataFrame(obj, columns=col_names)

    df_obj[ap1_flux_col] = ap1_flux
    df_obj[ap1_ferr_col] = ap1_ferr
    df_obj[ap1_flag_col] = ap1_flag



    
    if use_segmap:
        return df_obj, data_segmap
    if not use_segmap:
        return df_obj, err


def dist (x1, y1, x2, y2):
    d = np.sqrt((x1-x2)**2 + (y1 - y2)**2)
    return  d




def moffat_modelfunc(x,seeing,sigbkg,thresh):
    if x[0]==0:
        x[0]=x[0]+0.001
    alpha = 1.4874459*seeing**0.74605-0.42560065
    beta = -0.009436459043*alpha**1.99057+2.585906

    y =   2*x/(np.sqrt(np.pi*alpha**2*((x/sigbkg/thresh)**(1/beta)-1)))  

    return y

def moffat_bP08(x,seeing,sigbkg,thresh):
    if x[0]==0:
        x[0]=x[0]+0.001
    alpha = 1.4874459*seeing**0.74605-0.42560065
    beta = -0.009436459043*alpha**1.99057+3.385906

    y =   2*x/(np.sqrt(np.pi*alpha**2*((x/sigbkg/thresh)**(1/beta)-1)))  

    return y

def moffat_bM08(x,seeing,sigbkg,thresh):
    if x[0]==0:
        x[0]=x[0]+0.001
    alpha = 1.4874459*seeing**0.74605-0.42560065
    beta = -0.009436459043*alpha**1.99057+1.785906

    y =   2*x/(np.sqrt(np.pi*alpha**2*((x/sigbkg/thresh)**(1/beta)-1)))  

    return y








if __name__ == "__main__":
    df = pd.read_csv('test_sn5.csv')
    c=0
    INDEX =[]
    SEEinsec =[]
    sX=[]
    sY=[]
    TNSnum=[]
    total=[]
    CAND=[]
    TRUE=[]
    HPT=[]
    PPT=[]
    SPT=[]
    CLASST=[]

    for i in range(0,235):
            dfx=df['xpos'][i]
            dfy=df['ypos'][i]
            filename =df['filename'][i]
            TNSname = df['TNSname'][i]

            
            i,filename, TNSname, seeing,sigmaX,sigmaY,TNS_num,tot,cand,true,hpT,ppT,spT,classify = main(i,dfx,dfy,filename,TNSname)
            INDEX.append(i)
            SEEinsec.append(seeing*1.19)
            sX.append(sigmaX)
            sY.append(sigmaY)
            TNSnum.append(TNS_num)
            total.append(tot)
            CAND.append(cand)
            TRUE.append(true)
            HPT.append(hpT)
            PPT.append(ppT)
            SPT.append(spT)
            CLASST.append(classify)

            # break



    dfresult = pd.DataFrame(
        data={'filename': df['filename'],
            'TNSname':  df['TNSname'],
            'sigmaX': sX,
            'sigmaY': sY,
            'seeing in arcsec': SEEinsec,
            'TNS_num': TNSnum,
            'total detction': total,
            'cand': CAND,
            'true': TRUE
                }
    )

    timeres = pd.DataFrame(
            data ={
            'hpT':HPT, #hotpants time
            'ppT':PPT, #pythonphot time
            'spT':SPT, #sep time
            'cppT':CLASST #cpp-classification time
            }
    )
    timeres.to_csv('./timeres0714.csv')  
    dfresult.to_csv('./TGtransient0714.csv')
    print('')
    print('')
    print('ended')
    print('')
    print('')



