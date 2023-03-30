#!/usr/bin/env python3
# coding: utf-8

# Python Standard Library
import sys,os
import datetime
import time

# Python External Library
import numpy as np

# My modules
import run_hotpants
import tomoeutils
import mkpsf_func
import subprocess



def main(sci,ref,thresh,outdir,debug,corr):
    objnum_lst = []
    hp_time_lst = []
    pp_time_lst = []
    sep_time_lst = []

    start_date = datetime.datetime.now()
    if os.path.isdir(sci):
        print(sci)
        file_names, file_basenames = tomoeutils.get_filename(sci)
        for i in range(len(file_names)):
            sci = file_names[i]
            # df_obj, hp_time, pp_time, sep_time, bkg_rms = run_all(sci,ref,outdir,debug,corr)
            df_obj, hp_time, pp_time, sep_time, bkg_rms,alart_or_not = run_all(sci,ref,thresh,outdir,debug,corr)
            objnum_lst.append(len(df_obj))
            hp_time_lst.append(hp_time)
            pp_time_lst.append(pp_time)
            sep_time_lst.append(sep_time)
    else:
        # df_obj, hp_time, pp_time, sep_time, bkg_rms = run_all(sci,ref,outdir,debug,corr)
        df_obj, hp_time, pp_time, sep_time, bkg_rms,alart_or_not = run_all(sci,ref,thresh,outdir,debug,corr)
    
    objnum_arr = np.array(objnum_lst, dtype=int)
    hp_time_arr = np.array(hp_time_lst)
    pp_time_arr = np.array(pp_time_lst)
    sep_time_arr = np.array(sep_time_lst)
    
    np.savetxt(outdir + '/' + 'objnum.txt', objnum_arr)
    # np.savetxt(outdir + '/' + 'time_hotpants.txt', hp_time_arr)
    # np.savetxt(outdir + '/' + 'time_pythonphot.txt', pp_time_arr)
    # np.savetxt(outdir + '/' + 'sep_time.txt', sep_time_arr)

    funcs_time_arr = np.vstack((hp_time_arr, pp_time_arr, sep_time_arr))
    funcs_time_arr_t = funcs_time_arr.T
    np.savetxt(outdir + '/' + 'time_funcs.txt', funcs_time_arr_t)

    ####SASAOKA###
    return df_obj, bkg_rms,alart_or_not




def run_all(sci,ref,thresh,outdir,debug,corr):
    start_date = datetime.datetime.now()
    print('# input science file = {}'.format(sci))
    # Run hotpants (image subtraction)
    start_time_hotpans = time.time()
    out_fitsname = run_hotpants.runhotpants(sci, ref, outdir, debug)
    elapsed_time_hotpants = time.time() - start_time_hotpans
    print('-------------------------------------------------------')
    print('| Elapsed time of hotpants image subtraction:', str(datetime.timedelta(seconds=elapsed_time_hotpants)))
    print('-------------------------------------------------------')
    out_fitsname = outdir + '/' + sci.replace('.fits','_hp_diff.fits')

    # Run PythonPhot.getpsf (make PSF model)
    start_time_mkpsf = time.time()
    psf_data, psf_header, seeingFWHM, df_psfstars ,seg_psf= mkpsf_func.make_psfmodel(sci, outdir, star_num=25, psfrad=5, fitrad=4)
    elapsed_time_mkpsf = time.time() - start_time_mkpsf
    print('-------------------------------------------------------')
    print('| Elapsed time of PythonPhot.getpsf:', str(datetime.timedelta(seconds=elapsed_time_mkpsf)))
    print('-------------------------------------------------------')

    # Open subtracted image
    diff_header, diff_data = tomoeutils.openfits(out_fitsname, debug=False)
    new_header, new_data = tomoeutils.openfits(sci, debug=False)

    # Run SEP.extract (object detection)
    start_time_sep = time.time()
#!/usr/bin/env python3
# coding: utf-8

# Python Standard Library
import sys,os
import datetime
import time

# Python External Library
import numpy as np

# My modules
import run_hotpants
import tomoeutils
import mkpsf_func
import subprocess



def main(sci,ref,thresh,outdir,debug,corr):
    objnum_lst = []
    hp_time_lst = []
    pp_time_lst = []
    sep_time_lst = []

    start_date = datetime.datetime.now()
    if os.path.isdir(sci):
        print(sci)
        file_names, file_basenames = tomoeutils.get_filename(sci)
        for i in range(len(file_names)):
            sci = file_names[i]
            # df_obj, hp_time, pp_time, sep_time, bkg_rms = run_all(sci,ref,outdir,debug,corr)
            df_obj, hp_time, pp_time, sep_time, bkg_rms,tobemasked = run_all(sci,ref,thresh,outdir,debug,corr)
            objnum_lst.append(len(df_obj))
            hp_time_lst.append(hp_time)
            pp_time_lst.append(pp_time)
            sep_time_lst.append(sep_time)
    else:
        # df_obj, hp_time, pp_time, sep_time, bkg_rms = run_all(sci,ref,outdir,debug,corr)
        df_obj, hp_time, pp_time, sep_time, bkg_rms,tobemasked = run_all(sci,ref,thresh,outdir,debug,corr)
    
    objnum_arr = np.array(objnum_lst, dtype=int)
    hp_time_arr = np.array(hp_time_lst)
    pp_time_arr = np.array(pp_time_lst)
    sep_time_arr = np.array(sep_time_lst)
    
    np.savetxt(outdir + '/' + 'objnum.txt', objnum_arr)
    # np.savetxt(outdir + '/' + 'time_hotpants.txt', hp_time_arr)
    # np.savetxt(outdir + '/' + 'time_pythonphot.txt', pp_time_arr)
    # np.savetxt(outdir + '/' + 'sep_time.txt', sep_time_arr)

    funcs_time_arr = np.vstack((hp_time_arr, pp_time_arr, sep_time_arr))
    funcs_time_arr_t = funcs_time_arr.T
    np.savetxt(outdir + '/' + 'time_funcs.txt', funcs_time_arr_t)

    ####SASAOKA###
    return df_obj, bkg_rms,tobemasked




def run_all(sci,ref,thresh,outdir,debug,corr):
    start_date = datetime.datetime.now()
    print('# input science file = {}'.format(sci))
    # Run hotpants (image subtraction)
    start_time_hotpans = time.time()
    out_fitsname = run_hotpants.runhotpants(sci, ref, outdir, debug)
    elapsed_time_hotpants = time.time() - start_time_hotpans
    print('-------------------------------------------------------')
    print('| Elapsed time of hotpants image subtraction:', str(datetime.timedelta(seconds=elapsed_time_hotpants)))
    print('-------------------------------------------------------')
    out_fitsname = outdir + '/' + sci.replace('.fits','_hp_diff.fits')

    # Run PythonPhot.getpsf (make PSF model)
    start_time_mkpsf = time.time()
    psf_data, psf_header, seeingFWHM, df_psfstars ,seg_psf= mkpsf_func.make_psfmodel(sci, outdir, star_num=25, psfrad=5, fitrad=4)
    elapsed_time_mkpsf = time.time() - start_time_mkpsf
    print('-------------------------------------------------------')
    print('| Elapsed time of PythonPhot.getpsf:', str(datetime.timedelta(seconds=elapsed_time_mkpsf)))
    print('-------------------------------------------------------')

    # Open subtracted image
    diff_header, diff_data = tomoeutils.openfits(out_fitsname, debug=False)
    new_header, new_data = tomoeutils.openfits(sci, debug=False)

    # Run SEP.extract (object detection)
    start_time_sep = time.time()

    # thresh=3
    minarea=1
    # df_obj, data_bkgsub, data_segmap, bkg_mean, bkg_rms = tomoeutils.detectobj2(sci, diff_data, corr, thresh, minarea, filter_kernel=None, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23)
    maskim = tomoeutils.mk_badpixmask(sci)       
    # defaultconv = np.array([[1,2,1],[2,4,2],[1,2,1]])
    dfcorr, data_bkgsub, data_segmap, bkg_mean_x, bkg_rms_x = tomoeutils.detectobj2(sci, diff_data,corr, thresh, minarea, filter_kernel=psf_data, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23,mask=maskim)
    dfnew, data_bkgsub, data_segmap, bkg_mean_x, bkg_rms_new_x = tomoeutils.detectobj2(sci, new_data,corr, thresh, minarea, filter_kernel=psf_data, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23,mask=maskim)
    print('### {0} objects were detected   ( {1} sigma, {2} pixel ) '.format(len(df_obj), thresh, minarea)  )
    elapsed_time_sep = time.time() - start_time_sep
    print('-------------------------------------------------------')
    print('| Elapsed time of SEP.extract:', str(datetime.timedelta(seconds=elapsed_time_sep)))
    print('-------------------------------------------------------')

    end_date = datetime.datetime.now()
    print('Start date: {}'.format(start_date))
    print('End date: {}'.format(end_date))


    tobemasked = tomoeutils.region_for_check(sci,'ref_renorm.fits',seg_psf)
    df_obj, alart_or_not = tomoeutils.select_for_alart(dfnew,dfcorr,tobemasked)
    subprocess.call(['rm','ref_renorm.fits'])
    return df_obj, elapsed_time_hotpants, elapsed_time_mkpsf, elapsed_time_sep, bkg_rms_x,alart_or_not
    # thresh=3
    minarea=1
    # df_obj, data_bkgsub, data_segmap, bkg_mean, bkg_rms = tomoeutils.detectobj2(sci, diff_data, corr, thresh, minarea, filter_kernel=None, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23)
    maskim = tomoeutils.mk_badpixmask(sci)       
    # defaultconv = np.array([[1,2,1],[2,4,2],[1,2,1]])
    dfcorr, data_bkgsub, data_segmap, bkg_mean_x, bkg_rms_x = tomoeutils.detectobj2(sci, diff_data,corr, thresh, minarea, filter_kernel=psf_data, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23,mask=maskim)
    dfnew, data_bkgsub, data_segmap, bkg_mean_x, bkg_rms_new_x = tomoeutils.detectobj2(sci, new_data,corr, thresh, minarea, filter_kernel=psf_data, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23,mask=maskim)
    print('### {0} objects were detected   ( {1} sigma, {2} pixel ) '.format(len(df_obj), thresh, minarea)  )
    elapsed_time_sep = time.time() - start_time_sep
    print('-------------------------------------------------------')
    print('| Elapsed time of SEP.extract:', str(datetime.timedelta(seconds=elapsed_time_sep)))
    print('-------------------------------------------------------')

    end_date = datetime.datetime.now()
    print('Start date: {}'.format(start_date))
    print('End date: {}'.format(end_date))


    tobemasked = tomoeutils.region_for_check(sci,'ref_renorm.fits',seg_psf)
    df_obj, alart_or_not = tomoeutils.select_for_alart(dfnew,dfcorr,tobemasked)
    subprocess.call(['rm','ref_renorm.fits'])
    return df_obj, elapsed_time_hotpants, elapsed_time_mkpsf, elapsed_time_sep, bkg_rms_x,alart_or_not


import pandas as pd 
files = pd.read_csv('choice_8fits.csv')

def run():
    for i in range(len(files['filename'])):
        main(files['filename'][i],files['filename'][i].replace('.fits','_ref_tg.fits'),1.00,'.',False,True)

if __name__ == '__main__':
    run()
