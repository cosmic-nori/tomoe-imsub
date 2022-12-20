#!/usr/bin/env python3
# coding: utf-8

# Python Standard Library
import sys,os
from argparse import ArgumentParser as ap
import datetime
import time

# Python External Library
import numpy as np

# My modules
import run_hotpants
import tomoeutils
import mkpsf_func

def main(args):
    objnum_lst = []
    hp_time_lst = []
    pp_time_lst = []
    sep_time_lst = []

    start_date = datetime.datetime.now()
    if os.path.isdir(args.sci):
        print(args.sci)
        file_names, file_basenames = tomoeutils.get_filename(args.sci)
        for i in range(len(file_names)):
            args.sci = file_names[i]
            df_obj, hp_time, pp_time, sep_time = run_all(args)
            objnum_lst.append(len(df_obj))
            hp_time_lst.append(hp_time)
            pp_time_lst.append(pp_time)
            sep_time_lst.append(sep_time)
    else:
        run_all(args)
    
    objnum_arr = np.array(objnum_lst, dtype=int)
    hp_time_arr = np.array(hp_time_lst)
    pp_time_arr = np.array(pp_time_lst)
    sep_time_arr = np.array(sep_time_lst)
    
    np.savetxt(args.outdir + '/' + 'objnum.txt', objnum_arr)
    # np.savetxt(args.outdir + '/' + 'time_hotpants.txt', hp_time_arr)
    # np.savetxt(args.outdir + '/' + 'time_pythonphot.txt', pp_time_arr)
    # np.savetxt(args.outdir + '/' + 'sep_time.txt', sep_time_arr)

    funcs_time_arr = np.vstack((hp_time_arr, pp_time_arr, sep_time_arr))
    funcs_time_arr_t = funcs_time_arr.T
    np.savetxt(args.outdir + '/' + 'time_funcs.txt', funcs_time_arr_t)

def run_all(args):
    start_date = datetime.datetime.now()
    print('# input science file = {}'.format(args.sci))
    # Run hotpants (image subtraction)
    start_time_hotpans = time.time()
    out_fitsname = run_hotpants.runhotpants(args.sci, args.ref, args.outdir, args.debug)
    elapsed_time_hotpants = time.time() - start_time_hotpans
    print('-------------------------------------------------------')
    print('| Elapsed time of hotpants image subtraction:', str(datetime.timedelta(seconds=elapsed_time_hotpants)))
    print('-------------------------------------------------------')

    # Run PythonPhot.getpsf (make PSF model)
    start_time_mkpsf = time.time()
    psf_data, psf_header, df_psfstars = mkpsf_func.make_psfmodel(args.sci, args.outdir, star_num=25, psfrad=5, fitrad=4)
    elapsed_time_mkpsf = time.time() - start_time_mkpsf
    print('-------------------------------------------------------')
    print('| Elapsed time of PythonPhot.getpsf:', str(datetime.timedelta(seconds=elapsed_time_mkpsf)))
    print('-------------------------------------------------------')

    # Open subtracted image
    diff_header, diff_data = tomoeutils.openfits(out_fitsname, debug=False)

    # Run SEP.extract (object detection)
    start_time_sep = time.time()
    df_obj, data_bkgsub, data_segmap, bkg_mean, bkg_rms = tomoeutils.detectobj(diff_data, thresh=1.5, minarea=5, filter_kernel=psf_data, use_segmap=True, radius1=5.0, radius2=7.0, gain=0.23)
    print('### {} objects were detected'.format(len(df_obj)))
    elapsed_time_sep = time.time() - start_time_sep
    print('-------------------------------------------------------')
    print('| Elapsed time of SEP.extract:', str(datetime.timedelta(seconds=elapsed_time_sep)))
    print('-------------------------------------------------------')

    end_date = datetime.datetime.now()
    print('Start date: {}'.format(start_date))
    print('End date: {}'.format(end_date))

    return df_obj, elapsed_time_hotpants, elapsed_time_mkpsf, elapsed_time_sep

    
if __name__ == '__main__':
    #sci_default = '/home/arima/tomoe/tomoeflash/imsub/test/fits/frame_fits/rTMQ2202003220028809433_frame001.fits'
    sci_default = '/home/arima/tomoe/tomoeflash/imsub/test/fits/frame_fits'
    ref_default = '/home/arima/tomoe/tomoeflash/imsub/test/fits/stack_fits/rTMQ2202003220028809433_median_stack.fits'
    outdir_default = '/home/arima/tomoe/tomoeflash/imsub/test/out_dir'
    # make parser
    parser = ap(description='Execute hotpants image subtraction anaysis')
    parser.add_argument('-s', '--sci', type=str, nargs='*', default=sci_default, help='Input science FITS file')
    parser.add_argument('-r', '--ref', type=str, default=ref_default, help='Input reference FITS file')
    parser.add_argument('-o', '--outdir', type=str,default=outdir_default, help='Output directory')
    parser.add_argument('-d', '--debug', action='store_true', help='Print the hotpants stdout messages')
    # analyse arguments
    args = parser.parse_args()
    main(args)