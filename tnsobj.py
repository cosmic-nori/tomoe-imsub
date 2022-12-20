#!/usr/bin/env python3
# coding: utf-8


# class TNSObjDetect:
#     def __init__(self, sci_fits, ref_fits, obj_name, ra, dec):
#         self.sci_fits = sci_fits
#         self.ref_fits = ref_fits
#         self.sci_fits_base = os.path.splitext(os.path.basename(sci_fits))
#         self.ref_fits_base = os.path.splitext(os.path.basename(ref_fits))
#         self.obj_name = obj_name
#         self.ra = ra
#         self.dec = dec


class TNSObj:
    def __init__(self, name, ra, dec):
        self.name = name
        self.ra = ra
        self.dec = dec