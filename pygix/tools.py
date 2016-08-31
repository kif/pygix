#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def quadrant_average(data, x=None, y=None, dummy=0):
    """
    Function to perform four quadrant averaging of fiber diffraction
    patterns. If only the data array is given, the function 
    assumes the center of reciprocal space is the central pixel
    and the returned averaged array will have the same shape as 
    the input data. 

    If x and y scaling are given, it deduces the
    center from these arrays. One quadrant will have the size of 
    the largest quadrant and the resulting array will be larger 
    than the input array. In this case a new x and y will be
    calculated and returned.
    
    Args:
        data (ndarray): Input reciprocal space map array.
        x (ndarray): x scaling of the image.
        y (ndarray: y scaling of the image.
        dummy (int): Value of masked invalid regions.

    Returns:
        out_full (ndarray): The four quadrant averaged array
    """
    if (x is not None) and (y is not None):
        xcen = np.argmin(abs(x))
        ycen = np.argmin(abs(y))
    elif (x is None) and (y is None):
        ycen = data.shape[0] / 2.0
        xcen = data.shape[1] / 2.0
    else:
        raise RuntimeError('Must pass both x and y scales or neither')

    quad1 = np.flipud(np.fliplr(data[0:ycen, 0:xcen]))
    quad2 = np.fliplr(data[ycen:, 0:xcen])
    quad3 = data[ycen:, xcen:]
    quad4 = np.flipud(data[0:ycen, xcen:])

    quad_shapey = max(data.shape[0] - ycen, data.shape[0] - (data.shape[0] - ycen))
    quad_shapex = max(data.shape[1] - xcen, data.shape[1] - (data.shape[1] - xcen))
    mask = np.zeros((quad_shapey, quad_shapex))
    out = np.zeros((quad_shapey, quad_shapex))

    out[np.where(quad1 > dummy)] += quad1[np.where(quad1 > dummy)]
    out[np.where(quad2 > dummy)] += quad2[np.where(quad2 > dummy)]
    out[np.where(quad3 > dummy)] += quad3[np.where(quad3 > dummy)]
    out[np.where(quad4 > dummy)] += quad4[np.where(quad4 > dummy)]

    mask[np.where(quad1 > dummy)] += 1
    mask[np.where(quad2 > dummy)] += 1
    mask[np.where(quad3 > dummy)] += 1
    mask[np.where(quad4 > dummy)] += 1

    out[np.where(mask > 0)] /= mask[np.where(mask > 0)]
    out[np.where(mask == 0)] = dummy

    out_full = np.zeros((out.shape[0] * 2, out.shape[1] * 2))
    xcen = out_full.shape[1] / 2.0
    ycen = out_full.shape[0] / 2.0

    out_full[0:ycen, 0:xcen] = np.flipud(np.fliplr(out))
    out_full[0:ycen, xcen:] = np.flipud(out)
    out_full[ycen:, xcen:] = out
    out_full[ycen:, 0:xcen] = np.fliplr(out)

    if (x is not None) and (y is not None):
        xout = np.linspace(-abs(x).max(), abs(x).max(), out_full.shape[1])
        yout = np.linspace(-abs(y).max(), abs(y).max(), out_full.shape[0])
        return out_full, xout, yout
    else:
        return out_full


def sector_roi(chi_pos=None, chi_width=None, radial_range=None):
    """Generate array defining region of interest for sector integration.

    Args:
        chi_pos (float): chi angle (deg) defining the centre of the sector.
        chi_width (float): width (deg) of sector.
        radial_range (tuple): integration range (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    if (len([x for x in [chi_pos, chi_width, radial_range] if x is not None]) is 0) \
            or (radial_range is None):
        raise RuntimeError('Integration over whole image, no ROI to display.')
    return calc_sector(radial_range, chi_pos, chi_width)


def chi_roi(radial_pos, radial_width, chi_range=None):
    """Generate array defining region of interest for chi integration.

    Args:
        radial_pos (float): position defining the radius of the sector.
        radial_width (float): width (q or 2th) of sector.
        chi_range (tuple): azimuthal range (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    if (chi_range is None) or (chi_range[0] + chi_range[1] is 360):
        chi_width = None
        chi_pos = None
    else:
        chi_width = chi_range[1] - chi_range[0]
        chi_pos = chi_range[0] + chi_width / 2.0

    radial_min = radial_pos - radial_width / 2.0
    radial_max = radial_pos + radial_width / 2.0
    return calc_sector((radial_min, radial_max), chi_pos, chi_width)


def op_box_roi(ip_pos, ip_width, op_range):
    """Generate array defining region of interest for out-of-plane box integration.

    Args:
        ip_pos (float): in-plane centre of integration box.
        ip_width (float): in-plane width of integration box.
        op_range (tuple): out-of-plane range (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    ip_min = ip_pos - ip_width / 2.0
    ip_max = ip_pos + ip_width / 2.0
    return calc_box((ip_min, ip_max), op_range)


def ip_box_roi(op_pos, op_width, ip_range):
    """Generate array defining region of interest for in-plane box integration.

    Args:
        op_pos (float): out-of-plane centre of integration box.
        op_width (float): out-of-plane width of integration box.
        ip_range (tuple): in-plane range (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    op_min = op_pos - op_width / 2.0
    op_max = op_pos + op_width / 2.0
    return calc_box(ip_range, (op_min, op_max))


def calc_sector(radial_range, chi_pos, chi_width):
    """Main function for calculating sector region of interest.
    Called by sector_roi and chi_roi.

    Args:
        radial_range (tuple): integration range (min, max).
        chi_pos (float): chi angle (deg) defining the centre of the sector.
        chi_width (float): width (deg) of sector.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    if len([x for x in [chi_pos, chi_width] if x is not None]) is 1:
        raise RuntimeError('both chi_pos and chi_width must be supplied or neither')

    if (chi_pos is None) and (chi_width is None):
        chi_min = 0
        chi_max = 359
        npts = 360
    else:
        chi_min = -(chi_pos - chi_width / 2.0 - 90.0)
        chi_max = -(chi_pos + chi_width / 2.0 - 90.0)
        npts = abs(int(chi_max - chi_min))

    chi = np.radians(np.linspace(chi_min, chi_max, npts))

    # lower
    if radial_range[0] is 0:
        lo_qr = np.array(0)
        lo_qz = np.array(0)
    else:
        lo_qr = radial_range[0] * np.cos(chi)
        lo_qz = radial_range[0] * np.sin(chi)

    # upper
    hi_qr = (radial_range[1] * np.cos(chi))[::-1]
    hi_qz = (radial_range[1] * np.sin(chi))[::-1]

    qr = np.hstack((lo_qr, hi_qr))
    qz = np.hstack((lo_qz, hi_qz))

    if (chi_pos is not None) and (chi_width is not None):
        qr = np.append(qr, qr[0])
        qz = np.append(qz, qz[0])
    return qr, qz


def calc_box(ip_range, op_range):
    """Main function for calculating box regions of interest.
    Called by op_box_roi and ip_box_roi.

    Args:
        ip_range (tuple): in-plane (min, max).
        op_range (tuple): out-of-plane (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    qr = [ip_range[0], ip_range[0], ip_range[1], ip_range[1], ip_range[0]]
    qz = [op_range[0], op_range[1], op_range[1], op_range[0], op_range[0]]
    return qr, qz