#!/usr/bin/env python
# -*- coding: utf-8 -*-
# REQUIRES pyFAI v 0.10 or greater 

__author__ = "Thomas Dane, Jerome Kieffer"
__contact__ = "dane@esrf.fr"
__license__ = "GPLv3+"
__copyright__ = "ESRF - The European Synchrotron, Grenoble, France"
__date__ = "18/11/2014"
__status__ = "Development"
__docformat__ = "restructuredtext"

import os
import logging
logger = logging.getLogger("pygix.transform")
import types
import threading
import gc
import numpy
from math import pi
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)

from . import grazing_geometry
GrazingGeometry = grazing_geometry.GrazingGeometry
from . import grazing_units

import matplotlib.pyplot as plt

import pyFAI
import fabio
error = None

try:
    from pyFAI import ocl_azim  # IGNORE:F0401
    from pyFAI import opencl
except ImportError as error:  # IGNORE:W0703
    print 'Unable to import pyFAI.ocl_azim' 
    logger.warning("Unable to import pyFAI.ocl_azim")
    ocl_azim = None
    ocl = None
else:
    ocl = opencl.OpenCL()
    
print 'ocl =', ocl

try:
    from pyFAI import splitBBoxLUT
except ImportError as error:  # IGNORE:W0703
    logger.warning("Unable to import pyFAI.splitBBoxLUT for"
                 " Look-up table based azimuthal integration")
    splitBBoxLUT = None

try:
    from pyFAI import ocl_azim_lut
except ImportError as error:  # IGNORE:W0703
    logger.error("Unable to import pyFAI.ocl_azim_lut for"
                 " Look-up table based azimuthal integration on GPU")
    ocl_azim_lut = None

try:
    from pyFAI.fastcrc import crc32
except ImportError:
    from zlib import crc32

try:
    from pyFAI import splitPixel  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.splitPixel"
                  " full pixel splitting: %s" % error)
    splitPixel = None

try:
    from pyFAI import splitBBox  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.splitBBox"
                 " Bounding Box pixel splitting: %s" % error)
    splitBBox = None

try:
    from pyFAI import histogram  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.histogram"
                 " Cython OpenMP histogram implementation: %s" % error)
    histogram = None
del error  # just to see how clever pylint is !


class Transform(GrazingGeometry):
    """
    This class contains methods for performing transformation and
    integration of images collected in the grazing-incidence geometry.
 
    The operations are applicable to the related techniques of:
    - grazing-incidence X-ray diffraction (GIXRD, GID)
    - grazing-incidence wide-angle X-ray scattering (GIWAXS)
    - grazing-incidence small-angle X-ray scattering (GISAXS)
    These techniques will collectively be referred to as grazing-
    incidence X-ray scattering (GIXS).

    The class is based on pyFAI's AzimuthalIntegrator class. 
    GrazingTransform inherets from GrazingGeometry, which itself 
    inherets from pyFAI's Geometry class. All geometry calculation
    are done in the GrazingGeometry or pyFAI Geometry class.

    The development of these classes was necessary to introduce the 
    reciprocal space transformation equations (see grazingGeometry) 
    and to handle additional variables that could not be accounted for 
    in pyFAI:
      - grazingGeometry : the orientaition of the surface plane 
        (defines orientaition and sign of q vectors relative to input 
         image)
      - alpha_i :  the incident angle
      - misalign : an angle defining a misaligned surface plane (i.e. 
        not perfectly horizontal/vertical)

    These variables are class attributes to GrazingTransform. This is
    beneficial if the geometry and incident angle are constant (or at 
    least there is a commonly used incident angle) as look-up tables
    (LUTs) describing transformation and integration can be stored,
    speeding up processing time. When the geometry, incident angle or
    misalignment angle change either the class attributes can be 
    changed using the relevant getters or their values can be passed
    directly to the transformation/integration functions (therefore 
    not altering the class attributes).

    The grazing-incidence geometry ("giGeometry") can take one of 4 
    int values (0, 1, 2 or 3) is defined as follows:
        0: Sample plane horizontal; bottom-top = +ve Qz
        1: Sample plane vertical;   left-right = +ve Qz
        2: Sample plane horizontal; bottom-top = -ve Qz
        3: Sample plane vertical;   left-right = -ve Qz

    N.B. that it is not possible to rotate images before-hand as this 
    will nullify the poni file (as well as darks, flats, splines etc).
    For more details see the main documentation.
    
    The main methods are:
        >>> I, qxy, qz = gi.transform_image(data)
                Transform raw image to angular or q coordinates
        >>> I, q, chi  = gi.integrate_2d(data)
                Transform image to cake plot (chi vs q or 2theta)
        >>> I, q = gi.integrate_1d(data, npt)
                1d integration

    There are four wrappers for the 1d integration:
        >>> profile_sector
        >>> profile_chi
        >>> profile_op_box
        >>> profile_ip_box

    The error can be calculated and returned for each method if either
    a variance ndarray or error model is provided.

    For each method the resulting intensty data is always taken 
    directly from the source raw image (e.g. rather than performing 2d 
    integration on GIXS reciprocal space maps). This prevents mutliple 
    re-sampling, which can lead to the smearing of reflections and 
    inaccurate results.
    """
    
    DEFAULT_METHOD = "splitbbox"
    
    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
                 pixel1=0, pixel2=0, splineFile=None, detector=None, 
                 wavelength=None, useqx=True,
                 sample_orientation=None, incident_angle=None, tilt_angle=0):
        """
        Parameters
        ----------
        dist : float
            Distance from sample to detector plane (orthogonal 
            distance, not along the beam), in meters.
        poni1 : float
            Coordinate of the point-of-normal-incidence along the 
            detector's first dimension, in meters.
        poni2 : float
            Coordinate of the point-of-normal-incidence along the 
            detector's second dimension, in meters.
        rot1 : float
            First rotation from sample reference to detector 
            reference, in radians.
        rot2 : float
            Second rotation from sample reference to detector 
            reference, in radians.
        rot3 : float
            Third rotation from sample reference to detector 
            reference, in radians.
        pixel1 : float
            Pixel size of the fist dimension of the detector, in 
            meters.
        pixel2 : float
            Pixel size of the second dimension of the detector, in 
            meters.
        splineFile : str
            File containing the geometric distortion of the detector. 
            Overrides the pixel size.
        detector : str or pyFAI.detector
            Name of the detector or Detector instance.
        wavelength : float
            X-ray wavelength in meters.
        giGeometry : int
            Orientation of surface plane relative to image axes. Can 
            be 0, 1, 2 or 3. 
        alpha_i : float
            Incident angle (degrees).
        misalign : float
            Misalignment tilt angle of surface plane (degrees).
        """
        GrazingGeometry.__init__(self, dist, poni1, poni2,
                                 rot1, rot2, rot3,
                                 pixel1, pixel2, 
                                 splineFile, detector, 
                                 wavelength, useqx,
                                 sample_orientation, 
                                 incident_angle, tilt_angle)

        self._flatfield = None
        self._darkcurrent = None
        self._flatfield_crc = None
        self._darkcurrent_crc = None
        self.flatfiles = None
        self.darkfiles = None
        
        self.header = None
        
        #self._ocl_integrator = None
        #self._ocl_lut_integr = None
        #self._ocl_csr_integr = None
        #self._lut_integrator = None
        #self._csr_integrator = None
        #self._ocl_sem = threading.Semaphore()
        #self._lut_sem = threading.Semaphore()
        #self._csr_sem = threading.Semaphore()
        #self._ocl_csr_sem = threading.Semaphore()
        #self._ocl_lut_sem = threading.Semaphore()

        self._lut_sem = threading.Semaphore()
        self._ocl_lut_sem = threading.Semaphore()

        # Look-up tables for gi transformation
        self._ocl_lut_gi_transformer = None
        self._lut_gi_transformer = None
        self._csr_gi_transformer = None 

        # # Look-up tables for gi integration
        self._ocl_lut_gi_integrator = None
        self._lut_gi_integrator = None
        self._empty = 0.0
        
    def reset(self):
        """
        Reset GrazingTransform class. 
        """
        GrazingGeometry.reset(self)

        with self._lut_sem:
            self._lut_gi_transformer = None
            self._lut_gi_integrator = None

    def make_mask(self, data, mask=None,
                 dummy=None, delta_dummy=None, mode="normal"):
        """
        Taken from:
        pyFAI.azimuthalIntegrator.AzimuthalIntegrator.mask_mask

        This method combine two masks (dynamic mask from *data &
        dummy* and *mask*) to generate a new one with the 'or' binary
        operation.  One can adjust the level, with the *dummy* and
        the *delta_dummy* parameter, when you consider the *data*
        values needs to be masked out.

        This method can work in two different *mode*:

            * "normal": False for valid pixels, True for bad pixels
            * "numpy": True for valid pixels, false for others

        This method tries to accomodate various types of masks (like
        valid=0 & masked=-1, ...) and guesses if an input mask needs
        to be inverted.

        Parameters
        ----------
        data : ndarray
            Input array of data.
        mask : ndarray
            Input mask (if None, self.mask is used).
        dummy : float
            Dummy value of dead pixels.
        delta_dummy : float
            Precision of dummy pixels.
        mode : str
            Mode can be "normal" or "numpy" (inverted) or "where" 
            applied to the mask.

        Returns
        -------
        mask : ndarray of bool
            The new mask
        """
        shape = data.shape
        if mask is None:
            mask = self.mask
        if mask is None:
            mask = numpy.zeros(shape, dtype=bool)
        elif (mask.min() < 0) and (mask.max() == 0):  # 0 is valid, <0 is invalid
            mask = (mask < 0)
        else:
            mask = mask.astype(bool)
        if mask.sum(dtype=int) > mask.size // 2:
            logger.warning("Mask likely to be inverted as more"
                           " than half pixel are masked !!!")
            numpy.logical_not(mask, mask)
        if (mask.shape != shape):
            try:
                mask = mask[:shape[0], :shape[1]]
            except Exception as error:  # IGNORE:W0703
                logger.error("Mask provided has wrong shape:"
                             " expected: %s, got %s, error: %s" %
                             (shape, mask.shape, error))
                mask = numpy.zeros(shape, dtype=bool)
        if dummy is not None:
            if delta_dummy is None:
                numpy.logical_or(mask, (data == dummy), mask)
            else:
                numpy.logical_or(mask, abs(data-dummy) <= delta_dummy, mask)
        if mode == "numpy":
            numpy.logical_not(mask, mask)
        elif mode == "where":
            mask = numpy.where(numpy.logical_not(mask))
        return mask

    def gi_setup_lut(self, shape, npt, process, mask=None,
                     pos0_range=None, pos1_range=None,
                     mask_checksum=None, unit=grazing_units.Q):
        """
        Prepare a look-up-table for LUT-based methods. Used for
        giTransformImage, giIntegrate2d and giIntegrate1d.
        """
        if "__len__" in dir(npt) and len(npt) == 2:
            int2d = True
        else:
            int2d = False
        
        pos1, pos0   = self.giarray_from_unit(shape, process, "center", unit)
        dpos1, dpos0 = self.giarray_from_unit(shape, process, "delta", unit)

        if ("__len__" in dir(pos0_range)) and (len(pos0_range) > 1):
            pos0_min = min(pos0_range)
            pos0_maxin = max(pos0_range)
            default_pos0 = False
        else:
            pos0_min = pos0.min()
            pos0_maxin = pos0.max()
            default_pos0 = True
        if ("__len__" in dir(pos1_range)) and (len(pos1_range) > 1):
            pos1_min = min(pos1_range)
            pos1_maxin = max(pos1_range)
            default_pos1 = False
        else:
            pos1_min = pos1.min()
            pos1_maxin = pos1.max()
            default_pos1 = True

        pos0Range = (pos0_min, pos0_maxin * EPS32)
        pos1Range = (pos1_min, pos1_maxin * EPS32)
        
        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape

        if int2d:
            gi_lut = splitBBoxLUT.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                              bins=npt,
                                              pos0Range=pos0Range,
                                              pos1Range=pos1Range,
                                              mask=mask,
                                              mask_checksum=mask_checksum,
                                              allow_pos0_neg=True,
                                              unit=unit)
        else:
            gi_lut = splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                              bins=npt,
                                              pos0Range=pos0Range,
                                              pos1Range=pos1Range,
                                              mask=mask,
                                              mask_checksum=mask_checksum,
                                              allow_pos0_neg=True,
                                              unit=unit)

        return gi_lut, default_pos0, default_pos1

    csr_progress = """
    def gi_setup_CSR(self, shape, npt, process, mask=None,
                     pos0_range=None, pos1_range=None,
                     mask_checksum=None, unit=grazing_units.Q, split="bbox"):
        \"""
        Prepare a look-up-table for LUT-based methods. Used for
        giTransformImage, giIntegrate2d and giIntegrate1d.
        \"""
        if "__len__" in dir(npt) and len(npt) == 2:
            int2d = True
        else:
            int2d = False
        if split == "full":
            pos = self.giarray_from_unit(shape, process, "corner", unit)
        else:
            pos1, pos0 = self.giarray_from_unit(shape, process, "center", unit)
            if split = "no": 
                dpos1, dpos0 = (None, None)
            else:
                dpos1, dpos0 = self.giarray_from_unit(shape, process, 
                                                      "delta", unit)
            if (pos1_range is None) and (not int2d):
                
                
                
        pos1, pos0   = self.giarray_from_unit(shape, process, "center", unit)
        dpos1, dpos0 = self.giarray_from_unit(shape, process, "delta", unit)

        if ("__len__" in dir(pos0_range)) and (len(pos0_range) > 1):
            pos0_min = min(pos0_range)
            pos0_maxin = max(pos0_range)
            default_pos0 = False
        else:
            pos0_min = pos0.min()
            pos0_maxin = pos0.max()
            default_pos0 = True
        if ("__len__" in dir(pos1_range)) and (len(pos1_range) > 1):
            pos1_min = min(pos1_range)
            pos1_maxin = max(pos1_range)
            default_pos1 = False
        else:
            pos1_min = pos1.min()
            pos1_maxin = pos1.max()
            default_pos1 = True

        pos0Range = (pos0_min, pos0_maxin * EPS32)
        pos1Range = (pos1_min, pos1_maxin * EPS32)
        
        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape

        if int2d:
            gi_lut = splitBBoxLUT.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                              bins=npt,
                                              pos0Range=pos0Range,
                                              pos1Range=pos1Range,
                                              mask=mask,
                                              mask_checksum=mask_checksum,
                                              allow_pos0_neg=True,
                                              unit=unit)
        else:
            gi_lut = splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                              bins=npt,
                                              pos0Range=pos0Range,
                                              pos1Range=pos1Range,
                                              mask=mask,
                                              mask_checksum=mask_checksum,
                                              allow_pos0_neg=True,
                                              unit=unit)

        return gi_lut, default_pos0, default_pos1
    """
    
    def giarray_from_unit(self, shape, process, typ, unit):
        """
        Generate an array of coordinates in different dimensions 
        (tthf_alpf, qxy_qz or qy_qz) for GIXS transformation.

        intmethod = ["angular", "reciprocal", "polar", 
                     "sector", "chi", "ipbox", "opbox"]
       
           | methods:              |   process name:
        ----------------------------------------------
        2D | transform_angular     |  angular
           | transform_reciprocal  |  reciprocal
           | transform_polar       |  polar
        ----------------------------------------------- 
        1D | profile_sector        |  sector
           | profile_chi           |  chi
           | profile_ipbox         |  ipbox
           | profile_op_box        |  opbox
       
        """
        if not typ in ("center", "corner", "delta"):
            raise RuntimeError(("Unkown type of array %s," % typ))
        
        if process in ["polar", "sector", "chi"]:
            typ = "abs_"+typ

        unit = grazing_units.to_unit(unit)
        out = GrazingGeometry.__dict__[unit[typ]](self, shape)

        if process in ["chi", "opbox"]:
            if "corner" in typ:
                temp = numpy.zeros((shape[0], shape[1], 4, 2))
                temp[:,:,:,0] = out[:,:,:,1]
                temp[:,:,:,1] = out[:,:,:,0]
                out = temp
            else:
                out = (out[1], out[0])
        return out


    #---------------------------------------------------------------------------
    #   Reciprocal space transformation function
    #---------------------------------------------------------------------------
    
    def transform_angular(self, data, npt=None, 
                          ip_range=None, op_range=None,
                          filename=None, correctSolidAngle=False,
                          variance=None, error_model=None,
                          mask=None, dummy=None, delta_dummy=None,
                          polarization_factor=None, dark=None, flat=None,
                          method='splitpix', unit='deg',
                          safe=True, normalization_factor=None, all=False):
        """ 
        Wrapper for transform_image to project input array into
        angluar coordinates I(2th_f,a_f).
        """
        unit = '2theta_%s' % unit
        out = self.transform_image(data, process='angular', npt=npt,
                      x_range=ip_range, y_range=op_range, filename=filename, 
                      correctSolidAngle=correctSolidAngle,
                      variance=variance, error_model=error_model,
                      mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                      polarization_factor=polarization_factor,
                      dark=dark, flat=flat, method=method, unit=unit, safe=safe,
                      normalization_factor=normalization_factor, all=all)
        return out
    
    def transform_reciprocal(self, data, npt=None, 
                             ip_range=None, op_range=None,
                             filename=None, correctSolidAngle=False,
                             variance=None, error_model=None,
                             mask=None, dummy=None, delta_dummy=None,
                             polarization_factor=None, dark=None, flat=None,
                             method='splitpix', unit='nm',
                             safe=True, normalization_factor=None, all=False):
                             
        """         
        Wrapper for transform_image to project input array into
        reciprocal space coordinates I(q_xy,q_z).
        """
        unit = 'q_%s^-1' % unit
        out = self.transform_image(data, process='reciprocal', npt=npt,
                      x_range=ip_range, y_range=op_range, filename=filename, 
                      correctSolidAngle=correctSolidAngle,
                      variance=variance, error_model=error_model,
                      mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                      polarization_factor=polarization_factor,
                      dark=dark, flat=flat, method=method, unit=unit, safe=safe, 
                      normalization_factor=normalization_factor, all=all)
        return out
    
    def transform_polar(self, data, npt=(1000,200), 
                        q_range=None, chi_range=(-100,100),
                        filename=None, correctSolidAngle=False,
                        variance=None, error_model=None,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method='splitpix', unit='nm', 
                        safe=True, normalization_factor=None, all=False):
        """ 
        Wrapper for transform_image to project input array into
        polar coordinates I(q,chi).
        """
        unit = 'q_%s^-1' % unit
        out = self.transform_image(data, process='polar', npt=npt,
                      x_range=q_range, y_range=chi_range, filename=filename,
                      correctSolidAngle=correctSolidAngle,
                      variance=variance, error_model=error_model,
                      polarization_factor=polarization_factor,
                      dark=dark, flat=flat, method=method, unit=unit, safe=safe,
                      normalization_factor=normalization_factor, all=all)
        return out
    
    def transform_image(self, data, process='transform', npt=None,
                        x_range=None, y_range=None,
                        filename=None, correctSolidAngle=False, 
                        variance=None, error_model=None,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method='splitpix', unit=grazing_units.Q, 
                        safe=True, normalization_factor=None, all=False):
        """
        Project a raw GIXS pattern into reciprocal space qxy(nm^-1) vs 
        qz(nm^-1) by default. Based on pyFAI integrate 2d. Multi 
        algorithm implementation (tries to be bullet proof).

        Parameters
        ----------
        data : ndarray
            2D array from detector (raw image).
        filename : str
            Output filename in 2/3 column ascii format.
        correctSolidAngle : bool
            Correct for solid angle of each pixel if True.
        variance : ndarray
            Array containing the variance of the data. If not available, 
            no error propagation is done.
        error_model : str
            When variance is unknown, an error model can be given: 
            "poisson" (variance = I), "azimuthal" (variance = 
            (I-<I>)^2).
        x_range : (float, float), optional
            The lower and upper unit of the in-plane unit. If not 
            provided, range is simply (data.min(), data.max()). Values 
            outside the range are ignored.
        y_range : (float, float), optional
            The lower and upper range of the out-of-plane unit. If not 
            provided, range is simply (data.min(), data.max()). Values 
            outside the range are ignored.
        mask : ndarray
            Masked pixel array (same size as image) with 1 for masked 
            pixels and 0 for valid pixels.
        dummy : float
            Value for dead/masked pixels.
        delta_dummy : float
            Precision for dummy value
        polarization_factor : float
            Polarization factor between -1 and +1. 0 for no correction.
        dark : ndarray
            Dark current image.
        flat : ndarray
            Flat field image.
        method : str
            Integration method. Can be "numpy", "cython", "bbox",
            "splitpix", "lut" or "lut_ocl" (if you want to go on GPU).
        unit : str
            Grazing-incidence units. Can be "2th_af_deg", "2th_af_rad", 
            "qxy_qz_nm^-1", "qxy_qz_A^-1", "qxy_qz_nm^-1" or 
            "qxy_qz_A^-1". For GISAXS qy vs qz is typically preferred;
            for GIWAXS, qxy vs qz.
            (TTH_AF_DEG, TTH_AF_RAD, QY_QZ_NM, QY_QZ_A, QXY_QZ_NM, 
            QXY_QZ_A).
        safe : bool
            Do some extra check to ensure LUT is still valid. False is
            faster.
        normalization_factor : float
            Value of a normalization monitor.

        Returns
        -------
        I, bins_x, bins_y : 3-tuple of ndarrays
            Regrouped intensity, in-plane bins, out-of-plane bins.

        or
        
        I, bins_x, bins_y, sigma : 4-tuple of ndarrays
            Regrouped intensity, in-plane bins, out-of-plane bins and 
            error.
        """
        method = method.lower()
        unit = grazing_units.to_unit(unit)
        print unit
        pos_scale  = unit.scale
        if mask is None:
            mask = self.mask
        shape = data.shape
        
        if (npt is None) and ('polar' not in process):
            if self._sample_orientation in [1, 3]:
                npt = (shape[1], shape[0])
            else:
                npt = (shape[0], shape[1])
        
        if x_range:
            x_range = tuple([i / pos_scale for i in x_range])      
        if ('polar' in process):
            if y_range is None:
                y_range = (-95,95)
            if npt is None:
                npt = (1000,(abs(y_range[0])+ abs(y_range[1])))
            y_range = tuple([numpy.deg2rad(i)+pi for i in y_range])
        else:
            if y_range:
                y_range = tuple([i / pos_scale for i in y_range])
            
        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None
        
        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(shape, polarization_factor)
        
        if dark is None:
            dark = self.darkcurrent
        
        if flat is None:
            flat = self.flatfield

        I = None
        sigma = None
        sum = None
        count = None

        if (I is None) and ("lut" in method):
            logger.debug("in lut")
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._lut_gi_transformer is None:
                    reset = "init"
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                if (not reset) and safe:
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                    if self._lut_gi_transformer.unit != unit:
                        reset = "unit changed"
                    if self._lut_gi_transformer.bins != npt:
                        reset = "number of points changed"
                    if self._lut_gi_transformer.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and \
                            (not self._lut_gi_transformer.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and \
                            (self._lut_gi_transformer.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and \
                            (self._lut_gi_transformer.mask_checksum != \
                            mask_crc):
                        reset = "mask changed"
                    if (x_range is None) and not \
                            self._lut_gi_transformer.default_ipl:
                        reset = "x_range was defined in LUT"
                    elif (x_range is not None) and \
                            (self._lut_gi_transformer.pos0Range != \
                            (min(x_range), max(x_range) * EPS32)):
                        reset = ("x_range is defined"
                                 " but not the same as in LUT")
                    if (y_range is None) and not \
                            self._lut_gi_transformer.default_opl:
                        reset = ("y_range not defined and"
                                 " LUT had y_range defined")
                    elif (y_range is not None) and \
                            (self._lut_gi_transformer.pos1Range != \
                            (min(y_range), max(y_range) * EPS32)):
                        reset = ("y_range requested and"
                                 " LUT's y_range don't match")
                error = False
                if reset:
                    logger.info("giTransformImage: Resetting transformer"
                                " because %s" % reset)
                    try: 
                        lut_giTrnsfrmr, def_ip, def_op = \
                            self.gi_setup_lut(shape, npt, 
                                              "transform", mask, 
                                              x_range, 
                                              y_range, 
                                              mask_checksum=mask_crc, 
                                              unit=unit)

                        self._lut_gi_transformer = lut_giTrnsfrmr
                        self._lut_gi_transformer.default_ipl = def_ip
                        self._lut_gi_transformer.default_opl = def_op
                        error = False
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on" 
                                       " forward implementation")
                        self._ocl_lut_gi_transformer = None
                        gc.collect()
                        method = "splitbbox"
                        error = True
                if not error:
                    if  ("ocl" in method) and ocl_azim_lut:
                        with self._ocl_lut_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_lut_gi_transformer is None) or \
                                    (self._ocl_lut_gi_transformer.on_device["lut"] \
                                    != self._lut_gi_transformer.lut_checksum):
                                self._ocl_lut_gi_transformer = \
                                    ocl_azim_lut.OCL_LUT_Integrator(
                                        self._lut_gi_transformer.lut,
                                        self._lut_gi_transformer.size,
                                        devicetype=devicetype,
                                        platformid=platformid,
                                        deviceid=deviceid,
                                        checksum=\
                                        self._lut_gi_transformer.lut_checksum)
                            
                            print 'error = %s' % error
                            if not error:
                                I, sum, count = self._ocl_lut_gi_transformer.integrate(
                                        data, dark=dark, flat=flat,
                                        solidAngle=solidangle,
                                        solidAngle_checksum=self._dssa_crc,
                                        dummy=dummy, delta_dummy=delta_dummy,
                                        polarization=polarization,
                                        polarization_checksum=\
                                        self._polarization_crc,
                                        safe=safe)
                            I.shape = npt
                            I = I.T

                            bins_x  = self._lut_gi_transformer.outPos0  # this will be copied later
                            bins_y = self._lut_gi_transformer.outPos1
                    else:
                        print 'reset = %s' % reset
                        I, bins_x, bins_y, sum, count = \
                            self._lut_gi_transformer.integrate(
                                data, dark=dark, flat=flat,
                                solidAngle=solidangle,
                                dummy=dummy, delta_dummy=delta_dummy,
                                polarization=polarization)   
        
        # CSR HERE
        csr_transform = """
        if (I  is None) and ("csr" in method):
            logger.debug("in csr")
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._csr_transformer is None:
                    reset = "init"
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                if (not reset) and safe:
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                    if self._csr_gi_transformer.unit != unit:
                        reset = "unit changed"
                    if self._csr_gi_transformer.bins != npt:
                        reset = "number of points changed"
                    if self._csr_gi_transformer.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and \
                            (not self._csr_gi_transformer.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and \
                            (self._csr_gi_transformer.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and \
                            (self._csr_gi_transformer.mask_checksum != \
                            mask_crc):
                        reset = "mask changed"
                    if (x_range is None) and not \
                            self._csr_gi_transformer.default_ipl:
                        reset = "x_range was defined in CSR"
                    elif (x_range is not None) and \
                            (self._csr_gi_transformer.pos0Range != \
                            (min(x_range), max(x_range) * EPS32)):
                        reset = ("x_range is defined"
                                 " but not the same as in CSR")
                    if (y_range is None) and not \
                            self._csr_gi_transformer.default_opl:
                        reset = ("y_range not defined and"
                                 " CSR had y_range defined")
                    elif (y_range is not None) and \
                            (self._csr_gi_transformer.pos1Range != \
                            (min(y_range), max(y_range) * EPS32)):
                        reset = ("y_range requested and"
                                 " CSR's y_range don't match")
                error = False
                if reset:
                    logger.info("gi.transform_image: resetting CSR transformer"
                                " because %s" % reset)
                    if "no" in method:
                        split = "no"
                    elif "full" in method:
                        split = "full"
                    else:
                        split = "bbox"
                    try:
                        sef._csr_gi_transformer = self.gi_setup_CSR(
        """                        
        
        if (I is None) and ("splitpix" in method):
            if splitPixel is None:
                logger.warning("splitPixel is not available;"
                               " falling back on default method")
                method = self.DEFAULT_METHOD
            else:
                logger.debug("transform_image uses SplitPixel implementation")
                pos = self.giarray_from_unit(shape, process, "corner", unit)
                
                I, bins_x, bins_y, sum, count = splitPixel.fullSplit2D(
                                                pos=pos,
                                                weights=data,
                                                bins=npt,
                                                pos0Range=x_range,
                                                pos1Range=y_range,
                                                dummy=dummy,
                                                delta_dummy=delta_dummy,
                                                mask=mask, dark=dark, flat=flat,
                                                solidangle=solidangle,
                                                polarization=polarization)
        
        if (I is None) and ("bbox" in method):
            if splitBBox is None:
                logger.warning("splitBBox is not available;"
                               " falling back on cython histogram method")
                method = "cython"
            else:
                logger.debug("giTransformImage uses BBox implementation")
                pos1, pos0 = self.giarray_from_unit(shape, process, "center", unit)
                dpos1, dpos0 = self.giarray_from_unit(shape, process, "delta", unit)
                
                I, bins_x, bins_y, sum, count = splitBBox.histoBBox2d(
                                                weights=data,
                                                pos0=pos0,
                                                delta_pos0=dpos0,
                                                pos1=pos1,
                                                delta_pos1=dpos1,
                                                bins=npt,
                                                pos0Range=x_range,
                                                pos1Range=y_range,
                                                dummy=dummy,
                                                delta_dummy=delta_dummy,
                                                mask=mask, dark=dark, flat=flat,
                                                solidangle=solidangle,
                                                polarization=polarization,
                                                allow_pos0_neg=True)
                        
        
        if I is None:
            #Common part for Numpy and Cython
            logger.debug("giTransformImage uses cython implementation")
            npt = (npt[1],npt[0])
            data = data.astype(numpy.float32)
            mask = self.make_mask(data, mask, dummy, delta_dummy, mode="numpy")
            yarray, xarray = self.giarray_from_unit(shape, process, "center", unit)
            
            if x_range is not None:
                mask *= (xarray >= min(x_range)) * (xarray<= max(x_range))
            else:
                x_range = [xarray.min(), xarray.max() * EPS32]
            
            if y_range is not None:
                mask *= (yarray >= min(y_range)) * \
                    (yarray <= max(y_range))
            else:
                y_range = [yarray.min(), yarray.max() * EPS32]

            if variance is not None:
                variance = variance[mask]
            
            if dark is not None:
                data -= dark
            
            if flat is not None:
                data /= flat
            
            if polarization is not None:
                data /= polarization
            
            if solidangle is not None:
                data /= solidangle

            data = data[mask]
            xarray = xarray[mask]
            yarray = yarray[mask]

            if ("cython" in method):
                if histogram is None:
                    logger.warning("Cython histogram is not available;"
                                   " falling back on numpy histogram")
                    method = "numpy"
                else:
                    I, bins_y, bins_x, sum, count = histogram.histogram2d(
                            pos0=yarray, pos1=xarray,
                            weights=data, bins=npt,
                            split=False,
                            empty=dummy if dummy is not None else self._empty)
        
        if I is None:
            method = 'numpy'
            logger.debug("transform_image uses Numpy implementation")
            count, b, c = numpy.histogram2d(yarray, xarray, npt,
                                          range=[y_range, x_range])
            bins_y = (b[1:] + b[:-1]) / 2.0
            bins_x  = (c[1:] + c[:-1]) / 2.0
        
            count1    = numpy.maximum(1, count)
            sum, b, c = numpy.histogram2d(yarray, xarray, npt,
                                          weights=data,
                                          range=[y_range, x_range])
            I = sum / count1
        
        qmasking = '''
        if ("q" in str(unit)) and (self._useqx is True) and \
                (method not in ["numpy", "cython"]):
            
            if self._giqmask is None:
                self._transformedmask = self.make_qmask(process, npt, bins_x,
                                                        bins_y)
            #if self._transformedmask is None:
                #pos = self.giarray_from_unit(shape, process, "corner", unit)
                #self.make_transformedmask(mask, pos, npt, x_range, y_range) 
                        
            if dummy is None:
                dummy = 0
            
            I[self._transformedmask == 0] = dummy
            if sigma is not None:
                sigma[self._transformedmask == 0] = dummy
        '''
        
        print 'Method used: %s' % method    
        bins_x *= pos_scale
        if 'polar' in process:
            bins_y = (bins_y - pi)  * 180.0/pi
        else:
            bins_y *= pos_scale
        
        if normalization_factor:
            I /= normalization_factor

        self.save_transformed(filename, I, bins_x, bins_y, 
                              sigma, unit, dark=dark, flat=flat, 
                              polarization_factor=polarization_factor,
                              normalization_factor=normalization_factor)
        
        if all:
            res = {"I":I,
                   "in_plane":bins_x,
                   "out_plane":bins_y,
                   "count":count,
                   "sum": sum}
            if sigma is not None:
                res["sigma"] = sigma
        else:
            if sigma is not None:
                res = I, bins_x, bins_y, sigma
            else:
                res = I, bins_x, bins_y
        return res
        
    def make_transformedmask(self, mask, pos, npt, x_range, y_range):
        """ """
        if mask is None:
            mask = np.zeros((npt))

        import matplotlib.pyplot as plt
        import sys
        
        trmask, bipl, bopl, s, ct = splitPixel.fullSplit2D(
                                                pos=pos,
                                                weights=mask,
                                                bins=npt,
                                                pos0Range=x_range,
                                                pos1Range=y_range,
                                                dummy=None,
                                                delta_dummy=None,
                                                mask=None, dark=None, flat=None,
                                                solidangle=None,
                                                polarization=None)
        
        totmask = numpy.ones((trmask.shape))
        totmask[numpy.where(trmask > 0.0)] = 0
        totmask[numpy.where(self._giqmask == 0)] = 0
        
        self._transformedmask = totmask
        return self._transformedmask

    #---------------------------------------------------------------------------
    #   Integration functions
    #---------------------------------------------------------------------------
    
    def profile_sector(self, data, npt,
                       filename=None, correctSolidAngle=False,
                       variance=None, error_model=None,
                       chi_pos=None, chi_width=None, radial_range=None,
                       mask=None, dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None,
                       method="splitpix", unit=grazing_units.Q, 
                       normalization_factor=None):
        """
        Sector integration wrapper for integrate_1d. More intuatively
        for GIXS data, the function takes chi_pos and chi_width, 
        instead of chi_min, chi_max. Chi = 0 is defined as surface
        normal. E.g. out-of-plane (qz) line profile would be at chi = 0
        with the angular opening of the sector given by chi_width.

        Parameters
        ----------

        """
        azim_min = chi_pos - (chi_width/2.0)
        azim_max = chi_pos + (chi_width/2.0)
        azimuth_range = (azim_min, azim_max)
        
        out = self.integrate_1d(data, npt, process="sector",
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=variance, error_model=error_model,
                                p0_range=radial_range,
                                p1_range=azimuth_range,
                                mask=mask, 
                                dummy=dummy, delta_dummy=delta_dummy,
                                polarization_factor=polarization_factor,
                                dark=dark, flat=flat,
                                method=method,
                                unit=unit,
                                normalization_factor=normalization_factor)
        return out

    def profile_chi(self, data, npt,
                    filename=None, correctSolidAngle=False,
                    variance=None, error_model=None,
                    radial_pos=None, radial_width=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    method="bbox", unit=grazing_units.Q, 
                    normalization_factor=None):
        """
        Chi integration wrapper for integrate_1d.

        """
        rad_min = radial_pos - (radial_width/2.0)
        rad_max = radial_pos + (radial_width/2.0)
        radial_range = (rad_min, rad_max)

        out = self.integrate_1d(data, npt, process="chi",
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=variance, error_model=error_model,
                                p0_range=azimuth_range,
                                p1_range=radial_range,
                                mask=mask, 
                                dummy=dummy, delta_dummy=delta_dummy,
                                polarization_factor=polarization_factor,
                                dark=dark, flat=flat,
                                method=method,
                                unit=unit,
                                normalization_factor=normalization_factor)
        
        if ("q" in str(unit).lower()) and (self._useqx is True) and \
                (method not in ["numpy", "cython"]):
            I = out[0]
            chi = out[1]

            if self._chimask is None:
                self.make_chimask(rad_max, chi)  
            
            if dummy is None:
                dummy = numpy.nan

            I[self._chimask == 0] = dummy
            
            if len(out) == 3:
                sigma = out[2]
                sigma[self._chimask == 0] = dummy
                out = (I, chi, sigma)
            else:
                out = (I, chi)
        return out

    def profile_op_box(self, data, npt,
                       filename=None, correctSolidAngle=False,
                       variance=None, error_model=None,
                       ip_pos=0.0, ip_width=30.0, op_range=None,
                       mask=None, dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None,
                       method="bbox", unit=grazing_units.Q, 
                       normalization_factor=None):
        """
        Out-of-plane box integration wrapper for integrate_1d.

        """
        ip_min = ip_pos - (ip_width/2.0)
        ip_max = ip_pos + (ip_width/2.0)
        ip_range = (ip_min, ip_max)

        out = self.integrate_1d(data, npt, process="opbox",
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=variance, error_model=error_model,
                                p0_range=op_range,
                                p1_range=ip_range,
                                mask=mask, 
                                dummy=dummy, delta_dummy=delta_dummy,
                                polarization_factor=polarization_factor,
                                dark=dark, flat=flat,
                                method=method,
                                unit=unit,
                                normalization_factor=normalization_factor)
        return out

    def profile_ip_box(self, data, npt,
                       filename=None, correctSolidAngle=False,
                       variance=None, error_model=None,
                       op_pos=0.0, op_width=30.0, ip_range=None,
                       mask=None, dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None,
                       method="bbox", unit=grazing_units.Q, 
                       normalization_factor=None):
        """
        Out-of-plane box integration wrapper for integrate_1d.

        """
        op_min = op_pos - (op_width/2.0)
        op_max = op_pos + (op_width/2.0)
        op_range = (op_min, op_max)

        out = self.integrate_1d(data, npt, process="ipbox",
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=variance, error_model=error_model,
                                p0_range=ip_range,
                                p1_range=op_range,
                                mask=mask, 
                                dummy=dummy, delta_dummy=delta_dummy,
                                polarization_factor=polarization_factor,
                                dark=dark, flat=flat,
                                method=method,
                                unit=unit,
                                normalization_factor=normalization_factor)
        return out

    def integrate_1d(self, data, npt, process="sector",
                     filename=None,
                     correctSolidAngle=False,
                     variance=None, error_model=None,
                     p0_range=None, p1_range=None,
                     mask=None, dummy=None, delta_dummy=None,
                     polarization_factor=None, dark=None, flat=None,
                     method="bbox", unit=grazing_units.Q, 
                     safe=True, normalization_factor=None):
        """
        Calculate azimuthally integrated 1d curve in q(nm^-1) by 
        efault. Based on pyFAI integrate1d. Multi algorithm 
        implementation (tries to be bullet proof).

        Parameters
        ----------
        data : ndarray
            2D array from detector (raw image).
        npt : int
            Number of points in output data.
        filename : str
            Output filename in 2/3 column ascii format.
        correctSolidAngle : bool
            Correct for solid angle of each pixel if True.
        variance : ndarray
            Array containing the variance of the data. If not 
            available, no error propagation is done.
        error_model : str
            When variance is unknown, an error model can be given: 
            "poisson" (variance = I), "azimuthal" (variance = 
            (I-<I>)^2).
        p0_range : (float, float), optional
            The lower and upper unit of the radial unit. If not 
            provided, range is simply (data.min(), data.max()). Values
            outside the range are ignored.
        p1_range : (float, float), optional
            The lower and upper range of the azimuthal angle in degree.
            If not provided, range is simply (data.min(), data.max()). 
            Values outside the range are ignored.
        mask : ndarray
            Masked pixel array (same size as image) with 1 for masked 
            pixels and 0 for valid pixels.
        dummy : float
            Value for dead/masked pixels.
        delta_dummy : float
            Precision for dummy value
        polarization_factor : float
            Polarization factor between -1 and +1. 0 for no correction.
        dark : ndarray
            Dark current image.
        flat : ndarray
            Flat field image.
        method : str
            Integration method. Can be "numpy", "cython", "bbox",
            "splitpix", "lut" or "lut_ocl" (if you want to go on GPU).
        unit : str
            Radial units. Can be "2th_deg", "2th_rad", "q_nm^-1" or
            "q_A^-1" (TTH_DEG, TTH_RAD, Q_NM, Q_A).
        safe : bool
            Do some extra check to ensure LUT is still valid. False is
            faster.
        normalization_factor : float
            Value of a normalization monitor.

         Returns
        -------
        qAxis, I : 2-tuple of ndarrays
            Radial bins and integrated intensity.

        or
        
        qAxis, I, sigma : 3-tuple of ndarrays
            Radial bins, integrated intensity and error.
        """
        method = method.lower()
        unit = grazing_units.to_unit(unit)
        pos0_scale = 1.0  # nota we need anyway to make a copy !

        if mask is None:
            mask = self.mask

        shape = data.shape
        pos0_scale = unit.scale

        if p0_range:
            if process == "chi":
                p0_range = tuple([numpy.deg2rad(i)+pi for i in p0_range])
            else:
                p0_range = tuple([i / pos0_scale for i in p0_range])

        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if p1_range is not None:
            if process == "sector":
                p1_range = tuple([numpy.deg2rad(i)+pi for i in p1_range])
            else:
                p1_range = tuple([i / pos0_scale for i in p1_range])

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(shape, float(polarization_factor))

        if dark is None:
            dark = self.darkcurrent

        if flat is None:
            flat = self.flatfield

        I = None
        sigma = None

        if (I is None) and ("lut" in method):
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._lut_gi_integrator is None:
                    reset = "init"
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                if (not reset) and safe:
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                    if self._lut_gi_integrator.unit != unit:
                        reset = "unit changed"
                    if self._lut_gi_integrator.bins != npt:
                        reset = "number of points changed"
                    if self._lut_gi_integrator.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and \
                            (not self._lut_gi_integrator.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and (self._lut_gi_integrator.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and \
                            (self._lut_gi_integrator.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (p0_range is None) and not \
                            self._lut_gi_integrator.default_p0:
                        reset = "p0_range was defined in LUT"
                    elif (p0_range is not None) and \
                            (self._lut_gi_integrator.pos0Range != \
                            (min(p0_range), max(p0_range) * EPS32)):
                        reset = ("p0_range is defined"
                                 " but not the same as in LUT")
                    if (p1_range is None) and not \
                            self._lut_gi_integrator.default_p1:
                        reset = ("p1_range not defined and"
                                 " LUT had p1_range defined")
                    elif (p1_range is not None) and \
                            (self._lut_gi_integrator.pos1Range != \
                            (min(p1_range), max(p1_range) * EPS32)):
                        reset = ("p1_range requested and"
                                 " LUT's p1_range don't match")
                if reset:
                    logger.info("integrate_1d: Resetting integrator" 
                                " because %s" % reset)
                    try:
                        lut_giIntegrator, def_p0, def_p1 = \
                            self.gi_setup_lut(shape, npt, 
                                              process, mask, 
                                              p0_range, 
                                              p1_range, 
                                              mask_checksum=mask_crc, 
                                              unit=unit)

                        self._lut_gi_integrator = lut_giIntegrator
                        self._lut_gi_integrator.default_p0 = def_p0
                        self._lut_gi_integrator.default_p1 = def_p1
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on" 
                                       " forward implementation")
                        self._ocl_lut_gi_integrator = None
                        self._lut_gi_integrator = None
                        gc.collect()
                        method = "splitbbox"
                if self._lut_gi_integrator:
                    if ("ocl" in method) and ocl_azim_lut:
                        with self._ocl_lut_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_lut_gi_integrator is None) or\
                                    (self._ocl_lut_gi_integrator.on_device["lut"] != \
                                    self._lut_gi_integrator.lut_checksum):
                                self._ocl_lut_gi_integrator = \
                                    ocl_azim_lut.OCL_LUT_Integrator(
                                        self._lut_gi_integrator.lut,
                                        self._lut_gi_integrator.size,
                                        devicetype=devicetype,
                                        platformid=platformid,
                                        deviceid=deviceid,
                                        checksum=\
                                            self._lut_gi_integrator.lut_checksum)
                            
                            I, _, _ = self._ocl_lut_gi_integrator.integrate(
                                        data, dark=dark, flat=flat,
                                        solidAngle=solidangle,
                                        solidAngle_checksum=self._dssa_crc,
                                        dummy=dummy,
                                        delta_dummy=delta_dummy,
                                        polarization=polarization,
                                        polarization_checksum=\
                                            self._polarization_crc)

                            qAxis = self._lut_gi_integrator.outPos  # this will be copied later
                            if error_model == "azimuthal":
                                variance = (data - self.calcfrom1d(qAxis * \
                                        pos0_scale, I, dim1_unit=unit)) ** 2
                            if variance is not None:
                                var1d, a, b = self._ocl_lut_gi_integrator.integrate(
                                                variance,
                                                solidAngle=None,
                                                dummy=dummy,
                                                delta_dummy=delta_dummy)
                                sigma = numpy.sqrt(a) / numpy.maximum(b, 1)
                    else:
                        qAxis, I, a, b = self._lut_gi_integrator.integrate(
                                            data, dark=dark, flat=flat,
                                            solidAngle=solidangle,
                                            dummy=dummy,
                                            delta_dummy=delta_dummy,
                                            polarization=polarization)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * \
                                pos0_scale, I, dim1_unit=unit)) ** 2
                        if variance is not None:
                            _, var1d, a, b = self._lut_gi_integrator.integrate(
                                                variance,
                                                solidAngle=None,
                                                dummy=dummy,
                                                delta_dummy=delta_dummy)
                            sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if (I is None) and ("splitpix" in method):
            if splitPixel is None:
                logger.warning("SplitPixel is not available,"
                            " falling back on splitbbox histogram !")
                method = "splitbbox"
            else:
                logger.debug("integrate1d uses SplitPixel implementation")
                pos = self.giarray_from_unit(shape, process, "corner", unit)
                
                qAxis, I, a, b = splitPixel.fullSplit1D(
                                    pos=pos,
                                    weights=data,
                                    bins=npt,
                                    pos0Range=p0_range,
                                    pos1Range=p1_range,
                                    dummy=dummy,
                                    delta_dummy=delta_dummy,
                                    mask=mask,
                                    dark=dark,
                                    flat=flat,
                                    solidangle=solidangle,
                                    polarization=polarization)
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * \
                        pos0_scale, I, dim1_unit=unit)) ** 2
                if variance is not None:
                    _, var1d, a, b = splitPixel.fullSplit1D(
                                        pos=pos,
                                        weights=variance,
                                        bins=npt,
                                        pos0Range=p0_range,
                                        pos1Range=p1_range,
                                        dummy=dummy,
                                        delta_dummy=delta_dummy,
                                        mask=mask)
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if (I is None) and ("bbox" in method):
            if splitBBox is None:
                logger.warning("pyFAI.splitBBox is not available,"
                               " falling back on cython histograms")
                method = "cython"
            else:
                logger.debug("giIntegrate1d uses BBox implementation")
                chi, pos0   = self.giarray_from_unit(shape, process, "center", unit)
                dchi, dpos0 = self.giarray_from_unit(shape, process, "delta", unit)
                
                qAxis, I, a, b = splitBBox.histoBBox1d(
                                    weights=data,
                                    pos0=pos0,
                                    delta_pos0=dpos0,
                                    pos1=chi,
                                    delta_pos1=dchi,
                                    bins=npt,
                                    pos0Range=p0_range,
                                    pos1Range=p1_range,
                                    dummy=dummy,
                                    delta_dummy=delta_dummy,
                                    mask=mask,
                                    dark=dark,
                                    flat=flat,
                                    solidangle=solidangle,
                                    polarization=polarization)
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * \
                        pos0_scale, I, dim1_unit=unit)) ** 2
                if variance is not None:
                    _, var1d, a, b = splitBBox.histoBBox1d(
                                        weights=variance,
                                        pos0=pos0,
                                        delta_pos0=dpos0,
                                        pos1=chi,
                                        delta_pos1=dchi,
                                        bins=npt,
                                        pos0Range=p0_range,
                                        pos1Range=p1_range,
                                        dummy=dummy,
                                        delta_dummy=delta_dummy,
                                        mask=mask)
                    b[b == 0] = 1
                    sigma = numpy.sqrt(a) / b

        if I is None:
            #Common part for  Numpy and Cython
            data = data.astype(numpy.float32)
            mask = self.make_mask(data, mask, dummy, delta_dummy, mode="numpy")
            pos1, pos0 = self.giarray_from_unit(shape, process, "center", unit)
            
            if p0_range is not None:
                mask *= (pos0 >= min(p0_range)) * \
                    (pos0 <= max(p0_range))
            else:
                p0_range = [pos0.min(), pos0.max() * EPS32]
            if p1_range is not None:
                mask *= (pos1 >= min(p1_range)) * \
                    (pos1 <= max(p1_range))
            else:
                p1_range = [pos1.min(), pos1.max() * EPS32]

            mask = numpy.where(mask)

            if dark is not None:
                data -= dark

            if flat is not None:
                data /= flat

            if polarization is not None:
                data /= polarization

            if solidangle is not None:
                data /= solidangle

            if variance is not None:
                variance = variance[mask]

            data = data[mask]
            pos0 = pos0[mask]
            pos1 = pos1[mask]

            if ("cython" in method):
                if histogram is not None:
                    logger.debug("giIntegrate1d uses cython implementation")
                    if dummy is None:
                        dummy = 0
                    
                    qAxis, I, a, b = histogram.histogram(pos=pos0,
                                                         weights=data,
                                                         bins=npt,
                                                         pixelSize_in_Pos=0,
                                                         empty=dummy)
                    if error_model == "azimuthal":
                        variance = (data - self.calcfrom1d(
                                            qAxis * pos0_scale, 
                                            I, dim1_unit=unit, 
                                            correctSolidAngle=False)[mask]) ** 2
                    if variance is not None:
                        _, var1d, a, b = histogram.histogram(pos=pos0,
                                                             weights=variance,
                                                             bins=npt,
                                                             pixelSize_in_Pos=1,
                                                             empty=dummy)
                        sigma = numpy.sqrt(a) / numpy.maximum(b, 1)
                else:
                    logger.warning("pyFAI.histogram is not available,"
                                   " falling back on numpy")
                    method = "numpy"

        if I is None:
            logger.debug("integrate1d uses Numpy implementation")
            method = "numpy"
            ref, b = numpy.histogram(pos0, npt, range=p0_range)
            qAxis = (b[1:] + b[:-1]) / 2.0
            count = numpy.maximum(1, ref)
            val, b = numpy.histogram(pos0, npt, 
                                     weights=data, range=p0_range)
            if error_model == "azimuthal":
                variance = (data - self.calcfrom1d(
                                    qAxis * pos0_scale, 
                                    I, dim1_unit=unit, 
                                    correctSolidAngle=False)[mask]) ** 2
            if variance is not None:
                var1d, b = numpy.histogram(pos0, npt, 
                                           weights=variance, 
                                           range=p0_range)
                sigma = numpy.sqrt(var1d) / count
            I = val / count
        
        print 'Method used: %s' % method   
        if process == "chi":
            qAxis = numpy.rad2deg(qAxis-pi)
        elif pos0_scale:
            qAxis = qAxis * pos0_scale

        if normalization_factor:
            I /= normalization_factor
            if sigma is not None:
                sigma /= normalization_factor

        self.save_1d(filename, I, qAxis, sigma, unit,
                     dark=dark, flat=flat, 
                     polarization_factor=polarization_factor,
                     normalization_factor=normalization_factor)
        
        if sigma is not None:
            return I, qAxis, sigma
        else:
            return I, qAxis

#-------------------------------------------------------------------------------
#   Save functions
#-------------------------------------------------------------------------------

    def save_transformed(self, filename, I, dim1, dim2, 
                         error=None, unit=grazing_units.Q,
                         dark=None, flat=None, polarization_factor=None, 
                         normalization_factor=None):
        """
        This method saves the result of grazing-incidence transformation.
        If no filename is given the function will exit.

        Parameters
        ----------
        filename : str
            The file name to save the transformed image.
        I : ndarray
            Transformed image (intensity).
        dim1 : ndarray
            In-plane coordinates.
        dim2 : ndarray
            Out-of-plane coordinates.
        error : ndarray or None
            The error bar for each intensity.
        unit : PyGIX.grazing_units.Unit
            The in-plane and out-of-plane units.
        dark :  ???
            Save the dark filename(s) (default: no).
        flat : ???
            Save the flat filename(s) (default: no).
        polarization_factor : float
            The polarization factor.
        normalization_factor: float
            The normalization factor..
        """
        if not filename:
            return

        giUnit = grazing_units.to_unit(unit)
        ipl_giUnit = grazing_units.ip_unit(giUnit)
        opl_giUnit = grazing_units.op_unit(giUnit)
        header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                       "sample orientation", "incident angle", "tilt angle",
                       ipl_giUnit.IPL_UNIT + "_min",
                       ipl_giUnit.IPL_UNIT + "_max",
                       opl_giUnit.OPL_UNIT + "_min",
                       opl_giUnit.OPL_UNIT + "_max",
                       "pixelX", "pixelY",
                       "dark", "flat", "polarization_factor", 
                       "normalization_factor"]

        header = {"dist": str(self._dist),
                  "poni1": str(self._poni1),
                  "poni2": str(self._poni2),
                  "rot1": str(self._rot1),
                  "rot2": str(self._rot2),
                  "rot3": str(self._rot3),
                  "sample orientation" : str(self._sample_orientation),
                  "incident angle" : str(self._incident_angle),
                  "tilt angle" : str(self._tilt_angle),
                  ipl_giUnit.IPL_UNIT + "_min": str(dim1.min()),
                  ipl_giUnit.IPL_UNIT + "_max": str(dim1.max()),
                  opl_giUnit.OPL_UNIT + "_min": str(dim2.min()),
                  opl_giUnit.OPL_UNIT + "_max": str(dim2.max()),
                  "pixelX": str(self.pixel2),  # this is not a bug ... most people expect dim1 to be X
                  "pixelY": str(self.pixel1),  # this is not a bug ... most people expect dim2 to be Y
                  "polarization_factor": str(polarization_factor),
                  "normalization_factor":str(normalization_factor)
                  }

        if self.splineFile:
            header["spline"] = str(self.splineFile)

        if dark is not None:
            if self.darkfiles:
                header["dark"] = self.darkfiles
            else:
                header["dark"] = 'unknown dark applied'
        if flat is not None:
            if self.flatfiles:
                header["flat"] = self.flatfiles
            else:
                header["flat"] = 'unknown flat applied'
        f2d = self.getFit2D()
        for key in f2d:
            header[key] = f2d[key]
        try:
            img = fabio.edfimage.edfimage(data=I.astype("float32"),
                                          header=header,
                                          header_keys=header_keys)

            if error is not None:
                img.appendFrame(data=error, 
                                header={"EDF_DataBlockID": "1.Image.Error"})
            img.write(filename)
        except IOError:
            logger.error("IOError while writing %s" % filename)

    def save_2d(self, filename, I, dim1, dim2, 
                error=None, dim1_unit=grazing_units.Q,
                dark=None, flat=None, polarization_factor=None, 
                normalization_factor=None):
        """
        This method saves the result of 2D integration. If no filename is
        given the function will exit.

        Parameters
        ----------
        filename : str
            The file name to save the transformed image.
        I : ndarray
            Transformed image (intensity).
        dim1 : ndarray
            The 1st coordinates of the histogram (radial).
        dim2 : ndarray
            The 2nd coordinates of the histogram (azimuthal).
        error : ndarray or None
            The error bar for each intensity.
        dim1_unit : PyGIX.grazing_units.Unit
            Unit of the dim1 array.
        dark :  ???
            Save the dark filename(s) (default: no).
        flat : ???
            Save the flat filename(s) (default: no).
        polarization_factor : float
            The polarization factor.
        normalization_factor: float
            The normalization factor. 
        giGeometry : int
            Grazing-incidence geometry in [0, 1, 2, 3].
        alpha_i : float
            Incident angle.
        misalign : float
            Surface plane tilt angle.
        """
        if not filename:
            return

        absUnit = grazing_units.absolute_unit(dim1_unit)
        header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                       "wavelength",
                       "sample orientation", "incident angle", "tilt angle",
                       "chi_min", "chi_max",
                       dim1_unit.REPR + "_min",
                       dim1_unit.REPR + "_max",
                       "pixelX", "pixelY",
                       "dark", "flat", "polarization_factor", 
                       "normalization_factor"]

        header = {"dist": str(self._dist),
                  "poni1": str(self._poni1),
                  "poni2": str(self._poni2),
                  "rot1": str(self._rot1),
                  "rot2": str(self._rot2),
                  "rot3": str(self._rot3),
                  "wavelength" : str(self._wavelength),
                  "sample orientation" : str(self._sample_orientation),
                  "incident angle" : str(self._incident_angle),
                  "tilt angle" : str(self._tilt_angle),
                  "chi_min": str(dim2.min()),
                  "chi_max": str(dim2.max()),
                  dim1_unit.REPR + "_min": str(dim1.min()),
                  dim1_unit.REPR + "_max": str(dim1.max()),
                  "pixelX": str(self.pixel2),
                  "pixelY": str(self.pixel1),
                  "polarization_factor": str(polarization_factor),
                  "normalization_factor":str(normalization_factor)
                  }

        if self.splineFile:
            header["spline"] = str(self.splineFile)

        if dark is not None:
            if self.darkfiles:
                header["dark"] = self.darkfiles
            else:
                header["dark"] = 'unknown dark applied'
        if flat is not None:
            if self.flatfiles:
                header["flat"] = self.flatfiles
            else:
                header["flat"] = 'unknown flat applied'
        f2d = self.getFit2D()
        for key in f2d:
            header["key"] = f2d[key]
        try:
            img = fabio.edfimage.edfimage(data=I.astype("float32"),
                                          header=header,
                                          header_keys=header_keys)

            if error is not None:
                img.appendFrame(data=error, 
                                header={"EDF_DataBlockID": "1.Image.Error"})
            img.write(filename)
        except IOError:
            logger.error("IOError while writing %s" % filename)

    def make_headers(self, hdr="#", dark=None, flat=None,
                     polarization_factor=None, normalization_factor=None):
        """
        @param hdr: string used as comment in the header
        @type hdr: str
        @param dark: save the darks filenames (default: no)
        @type dark: ???
        @param flat: save the flat filenames (default: no)
        @type flat: ???
        @param polarization_factor: the polarization factor
        @type polarization_factor: float

        @return: the header
        @rtype: str
        """
        if self.header is None:
            headerLst = ["== pyFAI calibration =="]
            headerLst.append("SplineFile: %s" % self.splineFile)
            headerLst.append("PixelSize: %.3e, %.3e m" %
                             (self.pixel1, self.pixel2))
            headerLst.append("PONI: %.3e, %.3e m" % (self.poni1, self.poni2))
            headerLst.append("Distance Sample to Detector: %s m" %
                             self.dist)
            headerLst.append("Rotations: %.6f %.6f %.6f rad" %
                             (self.rot1, self.rot2, self.rot3))
            #headerLst += ["", "== Fit2d calibration =="]
            #f2d = self.getFit2D()
            #headerLst.append("Distance Sample-beamCenter: %.3f mm" %
            #                 f2d["directDist"])
            #headerLst.append("Center: x=%.3f, y=%.3f pix" %
            #                 (f2d["centerX"], f2d["centerY"]))
            #headerLst.append("Tilt: %.3f deg  TiltPlanRot: %.3f deg" %
            #                 (f2d["tilt"], f2d["tiltPlanRotation"]))
            headerLst.append("")
            
            if self._wavelength is not None:
                headerLst.append("Wavelength: %s" % self._wavelength)

            headerLst.append("Sample orientation: %s" % self._sample_orientation)
            headerLst.append("Incident angle: %.6f deg" % self._incident_angle)
            headerLst.append("Tilt angle: %.6f deg" % self._tilt_angle)

            if self.maskfile is not None:
                headerLst.append("Mask File: %s" % self.maskfile)
            if (dark is not None) or (self.darkcurrent is not None):
                if self.darkfiles:
                    headerLst.append("Dark current: %s" % self.darkfiles)
                else:
                    headerLst.append("Dark current: Done with unknown file")
            if (flat is not None) or (self.flatfield is not None):
                if self.flatfiles:
                    headerLst.append("Flat field: %s" % self.flatfiles)
                else:
                    headerLst.append("Flat field: Done with unknown file")
            if polarization_factor is None and self._polarization is not None:
                polarization_factor = self._polarization_factor
            headerLst.append("Polarization factor: %s" % polarization_factor)
            headerLst.append("Normalization factor: %s" % normalization_factor)
            self.header = os.linesep.join([hdr + " " + i for i in headerLst])
        return self.header

    def save_1d(self, filename, dim1, I, 
                error=None, dim1_unit=grazing_units.Q,
                dark=None, flat=None, polarization_factor=None, 
                normalization_factor=None):
        """
        This method saves the result of 2D integration. If no filename is
        given the function will exit.

        Parameters
        ----------
        filename : str
            The file name to save the transformed image.
        dim1 : ndarray
            The x coordinates fo the integrated curve.
        I : ndarray
            Transformed image (intensity).
        error : ndarray or None
            The error bar for each intensity.
        dim1_unit : PyGIX.grazing_units.Unit
            Unit of the dim1 array.
        dark :  ???
            Save the dark filename(s) (default: no).
        flat : ???
            Save the flat filename(s) (default: no).
        polarization_factor : float
            The polarization factor.
        normalization_factor: float
            The normalization factor. 
        giGeometry : int
            Grazing-incidence geometry in [0, 1, 2, 3].
        alpha_i : float
            Incident angle.
        misalign : float
            Surface plane tilt angle.
        """
        # dim1_unit = grazing_units.absolute_unit(dim1_unit)
        if filename:
            with open(filename, "w") as f:
                f.write(self.make_headers(
                            dark=dark, flat=flat,
                            polarization_factor=polarization_factor,
                            normalization_factor=normalization_factor))
                f.write("%s# --> %s%s" % (os.linesep, filename, os.linesep))
                if error is None:
                    f.write("#%14s %14s %s" % (dim1_unit.REPR, "I ", os.linesep))
                    f.write(os.linesep.join(["%14.6e  %14.6e" % \
                        (t, i) for t, i in zip(dim1, I)]))
                else:
                    f.write("#%14s  %14s  %14s%s" %
                            (dim1_unit.REPR, "I ", "sigma ", os.linesep))
                    f.write(os.linesep.join(["%14.6e  %14.6e %14.6e" % \
                        (t, i, s) for t, i, s in zip(dim1, I, error)]))
                f.write(os.linesep)

    #---------------------------------------------------------------------------
    #   Some properties
    #---------------------------------------------------------------------------
    #       All taken from pyFAI.azimuthalIntegrator.AzimuthalIntegrator

    def set_maskfile(self, maskfile):
        self.detector.set_maskfile(maskfile)

    def get_maskfile(self):
        return self.detector.get_maskfile()

    maskfile = property(get_maskfile, set_maskfile)

    def set_mask(self, mask):
        self.detector.set_mask(mask)

    def get_mask(self):
        return self.detector.get_mask()

    mask = property(get_mask, set_mask)

    def set_darkcurrent(self, dark):
        self._darkcurrent = dark
        if dark is not None:
            self._darkcurrent_crc = crc32(dark)
        else:
            self._darkcurrent_crc = None

    def get_darkcurrent(self):
        return self._darkcurrent

    darkcurrent = property(get_darkcurrent, set_darkcurrent)

    def set_flatfield(self, flat):
        self._flatfield = flat
        if flat is not None:
            self._flatfield_crc = crc32(flat)
        else:
            self._flatfield_crc = None

    def get_flatfield(self):
        return self._flatfield

    flatfield = property(get_flatfield, set_flatfield)

    def set_darkfiles(self, files=None, method="mean"):
        """
        Set the dark current from one or mutliple files, avaraged
        according to the method provided.

        Parameters
        ----------
        files : str or list(str) or None
            File(s) used to compute the dark current image.
        method : str 
            Method used to compute the dark. Can be "mean" or "median".
        """
        if type(files) in types.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_darkcurrent(None)
        elif len(files) == 1:
            self.set_darkcurrent(
                fabio.open(files[0]).data.astype(numpy.float32))
            self.darkfiles = files[0]
        else:
            self.set_darkcurrent(pyFAI.utils.averageImages(
                                    files, filter_=method, 
                                    format=None, threshold=0))
            self.darkfiles = "%s(%s)" % (method, ",".join(files))

    def set_flatfiles(self, files, method="mean"):
        """
        Set the flat field from one or mutliple files, averaged
        according to the method provided.

        Parameters
        ----------
        files : str or list(str) or None
            File(s) used to compute the flat field image.
        method : str 
            Method used to compute the flat. Can be "mean" or "median".
        """
        if type(files) in types.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_flatfield(None)
        elif len(files) == 1:
            self.set_flatfield(fabio.open(files[0]).data.astype(numpy.float32))
            self.flatfiles = files[0]
        else:
            self.set_flatfield(pyFAI.utils.averageImages(
                                files, filter_=method, 
                                format=None, threshold=0))
            self.flatfiles = "%s(%s)" % (method, ",".join(files))
    
    
    
    
