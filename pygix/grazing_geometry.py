#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Thomas Dane, Jerome Kieffer"
__contact__ = "dane@esrf.fr"
__license__ = "GPLv3+"
__copyright__ = "ESRF - The European Synchrotron, Grenoble, France"
__date__ = "18/11/2014"
__status__ = "Development"
__docformat__ = "restructuredtext"

import threading
import logging
logger = logging.getLogger("pygix.grazing_geometry")
import numpy

from numpy import pi, radians, sin, cos, tan, sqrt, arcsin, arccos, arctan2

import matplotlib.pyplot as plt

from pyFAI import geometry
Geometry = geometry.Geometry
from pyFAI import detectors, utils

from . import grazing_units

try:
    from pyFAI.geometry import bilinear
except ImportError:
    bilinear = None

try:
    from pyFAI.fastcrc import crc32
except ImportError:
    from zlib import crc32


class GrazingGeometry(Geometry):
    """
    This class defines the projection of pixels in detector coordinates
    into angular or reciprocal space under in the grazing-incidence
    geometry. Additionally the pixels in detector coordinates are 
    further projected into polar coordinates for azimuthal integration 
    as in pyFAI. 

    The class is inherets from pyFAI.geometry.Geometry (Geometry). The
    basic assumptions on detector configuration are therefore as 
    defined in Geometry. The correction for detector tilt is handled in
    Geometry, refer to the pyFAI documentation for more details. 

    Briefly:
    - dim1 is the Y dimension of the image
    - dim2 is the X dimension of the image
    - dim3 is along the incoming X-ray beam

    The scattering angles are defined as:
        alpha_i  = incident angle
        alpha_f  = out-of-plane exit angle
        2theta_f = in-plane exit angle

    The q-vectors are described as: (in-plane)
        qy = orthogonal to plane defined by qx, qz (in-plane)
        qz = normal to the surface (out-of-plane)

    The total in-plane scattering vector is given by:
        qxy = sqrt(qx**2 + qy**2)

    Description of the grazing-incidence transformation:
    ----------------------------------------------------
    A pixel with coordinates (t1, t2, t3) as described in Geometry, is
    first corrected for misalignment of the surface plane (if 
    applicable) as follows, where the tilt angle is "misalign":
        t1 = t2 * sin(misalign) + t1 * cos(misalign)
        t2 = t2 * cos(misalign) - t1 * sin(misalign)

    The angular coordinates are given by:
        ang1 = arctan2(t1, sqrt(t2**2 + t3**2))
        ang2 = arctan2(t2, t3)

    When the surface plane is horizontal on the detector (i.e. 
    giGeometry is 0 or 2):
        alpha_f  = ang1
        2theta_f = ang2
    When the surface plane is vertical on the detector (i.e. 
    giGeometry is 1 or 3): 
        alpha_f  = ang2
        2theta_f = ang1

    An additional correction is applied to the exit angle alpha_f to
    account for a non-zero incident angle:
        alpha_f -= alpha_i * cos(2theta_f)

    From the scattering angles, the wavevector transfer components are
    calculated as:
        k = 2 * pi/wavelength

        qx = k * (cos(alpha_f) * cos(2theta_f) - cos(alpha_i))
        qy = k * (cos(alpha_f) * sin(2theta_f))
        qz = k * (sin(alpha_f) + sin(alpha_i))
        qxy = numpy.copysign(sqrt(qx**2 + qy**2))

    Refraction correction:
    ----------------------
    The refractive index of matter to X-rays is given by:
        n = 1 - delta + i*beta

    For a given incident angle (ai), the effective incident angle
    transmitted within the film (ai') is slightly smaller than ai given
    by Snell's law:
        cos(ai) = n*cos(ai')

    At a given ai (known as the critical angle, ac), the effective ai'
    becomes zero. The critical angle is determined by the delta value
    of the refractive index, n:
        ac = sqrt(2 * delta)

    Snell's law can be expanded to give:
        ai'**2 = ai**2 - 2*delta + i*beta
        ai'**2 = ai**2 - ac**2 + i*beta

    """

    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
                 pixel1=0, pixel2=0, splineFile=None, detector=None, 
                 wavelength=None, useqx=True,
                 sample_orientation=None, incident_angle=None, tilt_angle=0):
        """
        """
        Geometry.__init__(self, dist, poni1, poni2,
                          rot1, rot2, rot3,
                          pixel1, pixel2, splineFile, 
                          detector, wavelength)

        self._useqx = useqx
        self._sample_orientation = sample_orientation
        self._incident_angle = incident_angle
        self._tilt_angle = tilt_angle

        # coordinate arrays for reciprocal space transformation
        self._gia_cen = None
        self._gia_crn = None
        self._gia_del = None
        self._giq_cen = None
        self._giq_crn = None
        self._giq_del = None
            
        # coordinate arrays for 1d and 2d integration
        self._absa_cen = None
        self._absa_crn = None
        self._absa_del = None
        self._absq_cen = None
        self._absq_crn = None
        self._absq_del = None
        
        # masks for missing data
        self._giqmask = None
        self._absqmask = None
        self._chimask = None
        self._transformedmask = None

    def reset(self):
        """
        """
        Geometry.reset(self)

        self._gia_cen = None
        self._gia_crn = None
        self._gia_del = None
        self._giq_cen = None
        self._giq_crn = None
        self._giq_del = None

        self._absa_cen = None
        self._absa_crn = None
        self._absa_del = None
        self._absq_cen = None
        self._absq_crn = None
        self._absq_del = None
        
        self._giqmask = None
        self._absqmask = None
        self._chimask = None

    #---------------------------------------------------------------------------
    #   Geometry calculations for grazing-incidence transformations
    #---------------------------------------------------------------------------
    def calc_qzero(self, d1, d2, param=None):
        """ 
        Calculate qx, qy and qz as if tilt and incident angle were
        zero.
        
        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: q 
        @rtype: float or array of floats.
        """
        if not self.wavelength:
            raise RuntimeError(("Scattering vector q cannot be calculated"
                                " without knowing wavelength !!!"))
        
        x,z,y = self.calc_pos_zyx(d0=None, d1=d1, d2=d2, param=param)
        wavevector = 1.0e-11*2*pi/self._wavelength
        
        dd = sqrt(x**2 + y**2 + z**2)
        qx0 = wavevector*(x/dd - 1)
        qy0 = wavevector*(-y/dd)
        qz0 = wavevector*(z/dd)
        return (qx0, qy0, qz0)
    
    def calc_qxyz(self, d1, d2, param=None):
        """
        Calculate qx, qy, qz corrected for tilt and incident angle.
        
        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: q 
        @rtype: float or array of floats.
        """
        if param is None:
            param = self.param
            
        if self._sample_orientation is None:
            raise RuntimeError(("Cannot calculate angles without"
                                " sample orientation defined!!!"))
        if self._incident_angle is None:
            raise RuntimeError(("Cannot calculate angles without"
                                " incident angle defined!!!"))
        
        qx0, qy0, qz0 = self.calc_qzero(d1, d2, param)
        
        ai = radians(self._incident_angle)
        ep = radians(self._tilt_angle)
        ep += (self._sample_orientation-1.0)*radians(90.0)
        
        if (ai == 0) and (ep == 0):
            qx, qy, qz = qx0, qy0, qz0
        else:
            cos_ai = cos(ai)
            sin_ai = sin(ai)
            cos_ep = cos(ep)
            sin_ep = sin(ep)
            
            qx = qx0*cos_ai + qz0*cos_ep*sin_ai + qy0*sin_ep*sin_ai
            qy = qy0*cos_ep - qz0*sin_ep
            qz = qz0*cos_ep*cos_ai + qy0*cos_ai*sin_ep - qx0*sin_ai
        
        qxy = sqrt(qx**2 + qy**2)*numpy.sign(qy)
        return (qx, qy, qz)
        
    def calc_q(self, d1, d2, param=None):
        """
        """
        qx, qy, qz = self.calc_qxyz(d1, d2, param)
        qxy = sqrt(qx**2 + qy**2)*numpy.sign(qy)
        return (qz, qxy)
    
    def calc_angles(self, d1, d2, param=None):
        """ 
        27/01/2016 equations for angles re-derrived.
        # OLD EXPRESSIONS
        #tthf = 2*arcsin(qy*wl/4*pi)
        #alpf = 2*arcsin(qz*wl/4*pi)
        """
        wl = self._wavelength*1e10
        k = 2*pi/wl
        
        qx, qy, qz = self.calc_qxyz(d1, d2, param)
        alpf = arcsin(qz / k)
        tthf = arctan2(qy, (k - qx))
        
        return (alpf, tthf)
    
    def calc_q_corner(self, d1, d2):
        """
        Returns (qz, qxy) for the corner of a given pixel
        (or set of pixels) in (0.01*nm^-1).
        """
        return self.calc_q(d1 - 0.5, d2 - 0.5)
    
    def calc_angles_corner(self, d1, d2):
        """
        """
        return self.calc_angles(d1 - 0.5, d2 - 0.5)

    def giq_center_array(self, shape):
        """
        Generate an array of the given shape with (qz, qxy) for all
        elements.
        """
        if self._giq_cen is None:
            with self._sem:
                if self._giq_cen is None:
                    self._giq_cen = numpy.fromfunction(self.calc_q, shape,
                                                       dtype=numpy.float32)
        return self._giq_cen
    
    def gia_center_array(self, shape):
        """
        """
        if self._gia_cen is None:
            with self._sem:
                if self._gia_cen is None:
                    self._gia_cen = numpy.fromfunction(self.calc_angles, shape,
                                                       dtype=numpy.float32)
        return self._gia_cen
    
    def giq_corner_array(self, shape):
        """
        Note: in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n, m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            qxy[0] and qz[1]
        """
        if self._giq_crn is None:
            with self._sem:
                if self._giq_crn is None:
                    qout_crn, qin_crn = numpy.fromfunction(
                                            self.calc_q_corner,
                                            (shape[0] + 1, shape[1] + 1),
                                            dtype=numpy.float32)
                    #N.B. swap to (ipl, opl) from here on
                    if bilinear:
                        corners = bilinear.convert_corner_2D_to_4D(2, qin_crn, 
                                                                   qout_crn)
                    else:
                        corners = numpy.zeros((shape[0], shape[1], 4, 2),
                                              dtype=numpy.float32)
                        corners[:, :, 0, 0] = qin_crn[:-1, :-1]
                        corners[:, :, 1, 0] = qin_crn[1:, :-1]
                        corners[:, :, 2, 0] = qin_crn[1:, 1:]
                        corners[:, :, 3, 0] = qin_crn[:-1, 1:]
                        corners[:, :, 0, 1] = qout_crn[:-1, :-1]
                        corners[:, :, 1, 1] = qout_crn[1:, :-1]
                        corners[:, :, 2, 1] = qout_crn[1:, 1:]
                        corners[:, :, 3, 1] = qout_crn[:-1, 1:]

                    self._giq_crn = corners
        return self._giq_crn
    
    def gia_corner_array(self, shape):
        """
        Note: in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n, m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            qxy[0] and qz[1]
        """
        if self._gia_crn is None:
            with self._sem:
                if self._gia_crn is None:
                    aout_crn, ain_crn = numpy.fromfunction(
                                            self.calc_angles_corner,
                                            (shape[0] + 1, shape[1] + 1),
                                            dtype=numpy.float32)
                    #N.B. swap to (ipl, opl) from here on
                    if bilinear:
                        corners = bilinear.convert_corner_2D_to_4D(2, ain_crn, 
                                                                   aout_crn)
                    else:
                        corners = numpy.zeros((shape[0], shape[1], 4, 2),
                                              dtype=numpy.float32)
                        corners[:, :, 0, 0] = ain_crn[:-1, :-1]
                        corners[:, :, 1, 0] = ain_crn[1:, :-1]
                        corners[:, :, 2, 0] = ain_crn[1:, 1:]
                        corners[:, :, 3, 0] = ain_crn[:-1, 1:]
                        corners[:, :, 0, 1] = aout_crn[:-1, :-1]
                        corners[:, :, 1, 1] = aout_crn[1:, :-1]
                        corners[:, :, 2, 1] = aout_crn[1:, 1:]
                        corners[:, :, 3, 1] = aout_crn[:-1, 1:]

                    self._gia_crn = corners
        return self._gia_crn
    
    def giq_delta_array(self, shape):
        """
        Generate 2 3D arrays of the given shape with (i,j) with the max
        distance between the center and any corner for qz and qxy.
        
        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2 2D-arrays containing the max delta between a pixel 
        center and any corner in (qz, qxy).
        """
        qout_cen, qin_cen = self.giq_center_array(shape)
        
        if self._giq_del is None:
            with self._sem:
                if self._giq_del is None:
                    qout_delta = numpy.zeros([shape[0], shape[1], 4],
                                             dtype=numpy.float32)
                    qin_delta = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    if self._giq_crn is not None \
                            and self._giq_crn.shape[:2] == tuple(shape):
                        for i in range(4):
                            qout_delta[:, :, i] = \
                                self._giq_crn[:, :, i, 1] - qout_cen
                            qin_delta[:, :, i] = \
                                self._giq_crn[:, :, i, 0] - qin_cen
                    else:
                        qout_crn, qin_crn = numpy.fromfunction(
                                               self.calc_q_corner,
                                               (shape[0]+1, shape[1]+1),
                                               dtype=numpy.float32)
                        qout_delta[:, :, 0] = abs(qout_crn[:-1, :-1] - qout_cen)
                        qout_delta[:, :, 1] = abs(qout_crn[1:, :-1] - qout_cen)
                        qout_delta[:, :, 2] = abs(qout_crn[1:, 1:] - qout_cen)
                        qout_delta[:, :, 3] = abs(qout_crn[:-1, 1:] - qout_cen)
                        qin_delta[:, :, 0] = abs(qin_crn[:-1, :-1] - qin_cen)
                        qin_delta[:, :, 1] = abs(qin_crn[1:, :-1] - qin_cen)
                        qin_delta[:, :, 2] = abs(qin_crn[1:, 1:] - qin_cen)
                        qin_delta[:, :, 3] = abs(qin_crn[:-1, 1:] - qin_cen)
                    
                    qout_delta = qout_delta.max(axis=2)
                    qin_delta  =  qin_delta.max(axis=2)
                    # delta values are enormous along the line defining 
                    # the q split. Setting to zero prevents interpolation
                    qin_delta[numpy.where(qin_delta > 0.0004)] = 0
                    
                    self._giq_del = (qout_delta, qin_delta)
        return self._giq_del

    def gia_delta_array(self, shape):
        """
        Generate 2 3D arrays of the given shape with (i,j) with the max
        distance between the center and any corner for alpha_f and 2theta_f.
        
        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2 2D-arrays containing the max delta between a pixel 
        center and any corner in (alpha_f, 2theta_f).
        """
        aout_cen, ain_cen = self.gia_center_array(shape)

        if self._gia_del is None:
            with self._sem:
                if self._gia_del is None:
                    aout_delta = numpy.zeros([shape[0], shape[1], 4],
                                             dtype=numpy.float32)
                    ain_delta = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    if self._gia_crn is not None \
                            and self._gia_crn.shape[:2] == tuple(shape):
                        
                        for i in range(4):
                            aout_delta[:, :, i] = \
                                self._gia_crn[:, :, i, 1] - qout_cen
                            ain_delta[:, :, i] = \
                                self._gia_crn[:, :, i, 0] - qin_cen
                    else:
                        aout_crn, ain_crn = numpy.fromfunction(
                                               self.calc_angles_corner,
                                               (shape[0]+1, shape[1]+1),
                                               dtype=numpy.float32)
                        aout_delta[:, :, 0] = abs(aout_crn[:-1, :-1] - aout_cen)
                        aout_delta[:, :, 1] = abs(aout_crn[1:, :-1] - aout_cen)
                        aout_delta[:, :, 2] = abs(aout_crn[1:, 1:] - aout_cen)
                        aout_delta[:, :, 3] = abs(aout_crn[:-1, 1:] - aout_cen)
                        ain_delta[:, :, 0] = abs(ain_crn[:-1, :-1] - ain_cen)
                        ain_delta[:, :, 1] = abs(ain_crn[1:, :-1] - ain_cen)
                        ain_delta[:, :, 2] = abs(ain_crn[1:, 1:] - ain_cen)
                        ain_delta[:, :, 3] = abs(ain_crn[:-1, 1:] - ain_cen)

                    self._gia_del = (aout_delta.max(axis=2), \
                                      ain_delta.max(axis=2))
        return self._gia_del
    
    #---------------------------------------------------------------------------
    #   Geometry calculations for 2d and 1d integrations
    #---------------------------------------------------------------------------
    
    def calc_absq_corner(self, d1, d2):
        """
        Returns (alpf, tthf) for the corner of a given pixel
        (or set of pixels) in radians.
        """
        qout, qin = self.calc_q(d1 - 0.5, d2 - 0.5)

        q = sqrt(qin**2 + qout**2)
        chi = arctan2(qin, qout)+pi
        return (chi, q)
    
    def calc_absa_corner(self, d1, d2):
        """
        Returns (alpf, tthf) for the corner of a given pixel
        (or set of pixels) in radians.
        """
        aout, ain = self.calc_angles(d1 - 0.5, d2 - 0.5)

        ang = sqrt(ain**2 + aout**2)
        chi = arctan2(ain, aout)+pi
        return (chi, ang)
    
    def absq_center_array(self, shape):
        """
        """
        qout, qin = self.giq_center_array(shape)

        q = sqrt(qin**2 + qout**2)
        chi = arctan2(qin, qout)+pi

        self._absq_cen = (chi, q)
        return self._absq_cen
    
    def absa_center_array(self, shape):
        """
        """
        aout, ain = self.gia_center_array(shape)

        tth = sqrt(ain**2 + aout**2)
        chi = arctan2(ain, aout)+pi

        self._absa_cen = (chi, tth)
        return self._absa_cen
    
    def absq_corner_array(self, shape):
        """
        N.B. in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n,m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            tthf[0] and alpf[1]
        """
        q_crn = self.giq_corner_array(shape)
        corners = numpy.zeros((shape[0], shape[1], 4, 2),
                              dtype=numpy.float32)
        corners[:,:,:,0] = sqrt(q_crn[:,:,:,0]**2 + q_crn[:,:,:,1]**2)
        corners[:,:,:,1] = arctan2(q_crn[:,:,:,0], q_crn[:,:,:,1])+pi

        self._absq_crn = corners
        return self._absq_crn

    def absa_corner_array(self, shape):
        """
        N.B. in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n,m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            tthf[0] and alpf[1]
        """
        ang_crn = self.gia_corner_array(shape)
        corners = numpy.zeros((shape[0], shape[1], 4, 2),
                               dtype=numpy.float32)
        corners[:,:,:,0] = sqrt(ang_crn[:,:,:,0]**2 + ang_crn[:,:,:,1]**2)
        corners[:,:,:,1] = arctan2(ang_crn[:,:,:,0], ang_crn[:,:,:,1])+pi

        self._absa_crn = corners
        return self._absa_crn

    def absq_delta_array(self, shape):
        """
        """
        chi_cen, q_cen = self.absq_center_array(shape)

        if self._absq_del is None:
            with self._sem:
                if self._absq_del is None:
                    chi_delta = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    q_delta   = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    if (self._absq_crn is not None) \
                            and (self._absq_crn.shape[:2] == tuple(shape)):
                        for i in range(4):
                            chi_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 1] - chi_cen
                            q_delta[:, :, i]   = \
                                self._absq_crn[:, :, i, 0] - q_cen
                    else:
                        chi_crn, q_crn = numpy.fromfunction(
                                            self.calc_absq_corner,
                                            (shape[0]+1, shape[1]+1),
                                            dtype=numpy.float32)

                        chi_delta[:, :, 0] = abs(chi_crn[:-1, :-1] - chi_cen)
                        chi_delta[:, :, 1] = abs(chi_crn[1:, :-1] - chi_cen)
                        chi_delta[:, :, 2] = abs(chi_crn[1:, 1:] - chi_cen)
                        chi_delta[:, :, 3] = abs(chi_crn[:-1, 1:] - chi_cen) 
                        q_delta[:, :, 0]   = abs(q_crn[:-1, :-1] - q_cen)
                        q_delta[:, :, 1]   = abs(q_crn[1:, :-1] - q_cen)
                        q_delta[:, :, 2]   = abs(q_crn[1:, 1:] - q_cen)
                        q_delta[:, :, 3]   = abs(q_crn[:-1, 1:] - q_cen)
                    
                    chi_delta = chi_delta.max(axis=2)
                    q_delta   = q_delta.max(axis=2)
                    
                    
                    # delta values are enormous along the line defining 
                    # the q split. Setting to zero prevents interpolation
                    chi_delta[numpy.where(chi_delta > 0.02)] = 0
                    #q_delta[numpy.where(q_delta > 0.0004)] = 0
                    
                    ttt = '''
                    fig = plt.figure()
                    fig.add_subplot(121)
                    plt.imshow(chi_delta)
                    plt.colorbar()
                    
                    fig.add_subplot(122)
                    plt.imshow(q_delta)
                    plt.colorbar()
                    plt.show()
                    '''
                    self._absq_del = (chi_delta, q_delta)
        return self._absq_del
    
    def absa_delta_array(self, shape):
        """
        """
        chi_cen, ang_cen = self.absa_center_array(shape)

        if self._absa_del is None:
            with self._sem:
                if self._absa_del is None:
                    chi_delta = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    ang_delta = numpy.zeros([shape[0], shape[1], 4],
                                            dtype=numpy.float32)
                    if (self._absa_crn is not None) \
                            and (self._absa_crn.shape[:2] == tuple(shape)):
                        for i in range(4):
                            chi_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 1] - chi_cen
                            ang_delta[:, :, i]   = \
                                self._absq_crn[:, :, i, 0] - ang_cen
                    else:
                        chi_crn, ang_crn = numpy.fromfunction(
                                            self.calc_absa_corner,
                                            (shape[0]+1, shape[1]+1),
                                            dtype=numpy.float32)

                        chi_delta[:, :, 0] = abs(chi_crn[:-1, :-1] - chi_cen)
                        chi_delta[:, :, 1] = abs(chi_crn[1:, :-1] - chi_cen)
                        chi_delta[:, :, 2] = abs(chi_crn[1:, 1:] - chi_cen)
                        chi_delta[:, :, 3] = abs(chi_crn[:-1, 1:] - chi_cen) 
                        ang_delta[:, :, 0] = abs(ang_crn[:-1, :-1] - ang_cen)
                        ang_delta[:, :, 1] = abs(ang_crn[1:, :-1] - ang_cen)
                        ang_delta[:, :, 2] = abs(ang_crn[1:, 1:] - ang_cen)
                        ang_delta[:, :, 3] = abs(ang_crn[:-1, 1:] - ang_cen)
                    
                    chi_delta = chi_delta.max(axis=2)
                    ang_delta = ang_delta.max(axis=2)
                    self._absa_del = (chi_delta, ang_delta)
        return self._absa_del
        
    #---------------------------------------------------------------------------
    #   Masking functions for inaccessible data under grazing-incidence
    #--------------------------------------------------------------------------- 

    def make_qmask(self, process, npt, bins_x, bins_y):
        """
        Transformation routines bbox, splitpix, csr and lut (+ocl) result in 
        interpolation over the unaccessible regions of reciprocal space. This
        function makes a mask for these regions.

        This is determined by setting qy = 0 (i.e. when 2theta_f = 0) and 
        calculating qx as a function of qz:

        qx = k(cos(alpha_f)cos(2theta_f) - cos(alpha_i))
        qx = k(cos(alpha_f) - cos(alpha_i))

        qz = k(sin(alpha_f) + sin(alpha_i))
        alpha_f = arcsin(qz/k - sin(alpha_i))

        """
        ai = radians(self._incident_angle)
        wvec = 1.0e-11*2*pi/self._wavelength
        mask = numpy.ones(npt).astype(numpy.float32).ravel()
        cos_ai = cos(ai)
        sin_ai = sin(ai)
        
        if 'polar' in process:
            print 'Masking for polar transforms not yet implemented'
            mask = numpy.reshape(mask, (npt[1],npt[0]))
            self._absqmask = mask
            return self._absqmask

            #chi = radians(numpy.outer(bins_y, numpy.ones(npt[0])).ravel())
            #qz_max   = bins_x.max() * cos(numpy.degrees(abs(bins_y).max()-pi))
            
            #print bins_x.max(), qz_max
            #bins_qz  = numpy.linspace(bins_x.min(), qz_max, npt[0])

            #qLim_1d  = abs(wvec*(cos(arcsin((bins_x/wvec) - sin_ai)) - cos_ai))
            #chi_lim1d = arctan2(qLim_1d, bins_x)
            #chi_lim   = numpy.outer(numpy.ones(npt[1]), chi_lim1d).ravel()
            
            #mask[abs(chi) < chi_lim] = 0
        else:
            qxy = numpy.outer(numpy.ones(bins_y.shape[0]), bins_x).ravel()
                    
            qLim_1d = abs(wvec*(cos(arcsin((bins_y/wvec) - sin_ai)) - cos_ai))
            qLim = numpy.outer(qLim_1d, numpy.ones(bins_x.shape[0])).ravel()

            mask[abs(qxy) < qLim] = 0
            mask = numpy.reshape(mask, (npt[1],npt[0]))
	
            self._giqmask = mask
            return self._giqmask

    def make_chimask(self, q_max, chi):
        """
        """
        mask = numpy.ones(chi.shape).astype(numpy.float32)

        alpi = radians(self._incident_angle)
        wvec = 1.0e-9*2*pi/self._wavelength

        cos_alpi = cos(alpi)
        sin_alpi = sin(alpi)
        q_lim  = abs(wvec*(cos(arcsin((q_max/wvec) - sin_alpi)) - \
            cos_alpi))

        chi_lim = numpy.rad2deg(arctan2(q_lim, q_max))
        mask[abs(chi) < chi_lim] = 0

        self._chimask = mask
        return self._chimask

    #---------------------------------------------------------------------------
    #   Some properties
    #---------------------------------------------------------------------------  

    def set_useqx(self, useqx):
        """
        """
        if not isinstance(useqx, (bool)):
            raise RuntimeError(("useqx must be True or False"))
        else:
            self._useqx = useqx
        self.reset()

    def get_useqx(self):
        """
        """
        return self._useqx

    def set_sample_orientation(self, sample_orientation):
        """
        """
        orientation_dict = {
            "1" : "sample plane horizontal; +ve Qz = bottom-to-top",
            "2" : "sample plane vertical;   +ve Qz = left-to-right",
            "3" : "sample plane horizontal; +ve Qz = top-to-bottom",
            "4" : "sample plane vertical;   +ve Qz = right-to-left"}
        
        if (sample_orientation % 1 != 0) or (sample_orientation < 1) or \
                sample_orientation not in [1, 2, 3, 4]:

            print "Sample orientation must be defined as integeter (0 to 3):\n"
            for key, val in sorted(orientation_dict.items()):
                print (key + ": " + val)
        else:
            self._sample_orientation = sample_orientation
        self.reset()

    def get_sample_orientation(self):
        return self._sample_orientation

    def set_incident_angle(self, incident_angle):
        self._incident_angle = incident_angle
        self.reset()

    def get_incident_angle(self):
        return self._incident_angle

    def set_tilt_angle(self, tilt_angle):
        self._tilt_angle = tilt_angle
        self.reset()

    def get_tilt_angle(self):
        return self._tilt_angle

    def print_giParams(self):
        print "Incident angle = %g degrees, %g radians" % (self._alpha_i, \
            radians(self._alpha_i))
        print "Surface plane misalignment: %g radians" % self.misalign
        giGeometry = self._giGeometry

        print "Grazing-incidence geometry:"
        if giGeometry == 0:
            print "0: Sample plane horizontal; bottom-top = +ve Qz"
        elif giGeometry == 1:
            print "1: Sample plane vertical;   left-right = +ve Qz"
        elif giGeometry == 2:
            print "2: Sample plane horizontal; bottom-top = -ve Qz"
        elif giGeometry == 3:
            print "3: Sample plane vertical;   left-right = -ve Qz"


