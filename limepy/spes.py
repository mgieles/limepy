# -*- coding: utf-8 -*-

import numpy
import scipy
from scipy.interpolate import BPoly, interp1d, UnivariateSpline
from numpy import exp, sqrt, pi, sin, cos, log10
from scipy.special import gamma, gammainc, gammaincc, hyp1f1
from scipy.integrate import ode, simps
from math import factorial, sinh

#     Authors: Mark Gieles & Ian Clayon (Surrey 2018)

class spes:
    def __init__(self, phi0, **kwargs):
        r"""

        SPES models

        Spherical Potential Escapers Stitched models

        This code solves the models presented in Claydon et al. 2019 (C19)
        and calculates radial profiles for some useful quantities. The 
        models are defined by the distribution function (DF) of eq. (3) in 
        C19.

        Argument (required):
        ====================

        phi0 : scalar, required
             Central dimensionless potential

        Named arguments (required):
        ===========================

        B : scalar, required
          Reduction of the DF at trunction [0-1]
        eta  : scalar, required
          Velocity dispersion of PEs in model units [0-1]

        Input for scaling:
        ==================

        G : scalar, optional
          Gravitational const; 0.004302 (km/s)^2 pc/Msun if both M and 
                               scale radius are set (see below)
        M : scalar, optional
          Total mass of bound stars and PEs < rt
        r0, rh, rv, rt : scalar, optional
          Final scaled radius; default=rh=3 

        Options:
        ========

        project : bool, optional
                Compute model properties in projection; default=False
        nrt : scalar, optional
              Solve model until nrt*rt; default=2
        potonly : bool, optional
                Fast solution by solving potential only; default=False
        max_step : scalar, optional
                 Maximum step size for ode output; default=1e10
        verbose : bool, optional
                Print diagnostics; default=False
        ode_atol : absolute tolerance parameter for ode solver; default=1e-7
        ode_rtol : relative tolerance parameter for ode solver; default=1e-7

        Output variables:
        =================

        All models:
        -----------
         rhat, phihat, rhohat : radius, potential and density in model units
         r, phi, rho : as above, in scaled units 
         v2 : total mean-square velocity
         mc : enclosed mass profile
         r0, rh, rv, rt : radii (King, half-mass, virial, truncation)
         K : kinetic energy
         U, Q : potential energy, virial ratio

         volume : phase-space volume occupied by model
         nstep : number of integration steps (depends on ode_rtol&ode_atol)
         converged : bool flag to indicate whether model was solved

        Projected models:
        -----------------
         Sigma : surface (mass) density
         v2p : line-of-sight mean-square velocity
         v2R, v2T : radial and tangential component of mean-square velocity
                  on plane of the sky (same as v2p)

        Examples:
        =========

        Construct a SPES model with W0=7, B=0.99, eta=0.1 for physical 
        parameters M=1e5 Msun, rh=3 pc. Solve until 2rt and project on sky.

        >>> s = limepy.spes(7,B=0.99,eta=0.1,nrt=2,M=1e5,rh=3,project=True)

        Plot surface density profile
        >>> plt.plot(s.r, s.Sigma)

        """

        # Set parameters 
        self._set_kwargs(phi0, **kwargs)

        # Required constants
        self.rhoint0 = [self._rhoint(self.phi0, 0)]
        self.r0 = 1
        self._poisson(self.potonly)

        # Optional scaling
        if (self.scale): self._scale()

        # Optional computation of model properties in projection
        if (self.project): self._project()

        # Optional output
        if (self.verbose):
            print("\n Model properties: ")
            print(" ----------------- ")
            print(" phi0 = %5.2f; B = %12.6e; eta = %10.4e"%(self.phi0, self.B, self.eta))
            print(" fpe = %10.4e"%self.fpe)
            print(" Converged = %s"%(self.converged))
            if (self.potonly):
                print(" M = %10.3f; U = %10.4f "%(self.M, self.U))
            else:
                pV = self.rho[self.nbound-1]*self.v2[self.nbound-1]*(4.*pi/3)*self.rt**3
                out1 = (self.U,self.K,pV, -self.K/(self.U+3*pV))
                print(" M = %10.3e "%self.M)
                frm = " U = %9.3e; K = %9.3e; p_eV_e = %9.3e; "
                frm += "Q = -K/(U+3p_eV_e) = %5.3f "
                print(frm%out1)

            out2 = (self.rv/self.rh, self.rh/self.r0, self.rt/self.r0)
            frm = " rv/rh = %4.3f; rh/r0 = %6.3f; rt/r0 = %7.3f"

            print(frm%out2)

    def _set_kwargs(self, phi0, **kwargs):
        """ Set parameters and scales """

        self.model = "spes"
        
        self.phi0 = phi0
        self.B, self.C, self.eta, self.fpe = None, None, None, None
        self._MS, self._RS, self._GS = None, None, None
        self.scale_radius = None
        self.scale = False
        self.project = False
        self.maxr = 1e99
        self.rt = 1e99
        self.max_step = self.maxr
        self.max_arg_exp = 700  # Maximum argument for exponent 
        self.minimum_phi = 1e-8 # Stop criterion for integrator

        self.ode_atol = 1e-7
        self.ode_rtol = 1e-7
        self.nmbin = 1 # for future expansion to multimass

        self.G = 9.0/(4.0*pi)
        self.mu, self.alpha = numpy.array([1.0]),numpy.array([1.0])
        self.s2 = 1.0
        self.s2j = numpy.array([1.0])

        self.potonly, self.multi, self.verbose = [False]*3

        self.nstep=1
        self.converged=True
        self._interpolator_set=False

        # Spes specific
        self.nrt = 2
        if kwargs is not None:
            for key, value in kwargs.items():
                # Check for scaling input (similar to LIMEPY)
                if key is 'G':
                    self._GS, self.scale = value, True
                elif key is 'M':
                    self._MS, self.scale = value, True
                elif key is 'r0':
                    if self.scale_radius is None:
                        self._RS,self.scale_radius,self.scale = value,'r0', True
                    else:
                        error="Can not set scale radius to r0,already set to %s"
                        raise ValueError(error%self.scale_radius)
                elif key is 'rh':
                    if self.scale_radius is None:
                        self._RS, self.scale_radius, self.scale=value,'rh', True
                    else:
                        error="Can not set scale radius to rh,already set to %s"
                        raise ValueError(error%self.scale_radius)
                elif key is 'rv':
                    if self.scale_radius is None:
                        self._RS, self.scale_radius, self.scale=value,'rv', True
                    else:
                        error="Can not set scale radius to rv,already set to %s"
                        raise ValueError(error%self.scale_radius)
                elif key is 'rt':
                    if self.scale_radius is None:
                        self._RS, self.scale_radius, self.scale=value,'rt', True
                    else:
                        error="Can not set scale radius to rt,already set to %s"
                        raise ValueError(error%self.scale_radius)
                # SPES specific checks
                elif key is 'C' :
                    error="Can not set C, use B & eta. C and fpe are computed internally"
                    raise ValueError(error)
                else:
                    # Set input parameters
                    setattr(self, key, value)
        
            # Check if 2 required spes parameters are set
            npar = 0
            for par in [self.B, self.eta, self.fpe]:
                if (par != None): npar +=1
            if npar<2:
                error="Two model parameters from {B, eta, fpe} are needed"
                raise ValueError(error)

            # There are 3 options to define the model
            if (self.B != None) and (self.eta != None):
                # this is most straightforward, fpe is computed after solving
                self.C = 1 + (self.B-1)/self.eta**2
            else:
                # if fpe is set in input, iteration is required
                self._iterate_on_B()

            if (self.scale):
                if self._MS is None:
                    self._MS = 1e5
                    if (self.verbose): 
                        print(" No mass-scale provided, set to default M = 1e5")
                if self._RS is None:
                    self._RS, self.scale_radius = 3, 'rh'
                    if (self.verbose): 
                        print(" No radius-scale provided, set to default rh = 3")
                if self._GS is None:
                    self._GS = 0.004302
                    if (self.verbose): 
                        print(" No G provided, set to default: G = 0.004302")
                if (self.verbose): 
                    vars=(self._GS, self._MS, self.scale_radius, self._RS)
                    print(" Model scaled to: G = %s, M = %s, %s = %s"%vars)


        return

    def _iterate_on_B(self):
        # Iteratively find B based on input fpe and eta
        frac = 0

        # Check fpe input range
        if (self.fpe < 0) or (self.fpe > 1):
            error="error: fpe needs to be between 0 and 1"
            raise ValueError(error)

        # Option 1: eta & fpe are set, find B
        if (self.eta != None):
            # Check value of eta within allowed range
            if (self.eta < 0) or (self.eta > 1):
                error="error: eta needs to be between 0 and 1"
                raise ValueError(error)

            self.B = 0.999 # trial
            self.C = 1 + (self.B-1)/self.eta**2

            # Find B within 1% (TBD: user defined accuracy)
            i = 0
            
            while abs(frac - self.fpe)/self.fpe > 0.001:
                self.nstep=1
                # Set parameters and scales
                if frac > self.fpe:
                    self.B += 0.5*(1-self.B)
                else:
                    self.B -= 0.5*(1-self.B)
                
                self.C = 1 + (self.B-1)/self.eta**2

                self.rhoint0 = [self._rhoint(self.phi0, 0)]
                self.r0 = 1.0

                # Solve Poisson equation to get the potential
                self._poisson(self.potonly)
                frac = self.Mpe/self.M
                self.rhohat = self._rhohat(self.phihat, self.r, 0)            
                i+=1

        # Option 2: B is set, find eta
        else:
            # Check value of B within range
            if (self.B < 0) or (self.B > 1):
                error="error: B needs to be between 0 and 1"
                raise ValueError(error)

            # TBD
            raise ValueError("Combo B & fpe TBD")
            #            self.eta = sqrt((1.-self.B)/(1.0-self.C)) 

    def _logcheck(self, t, y):
        """ Logs steps and checks for final values """
        if (y[0]>self.minimum_phi):
            if (t>0): self.r, self._y = numpy.r_[self.r, t], numpy.c_[self._y, y]
            self.nstep+=1
            return 0
        else:
            return -1

    def _logcheck2(self, t, y):
        """ Logs steps and checks for final values """

        # Ensure that tidal radius value is not added again
        if (t>self.rt):
            self.r, self._y = numpy.r_[self.r, t], numpy.c_[self._y, y]
            self.nstep+=1

        return 0 if (t<=self.nrt*self.rt) else -1



    def _poisson(self, potonly):
        """ Solves Poisson equation """
        # y = [phi, u_j, U, K_j], where u = -M(<r)/G

        # Initialize
        self.r = numpy.array([0])
        self._y = numpy.r_[self.phi0, numpy.zeros(2*self.nmbin+1)]

        if (not potonly): self._y = numpy.r_[self._y,numpy.zeros(self.nmbin)]

        # Ode solving using Runge-Kutta integator of order 4(5) 'dopri5'
        # (Hairor, Norsett& Wanner 1993)
        max_step = self.maxr if (potonly) else self.max_step
        sol = ode(self._odes)
        sol.set_integrator('dopri5',nsteps=1e6,max_step=max_step,
                           atol=self.ode_atol,rtol=self.ode_rtol)
        sol.set_solout(self._logcheck)
        sol.set_f_params(potonly)
        sol.set_initial_value(self._y,0)
        sol.integrate(self.maxr)

        # Extrapolate to rt
        derivs = self._odes(self.r[-1], self._y[:,-1], self.potonly)
        self.rt = self.r[-1] - self._y[0][-1]/derivs[0]
        dr = self.rt - self.r[-1]
        ylast = [0]

        for i in range(len(derivs)-1):
            ylast.append(self._y[1+i,-1] + derivs[i+1]*dr)
        self._y = numpy.c_[self._y, ylast]

        # Set the converged flag to True if successful
        if (self.rt < self.maxr)&(sol.successful()):
            self.converged=True
        else:
            self.converged=False

        # Fill arrays needed if potonly=True
        self.r = numpy.r_[self.r, self.rt]
        self.rhat = self.r*1.0

        self.phihat = numpy.r_[self._y[0,:]]
        self.phi = self.phihat*1.0
        self.nbound = len(self.phihat)

        self._Mjtot = -sol.y[1:1+self.nmbin]/self.G

        self.M = sum(self._Mjtot)
        self.Mpe = -sol.y[1+self.nmbin]/self.G
        self.fpe = self.Mpe/self.M
        
        # Save the derivative of the potential for the potential interpolater
        dphidr = numpy.sum(self._y[1:1+self.nmbin,1:],axis=0)/self.r[1:]**2
        self.dphidrhat1 = numpy.r_[0, dphidr] 

        self.A = 1/(2*pi*self.s2j)**1.5/self.rhoint0

        self.mc = -self._y[1,:]/self.G
        

        # Compute radii to be able to scale in case potonly=True
        self.U = self._y[2+self.nmbin,-1]  - 0.5*self.G*self.M**2/self.rt

        # Get half-mass radius from cubic interpolation, because the half-mass
        # radius can be used as a scale length, this needs to be done
        # accurately, a linear interpolation does not suffice. Because the
        # density array is not known yet at this point, we need a temporary
        # evaluation of density in the vicinity of r_h
        ih = numpy.searchsorted(self.mc, 0.5*self.mc[-1])-1
        rhotmp=numpy.zeros(2)
        for j in range(self.nmbin):
            phi = self.phihat[ih:ih+2]/self.s2j[j]
            rhotmp += self.alpha[j]*self._rhohat(phi, self.r[ih:ih+2], j)
        drdm = 1./(4*pi*self.r[ih:ih+2]**2*rhotmp)
        rmc_and_derivs = numpy.vstack([[self.r[ih:ih+2]],[drdm]]).T

        self.rh = BPoly.from_derivatives(self.mc[ih:ih+2], rmc_and_derivs)(0.5*self.mc[-1])
        self.rv = -0.5*self.G*self.M**2/self.U

        # Solve (mass-less) part outside rt
        if (self.nrt > 1):
            # Continue to solve until rlast
            sol.set_solout(self._logcheck2)
            sol.set_f_params(potonly)
            sol.set_initial_value(self._y[:,-1],self.rt)
            sol.integrate(self.nrt*self.rt)

        self.rhat = self.r*1.0
        self.phihat = numpy.r_[self.phihat, self._y[0,self.nbound:]]
        self.mc = numpy.r_[self.mc, numpy.zeros(len(self._y[0,self.nbound:])) + self.mc[-1]]
        self.phi = self.phihat*1.0

        dphidr = numpy.sum(self._y[1:1+self.nmbin,self.nbound:],axis=0)/self.r[self.nbound:]**2
        self.dphidrhat1 = numpy.r_[self.dphidrhat1, dphidr] 

        # Additional stuff
        if (not potonly):
            # Calculate kinetic energy
            self.K = numpy.sum(sol.y[4:5])

            # Calculate density and velocity dispersion components
            if (not self.multi):
                self.rhohat = self._rhohat(self.phihat, self.r, 0)
                self.rhohatpe = self._rhohatpe(self.phihat, self.r, 0)
                self.rho = self.rhohat*1.0
                self.rhope = self.rhohatpe*1.0

                self.v2 = self._get_v2(self.phihat, self.rhohat, 0)

    def _rhohat(self, phi, r, j):
        """
        Wrapper for _rhoint when either: both phi or r are arrays, or both
        are scalar
        """
        if not hasattr(phi,"__len__"): phi = numpy.array([phi])
        if not hasattr(r,"__len__"): r = numpy.array([r])

        n = max([phi.size, r.size])
        rhohat = numpy.zeros(n)

        for i in range(n):
            if (phi[i]<self.max_arg_exp) or (numpy.isnan(phi[i])):
                rhohat[i] = self._rhoint(phi[i], r[i])
                rhohat[i] /= self.rhoint0[j]
            else:
                # For large phi compute the ratio in one go (Section 4.1, GZ15)
                rhohat[i] = exp(phi[i]-self.phi0j[j]) if (self.multi) else 0

        return rhohat

    def _rhohatpe(self, phi, r, j):
        """
        Wrapper for _rhointpe when either: both phi or r are arrays, or both
        are scalar
        """
        if not hasattr(phi,"__len__"): phi = numpy.array([phi])
        if not hasattr(r,"__len__"): r = numpy.array([r])

        n = max([phi.size, r.size])
        rhohatpe = numpy.zeros(n)

        for i in range(n):
            rhohatpe[i] = self._rhointpe(phi[i])
            rhohatpe[i] /= self.rhoint0[j]

        return rhohatpe

    def _rhoint(self, phi, r):
        """
        Dimensionless density integral as a function of phi and r (scalar)
        """
        if phi > 0 : 
            # Bound part
            rhoint = exp(phi)*gammainc(1.5, phi)
            rhoint -= self.B*phi**1.5/gamma(2.5)
            rhoint -= self.C*phi**2.5/gamma(3.5)
            # Add PEs
            rhoint += self._rhointpe(phi) 
        else:
            rhoint = self._rhointpe(phi)
 
        return rhoint

    def _rhointpe(self, phi):
        phie = phi/self.eta**2
        if phi > 0:
            # TBD: Check whether self.max_arg_exp hs correct value
            if phie < self.max_arg_exp:
                return (1 - self.B)*self.eta**3*exp(phie)*gammaincc(1.5, phie)
            else:
                # TBD: needed?
                return (1 - self.B)*self.eta**3*sqrt(phie)/gamma(1.5)
        else:
            return (1 - self.B)*self.eta**3*exp(phie) 

    def _get_v2(self, phi, rho, j):
        v2 = numpy.zeros(phi.size)
        for i in range(v2.size):
            v2[i] = self._rhov2int(phi[i])
            v2[i] /= rho[i]*self.rhoint0[j]
        return v2

    def _rhov2int(self, phi):
        """Compute dimensionless pressure integral for phi, r """

        # Isotropic 
        if phi > 0 : 
            # Bound part
            rhov2 = exp(phi)*gammainc(2.5, phi)
            rhov2 -= self.B*phi**2.5/gamma(3.5)
            rhov2 -= self.C*phi**3.5/gamma(4.5)

            rhov2 += self._rhov2intpe(phi)
        else:
            # Only PEs
            rhov2 = self._rhov2intpe(phi)

        rhov2  *= 3

        return rhov2

    def _rhov2intpe(self, phi):

        phie = phi/self.eta**2
        if phi >= 0:
            if phie < self.max_arg_exp:
                # exp(x) only works for x<700 
                return (1 - self.B)*self.eta**5*exp(phie)*gammaincc(2.5, phie)
            else:
                # 9/2/2019: correct asymptotic behaviour
                return (1 - self.B)*self.eta**5*phie**1.5/gamma(2.5)

        else:
            return (1 - self.B)*self.eta**5*exp(phie) 

    def _beta(self, r, v2r, v2t):
        # NOT IN USE
        """ Calculate the anisotropy profile """

        beta = numpy.zeros(r.size)
        if (self.ra < self.ramax):
            c = (v2r>0.)
            beta[c] = 1.0 - 0.5*v2t[c]/v2r[c]
        return beta

    def _odes(self, x, y, potonly):
        """ Solve ODEs """
        # y = [phi, u, u_pe, U, K_j], where u = -M(<r)/G

        # phi
        derivs = [y[1]/x**2] if (x>0) else [0]

        # u ~ enclosed mass
        derivs.append(-9.0*x**2*self._rhoint(y[0], x)/self.rhoint0[0])

        # u_pe (Keep track of PE mass distribution)
        derivs.append(-9.0*x**2*self._rhointpe(y[0])/self.rhoint0[0])

        dUdx  = 2.0*pi*derivs[1]*y[0]/9.

        derivs.append(dUdx)
        if (not potonly): #dK/dx
            rv2 = self._rhov2int(y[0]/self.s2)
            P = 2*pi*x**2*rv2/self.rhoint0[0]

            derivs.append(P)
        return derivs


    def _setup_phi_interpolator(self):
        """ Setup interpolater for phi, works on scalar and arrays """

        # Generate piecewise 3th order polynomials to connect the discrete
        # values of phi obtained from from Poisson, using dphi/dr
        self._interpolator_set = True

        if (self.scale):
            phi_and_derivs = numpy.vstack([[self.phi],[self.dphidr1]]).T
        else:
            phi_and_derivs = numpy.vstack([[self.phihat],[self.dphidrhat1]]).T

        self._phi_poly = BPoly.from_derivatives(self.r,phi_and_derivs)

    def _scale(self):
        """ Scales the model to the units set in the input: GS, MS, RS """

        Mstar = self._MS/self.M
        Gstar = self._GS/self.G
        if (self.scale_radius=='r0'): Rstar = self._RS/self.r0
        if (self.scale_radius=='rh'): Rstar = self._RS/self.rh
        if (self.scale_radius=='rv'): Rstar = self._RS/self.rv
        if (self.scale_radius=='rt'): Rstar = self._RS/self.rt
        v2star =  Gstar*Mstar/Rstar

        # Update the scales that define the system (see Section 2.1.2 of GZ15)
        self.G *= Gstar
        self.rs = Rstar
        self.s2 *= v2star
        self.s2j *= v2star

        # Scale all variable needed when run with potonly=True
        self.r, self.r0, self.rt = (q*Rstar for q in [self.rhat,
                                                      self.r0, self.rt])
        self.rh, self.rv = (q*Rstar for q in [self.rh,self.rv])

        self.M *= Mstar
        self.Mpe *= Mstar

        self.phi = self.phihat * v2star
        self.dphidr1 = self.dphidrhat1 * v2star/Rstar
        self.mc *= Mstar
        self.U *= Mstar*v2star
        self.A *= Mstar/(v2star**1.5*Rstar**3)

# TBD, needed?
#        self.volume *= v2star**1.5*Rstar**3

        # Scale density, velocity dispersion components, kinetic energy
        self.rho = self.rhohat*Mstar/Rstar**3
        self.rhope = self.rhohatpe*Mstar/Rstar**3

        if (not self.potonly):
            self.v2 *= v2star 
            self.K *= Mstar*v2star 

    def _tonp(self, q):
        q = numpy.array([q]) if not hasattr(q,"__len__") else numpy.array(q)
        return q

    def _project(self):
        """
        Compute projected mass density (Sigma) and projected <v2> profiles
        """

        # Initialise the projected quantities:
        # R is the projected (2d) distance from the center, Sigma is the
        # projected density, v2p is the line-of-sight velocity dispersion,
        # v2R and v2T are the radial and tangential velocity dispersion
        # components projected on the plane of the sky

        # Initialise some arrays
        R = self.r
        Sigma = numpy.zeros(self.nstep)
        v2p = numpy.zeros(self.nstep)
        v2R = numpy.zeros(self.nstep)
        v2T = numpy.zeros(self.nstep)
        mcp = numpy.zeros(self.nstep)


        # Project model properties for each R
        for i in range(self.nstep-1):
            c = (self.r >= R[i])
            r = self.r[c]
            z = sqrt(abs(r**2 - R[i]**2)) # avoid small neg. values
            Sigma[i] = 2.0*abs(simps(self.rho[c], x=z))
            v2p[i] = abs(2.0*simps(self.rho[c]*self.v2[c]/3., x=z))
            v2p[i] /= Sigma[i]

            v2R[i] = v2p[i]
            v2T[i] = v2p[i]

            # Cumulative mass in projection
            if (i>0):
                x = self.r[i-1:i+1]
                mcp[i] = mcp[i-1] + 2*pi*simps(x*Sigma[i-1:i+1], x=x)
            mcp[-1] = mcp[-2]

            # Radius containing half the mass in projection
            self.rhp = numpy.interp(0.5*mcp[-1], mcp, self.r)

        # Find final value from extrapolating logarithmic slope
        Sigma_slope = log10(Sigma[-2]/Sigma[-3])/log10(self.r[-2]/self.r[-3])
        v2p_slope = log10(v2p[-2]/v2p[-3])/log10(self.r[-2]/self.r[-3])


        Sigma[-1] = Sigma[-2]*(self.r[-1]/self.r[-2])**Sigma_slope
        v2p[-1] = v2p[-2]*(self.r[-1]/self.r[-2])**v2p_slope
        v2R[-1] = v2p[-1]
        v2T[-1] = v2p[-1]

        self.R, self.Sigma = R, Sigma
        self.v2p, self.v2R, self.v2T = v2p, v2R, v2T
#        self.mcp = mcp

        return
        
    def get_Paz(self, az_data, R_data, jns):
        """ 
        Computes probability of line of sight acceleration at projected R : P(az|R)  
        """

        # Under construction !!! 

        # Return P(az|R)

        az_data = abs(az_data) # Consider only positive values
        
        # Assumes units of az [m/s^2] if self.G ==0.004302, else models units
        # Conversion factor from [pc (km/s)^2/Msun] -> [m/s^2]
        az_fac = 1./3.0857e10 if (self.G==0.004302) else 1
        
        if (R_data < self.rt):
            nz = self.nstep                   # Number of z value equal to number of r values
            zt = sqrt(self.rt**2 - R_data**2) # maximum z value at R

            z = numpy.logspace(log10(self.r[1]), log10(zt), nz)

            spl_Mr = UnivariateSpline(self.r, self.mc, s=0, ext=1)  # Spline for enclosed mass

            r = sqrt(R_data**2 + z**2)                        # Local r array
            az = self.G*spl_Mr(r)*z/r**3                     # Acceleration along los
            az[-1] = self.G*spl_Mr(self.rt)*zt/self.rt**3    # Ensure non-zero final data point

            az *= az_fac # convert to [m/s^2]
            az_spl = UnivariateSpline(z, az, k=4, s=0, ext=1) # 4th order needed to find max (can be done easier?)
            
            zmax = az_spl.derivative().roots()  # z where az = max(az), can be done without 4th order spline?
            azt = az[-1]                        # acceleration at the max(z) = sqrt(r_t**2 - R**2)

            # Setup spline for rho(z)
            if jns == 0 and self.nmbin == 1:
                rho = self.rho
            else:
                rho = self.rhoj[jns]

            rho_spl = UnivariateSpline(self.r, rho, ext=1, s=0)
            rhoz = rho_spl(sqrt(z**2 + R_data**2))
            rhoz_spl = UnivariateSpline(z, rhoz, ext=1, s=0)

            # Now compute P(a_z|R)
            # There are 2 possibilities depending on R:
            #  (1) the maximum acceleration occurs within the cluster boundary, or 
            #  (2) max(a_z) = a_z,t (this happens when R ~ r_t)
            
            nr, k = nz, 3 # bit of experimenting

            # Option (1): zmax < max(z)
            if len(zmax)>0:
                zmax = zmax[0] # Take first entry for the rare cases with multiple peaks
                # Set up 2 splines for the inverse z(a_z) for z < zmax and z > zmax
                z1 = numpy.linspace(z[0], zmax, nr)
                z2 = (numpy.linspace(zmax, z[-1], nr))[::-1] # Reverse z for ascending az
                
                z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)
                z2_spl = UnivariateSpline(az_spl(z2), z2, k=k, s=0, ext=1)

            # Option 2: zmax = max(z)
            else: 
                zmax = z[-1]
                z1 = numpy.linspace(z[0], zmax, nr)
                z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)

            # Maximum acceleration along this los
            azmax = az_spl(zmax)

            # Now determine P(az_data|R)
            if (az_data < azmax):
                z1 = max([z1_spl(az_data), z[0]]) # first radius where az = az_data
                Paz = rhoz_spl(z1)/abs(az_spl.derivatives(z1)[1])

                if (az_data> azt):
                    # Find z where a_z = a_z,t
                    z2 = z2_spl(az_data)
                    Paz += rhoz_spl(z2)/abs(az_spl.derivatives(z2)[1])

                # Normalize to 1
                Paz /= rhoz_spl.integral(0, zt)
                self.z = z 
                self.az = az
                self.Paz = Paz
                self.azmax = azmax
                self.zmax = zmax
            else:
                self.Paz = 0
        else:
            self.Paz = 0
        
        return 

     
    def interp_phi(self, r):
        """ Returns interpolated potential at r, works on scalar and arrays """

        if not hasattr(r,"__len__"): r = numpy.array([r])
        if (not self._interpolator_set): self._setup_phi_interpolator()

        phi = numpy.zeros([r.size])
        inrt = (r<self.rt)
        # Use 3th order polynomials to interp, using phi'
        if (sum(inrt)>0): phi[inrt] = self._phi_poly(r[inrt])
        return phi

    def df(self, *arg):
        """
        Returns the value of the normalised DF at a given position in phase
        space, can only be called after solving Poisson's equation

        Arguments can be:
          - r, v                  
          - x, y, z, vx, vy, vz  

        Works with scalar and array input

        """

        if (len(arg) != 2) and (len(arg) != 6):
            raise ValueError("Error: df needs 2 or 6 arguments")

        if len(arg) == 2:
            r, v = (self._tonp(q) for q in arg)
            r2, v2 = r**2, v**2
            
        if len(arg) == 6:
            x, y, z, vx, vy, vz = (self._tonp(q) for q in arg[:-1])
            r2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            r, v = sqrt(r2), sqrt(v2)

        # Interpolate potential and calculate escape velocity as a
        # function of the distance from the center
        phi = self.interp_phi(r)
        vesc2 = 2.0*phi                  # Note: phi > 0

        DF = numpy.zeros([max(r.size, v.size)])

        if (len(r) == len(v2)):
            # Bitwise & needed to allow for possibility that r and v2 are arrays
            c = (r<self.rt) & (v2<vesc2)

        if (len(r) == 1) and (len(v2) > 1):
            c = (v2<vesc2)
            if (r>self.rt): c=False

        if (len(r) > 1) and (len(v2) == 1):
            # Bitwise & needed to allow for possibility that r and v2 are arrays
            c = (r<self.rt) & (numpy.zeros(len(r))+v2<vesc2)

        if (sum(c)>0):
            # Scaled energy
            E = (phi-0.5*v2)/self.s2
            
            # Compute the DF: eq. (3) in C19
            DF[c] = exp(E[c]) - self.B - self.C*E[c]
            DF[~c] = (1-self.B)*exp(E[~c]/self.eta**2)
            
            DF *= self.A
        else:
	    DF = numpy.zeros(max(len(r),len(v)))

        return DF

