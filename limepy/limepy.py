# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
import numpy
from numpy import exp, sqrt, pi, sin
from scipy.interpolate import PiecewisePolynomial, interp1d
from scipy.special import gamma, gammainc, dawsn, hyp1f1
from scipy.integrate import ode, quad, simps
from math import factorial

#     Authors: Mark Gieles, Alice Zocchi (Surrey 2015)

class limepy:
    def __init__(self, phi0, g, **kwargs):
        r"""

        (MM, A)limepy

        (Multi-Mass, Anisotropic) Lowered Isothermal Model Explorer in Python

        This code solves the models presented in Gieles & Zocchi 2015 [GZ15], 
        and calculates radial profiles for some useful quantities. The models 
        are defined by the distribution function (DF) of equation (1) in GZ15

        Model parameters:
        =================

        phi0 : scalar
           Central dimensionless potential
        g : scalar
          Order of truncation [0=Woolley, 1=King, 2=Wilson]; default=1        
        ra : scalar, required for anisotropic models
          Anisotropy radius; default=1e8
        mj : list, required for multi-mass system
          Mean mass of each component; default=None
        Mj : list, required for multi-mass system
           Total mass of each component; default=None
        delta : scalar, optional
              Index in s_j = v_0*mu_j**-delta; default=0.5
              See equation (20) in GZ
        eta : scalar, optional
              Index in ra_j = ra*mu_j**eta; default=0
              See equation (21) in GZ

        Input for scaling:
        ==================

        scale : bool, optional
           Scale model to desired G=GS, M=MS, R=RS; default=False
        MS : scalar, optional
           Final scaled mass; default=10^5 [Msun]
        RS : scalar, optional
           Final scaled mass; default=3 [pc]
        GS : scalar, optional
           Final scaled mass; default=0.004302 [(km/s)^2 pc/Msun]
        scale_radius : str, optional
                     Radius to scale ['rv' or 'rh']; default='rh'

        Options: 
        ========

        potonly : bool, optional
                Fast solution by solving potential only; default=False
        max_step : scalar, optional
                 Maximum step size for ode output; default=1e4
        verbose : bool, optional
                Print diagnostics; default=False


        Examples:
        =========

        Construct a Woolley model with phi0 = 7 and print r_t/r_0 and r_v/r_h

        >>> k = limepy(7, 0)
        >>> print k.rt/k.r0, k.rv/k.rh
        >>> 19.1293485709 1.17783655227

        Construct a Michie-King model and print r_a/r_h and 2*k.Kr/k.Kt

        >>> a = limepy(7, 1, ra=5)
        >>> print a.ra/a.rh, 2*a.Kr/a.Kt
        >>> 1.03378595801 1.36280947104

        Create a Wilson model with W_0 = 12 in Henon/N-body units: G = M =
        r_v = 1 and print the normalisation constant A of the DF and the
        value of the DF in the centre:

        >>> w = limepy(12, 2, scale=True, GS=1, MS=1, RS=1, scale_radius='rv')
        >>> print w.A, w.df(0,0)
        >>> [ 0.00800902] [ 1303.40245567]

        Multi-mass model in physical units with r_h = 1 pc and M = 10^5 M_sun
        and print central densities of each bin over the total central density

        >>> m = limepy(7, 1, mj=[0.3,1,5], Mj=[9,3,1], scale=True, MS=1e5,RS=1)
        >>> print m.alpha
        >>> array([ 0.3072483 ,  0.14100798,  0.55174372])

        """

        # Set parameters and scales
        self._set_kwargs(phi0, g, **kwargs)
        self.rhoint0 = [self._rhoint(self.phi0, 0, self.ramax)]

        # In case of multi-mass model, iterate to find central densities
        # (see Section 2.2 in GZ15)
        if (self.multi):
            self._init_multi(self.mj, self.Mj)

            while self.diff > self.diffcrit:
                self._poisson(True)

                if (not self.converged):
                    raise ValueError("Error: model did not converge in first iteration, try larger r_a / smaller phi_0")
                else:
                    self._set_alpha()
                    if self.niter > 200:
                        self.converged=False
                        raise ValueError("Error: mass function did not converge, try larger phi_0")

        self.r0 = 1.0
        if (self.multi): self.r0j = sqrt(self.s2j)*self.r0

        # Solve Poisson equation to get the potential
        self._poisson(self.potonly)
        if (self.multi): self.Mj = self._Mjtot

        # Optional scaling
        if (self.scale): self._scale()

        # Optional output 
        if (self.verbose):
            print "\n Model properties: "
            print " ----------------- "
            print " phi0 = %5.2f; g = %4.2f"%(self.phi0, self.g)
            print " Converged = %s"%(self.converged)
            if (self.potonly):
                print " M = %10.3f; U = %10.4f "%(self.M, self.U)
            else:
                out1=(self.M,self.U,self.K,-self.K/self.U,2*self.Kr/self.Kt)
                print " M = %10.3e; U = %9.3e; K = %9.3e; Q = %6.4f; 2Kr/Kt = %5.3f"%out1
            out2=(self.rv/self.rh,self.rh/self.r0,self.rt/self.r0,self.ra/self.rh)
            print " rv/rh = %4.3f; rh/r0 = %6.3f; rt/r0 = %7.3f; ra/rh = %7.3f"%out2

    def _set_kwargs(self, phi0, g, **kwargs):
        """ Set parameters and scales """

        if (g<0): raise ValueError("Error: g must be larger or equal to 0")
        if (g>=3.5): raise ValueError("Error: for g>=3.5 models are infinite")

        self.phi0, self.g = phi0, g

        self.MS, self.RS, self.GS = 1e5, 3, 0.004302
        self.scale_radius = 'rh'
        self.scale = False
        self.maxr = 1e10
        self.max_step = self.maxr
        self.diffcrit = 1e-8
        self.max_arg_exp = 700
        self.mf_iter_index = 0.5
        self.ode_atol = 1e-6
        self.ode_rtol = 1e-6
        self.nmbin, self.delta, self.eta = 1, 0.5, 0.0

        self.G = 9.0/(4.0*pi)
        self.mu, self.alpha = numpy.array([1.0]), numpy.array([1.0])
        self.s2 = 1.0
        self.s2j = numpy.array([1.0])
        self.niter = 0

        self.potonly, self.multi, self.verbose = [False]*3
        self.ra, self.ramax = 1e8, 1e8

        self.nstep=1
        self.converged=True
        self._interpolator_set=False

        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)
            if 'mj' in kwargs and 'Mj' in kwargs:
                self.multi=True
                if len(self.Mj) is not len(self.mj):
                    raise ValueError("Error: Mj and mj must have same length")
            if ('mj' not in kwargs and 'Mj' in kwargs) or \
               ('Mj' not in kwargs and 'mj' in kwargs):
                raise ValueError("Error: Supply both mj and Mj")
        self.raj = numpy.array([self.ra])

        return

    def _logcheck(self, t, y):
        """ Logs steps and checks for final values """

        if (t>0): self.r, self.y = numpy.r_[self.r, t], numpy.c_[self.y, y]
        self.nstep+=1
        return 0 if (y[0]>1e-8) else -1

    def _set_mass_function_variables(self):
        """ Multi-mass models: Set properties for each mass bin """ 
                
        self.mmean = sum(self.mj*self.alpha)    # equation (26) GZ15
        self.mu = self.mj/self.mmean            
        self.s2j = self.mu**(-2*self.delta)   # equation (24) GZ15
        self.raj = self.ra*self.mu**self.eta    # equation (25) GZ15

        self.phi0j = self.phi0/self.s2j
        self.rhoint0 = numpy.zeros(self.nmbin)

        for j in range(self.nmbin):
            self.rhoint0[j] = self._rhoint(self.phi0j[j], 0, self.ramax)


    def _init_multi(self, mj, Mj):
        """ Initialise parameters and arrays for multi-mass system (Section 2.2 in GZ15) """

        self.multi=True
        self.mj = numpy.array(mj)
        self.Mj = numpy.array(Mj)
        self.nmbin = len(mj)

        # Set trial value for alpha_j array, will be updated in iterations
        self.alpha = self.Mj/sum(self.Mj) 
        self._set_mass_function_variables()
        self.diff = 1

    def _set_alpha(self):
        """ Set central rho_j for next iteration """

        # The power of 0.5 is used. This is different from the recommendation of Da Costa & Freeman 1976
        # and Gunn & Griffin 1979, but it leads to better convergens for low phi0 and wide MFs 
        # (i.e. when black-holes are considered), which can fail with an index of 1 (see Section 2.2, GZ15)
        
        self.alpha *= (self.Mj/self._Mjtot)**self.mf_iter_index
        self.alpha/=sum(self.alpha)

        self._set_mass_function_variables()
        self.diff = sum((self._Mjtot/sum(self._Mjtot) -
                         self.Mj/sum(self.Mj))**2)/len(self._Mjtot)
        self.niter+=1
        self.nstep=1
        if (self.verbose):
            fracd,Mjit="", ""
            for j in range(self.nmbin):
                fracd=fracd+"%12.8f "%((self._Mjtot[j]/sum(self._Mjtot) - self.Mj[j]/sum(self.Mj))/(self.Mj[j]/sum(self.Mj)))
                Mjit=Mjit+"%12.8f "%(self._Mjtot[j]/sum(self._Mjtot))
            out=(self.niter, self.diff, self.converged, fracd, Mjit)

            print " Iter %3i; diff = %10.3e; conv = %s; frac diff=%s; Mjtot=%s"%out

    def _poisson(self, potonly):
        """ Solves Poisson equation """
        # y = [phi, u_j, U, K_j], where u = -M(<r)/G

        # Initialize
        self.r = numpy.array([0])
        self.y = numpy.r_[self.phi0, numpy.zeros(self.nmbin+1)]
        if (not potonly): self.y = numpy.r_[self.y, numpy.zeros(2*self.nmbin)]
        self.y = numpy.r_[self.y, 0]

        # Ode solving using Runge-Kutta integator of order 4(5) 'dopri5' (Hairor, Norsett& Wanner 1993)
        max_step = self.maxr if (potonly) else self.max_step
        sol = ode(self._odes)
        sol.set_integrator('dopri5',nsteps=1e6,max_step=max_step,atol=self.ode_atol,rtol=self.ode_rtol)
        sol.set_solout(self._logcheck)
        sol.set_f_params(potonly)
        sol.set_initial_value(self.y,0)
        sol.integrate(self.maxr)

        # Extrapolate to r_t: 
        # phi(r) =~ a(r_t -r)
        # a = GM/r_t^2
        GM = -self.G*sum(sol.y[1:1+self.nmbin])
        p = 2*sol.y[0]*self.r[-1]/GM

        if (p<=0.5):
            rtfac = (1 - sqrt(1-2*p))/p
            self.rt = rtfac*self.r[-1] if (rtfac > 1) else 1.0000001*self.r[-1]
        else:
            self.rt = 1.000001*self.r[-1]

        # Set the converged flag to True if successful
        if (self.rt < self.maxr)&(sol.successful()):
            self.converged=True
        else:
            self.converged=False

        # Calculate the phase space volume occupied by the model
        dvol = (4./3*pi)**2*(self.rt**3 - self.r[-1]**3)*0.5*(2*self.y[0,-1])**1.5
        self.volume = self.y[-1][-1]+dvol

        # Fill arrays needed if potonly=True
        self.r = numpy.r_[self.r, self.rt]
        self.phihat = numpy.r_[self.y[0,:], 0]
        self._Mjtot = -sol.y[1:1+self.nmbin]/self.G

        self.M = sum(self._Mjtot)

        # Save the derivative of the potential for the potential interpolater
        dphidr = numpy.sum(self.y[1:1+self.nmbin,1:],axis=0)/self.r[1:-1]**2
        self.dphidrhat1 = numpy.r_[0, dphidr, -self.G*self.M/self.rt**2]

        self.A = self.alpha/(2*pi*self.s2j)**1.5/self.rhoint0

        if (not self.multi):
            self.mc = -numpy.r_[self.y[1,:],self.y[1,-1]]/self.G

        if (self.multi):
            self.mc = sum(-self.y[1:1+self.nmbin,:]/self.G)
            self.mc = numpy.r_[self.mc, self.mc[-1]]

        # Compute radii to be able to scale in case potonly=True
        self.U = self.y[1+self.nmbin,-1]  - 0.5*self.G*self.M**2/self.rt

        # Get half-mass radius from cubic interpolation. This seems an overkill, but linear interpolation
        # leads to spurious results.
        ih = numpy.searchsorted(self.mc, 0.5*self.mc[-1])-1
        rhotmp=numpy.zeros(2)
        for j in range(self.nmbin):
            rhotmp += self.alpha[j]*self._rhohat(self.phihat[ih:ih+2]/self.s2j[j], self.r[ih:ih+2], j)
        drdm = 1./(4*pi*self.r[ih:ih+2]**2*rhotmp)
        rmc_and_derivs = numpy.vstack([[self.r[ih:ih+2]],[drdm]]).T
        self.rh = PiecewisePolynomial(self.mc[ih:ih+2], rmc_and_derivs,direction=1)(0.5*self.mc[-1])

        self.rv = -0.5*self.G*self.M**2/self.U

        # Additional stuff
        if (not potonly):
            # Calculate kinetic energy (total, radial, tangential)
            self.K = numpy.sum(sol.y[2+self.nmbin:2+2*self.nmbin])
            self.Kr = numpy.sum(sol.y[2+2*self.nmbin:2+3*self.nmbin])
            self.Kt = self.K - self.Kr

            # Calculate density and velocity dispersion components
            if (not self.multi):
                self.rhohat = self._rhohat(self.phihat, self.r, 0)
                self.v2, self.v2r, self.v2t = \
                        self._get_v2(self.phihat, self.r, self.rhohat, 0)

            # For multi-mass models, calculate quantities for each mass bin
            if (self.multi):
                for j in range(self.nmbin):
                    phi = self.phihat/self.s2j[j]
                    rhohatj = self._rhohat(phi, self.r, j)
                    v2j, v2rj, v2tj = self._get_v2(phi, self.r, rhohatj, j)
                    v2j, v2rj, v2tj = (q*self.s2j[j] for q in [v2j,v2rj,v2tj])
                    betaj = self._beta(self.r, v2rj, v2tj)

                    kj = self.y[2+self.nmbin+j,:]
                    krj = self.y[2+2*self.nmbin+j,:]
                    ktj = kj - krj

                    mcj = -numpy.r_[self.y[1+j,:], self.y[1+j,-1]]/self.G
                    rhj = numpy.interp(0.5*mcj[-1], mcj, self.r)

                    if (j==0):
                        self.rhohatj = self.alpha[j]*rhohatj
                        self.rhohat = self.rhohatj
                        self.v2j, self.v2rj, self.v2tj = v2j, v2rj, v2tj
                        self.v2 = self._Mjtot[j]*v2j/self.M
                        self.v2r = self._Mjtot[j]*v2rj/self.M
                        self.v2t = self._Mjtot[j]*v2tj/self.M

                        self.betaj = betaj
                        self.kj, self.krj, self.ktj, self.Kj, self.Krj = kj, krj, ktj, kj[-1], krj[-1]
                        self.ktj = self.kj - self.krj
                        self.Ktj = self.Kj - self.Krj
                        self.rhj, self.mcj = rhj, mcj
                    else:
                        self.rhohatj = numpy.vstack((self.rhohatj, self.alpha[j]*rhohatj))
                        self.rhohat += self.alpha[j]*rhohatj

                        self.v2j = numpy.vstack((self.v2j, v2j))
                        self.v2rj = numpy.vstack((self.v2rj, v2rj))
                        self.v2tj = numpy.vstack((self.v2tj, v2tj))
                        self.v2 += self._Mjtot[j]*v2j/self.M
                        self.v2r += self._Mjtot[j]*v2rj/self.M
                        self.v2t += self._Mjtot[j]*v2tj/self.M

                        self.betaj = numpy.vstack((self.betaj, betaj))
                        self.kj = numpy.vstack((self.kj, kj))
                        self.krj = numpy.vstack((self.krj, krj))
                        self.ktj = numpy.vstack((self.ktj, ktj))
                        self.Kj = numpy.r_[self.Kj, kj[-1]]
                        self.Krj = numpy.r_[self.Krj, krj[-1]]
                        self.Ktj = numpy.r_[self.Ktj, ktj[-1]]
                        self.rhj = numpy.r_[self.rhj,rhj]
                        self.mcj = numpy.vstack((self.mcj, mcj))
            # Calculate anisotropy profile (equation 30 of GZ)
            self.beta = self._beta(self.r, self.v2r, self.v2t)


    def _rhohat(self, phi, r, j):
        """ Wrapper for _rhoint when either: both phi or r are arrays, or both are scalar """
        if not hasattr(phi,"__len__"): phi = numpy.array([phi])
        if not hasattr(r,"__len__"): r = numpy.array([r])

        n = max([phi.size, r.size])
        rhohat = numpy.zeros(n)

        for i in range(n):
            if (phi[i]<self.max_arg_exp) or (numpy.isnan(phi[i])):
                rhohat[i] = self._rhoint(phi[i], r[i], self.raj[j])/self.rhoint0[j]
            else:
                # For large phi compute the ratio in one go (see Section 4.1, GZ15)
                rhohat[i] = exp(phi[i]-self.phi0j[j]) if (self.multi) else 0

        return rhohat

    def _rhoint(self, phi, r, ra):
        """ Dimensionless density integral as a function of phi and r (scalars only) """

        # Isotropic case first (equation 8, GZ15)
        rhoint = exp(phi)*gammainc(self.g + 1.5, phi)

        # For anisotropic models, add r-dependent parts explicitly (equation 11, GZ15)
        if (self.ra < self.ramax) and (phi>0) and (r>0):
            p, g = r/ra, self.g
            p2 = p**2
            g3, g5, fp2 = g+1.5, g+2.5, phi*p2

            func = hyp1f1(1, g5, -fp2)  if fp2 < self.max_arg_exp else g3/fp2
            rhoint += p2*phi**(g+1.5)*func/gamma(g5)
            rhoint /= (1+p2)
        return rhoint

    def _get_v2(self, phi, r, rho, j):
        v2, v2r, v2t = numpy.zeros(r.size), numpy.zeros(r.size), numpy.zeros(r.size)
        for i in range(r.size-1):
            v2[i], v2r[i], v2t[i] = self._rhov2int(phi[i], r[i], self.raj[j])/rho[i]/self.rhoint0[j]
        return v2, v2r, v2t

    def _rhov2int(self, phi, r, ra):
        """Compute dimensionless pressure integral for phi, r """

        # Isotropic case first (equation 8, GZ15)
        rhov2r = exp(phi)*gammainc(self.g + 2.5, phi)
        rhov2  = 3*rhov2r
        rhov2t = 2*rhov2r

        # Add anisotropy, add parts depending explicitly on r
        # (see equations 12, 13, and 14 of GZ)
        if (ra < self.ramax) and (r>0) and (phi>0):
            p, g = r/ra, self.g
            p2 = p**2
            p12 = 1+p2
            g3, g5, g7, fp2 = g+1.5, g+2.5, g+3.5, phi*p2

            P1 = p2*phi**g5/gamma(g7)
            H1 = hyp1f1(1, g7, -fp2) if fp2 < self.max_arg_exp else g5/fp2 
            H2 = hyp1f1(2, g7, -fp2) if fp2 < self.max_arg_exp else g5*g3/fp2**2

            rhov2r += P1*H1
            rhov2r /= p12

            rhov2t /= p12
            rhov2t += 2*P1*(H1/p12 + H2)
            rhov2t /= p12

            rhov2 = rhov2r + rhov2t
        return rhov2, rhov2r, rhov2t

    def _beta(self, r, v2r, v2t):
        """ Calculate the anisotropy profile """

        beta = numpy.zeros(r.size)
        if (self.ra < self.ramax):
            c = (v2r>0.)
            # Equation  (32), GZ15
            beta[c] = 1.0 - 0.5*v2t[c]/v2r[c]
        return beta

    def _odes(self, x, y, potonly):
        """ Solve ODEs """
        # y = [phi, u_j, U, K_j], where u = -M(<r)/G
        if (self.multi):
            derivs = [numpy.sum(y[1:1+self.nmbin])/x**2] if (x>0) else [0]
            for j in range(self.nmbin):
                phi = y[0]/self.s2j[j]
                derivs.append(-9.0*x**2*self.alpha[j]*self._rhohat(phi, x, j))

            dUdx  = 2.0*pi*numpy.sum(derivs[1:1+self.nmbin])*y[0]/9.
        else:
            derivs = [y[1]/x**2] if (x>0) else [0]
            derivs.append(-9.0*x**2*self._rhohat(y[0], x, 0))

            dUdx  = 2.0*pi*derivs[1]*y[0]/9.
        
        derivs.append(dUdx)

        # Calculate pressure tensor components for all the mass bins

        if (not potonly): #dK_j/dx
            rhov2j, rhov2rj = [], []
            for j in range(self.nmbin):
                rv2, rv2r, rv2t = self._rhov2int(y[0]/self.s2j[j], x, self.raj[j])
                rhov2j.append(self.alpha[j]*self.s2j[j]*2*pi*x**2*rv2/self.rhoint0[j])
                rhov2rj.append(self.alpha[j]*self.s2j[j]*2*pi*x**2*rv2r/self.rhoint0[j])

            for j in range(self.nmbin):
                derivs.append(rhov2j[j])

            for j in range(self.nmbin):
                derivs.append(rhov2rj[j])

        dVdvdr = (4*pi)**2*x**2 * (2*y[0])**1.5/3 if (x>0) and (y[0]>0) else 0
        derivs.append(dVdvdr)

        return derivs

    def _setup_phi_interpolator(self):
        """ Setup interpolater for phi, works on scalar and arrays """

        # Generate piecewise 3th order polynomials to connect the discrete values of phi
        # obtained from from Poisson, using phi'
        self._interpolator_set = True

        if (self.scale):
            phi_and_derivs = numpy.vstack([[self.phi],[self.dphidr1]]).T
        else:
            phi_and_derivs = numpy.vstack([[self.phihat],[self.dphidrhat1]]).T
        self._phi_poly = PiecewisePolynomial(self.r,phi_and_derivs,direction=1)

    def _scale(self):
        """ Scales the model to the units set in the input: GS, MS, RS """

        Mstar = self.MS/self.M
        Gstar = self.GS/self.G
        if (self.scale_radius=='rh'): Rstar = self.RS/self.rh
        if (self.scale_radius=='rv'): Rstar = self.RS/self.rv
        v2star =  Gstar*Mstar/Rstar

        # Update the scales that define the system (see Section 2.1.2 of GZ)

        self.G *= Gstar
        self.rs = Rstar
        self.s2 *= v2star
        self.s2j *= v2star

        # Anisotropy radii
        self.ra, self.raj, self.ramax = (q*Rstar for q in [self.ra,self.raj,self.ramax])

        # Scale all variable needed when run with potonly=True
        self.rhat = self.r*1.0
        self.r, self.r0, self.rt = (q*Rstar for q in [self.r,self.r0,self.rt])
        self.rh, self.rv = (q*Rstar for q in [self.rh,self.rv])

        self.M *= Mstar
        self.phi = self.phihat * v2star
        self.dphidr1 = self.dphidrhat1 * v2star/Rstar
        self.mc *= Mstar
        self.U *= Mstar*v2star
        self.A *= Mstar/(v2star**1.5*Rstar**3)
        self.volume *= v2star**1.5*Rstar**3

        if (self.multi):
            self.Mj *= Mstar
            self.r0j *= Rstar

        # Scale density, velocity dispersion components, kinetic energy
        if (not self.potonly):
            self.rho = self.rhohat*Mstar/Rstar**3
            self.v2, self.v2r, self.v2t = (q*v2star for q in [self.v2,
                                                          self.v2r,self.v2t])
            self.K,self.Kr,self.Kt=(q*Mstar*v2star for q in [self.K, self.Kr, self.Kt])

            if (self.multi):
                self.rhj = self.rhohatj * Mstar/Rstar**3
                self.mcj *= Mstar
                self.rhj *= Rstar
                self.v2j,self.v2rj,self.v2tj=(q*v2star for q in
                                              [self.v2j, self.v2rj,self.v2tj])
                self.kj,self.Krj,self.Ktj,self.Kj=(q*Mstar*v2star for q in [self.kj, self.Krj, self.Ktj, self.Kj])

    def _tonp(self, q):
        q = numpy.array([q]) if not hasattr(q,"__len__") else numpy.array(q)
        return q

    def project(self):
        """ Compute projected mass density (Sigma) and projected <v2> profiles """

        # Initialise the projected quantities:
        # R is the projected (2d) distance from the center, Sigma is the 
        # projected density, v2p is the line-of-sight velocity dispersion, 
        # v2Rp and v2Tp are the radial and tangential velocity dispersion 
        # components projected on the plane of the sky

        R = self.r
        Sigma = numpy.zeros(self.nstep)
        v2p = numpy.zeros(self.nstep)
        v2Rp = numpy.zeros(self.nstep)
        v2Tp = numpy.zeros(self.nstep)

        if (self.multi):
            Sigmaj = numpy.zeros((self.nmbin, self.nstep))
            v2jp = numpy.zeros((self.nmbin, self.nstep))
            v2jRp = numpy.zeros((self.nmbin, self.nstep))
            v2jTp = numpy.zeros((self.nmbin, self.nstep))

        for i in range(self.nstep-1):
            c = (self.r >= R[i])
            r = self.r[c]
            z = sqrt(abs(r**2 - R[i]**2)) # avoid small neg. values due to round off

            Sigma[i] = 2.0*simps(self.rho[c], x=z)
            betaterm1 = 1 if i==0 else 1 - self.beta[c]*R[i]**2/self.r[c]**2
            betaterm2 = 1 if i==0 else 1 - self.beta[c]*(1-R[i]**2/self.r[c]**2)
            v2p[i] = abs(2.0*simps(betaterm1*self.rho[c]*self.v2r[c], x=z)/Sigma[i])
            v2Rp[i] = abs(2.0*simps(betaterm2*self.rho[c]*self.v2r[c], x=z)/Sigma[i])
            v2Tp[i] = abs(2.0*simps(self.rho[c]*self.v2t[c]/2., x=z)/Sigma[i])

            if (self.multi):
                for j in range(self.nmbin):
                    Sigmaj[j,i] = 2.0*simps(self.rhoj[j,c], x=z)
                    betaterm1 = 1 if i==0 else 1 - self.betaj[j,c]*R[i]**2/self.r[c]**2
                    betaterm2 = 1 if i==0 else 1 - self.betaj[j,c]*(1-R[i]**2)/self.r[c]**2

                    v2jp[j,i] = abs(2.0*simps(betaterm1*self.rhoj[j,c]*self.v2rj[j,c], x=z))
                    v2jp[j,i] /= Sigmaj[j,i]

                    v2jRp[j,i] = abs(2.0*simps(betaterm2*self.rhoj[j,c]*self.v2rj[j,c], x=z))
                    v2jRp[j,i] /= Sigmaj[j,i]

                    v2jTp[j,i] = abs(simps(self.rhoj[j,c]*self.v2tj[j,c], x=z))
                    v2jTp[j,i] /= Sigmaj[j,i]


        self.R, self.Sigma, self.v2p, self.v2Rp, self.v2Tp = R, Sigma, v2p, v2Rp, v2Tp
        if (self.multi):
            self.Sigmaj, self.v2jp, self.v2jRp, self.v2jTp = Sigmaj, v2jp, v2jRp, v2jTp 
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
        Returns the value of the normalised DF at a given position in the phase 
        space; this function can only be called after solving the Poisson equation

        Arguments can be:
          - r, v                   (isotropic single-mass models)
          - r, v, j                (isotropic multi-mass models)
          - r, v, theta, j         (anisotropic models)
          - x, y, z, vx, vy, vz, j (all models)
        Here j specifies the mass bin, j=0 for single mass
        Works with scalar and array input

        """

        if (len(arg)<2) or (len(arg)==5) or (len(arg)==6) or (len(arg)>7):
            raise ValueError("Error: df needs 2, 3, 4 or 7 arguments")

        if (len(arg)<=3) and (self.ra<self.ramax):
            raise ValueError("Error: model is anisotropy, more input needed")

        if len(arg) == 2:
            r, v = (self._tonp(q) for q in arg)
            j = 0

        if len(arg) == 3:
            r, v = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]

        if len(arg) == 4:
            r, v, theta = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]

        if len(arg) < 7: r2, v2 = r**2, v**2

        if len(arg) == 7:
            x, y, z, vx, vy, vz = (self._tonp(q) for q in arg[:-1])
            j = arg[-1]
            r2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            r, v = sqrt(r2), sqrt(v2)

        # Interpolate potential and calculate escape velocity as a 
        # function of the distance from the center
        phi = self.interp_phi(r)
        vesc2 = 2.0*phi                        # Note: phi > 0

        DF = numpy.zeros([max(r.size, v.size)])

        if (len(r) == len(v2)):
            c = (r<self.rt) and (v2<vesc2)

        if (len(r) == 1) and (len(v2) > 1):
            c = (v2<vesc2)
            if (r>self.rt): c=False

        if (len(r) > 1) and (len(v2) == 1):
            c = (r<self.rt) and (numpy.zeros(len(r))+v2<vesc2)
            

        if (sum(c)>0):
            # Compute the DF: equation (1), GZ15

            E = (phi-0.5*v2)/self.s2j[j]          # Dimensionless positive energy
            DF[c] = exp(E[c])

            if (self.g>0): DF[c] *= gammainc(self.g, E[c])
            
            if (len(arg)==7): J2 = v2*r2 - (x*vx + y*vy + z*vz)**2
            if (len(arg)==4): J2 = sin(theta)**2*v2*r2
            if (len(arg)<=3): J2 = numpy.zeros(len(c)) 
            
            DF[c] *= exp(-J2[c]/(2*self.raj[j]**2*self.s2j[j]))
            
            DF[c] *= self.A[j]

        else:
	    DF = numpy.zeros(max(len(r),len(v)))

        return DF



