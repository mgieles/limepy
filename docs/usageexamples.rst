Examples
---------

Construct a Woolley model with :math:`\hat{\phi}_0 = 7` and print
:math:`r_{\rm t}/r_0` and :math:`r_{\rm v}/r_{\rm h}`

>>> from limepy import limepy
>>> k = limepy(7, 0)
>>> print k.rt/k.r0, k.rv/k.rh

Construct a Michie-King model and print :math:`r_{\rm
a}/r_{\rm h}` and the Polyachenko & Shukhman (1981) anisotropy parameter 

>>> a = limepy(7, 1, ra=5)
>>> print a.ra/a.rh, 2*a.Kr/a.Kt

Create a Wilson model with :math:`\hat{\phi}_0 = 12` in Henon/N-body
units: :math:`G=M=r_{\rm v}=1` and print the normalisation
constant :math:`A` of the DF and the DF in the centre:

>>> w = limepy(12, 2, G=1, M=1, rv=1)
>>> print w.A, w.df(0,0)

Multi-mass in physical units with :math:`r_{\rm h} = 3` pc and :math:`M = 10^5\,M_\odot` and print central densities of each bin over the total central density and the half-mass radius + half-mass radius in projection

>>> m = limepy(7, 1, mj=[0.3,1,5], Mj=[9,3,1], M=1e5, rh=3 project=True)

Create a discrete sample of points sampled from this multi-mass model 

>>> from limepy import sample
>>> ics = sample(m, seed=1234, verbose=True)
