Examples
---------

Construct a Woolley model with :math:`\hat{\phi}_0 = 7` and print
:math:`r_{\rm t}/r_0` and :math:`r_{\rm v}/r_{\rm h}`

>>> from limepy import limepy
>>> k = limepy(7, 0)
>>> print k.rt/k.r0, k.rv/k.rh

Construct a Michie-King model and print :math:`r_{\rm
a}/r_{\rm h}`

>>> a = limepy(7, 1, ra=2)
>>> print a.ra/a.rh

Create a Wilson model with :math:`\hat{phi}_0 = 12` in Henon/N-body
units: :math:`G=M=r_{\rm v}=1` and print the normalisation
constant :math:`A` of the DF and the DF in the centre:

>>> w = limepy(12, 2, scale=True, GS=1, MS=1, RS=1, scale_radius='rv')
>>> print w.A, w.df(0,0)

Multi-mass King model in physical units with :math:`r_{\rm h}
= 1\,{\rm pc}` and :math:`M = 10^4\,{\rm M_{\odot}}`

>>> m = limepy(7, 1, mj=[0.3,1,5], Mj=[9,3,1], scale=True, MS=1e4, RS=1)

Create a discrete sample of points sampled from this multi-mass model 

>>> from limepy import sample
>>> ics = sample(m, seed=1234, verbose=True)
