========================
Description of the model
========================



(MM, A)limepy: (Multi-Mass, Anisotropic) Lowered Isothermal Model Explorer in Python 
-------------------------------------------------------------------------------------

Isotropic models
^^^^^^^^^^^^^^^^


The isotropic distribution functions are defined as `(Gomez-Leyton \&
Velazquez 2014) <http://adsabs.harvard.edu/abs/2014JSMTE..04..006G>`_

.. math::
   f_g(E) = \displaystyle \begin{cases}
   A\exp(E), &g=0 \\
   \displaystyle A\exp(E)P(g, E), &g>1
   \end{cases}

where :math:`\displaystyle E = \frac{\phi - \phi(r_{\rm
t}) - v^2/2}{\sigma^2}`, :math:`\sigma` is a velocity scale, :math:`0 <
\phi-\phi(r_{\rm t}) <W_0/\sigma^2` is the (positive) potential and :math:`P(s,x)` is the
regularised lower incomplete gamma function :math:`P(s,x) =
\gamma(s,x)/\Gamma(x)`. For some integer values of *g* several well
known models are found

*  g = 0 : `Woolley (1954) <http://adsabs.harvard.edu/abs/1954MNRAS.114..191W>`_
*  g = 1 : `King (1966) <http://adsabs.harvard.edu/abs/1966AJ.....71...64K>`_
*  g = 2 : `Wilson (1975) <http://adsabs.harvard.edu/abs/1975AJ.....80..175W>`_

Anisotropic models
^^^^^^^^^^^^^^^^^^

Radial anisotropy a la `Michie (1963)
<http://adsabs.harvard.edu/abs/1963MNRAS.125..127M>`_ can be
included as follows

.. math::
   f_g(E, J^2) = \exp(-J^2)f_g(E),

where :math:`J^2 = (rv_t)^2/(2r_{\rm a}^2\sigma^2)`, here :math:`r_{\rm a}` is the user-defined anisotropy radius.

Multi-mass model
^^^^^^^^^^^^^^^^

Multi-mass models are found by summing the DFs of individual mass
components and adopting for each component (following `Gunn &
Griffin (1979) <http://adsabs.harvard.edu/abs/1979AJ.....84..752G>`_)

.. math::
   \sigma_j       &\propto  \mu_j^{-\delta}\\
   r_{{\rm a},j}  &\propto  \mu_j^{\eta}

where :math:`\mu_j = m_j/\bar{m}` and :math:`\bar{m}` is the central density weighted mean mass.

