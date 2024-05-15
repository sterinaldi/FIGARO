---
title: 'FIGARO: hierarchical non-parametric inference for population studies'
tags:
  - Python
  - astronomy
  - astrophysics
  - nonparametric methods
  - black holes
  - gravitational waves
authors:
  - given: Stefano
    surname: Rinaldi
    orcid: 0000-0001-5799-4155
    affiliation: "1, 2" 
    corresponding: true
  - given: Walter 
    surname: Del Pozzo
    orcid: 0000-0003-3978-2030
    affiliation: "3, 4" 

affiliations:
 - name: Institut für Theoretische Astrophysik, ZAH, Universität Heidelberg, Albert-Ueberle-Str. 2, 69120 Heidelberg, Germany
   index: 1
 - name: Dipartimento di Fisica e Astronomia "G. Galilei", Università di Padova, Via F. Marzolo 8, 35121 Padova, Italy
   index: 2
 - name: Dipartimento di Fisica "E. Fermi", Università di Pisa, Largo Bruno Pontecorvo 3, 56127 Pisa, Italy
   index: 3
 - name: INFN, Sezione di Pisa, Largo Bruno Pontecorvo 3, 56127 Pisa, Italy
   index: 4
   
date: 15 May 2024
bibliography: paper.bib

---

# Summary
The astrophysical graveyard is populated by black holes (BHs) and neutron stars (NSs). These are the remains of the most massive stars, and studying them will teach us about the physics that governs the stars’ lives and deaths. Gravitational wave (GW) observations are now revealing the population of BHs via the detection of binary black hole mergers, and the numbers are set to grow rapidly in the coming years. The astrophysical distribution of BHs is inferred making use of the probability distribution of the parameters of detected BHs, combined together in a hierarchical population study. Currently, the characterisation and interpretation of the available observations is mainly guided by astrophysical models which depend on the not-well-understood physics of massive binary evolution [see @astrodistGWTC3:2023 and references therein]: this approach is, however, intrinsically prone to potential biases induced by inaccurate modelling of the underlying population.

Bayesian non-parametric methods are models with a countably infinite number of parameters to model arbitrary probability densities. These parameters do not have any connection with the modelled distribution, making these a convenient and agnostic way of describing some unknown population. These are key tools to reconstruct probability densities without being committal to a specific functional form: the basic idea is to let the data speak for themselves, retrieving the distribution that is the most likely to have generated the observed data.
In a certain sense, this is the most phenomenological approach possible: unlike the standard parametric approach, where we specify a functional form inspired by what we expect to find in the data, with non-parametric methods all the information comes from the data, thus avoiding the risk of biasing the inference with inaccurate models. Features in the inferred distribution will arise naturally without the need of including them in the model, leaving astrophysicists tasked with explaining them in terms of formation channels and astrophysical processes. 

The GW community is currently exploring this direction [@tiwari:2021:vamana; @edelman:2023; @toubiana:2023; @callister:2024]: `FIGARO` fits in this framework, being a non-parametric inference scheme designed to reconstruct arbitrary probability densities in a hierarchical fashion under the requirement of minimal mathematical assumptions.

# Statement of need

`FIGARO` (**F**ast **I**nference for **G**W **A**stronomy, **R**esearch and **O**bservations) is a `python` package that implements a variation of the Gibbs sampling scheme with *a Hierarchy of Dirichlet Process Gaussian Mixture Models* [@rinaldi:2022:hdpgmm], or (H)DPGMM for short, as non-parametric model^[(H)DPGMM is based on the Dirichlet Process Gaussian Mixture Model, introduced in @escobar:1995. Mathematical details about the Dirichlet Process can be found in @teh:2010.] to reconstruct arbitrary probability densities given a set of observations. These observations can either be samples from an unknown distribution^[In this case, the model used will be a DPGMM.] or a set of posterior probability densities represented by samples from these posteriors. Differently from other publicly-available DPGMM implementations such as the one included in `scikit-learn` [@pedregosa:2011], `FIGARO` performs a stochastic sampling over the (potentially infinite) parameter space of the DPGMM, thus allowing for an efficient marginalisation over such parameters.

Despite being originally developed in the context of GW physics and in particular to work with the data released by the LIGO-Virgo-KAGRA (LVK) collaboration, `FIGARO` can take as input generic data and therefore it can be applied to a variety of problems beyond GWs (see *Publications*). `FIGARO` output objects are modelled after `scipy.stats` [@virtanen:2020] classes to make them intuitive for users that are already familiar with the `scipy` package. 
The flexibility of (H)DPGMM in reconstructing arbitrary probability densities united with the speed provided by the Gibbs sampling variation we implemented in this package makes `FIGARO` an ideal tool for population studies.

# Availability and usage
`FIGARO` is available via [PyPI](https://pypi.org/project/figaro/) and is compatible with `python<3.12`. The code is hosted on [GitHub](https://github.com/sterinaldi/figaro) and the documentation can be found at [readthedocs.io](https://figaro.readthedocs.io). `FIGARO` comes with two CLIs to perform both the reconstruction of a probability density given a set of samples (``figaro-density``) and the hierarchical inference (``figaro-hierarchical``). The docmumentation also includes a guide on how to use `FIGARO` in a custom `python` script.

# Publications
This is a list of the publications that made use of `FIGARO` so far:

- @rinaldi:2022:figaro – online sky localisation of the potential electromagnetic (EM) counterpart of a GW signal to maximise the likelihood of a joint GW-EM detection;
- @rinaldi:2023:bigG – non-parametric analysis of systematic errors in the determination of Newton's constant of gravitation $G$;
- @rinaldi:2024:m1qz – non-parametric inference of the joint BH primary mass, mass ratio and redshift distribution. First evidence for the evolution of the BH mass function with redshift;
- @cheung:2023 – use of non-parametric methods to minimise the impact of population bias in the search for lensed GW events;
- @sgalletta:2023 – detailed study of the NS population in the Milky Way. `FIGARO` is used to approximate the predicted observed population of NSs;
- @morton:2023 – investigation of the potential origin of GW190521 in an AGN. `FIGARO` is used to approximate the multivariate posterior probability density for GW190521 parameters;
- @rallapalli:2023 – applies the framework described in @rinaldi:2023:bigG to the inference of the W boson mass.

# Acknowledgements

We acknowledge contributions from Daniele Sanfratello and Vera Delfavero. 
SR acknowledges financial support from the European Research Council for the ERC Consolidator grant DEMOBLACK, under contract no. 770017.

# References
