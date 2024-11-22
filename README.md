# GFN: A graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications
A resolution-invariant generalisation of feedforward networks for graphical data, applied to model order reduction (MOR).

<p align="center">
<img src="readme_images/gfn_rom.png"/>
</p>

## Why GFNs?
Many applications rely upon graphical data, which standard machine learning methods such as feedforward networks and convolutions cannot handle. GFNs present a novel approach of tackling this problem by extending existing machine learning approaches for use on graphical data. GFNs have very close links with neural operators and graph neural networks.

<p align="center">
<img src="readme_images/gfn.png"/>
</p>

Key advantages of GFNs:
- Resolution invariance
- Equivalence to feedforward networks for single fidelity data (no deterioration in performance)
- Provable guarantees on performance for super- and sub-resolution
- Both fixed and adapative multifidelity training possible

## GFN-ROM
We show the capability of GFNs for MOR by developing the graph feedforward network reduced order model (GFN-ROM).

Key advantages of GFN-ROM:
- First graph-based resolution-invariant ROM
- Lightweight and flexible architecture
- Computational efficiency
- Excellent generalisation performance
- Adaptive multifidelity training

## Codebase
The code implementing the GFN-ROM model is given in
- `gfn_rom/`

All results presented in the paper are fully reproducible and we provide pre-run jupyter notebooks containing all the necessary code in
- `graetz/`
- `advection/`
- `stokes/`

**Requirements**:
Running GFN-ROM requires
- `torch, numpy, sklearn, matplotlib, tqdm, pykdtree`

Additional modules are required if one wishes to rerun the data generation or GCA-ROM experiments.

## Cite this work!
If this work is useful to you, please cite

[1] Morrison, O. M., Pichi, F. and Hesthaven, J. S. (2024) ‘GFN: A graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications’. Available at: [arXiv](https://arxiv.org/abs/2406.03569) and [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2024.117458)
```
@article{Morrison2024,
  title={{GFN}: {A} graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications},
  author={Morrison, Oisín M. and Pichi, Federico and Hesthaven, Jan S.},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {432},
  pages = {117458},
  year = {2024},
  doi = {10.1016/j.cma.2024.117458},
}
```
