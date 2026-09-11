# Third-party notices

The repository-level Apache-2.0 license applies to the original Python code,
tests, scripts, and project documentation authored for this project. It does
not replace the licenses of bundled styles, dependencies, models, or datasets.

## Algorithms and reference implementations

The incremental/decremental active-set implementation follows the algorithmic
ideas in:

> Gert Cauwenberghs and Tomaso Poggio. Incremental and Decremental Support
> Vector Machine Learning. NeurIPS, 2000.

The Python reference solver is Vishwajith Ramesh's translation and adaptation
of Gert Cauwenberghs's original incremental-SVM MATLAB release:

- https://isn.ucsd.edu/svm/incremental/
- https://isn.ucsd.edu/svm/incremental/README

The upstream README states, “This software is in the public domain; there are
no implied warranties of any kind.” The original MATLAB source, fixture data,
and workspace files are not redistributed here. The Python translation and
project-specific one-class extensions are distributed under this repository's
Apache-2.0 license. Chris Diehl's separate GPL incremental-SVM extension was
not used for this translation.

## Bundled LaTeX files

- `paper/fancyhdr.sty` is distributed under the LaTeX Project Public License
  (LPPL), as stated in its file header.
- `paper/tmlr.bst` carries the upstream `plainnat`/ICML-derived copyright and
  LPPL notice stated in its file header.
- `paper/tmlr.sty` comes from the official
  [`JmlrOrg/tmlr-style-file`](https://github.com/JmlrOrg/tmlr-style-file)
  project, which is distributed under Apache-2.0. Its upstream attribution is
  retained.

## External dependencies and data

NumPy, SciPy, CVXPY, PyTorch, Matplotlib, pandas, MLX, and their transitive
dependencies retain their respective upstream licenses.

MIMIC-IV, enwik8, and TinyStories are not distributed. Users must obtain them
from their original providers and comply with the applicable terms. No
MIMIC-IV source record, identifier, cache, or per-stay result is included.
