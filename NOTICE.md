# License and Third-Party Notices

The root MIT license applies to the original TopoRecover implementation, tests,
and documentation. It does not relicense upstream datasets, generated artifacts
derived from those datasets, models, CAD kernels, or other external packages.

## Data

Text2CAD, by Sadil Khan and collaborators, is the source of the public design
descriptions and CAD examples used in the released derived evaluation artifacts.
The official dataset declares **Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International**. The derived files under `data/` and `results/`
are distributed under those terms, not MIT. See `data/LICENSE` and `data/README.md`.

- Official dataset and attribution: https://huggingface.co/datasets/SadilKhan/Text2CAD
- Official Text2CAD project: https://github.com/SadilKhan/Text2CAD
- DeepCAD source dataset/project: https://github.com/rundiwu/DeepCAD
- Dataset license: https://creativecommons.org/licenses/by-nc-sa/4.0/

The native reference corpus is not bundled. Obtain it through the upstream
dataset and comply with upstream attribution and use restrictions. The dataset's
noncommercial condition is not removed by the software's MIT license.

## External Software and Models

CadQuery/Open CASCADE, NumPy, SciPy, Shapely, trimesh, PyTorch, Transformers,
lm-format-enforcer, and Qwen are external dependencies, not copied source trees.
Their installations and weights remain subject to their own licenses. The
Qwen model identities used for experiments are listed in `configs/`.

The CADReview/ReCAD, CADCodeVerify and CADDesigner comparisons are method-inspired
adaptations to the supported command language, not releases of those authors'
original implementations. Adaptation prompts and runner code are included.
Upstream citations and acknowledgments must be preserved in subsequent use.

Third-party attribution above is intentionally retained in this review release;
it is not author-identifying metadata about the submitting work.
