# Third-party models and data

The environment and orchestration code in this repository use the repository's
[MIT license](LICENSE). Dependencies, source datasets, and pretrained weights
retain their own licenses.

The detector experiments fine-tune **RF-DETR Nano**, developed by Roboflow.
Roboflow identifies the open-source RF-DETR package and Nano model weights as
Apache 2.0. A copy of that license is included at
[licenses/RF-DETR-APACHE-2.0.txt](licenses/RF-DETR-APACHE-2.0.txt).
Our detector artifacts modify the pretrained weights through training; each
published model card records its source checkpoint, training data, and changes.
The repository's MIT license does not replace the pretrained model license.
See the [official RF-DETR repository](https://github.com/roboflow/rf-detr#license)
for the component-specific licenses; this project uses the Nano model.

The downloaded RF100-VL pill and stomata exports identify their dataset license
as MIT. Their preparation scripts preserve the source URLs, export versions,
annotation identities, and file checksums. Dataset images are not included in
this repository. The derived stomata split changes group allocation and is
explicitly separate from the official native benchmark.

Scientific methods and benchmark sources are attributed in the
[detection research notes](docs/detection-research.md). This is an independent
project and is not an official Roboflow Labs product.
