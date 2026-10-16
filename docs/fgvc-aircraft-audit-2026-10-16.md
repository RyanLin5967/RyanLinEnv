# FGVC-Aircraft author metadata audit

This report records the metadata-only preparation stage on October 16, 2026. At that stage no aircraft images were downloaded, decoded, viewed or supplied to a model. Public test annotations were examined only for automated membership/count checks. The subsequently authorized image acquisition and bounded training probe are recorded separately in [the image preparation update](fgvc-aircraft-image-preparation-2026-10-16.md); historical metadata-only findings below retain their original scope.

The current [author annotation archive](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz) contains **100 variants and 10,000 image IDs**, despite the [project page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and bundled README describing 102 variants and 10,200 images. Actual archive contents define this prospective experiment. Its directory and release name are `fgvc-aircraft-2013b`; the HTTPS response reports January 17, 2014 as its modification date.

| Native split | Images | Variants | Images per variant |
| --- | ---: | ---: | --- |
| Train | 3,334 | 100 | 34 classes have 34; 66 have 33 |
| Validation | 3,333 | 100 | 33 classes have 34; 67 have 33 |
| Test | 3,333 | 100 | 33 classes have 34; 67 have 33 |

There are 70 families and **30 manufacturers**, rather than the 41 manufacturers described on the page. Every image ID is unique across the three native splits. Variant/family/manufacturer annotation rows align exactly with each split's image list. Native `trainval` contains the union of train and validation, 6,667 images; it is not an additional independent split. Every variant maps consistently to one family and manufacturer.

The [machine-readable audit](fgvc-aircraft-audit-2026-10-16.json) contains all canonical labels, per-class split counts, source-file hashes, split fingerprints and the prospective few-shot IDs. The exact sorted variant vocabulary is `canonical_labels.variant`; retain its spelling, punctuation and order. IDs are seven-character strings, including leading zeroes.

That JSON is retained byte-for-byte as the historical metadata-stage record (SHA-256 `3262fc7b24ad6ca03d17e30d8a0f35bcbc25dae2c22a30c0fd9855a70da465e7`). Its prospective `trainer_contract.row_fields.image` wording described already-cropped images. The subsequently adopted contract uses **original JPEG paths with one shared lazy bottom-20-row removal**, as documented in the image preparation update. The generator now emits the corrected original-JPEG contract for future metadata audits; this clarification does not rewrite the historical record or any frozen selection.

## Source pin and checksum discrepancy

The annotation archive is **407,927 bytes**. Repeated HTTPS retrievals produced:

- Observed SHA-256: `50e4419bd330b68a2e85dbcb32339d31120190efe6944682e87c9b431f5f202a`.
- Observed MD5: `6939ddc41c731b6d5e82b20b2255f999`.
- [Author-declared annotation MD5](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.html): `e3799900175f63abbc48a5bc5b16427e` — **does not match**.

The cause of the mismatch is unverified. The downloader pins the actual received SHA-256 and records the author-declared MD5 failure; it does not claim that check passed. Transport used standard HTTPS certificate verification. The [full image archive checksum page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html) declares MD5 `32eca553f897e747706144f3bb444df0`; the full archive was not downloaded or verified. Neither filename equality nor annotation hashes establish image-byte equivalence to another mirror.

[`prepare_aircraft.py`](../scripts/prepare_aircraft.py) downloads only the bounded annotation archive, validates its pinned size/SHA-256, rejects image/link/path-traversal members, and processes the metadata without extracting executable files or running the bundled MATLAB code.

```sh
python scripts/prepare_aircraft.py --download-metadata \
  --archive runs/aircraft-audit/annotations.tar.gz \
  --output runs/aircraft-audit/audit.json \
  --metadata-rows-output runs/aircraft-audit/native-metadata-rows.json
pytest tests/test_aircraft_data.py
```

## Evaluation contract and remaining parity work

The author archive's `evaluation.m` has SHA-256 `4e25f8db8a0d0163b10b167299c04c7a437a5f0de637f1e79628066ec297676a`. Its primary result is the mean diagonal of the row-normalized confusion matrix: **equal-class accuracy**. Microaccuracy is supplemental because class counts differ slightly. The source was read completely. No MATLAB/Octave runtime is installed, so native execution parity is **not yet verified**. The author page also says this evaluator can differ from the 2013 challenge evaluator; no challenge-leaderboard equivalence is claimed.

The source selects the highest-scoring class per image after creating a dense score matrix. Missing predictions remain unclassified and contribute an error to their true class denominator. Finite score ties select the first class in sorted order; all scores at negative infinity select the unclassified sentinel. The sorted `unique(labels0)` order matches the supplied `variants.txt` exactly. Negative finite scores remain legitimate predictions.

Malformed-input paths need validation before the native function: its unknown-image branch checks `y == 0` rather than `x == 0`, and its unknown-label path filters image/class indices without correspondingly filtering scores. Duplicate image/class entries use indexed overwrite semantics. A wrapper should reject unknown IDs/classes, duplicate pairs and NaNs before invoking the evaluator; otherwise malformed responses can crash or produce ambiguous results. A strict one-label-per-image VLM output has no score ties or duplicate pairs. Request/parse failures must remain reported as errors, not silently disappear from the evaluated set.

Prospective parity fixtures are recorded in the JSON: perfect/wrong predictions; empty output; one missing output; tied finite scores; all-negative-infinity scores; a finite negative correct score; and an imbalanced two-class case. That last fixture has three correct images in class A and one incorrect image in B, yielding mean-class accuracy **0.50** versus microaccuracy **0.75**. These are expected source-derived values, not executed native results or model measurements. Before any Aircraft model claim, execute these fixtures against the unchanged official code and the Python scorer on identical valid inputs.

## Image preprocessing, permitted use and manifests

The author requires removing the **bottom 20 pixel rows** from every training and evaluation image. The future pipeline should decode RGB, validate that height exceeds 20, remove that strip, and feed the same remaining image to every baseline/control/RL arm. A prospective full-image classification protocol should not use ground-truth boxes as crops. The metadata has no image-size table; dimensions and duplicate pixels remain unaudited.

The source prose describes boxes as one-based XYXY, but 130 actual box rows contain a zero coordinate. Their original values remain unchanged and all affected IDs are recorded. This discrepancy is another reason not to infer an object-cropping policy from unchecked geometry. Original image hashes, banner-removed hashes and the preprocessing transform must be recorded if images are later authorized; none exist yet.

The image terms restrict use to **non-commercial research**, with photographers retaining copyright. This is not a general commercial/product license or an authorization to redistribute images. The public-domain declaration in `evaluation.m` applies to its code, not to the photographs. Keep dataset images out of the public repository; a future Roboflow onboarding or commercial deployment needs an appropriate use decision or a differently licensed dataset. The audit itself publishes metadata/provenance, not photographs.

The initial metadata-stage trainer proposal used strict JSONL rows with only `id`, `split`, `image` and `label`, and contemplated already-cropped image paths. The final agreed contract retains those four fields but points `image` to the **original JPEG**; the declared shared loader removes the banner lazily exactly once. Boxes, dimensions, family and manufacturer stay in a metadata sidecar rather than prompts. A separate `data_spec.json` contains the question, all 100 labels, split policy, preprocessing and source URLs/hashes. Test rows must never enter the training loader. No image-ready training JSONL was emitted during this historical metadata stage; the later 16-image and 400-image candidate manifests are described in the image preparation update.

## Prospective train-only few-shot plan

The audit freezes an optional **16 examples per class, 1,600 total**, using only native training IDs and seed **2718**. Each class ranks IDs by SHA-256 of compact JSON `["fgvc-aircraft-2013b", seed, variant, image_id]`; the first 16 are selected. This is reproducible regardless of source row order. Full validation remains available for prospective development and native test stays sealed. The selected metadata-row fingerprint is `97ce7595a88c63d419c9760efbbf6ab89c6c7361f8720ce71abc2fa91216387d`.

This defines a candidate sampling plan, not an activated experiment. If used, all matched controls must receive the same 1,600 training examples and evaluation conditions. Freeze model identities, prompts, class parser, optimization budgets and acceptance criteria before fits. The metadata does not establish that frontier models are weak here, and no performance claim follows from this audit.

Validation at completion of the metadata stage: eight preparation tests passed and Ruff check/format passed. A fresh 407,927-byte metadata-only download reproduced every audit field exactly at that time; evidence is retained in `runs/aircraft-audit/reproduction-verification.json`. No download or audit jobs remained at that stage. The later full-archive acquisition and candidate preparation have their own status and evidence in the image preparation update.
