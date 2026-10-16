# Global Wheat metadata audit

Audited October 16, 2026. This is dataset and evaluator research, **not a model result**. The initial metadata-only audit was followed by the root-authorized full image integrity audit below. No images were visually inspected and no wheat inference or training ran. The pills and stomata tests were not accessed by this work.

The author's Hugging Face release is a credible public acquisition-domain-held-out candidate. Its 6,512 unique image IDs, boxes, and domains match the unique IDs and corresponding annotation rows in the original competition CSVs. The exact 2021 competition assignment implementation remains unverified. WILDS offers a separately named benchmark with published scoring code, but its different native manifests and annotations must be used. The WILDS manifest audit is incomplete and has already identified one train/ID-test image overlap.

## Pinned sources and reproduction

- Author HF release: [`Etienne-David/GlobalWheatHeadDataset2021`](https://huggingface.co/datasets/Etienne-David/GlobalWheatHeadDataset2021/tree/d223a227ee21fca3dbab9605d3fbcdcc2e33abc5), revision `d223a227ee21fca3dbab9605d3fbcdcc2e33abc5`, CC-BY-4.0. Four parquet shards total **1,260,938,177 bytes**. Full shard checksums were obtained from metadata, not verified by downloading the shards.
- Original author archive: [Zenodo 5092309](https://zenodo.org/records/5092309), version 1.0, July 12, 2021, CC-BY-4.0. The 10,215,190,132-byte ZIP declares MD5 `22b4b542c9ae7e056d7fcdeae9ecaed5`; the complete archive was not downloaded or hashed. CSV bytes were extracted through bounded HTTP ranges and hashed locally.
- [Machine-readable audit](wheat-data-audit-2026-10-16.json) includes all CSV SHA-256s, declared parquet hashes, canonical metadata fingerprints, domain/country counts, annotation flags, full affected IDs, duplicate candidates, and limitations.
- [Audit script](../scripts/prepare_wheat.py) downloads only parquet metadata columns and ZIP CSV entries. It refuses oversized byte ranges and responses that ignore Range. PyArrow 23.0.1 was used for the metadata projection. No production TLS-verification bypass exists in this script.

```sh
python scripts/prepare_wheat.py --download-metadata \
  --metadata-dir runs/wheat-audit \
  --output runs/wheat-audit/audit.json
python scripts/prepare_wheat.py --verify-wilds-metric \
  --output docs/wheat-data-audit-2026-10-16.json \
  --protocol-output docs/gwhd2021-hf-domain-ada-v1.json
pytest tests/test_wheat_data.py
```

The evaluator verification command requires WILDS 2.0.0 and torch-scatter 2.1.2 in addition to PyTorch/torchvision. Our execution used PyTorch 2.14.1 and torchvision 0.29.1. These optional audit dependencies were isolated under `runs/wheat-audit/wilds-lib`; the detector training environment was not modified.

A fresh HF/Zenodo metadata-only reproduction pass transferred **5,723,172 bytes** and exactly reproduced all split, source, and storage fingerprints. Independent source checks, ZIP header probes, and WILDS CSV requests transferred additional metadata; that number is not a session-wide network total. Optional WILDS comparison uses existing local `wilds-*.response` files and the pinned dataset source under the metadata directory. The downloader does not fetch those files or silently reconstruct missing WILDS manifests.

## Actual competition-release membership

| Author HF split | Unique images | Original CSV rows | Boxes | Empty images | Domains | Maximum boxes/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 3,655 | 3,657 | 163,651 | 50 | 18 | 129 |
| Validation | 1,476 | 1,476 | 44,347 | 28 | 11 | 179 |
| Test | 1,381 | 1,382 | 67,373 | 47 | 18 | 190 |

There are **zero image-ID overlaps and zero acquisition-domain overlaps** between these three splits. All 47 domains are assigned to one split. Training countries are Belgium, France, Norway, Switzerland, and the UK; validation and test contain none of those countries. Validation and test share Australia, China, and Japan, so the claim is acquisition-domain separation rather than entirely disjoint validation/test countries. Actual metadata has 47 domains and 12 countries, superseding older descriptive counts.

The [original archive description](https://zenodo.org/records/5092309) identifies competition validation as the old 2020 public test and competition test as the private 2021 test. Membership was checked against those actual files, not inferred from the HF split names.

Image IDs are filenames, not computed pixel hashes. Full file/pixel duplicate audits and the possibility of pretraining exposure remain unresolved. Zero filename overlap does not establish absence of near duplicates or model contamination.

## Conversion and annotation preservation

**HF `objects.boxes` are XYWH.** The card's XYXY description corresponds to the original CSV `BoxesString`. Every HF annotation list and domain matches an original CSV row after explicit `[xmin, ymin, xmax, ymax]` to `[xmin, ymin, width, height]` conversion. Coordinate-format guessing would corrupt this dataset.

Three original image IDs have two CSV rows. HF retains the later row for each:

| Split | Image ID | First/later box counts |
| --- | --- | ---: |
| Train | `1961bcf453d5b2206c428c1c14fe55d1f26f3c655db0a2b6a83094476e8edb5b` | 14 / 45 |
| Train | `d88963636d49127bda0597ef73f1703e92d6f111caefc44902d5932b8cd3fa94` | 25 / 42 |
| Test | `da9846512ff19b8cd7278c8c973f75d36de8c4eb4e593b8285f6821aae1f4203` | 58 / 24 |

This is the author's release behavior, not a cleanup applied here. Within-image identical GT boxes remain in five train images and one test image, each repeated twice. Preserve them for native-release parity.

There are **19 out-of-frame GT boxes in five test images**. The five original image headers and HF image metadata independently confirm 1024×1024 dimensions. Only 623 JPEG header bytes per image were decoded for dimensions, with no pixel arrays or visual inspection; the ZIP container range reader transferred more bytes than these header lengths. These extents may describe full truncated objects and are **not automatically label errors**. Do not clip or discard them silently. Exact boxes and image IDs are in the JSON audit. Other image dimensions still need validation before conversion.

The ZIP contains three image names absent from every competition CSV. Do not add them to a split. One same-size/same-CRC32 candidate pair is entirely within test, domain Terraref_2: `12906c1e75f9d4742f697961678cc4db0d172389dbfec48fc8fd0d2e49dddbc0` has zero boxes, and `d747d8ff8dda97f935792d36fd8d6cb94f7b72a0e402b6a2419c3974db5f0e58` has one. CRC32 is not proof of identical image bytes or pixels; retain both until a full-byte audit can establish what the candidate means.

A later COCO converter should preserve HF split membership, every box, empty images, original 64-character IDs, domain, country, and source row provenance. Map single class HF `0` to COCO `1`; assign unique integer image/annotation IDs without replacing the original identifiers. Read image dimensions and validate geometry without mutating native evaluation labels. Any training-only clipping policy must be explicit and separate from scoring ground truth. A WILDS converter must instead preserve native CSV row identity, including repeated image filenames, and its integer domain mapping.

| Cap | Train images above cap | Validation images above cap | Test images above cap |
| --- | ---: | ---: | ---: |
| 100 | 23 | 29 | 75 |
| 128 | 1 | 16 | 13 |
| 200 | 0 | 0 | 0 |
| 300 | 0 | 0 | 0 |
| 500 | 0 | 0 | 0 |

A 500-output evaluation cap does not truncate GT counts; RF-DETR's 300 native queries also exceed the largest native count. A 128-selection refinement limit must preserve all remaining base predictions. These counts are capacity checks, not measured detector recall.

## Competition metric versus WILDS

The [2021 challenge page](https://www.aicrowd.com/challenges/global-wheat-challenge-2021) defines Average Domain Accuracy as per-image `TP / (TP + FP + FN)`, averaged within each domain, then averaged equally across domains. It describes IoU strictly above 0.5. Empty GT plus empty predictions scores 1; either one alone being empty scores 0. Weighting every image equally across the entire dataset is a different metric.

The exact official 2021 matching/tie implementation was not located. The organizer's [starter tutorial](https://www.aicrowd.com/showcase/tutorial-with-pytorch-torchvision-and-pytorch-lightning) explicitly distinguishes its helper from the evaluator. `EtienneDavid/GWC-codalab-public/evaluate.py` is a 2020 source and does not prove the 2021 semantics. Do not label our Hungarian assignment or the WILDS matcher as verified competition equivalence.

WILDS pins an executable definition: [dataset/evaluation code](https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/datasets/globalwheat_dataset.py) and [DetectionAccuracy](https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/common/metrics/all_metrics.py), commit `472677590de351857197a9bf24958838c39c272b`. Predictions must have scores **strictly above 0.5**. `torchvision.Matcher(0.5, 0.5)` accepts IoU **at least 0.5**; each prediction selects its best GT, and repeated selections add false positives. It does not maximize matching cardinality. The dataset then reports the equal-session mean as `detection_acc_avg_dom`.

**The official WILDS 2.0.0 library was executed successfully**, without replacing the matcher or grouped-average implementation. Its installed dataset/metric file SHA-256s exactly match the pinned GitHub source. Ten synthetic cases verify ambiguous overlap (1/3), exact IoU 0.5 (1), score exactly 0.5 (0), score just above 0.5 (1), duplicate predictions (1/2), repeated native GT boxes (1/3), unchanged out-of-frame GT (1), and the three empty cases (1/0/0).

We also instantiated the real `GlobalWheatDataset` with temporary synthetic CSVs and called its actual subset evaluator. Three correct images in one domain and one incorrect image in another produce **image average 0.75 and official equal-domain average 0.50**. This checks the full grouping path, including the actual torch-scatter dependency, rather than reimplementing its average. No image files were created or loaded. All checks are reproducible with `--verify-wilds-metric`, and actual outputs and package/source identities are in the JSON audit. These are algorithm checks, not wheat model results.

## WILDS-native alternative: promising, incomplete audit

GlobalWheat-WILDS **dataset v1.1**, `split_scheme="official"`, uses CodaLab bundle `0x36e16907b7254571b708b725f8beae52`. The [WILDS paper](https://openreview.net/pdf?id=z7p2V6KROOV) defines 2,943 train, 357 ID-val, 357 ID-test, 1,424 OOD-val, and 1,434 OOD-test rows. OOD validation has eight Asian/African domains and OOD test has 21 Australian/North American domains. These are **not the HF competition splits**. The [release changelog](https://wilds.stanford.edu/changelog/) describes v1.1 corrections to location/stage metadata without changing labeled splits or evaluation.

The source code declares MIT for the WILDS dataset; the original archive and HF release declare CC-BY-4.0. Preserve original dataset attribution and document both declarations; do not assume the library's software license replaces the original data provenance.

Actual metadata retrieved so far:

- `official_train.csv`: 2,943 rows, 2,942 unique image IDs, 131,864 boxes, 43 empty rows, 18 domains. SHA-256 `efe5eebc03e4db135d2771b0f301e75f3669825c1eb4393ac7a9887cfd79ed74`.
- `fixed_train_test.csv`: 357 unique rows, 16,093 boxes, four empty rows, 18 domains. SHA-256 `382a6ad0975170c4c71cab45eb6a54b799fdb6d5b67699382bd488046c2ccb3d`.
- `metadata_domain.csv`: SHA-256 `23aedf00c2a8ebc07db5c96fd5cd0e1d73b18807026eabba9a3b1222d88a6bc4`.
- The bundle directory lists all five native split files, but OOD-val, OOD-test, and ID-val retrieval attempts timed out or failed. Their actual membership/annotation audits remain pending.

The repeated train ID is `1961bcf453d5b2206c428c1c14fe55d1f26f3c655db0a2b6a83094476e8edb5b`. **Train and ID-test share** `d88963636d49127bda0597ef73f1703e92d6f111caefc44902d5932b8cd3fa94`: train contains the 25-box Arvalis_10 row while ID-test contains the 42-box Arvalis_12 row. This is an observed ID-split leak, not evidence of an OOD split leak. It also proves that replacing WILDS labels with the HF later annotations changes its native definition.

The CodaLab host failed normal TLS verification with an expired certificate. Limited, unauthenticated public metadata probes disabled verification; no credentials or image payloads were involved. Recorded hashes pin received bytes but do not authenticate that transport. Revalidate manifests over trusted transport before freezing a WILDS experiment. This transport exception is not built into the reproducible downloader.

A valid native experiment could be named **GlobalWheat-WILDS v1.1, official OOD test, official WILDS library metric**, once all manifests and original image provenance have been verified. The missing manifests remain blocked: standard TLS verification fails on the original host, the Stanford alias times out, the official GitHub tree/release assets contain no CSVs, and checked maintainer issue responses provide no current mirror. The public Internet Archive index returned no captured CSVs. Do not reconstruct those files from paper counts and label them native.

HF's much smaller storage and `.jpg` paths do not establish decoded-pixel equivalence to original WILDS images. Matching filenames and annotations alone is insufficient to substitute HF image bytes into a WILDS-native result.

## Reviewable protocol with verified metadata and evaluator

The root-approved proposal is **[`gwhd2021-hf-domain-ada-v1`](gwhd2021-hf-domain-ada-v1.json)**: the author's unchanged HF splits and annotations, evaluated with the unchanged WILDS 2.0.0 equal-domain metric. It is explicitly a named HF-release evaluation protocol, with **no claimed competition-leaderboard or WILDS-native equivalence**.

- Train: **3,655 images, 18 domains**; only these labels update model parameters.
- Validation: **1,476 images, 11 domains**; model, prompt, postprocessing, and confidence-calibration selection.
- Sealed final test: **1,381 images, 18 domains**; no model inference until choices are frozen. Existing test access is limited to provenance, domain, annotation geometry, and density auditing.
- Preserve all 275,371 boxes, 125 empty images, repeated boxes, and out-of-frame GT. Native HF annotations are not replaced by WILDS rows. Models without confidence scores assign score 1.0 to retained predictions; any score calibration must be selected on validation and frozen.
- Primary metric: exact `detection_acc_avg_dom`; 500-prediction ceiling; supplemental COCO AP must remain separately named. Compare every model on the same complete test split and report per-domain scores.

The protocol JSON records dataset and evaluator hashes, split fingerprints, explicit non-equivalence claims, sealing rules, and a canonical protocol hash. It is a **reviewable proposal, not an experiment result or proof that frontier models will perform poorly**.

Parquet-footer metadata provides a concrete storage bound: full shards **1,260,938,177 bytes**, extracted encoded image payloads at most **1,258,572,519 bytes**, combined **2,519,510,696 bytes (2.35 GiB)** before filesystem overhead, caches, and training outputs. Reading these footers transferred 262,144 bytes and no image columns. The original WILDS bundle is roughly 9.58 GiB, and retaining both its archive and extracted files would require substantially more space. This was the storage estimate before full-image authorization; the completed extraction below supersedes the initial deferral.

The initial protocol JSON remains unchanged as historical evidence of the definition before image access. The following image audit is a separate advancement of its integrity checks; its historical `images_not_downloaded` status must not be read as the current dataset state.


## Completed image integrity audit and usable scorer

All four pinned parquet shards were downloaded sequentially and their complete SHA-256s and sizes verified. The extractor preserved every original encoded image byte, native split, domain, empty image and annotation. All **6,512 decoded images are 1024×1024**, with no orientation transformations or re-encoding. Exact encoded-file and decoded-RGB hashes show **zero cross-split duplicate groups**. Domain overlap remains zero. Near-duplicate leakage and pretraining contamination have not been excluded by these exact hashes.

The former CRC32 candidate is now a confirmed identical-file and identical-pixel pair entirely within test: original IDs `12906c1e75f9d4742f697961678cc4db0d172389dbfec48fc8fd0d2e49dddbc0` and `d747d8ff8dda97f935792d36fd8d6cb94f7b72a0e402b6a2419c3974db5f0e58`, both domain Terraref_2, have zero and one GT boxes respectively. Both rows and their labels remain unchanged. The 19 out-of-frame test GT boxes also remain intact. No train or validation box extends outside its actual dimensions, and every GT box has positive area.

The complete [image-audit report](wheat-image-audit-2026-10-16.json) pins actual shard verification, conversion fingerprints, all affected duplicate IDs, out-of-frame annotations, domain counts and each image-manifest checksum. Individual encoded/decoded image hashes are generated in the ignored dataset's `audit/{train,validation,test}-images.json`. Encoded payloads total **1,258,536,943 bytes**. The real initial download and audit took **57.24 seconds** on this host; this is preparation latency, not model performance. Verified temporary parquet copies were removed only after extraction receipts and image hashes were written. A 6 GiB free-space floor was enforced throughout; the shared Qwen download was already complete.

```sh
python scripts/extract_wheat.py --reserve-gib 6 --remove-verified-shards
```

The script supports resumption through receipts: it verifies every existing encoded image against its recorded hash before reusing a shard. Unexpected source dimensions, changed metadata, malformed native boxes, bad full-shard checksums or the disk reserve guard stop preparation. Duplicate findings are reported and gate readiness; no images or labels are silently removed.

Only `data/gwhd2021-hf-domain-ada-v1/development/coco/train` and `development/coco/valid` should be exposed to training. Their native counts remain **3,655 / 1,476**, with **18 / 11** domains. Each COCO image contains `domain`, integer `session_id` in the pinned WILDS order, `original_image_id`, `native_split` and `source_shard`. `sealed/test` is separate; its 1,381 images were decoded solely by the integrity auditor. They remain unavailable to models, training and manual image inspection until the final freeze.

[`wheat_eval.py`](../src/vision_rl/wheat_eval.py) provides `evaluate_wheat` for COCO XYWH outputs and `official_image_rewards` for grouped XYXY tensor proposals. Both invoke the unchanged official WILDS metric; the full report invokes its native equal-domain evaluator. Scorer version, source hashes, selected image IDs, domain membership, errors and data/prediction hashes are recorded. Request/parse failures invalidate model comparisons. Full extents of native GT are preserved.

`calibrated_wheat_evaluation` selects a foreground logit offset on development data from the fixed grid **−1.5, −1.0, −0.5, 0, 0.5, 1.0, 1.5**. Ties prefer the smallest absolute offset, then the smaller signed value. It preserves all proposal counts, order and geometry and refuses a search on explicitly marked native test annotations. Apply the chosen frozen offset with `calibrate_wheat_predictions` for final scoring. The official score gate remains strictly above 0.5.

`paired_domain_comparison` computes paired acquisition-domain bootstrap intervals with 20,000 resamples and seed 2718. It compares identical image/domain membership and scorer fingerprints, rejects incomplete evaluations, and checks a 0.05 ADA gain plus a positive 95% interval lower bound. The interval is over observed domains and does not correct development selection bias. AP is a separately named supplemental metric and is not numerically comparable to ADA.


## Optional evaluator installation

[`requirements-wheat-eval.in`](../requirements-wheat-eval.in) pins the scorer packages used for this experiment. [`requirements-wheat-eval.lock`](../requirements-wheat-eval.lock) adds their full dependency closure with published distribution hashes, constrained to the shared versions in the detector lock. It deliberately excludes PyTorch and torchvision wheels so that it cannot silently replace the selected CPU/CUDA backend. WILDS' graph dependency `ogb` is included even though the wheat scorer does not use graph models. PyArrow 23.0.1 is a separate preparation-only dependency and is not required for scoring.

For a fresh Python 3.12 environment, install the selected PyTorch runtime first, then build the CPU scatter extension against that runtime. Linux needs a C++ compiler (`build-essential` on Debian); macOS needs the Xcode Command Line Tools. The commands below select CPU PyTorch; a CUDA deployment must first choose its own matching official torch/torchvision wheels. The ADA scorer itself remains on CPU.

```sh
uv venv --python 3.12 .venv-wheat
uv pip install --python .venv-wheat/bin/python \
  torch==2.14.1 torchvision==0.29.1 --torch-backend cpu
uv pip install --python .venv-wheat/bin/python setuptools==78.1.0
FORCE_ONLY_CPU=1 MAX_JOBS=2 uv pip install --python .venv-wheat/bin/python \
  --no-deps --require-hashes --no-build-isolation -r requirements-wheat-eval.lock
uv pip check --python .venv-wheat/bin/python
```

`--no-build-isolation` is required because torch-scatter imports the already installed torch while building. `FORCE_ONLY_CPU=1` avoids requiring a CUDA compiler for this CPU evaluator even on GPU hosts. Do not copy a compiled torch-scatter extension across Python versions, CPU architectures or incompatible torch runtimes. Hashes verify the source distribution; they do not promise identical compiled binary bytes across toolchains.

To add the scorer without modifying an existing detector environment, install the same lock into a separate target directory:

```sh
FORCE_ONLY_CPU=1 MAX_JOBS=2 uv pip install --python .venv-detection/bin/python \
  --target runs/wheat-eval-lib --no-deps --require-hashes --no-build-isolation \
  -r requirements-wheat-eval.lock
PYTHONPATH=runs/wheat-eval-lib:src .venv-detection/bin/python \
  -m pytest tests/test_wheat_eval.py -q
```

The test command also needs pytest 9.1.1 in the chosen Python environment. Test fixtures are synthetic boxes and temporary annotation files; they read no benchmark images. The scorer checks the exact WILDS 2.0.0 source hashes at runtime. For complete constructor/grouping parity without reading any dataset images or metadata:

```sh
PYTHONPATH=src .venv-wheat/bin/python -c \
  'import json, runpy; print(json.dumps(runpy.run_path("scripts/prepare_wheat.py")["verify_wilds_metric"](), indent=2))'
```

The experiment's original ignored `wilds-lib` directory remains unchanged. A fresh hash-verified target installation passed all 24 scorer/calibration/domain-bootstrap tests, and all 32 installed optional distributions had satisfied declared dependencies. The full official synthetic constructor still returned equal-domain accuracy **0.50**, versus **0.75** for an image average. A separate forced, uncached CPU source build verifies that installation does not rely on the original exploratory wheel cache. Build/install logs and full package identities are retained in `runs/wheat-audit/packaging-*.log` and `packaging-verification.json`; these are packaging checks, not model results.

The documented standalone recipe was also executed in a fresh Python 3.12 virtual environment: dependency checking reported all 39 packages compatible, all 24 tests passed, and the unchanged official constructor/grouped evaluator reproduced the expected scores. [Public installation evidence](wheat-evaluator-install-verification-2026-10-16.json) records package versions, input-lock hashes, uncached source-build details and compiled CPU extension hashes. This verification used macOS arm64; Linux/CUDA installation has not yet been exercised.
