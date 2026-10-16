# Aircraft image preparation and bounded training probe

This continues the [author metadata audit](fgvc-aircraft-audit-2026-10-16.md). It prepares non-commercial research data; it does not establish model performance. Aircraft became the primary candidate after the bounded wheat RL experiment failed its progression gates. Native image integrity and evaluation requirements still apply before benchmark claims. Test images are excluded from model calls and manual viewing.

The full author archive and image audit are now **complete**. All **10,000** original JPEGs preserve native membership: **3,334 TRAIN / 3,333 validation / 3,333 sealed TEST**, with 100 classes in every split. There are **zero exact duplicate groups**, either within or across splits, using original encoded bytes, decoded RGB pixels or banner-removed RGB pixels. All annotation and evaluator files match the pinned metadata archive byte-for-byte. The [completed audit](fgvc-aircraft-image-audit-2026-10-16.json) records these findings, native manifest hashes, preserved candidate inputs and checksum limitations. Near-duplicates, acquisition dependence and model pretraining exposure remain unassessed.

The author archive is **2,753,340,328 bytes**, SHA-256 `e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834`. Its observed MD5 is `d4acdd33327262359767eeaa97a4f732`, which **differs** from the [author's checksum page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html). The author index links that checksum alongside the downloaded archive, and the checksum text explicitly names `data/fgvc-aircraft-2013b.tar.gz`. It is therefore visibly associated with the same release basename; the evidence does not establish a different archive version as the cause. The precise discrepancy remains unexplained.

After download, the observed MD5 was independently corroborated against the [R torchvision maintainer's pinned source](https://raw.githubusercontent.com/mlverse/torchvision/2a29537f210de739fe9cb7b2e35c8cbbdc19be3f/R/dataset-fgvc.R). The separate [PyTorch torchvision v0.24.0 implementation](https://raw.githubusercontent.com/pytorch/vision/v0.24.0/torchvision/datasets/fgvc_aircraft.py) uses the same author URL and describes 100 variants, but supplies no archive checksum. The R-source corroboration does not erase the author-page discrepancy or establish a prospectively pinned full-archive SHA. The metadata archive has its own separately disclosed MD5 discrepancy.

The original transfer exited naturally with a read timeout after **2,088,763,392 bytes**. A single authorized continuation preserved the exact recorded prefix, matched a 65,536-byte tail against the author server, and used its strong ETag with `If-Range`. The successful response began exactly at the retained offset and reported the original total size. The continuation took **288.09 seconds** in one attempt; extraction and automated image auditing took **135.86 seconds**. Recovery refuses ignored ranges, changed validators, mismatched overlaps and competing file locks. Original JPEGs total **2,761,629,717 bytes**; only the verified compressed archive was removed afterward. No transformed image cache, lossy re-encoding, model call or manual test-image view occurred during preparation.

To avoid making a requested quick feasibility check wait for the slow author server, a separate, prospectively fixed **16-image native-training probe** was acquired from a [pinned mirror revision](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/16c6368266640c62cafa0cd59b9abb01291b7072/fgvc-aircraft-2013b). Selection used seed 791, chose 16 classes by hash and one training ID per class by hash, and was frozen before any image availability or model output was observed. The complete 100-label vocabulary remains in the question. This is separate from the earlier prospective 16-examples-per-class plan.

The mirror's training-ID list, training variant annotations, variant vocabulary and `evaluation.m` match the pinned author annotation archive byte-for-byte. All 16 JPEGs matched the mirror commit's individual LFS SHA-256 values. They total **4,019,641 bytes**, below the authorized 200 MiB additional-transfer bound. The initial per-image provenance is [recorded here](fgvc-aircraft-probe-provenance-2026-10-16.json), retaining its acquisition-time pending status. The subsequent author audit now confirms exact original JPEG bytes, decoded pixels, cropped pixels, dimensions, labels and training membership for all 16. That new result is recorded separately; no frozen probe input or historical provenance was rewritten. No benchmark score or held-out generalization claim follows from this preparation.

All arms use [`aircraft_data.py`](../src/vision_rl/aircraft_data.py), preprocessing ID `fgvc-aircraft-rgb-remove-bottom20-v1`: decode the original JPEG into RGB, remove exactly the bottom 20 rows, and pass the remaining image onward. It performs no resizing, ground-truth box crop, orientation correction or re-encoding. Unexpected EXIF orientation fails explicitly. The pixel hash includes RGB mode, width, height and exact bytes. The same transform runs lazily for every arm, avoiding a second transformed-image cache. Both source and cropped dimensions and pixel hashes are retained in the audit sidecar.

The strict probe JSONL contains only `id`, `split`, `image`, and `label`, with original JPEG paths relative to the manifest. `data_spec.json` names the shared preprocessing ID, immutable source hashes, task question and full class vocabulary. Geometry and hierarchical metadata never enter the question. The actual trainer's loader reproduced all 16 audited cropped-image hashes; synthetic tests also check that removing different banners can reveal an otherwise hidden pixel duplicate.

## Frozen four-shot candidate

The subsequent [four-shot plan](fgvc-aircraft-fourshot-plan-v1.json) fixes **400 native TRAIN images**, exactly four from each of all 100 canonical variants. It uses seed **791** and, within each class, ascending SHA-256 of compact JSON `["fgvc-aircraft-2013b", 791, variant, image_id]`. No model outputs, rewards, image availability, validation data or test data determine this selection. Its frozen file SHA-256 is `44be6969eb2582073f12f7a84594ee0bf1e3c211fa6ac82170a0efbf7302f423`. The earlier 16-image probe is unchanged; one selected ID overlaps by chance. The optional 1,600-image plan remains inactive.

A separately authorized mirror fetch acquired these exact 400 selected images from the same immutable revision, with a 200 MiB transfer cap and a minimum 10 GiB free-space reserve. Their total original payload is **110,960,896 bytes**. One already verified probe image was hardlinked, so the new transfer was **110,851,613 bytes**. Every selected JPEG matched its pinned LFS SHA-256 and decoded successfully. There are no exact encoded, decoded-RGB or cropped-RGB duplicate groups within this candidate. The completed author audit additionally confirms exact original bytes, both pixel representations, dimensions, labels and native TRAIN membership for all 400, together with the complete cross-split duplicate checks above.

The selected originals occupy their final `development/images/train` paths. Author extraction compared every existing file against its original encoded bytes and found no mismatch. The separate mirror provenance retains its original acquisition-time status; the new author-equivalence receipt establishes the later verification. No image has been re-encoded, and no PNG crop cache exists. [Per-image acquisition evidence](fgvc-aircraft-fourshot-image-preparation-2026-10-16.json) records original and cropped dimensions/hashes and the actual trainer-loader verification. Candidate artifacts are `data/fgvc-aircraft-2013b/candidates/fourshot-seed791/{train400.jsonl,data_spec.json,provenance.json}`. Their hashes and contents remain unchanged. Training budgets, model comparisons and authorization for fits are separate decisions.

```sh
# First download the metadata archive using the command sequence below.
python scripts/prepare_aircraft_fourshot.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz
PYTHONPATH=src python scripts/fetch_aircraft_fourshot.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz
pytest tests/test_aircraft_fourshot.py
```

The plan generator refuses to replace a changed frozen file. The acquisition requires that exact plan hash, verifies its rows against the original author training metadata, checks the complete expected selected-file byte count before downloading, and rejects incorrect mirror bytes before creating an accepted JPEG. Reused probe files retain a single underlying file payload.

Local probe artifacts are `runs/aircraft-shared-probe16/prepared/{train.jsonl,data_spec.json,provenance.json}`. Complete native manifests are under `data/fgvc-aircraft-2013b`; test originals occupy `sealed/images/test`, separate from `development/images/{train,validation}`. The completed data-integrity gate produced `development/train.jsonl`, `development/validation.jsonl` and `development/data_spec.json`. Native validation contains all 3,333 author validation IDs; no TEST JSONL was emitted into development. The optional `train-16shot.jsonl` is still inactive. Model fitting and evaluation must follow the separately frozen experiment protocol.

```sh
python scripts/prepare_aircraft.py --download-metadata \
  --archive runs/aircraft-audit/annotations.tar.gz \
  --output runs/aircraft-audit/audit.json

PYTHONPATH=src python scripts/prepare_aircraft_probe.py \
  --selection docs/fgvc-aircraft-probe16-plan-v1.json \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz \
  --output runs/aircraft-shared-probe16/prepared

PYTHONPATH=src python scripts/fetch_aircraft_fourshot.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz

PYTHONPATH=src python scripts/extract_aircraft.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz \
  --reserve-gib 10 --remove-verified-archive

# After the author extraction completes, verify both frozen mirror candidates:
PYTHONPATH=src python scripts/verify_aircraft_clearance.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz

# Requires the preceding author-equivalence receipt and freshly rechecks it:
PYTHONPATH=src python scripts/prepare_aircraft_manifests.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz

pytest tests/test_aircraft_data.py tests/test_aircraft_images.py \
  tests/test_aircraft_extraction.py tests/test_aircraft_fourshot.py \
  tests/test_aircraft_clearance.py
```

The [published 16-image selection](fgvc-aircraft-probe16-plan-v1.json) is an exact byte copy of the prospectively frozen local selection. The acquisition validates its IDs and labels directly against the pinned author metadata; it does not need the ignored local metadata path recorded inside that historical selection. All examples above name the same metadata archive explicitly, so they do not depend on existing local run files. Install the project's existing development dependencies before running the tests.

Preparation validation: **33 tests passed**, covering training-only selection, full-vocabulary retention, immutable image copies, sealed test placement, duplicate gating, cropped-pixel identity, disk guards, preservation of partial downloads, and metric parity. The additional four-shot tests verify exact balanced selection, reject changes to the frozen plan, reuse a single image inode without network transfer, and reject incorrect mirror bytes. Re-running a completed extraction verifies all retained original-byte hashes and source manifests without downloading or decoding again. Changed originals fail explicitly and are preserved for investigation. Development manifest publication requires the complete integrity gate, emits no test rows, and refuses to silently replace changed experiment inputs. This is preparation evidence, not model validation.

The data-only publication subset independently passes **24 tests** with only its own scripts, shared image helper, Pillow and pytest. This was verified in an isolated directory with the VLM trainer module physically absent and no dataset images or local audit artifacts copied into it. The other nine tests import `vision_rl.vlm_local_training.classification_metrics`; publish and run that optional Python replay test with the trainer module. The official Octave evidence and `.m` fixture have no dependency on the Python trainer.

The later clearance hardening adds an explicit author-equivalence receipt before any native validation manifest can be published. It compares the immutable 16- and 400-image manifests, acquisition provenance and selection plans against the author labels and training membership, including their full vocabulary, preprocessing identity, exact original-JPEG paths and retained byte hashes. A consistently hashed but incorrect label or split fails this semantic check. Publication requires the receipt to equal freshly reconstructed evidence, including source and input hashes; an absent or stale receipt fails before any output is written. These checks preserve the candidate inputs and perform no image decoding or model calls. The data-only suite passed **37 tests**; the subsequent range-recovery suite passed **8 additional tests**, including interruption recovery and competing-owner rejection. The real full-author pipeline exited successfully after executing extraction, semantic clearance and development publication in sequence.

## Executed official evaluator parity

The unchanged author `evaluation.m` and its five bundled MATLAB companions executed in GNU Octave **9.4.0**, Linux arm64, with one CPU, 1 GiB RAM and networking disabled. The read-only source mount contained only the six `.m` files; no benchmark metadata or image was exposed. All six file hashes matched the author audit. The [retained execution evidence](fgvc-aircraft-evaluator-parity-2026-10-16.json) includes container identity, fixture and source hashes, every resulting confusion matrix and resource measurements.

All **nine** synthetic cases passed: perfect predictions, all wrong, imbalanced classes, a missing response, no responses, finite score ties, all negative-infinity scores, negative finite scores, and the native last-assignment behavior for a repeated image/class pair. The imbalanced case executed at **0.50 mean-class accuracy** with **0.75 microaccuracy**. These are metric fixtures, not model measurements. The actual container build took 32.60 seconds and fixture execution approximately 0.423 seconds. VM free space remained above the required 1 GiB reserve; the observed Docker storage change exceeded the apt installed-size estimate and is recorded rather than hidden.

The Python canonical-label scorer now passes nine parity tests against those **recorded native outputs**, including invalid/missing-response denominators. These replay tests do not start Octave. Native score-triplet ties and duplicate assignment are tested by the `.m` fixture; the Python classification interface receives the corresponding single canonical label or invalid response and does not claim to implement a general score-triplet API. As before, no equivalence to the separate 2013 competition evaluator is claimed.

To repeat native execution in the prepared container, mount `scripts` at `/fixture`, a directory containing only the six hash-verified official `.m` files at `/source`, and a fresh writable directory at `/output`, then run:

```sh
octave --no-gui --quiet --eval \
  "addpath('/fixture'); aircraft_parity('/source','/output')"
```
