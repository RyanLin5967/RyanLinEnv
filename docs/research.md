# Research context and benchmark choices

Checked against primary sources on **October 15, 2026**. This is an independent learning project inspired by public research. It is not an official Roboflow Labs product or a reproduction of its internal training infrastructure.

## What Roboflow announced

Roboflow announced **Roboflow Labs on September 30, 2026**. The division describes three areas of work: open models, collaborations with frontier labs on models/data/environments, and enterprise evaluations for capabilities that remain weak in practice. The public announcement establishes the direction, but does not publish a general-purpose Harbor RL training stack. [Roboflow Labs announcement](https://roboflow.com/blog/roboflow-labs-launch)

The Labs site includes physical reasoning, visual perception, fine-grained inspection, document understanding, and temporal understanding among its capability areas. These provide useful starting points for defining a specialist's task. [Roboflow Labs](https://labs.roboflow.com/)

**Roboflow Agent**, announced October 14, 2026, is the closest public product to the agent-assisted workflow described in this project: it selects models, builds vision workflows, and can train efficient models on user data. Its announcement does not establish that it uses this repository's RL algorithm or Harbor. [Roboflow Agent announcement](https://roboflow.com/blog/roboflow-agent)

## Which benchmarks fit the goal?

| Benchmark or task | What it measures | Integration in this repository |
| --- | --- | --- |
| Generated spatial demo | Relative positions of colored objects | Runs locally; demonstrates learning, selection, and verification |
| Custom image multiple-choice tasks | A user-defined, objectively scored visual skill | JSONL manifests, specialist training, endpoint/offline evaluation |
| RF100-VL | Object detection across unusual visual domains | Research target; a counting-task import is a derived task, not RF100-VL evaluation |
| Humanity's Sixth Sense (HSS) | Open-ended intuitive reasoning from images and video | Research target; requires a separate native evaluation adapter |

**RF100-VL** is a Roboflow/Carnegie Mellon benchmark with 100 datasets across seven domains. Its paper and code investigate failures on concepts outside common web imagery. Detection uses bounding boxes and average precision, so accuracy on derived multiple-choice questions cannot be compared with its published AP scores. Use the benchmark's own protocol when adding a detector backend. [RF100-VL project](https://rf100-vl.org/), [official code](https://github.com/roboflow/rf100-vl/)

**HSS was created by Scale AI and Elorian**, independently of Roboflow. Its October 7 announcement reported 522 tasks and a human result of 93.1%, compared with 53.6% for the strongest model in that launch evaluation. The examples cover temporal, physical/spatial, social, and abstract inference. These motivate hypotheses about capabilities to improve; they do not show that our synthetic demo is difficult for frontier models. [Scale announcement](https://labs.scale.com/blog/introducing-humanitys-sixth-sense), [Elorian explanation](https://elorian.ai/blog/humanitys-sixth-sense)

The live HSS leaderboard had already changed by this research date: it listed Gemini-4-Argon at 55.4%. Historical paper numbers are not current baselines for a newly trained specialist. Its official result aggregates three attempts per task and uses task-clustered confidence intervals. [HSS leaderboard](https://labs.scale.com/leaderboard/hss)

## Requirements for a faithful HSS adapter

The official [ScaleAI/HSS dataset card](https://huggingface.co/datasets/ScaleAI/HSS) exposes a **test-only** split, open-ended answers, and per-task rubrics. Do not train on it. A correct answer must satisfy every rubric criterion. Models receive the whole video; evidence timestamps are analysis metadata. Annotations are MIT-licensed, while media retain their original rights and are supplied for non-commercial research/evaluation. The displayed download size is approximately 11.2 GB.

An adapter should preserve task IDs, media, prompts, rubric grading, judge settings, and the official version. Pin the dataset revision and grader before a comparison. Keep reference answers and evidence timestamps out of the candidate's inputs. Use independent, permitted data for training and validation. Label image-only or otherwise filtered results as subset experiments.

The [official leaderboard](https://labs.scale.com/leaderboard/hss) describes an evaluation harness, but the pages checked did not provide a verifiable repository link. Resolve that source before claiming protocol parity. Converting HSS to answer-letter questions would change the task and would not produce an official HSS score.

## Why Docker and Harbor?

[Harbor](https://docs.harborframework.com/) runs agents on tasks in sandbox environments and collects verifier results. Here, Docker supplies the execution environment; Harbor supplies the outer agent task and verifier lifecycle. The training code supplies the model optimizer. These are separate responsibilities: running a Harbor job alone does not update a model's weights.

There are also two meanings of **agent**. The outer training agent chooses candidate settings and submits a checkpoint. The inner learned policy produces predictions and receives correctness rewards. The CPU demonstration uses REINFORCE; the optional GPU path uses GRPO with LoRA. Neither means that this project trains a general frontier foundation model from scratch.

## How to choose a defensible specialist target

1. Collect independent examples of a useful skill, such as counting small objects or identifying a narrow defect type. Check that annotation category and spatial scope match the question: board-only counts cannot answer a question about all visible objects. Group related images before splitting so one scene does not cross train, validation, and test.
2. Run actual baseline model versions on validation. Find a stable capability gap across enough examples, rather than a single surprising failure.
3. Fix the task, metric, test set, model settings, and budget before optimizing. Train on the training split and choose candidates on validation.
4. Evaluate the frozen specialist and the baseline on the same untouched test examples. Report sample count, per-category results, uncertainty, latency/cost, and failed requests.
5. Limit the conclusion to the tested benchmark, versions, and conditions. A statistically supported win on one narrow task is useful evidence, without implying general superiority.

A Roboflow API key gives access permitted by that account and workspace. It is not a general GPU credential. Dataset access, Roboflow-hosted training, external model inference, and arbitrary CUDA training are distinct resources. This repository uses local compute by default and keeps provider credentials outside tracked files.
