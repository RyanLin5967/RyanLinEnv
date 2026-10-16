import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def wrapper():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/local_vision.py"))


@pytest.fixture
def wheat(tmp_path):
    root = tmp_path / "coco"
    train = root / "train"
    train.mkdir(parents=True)
    images, annotations = [], []
    for index, count in enumerate([3, 0, 8, 4, 1]):
        filename = f"{index}.jpg"
        Image.new("RGB", (32, 32), (index * 30, 0, 0)).save(train / filename)
        images.append({"id": index, "file_name": filename, "width": 32, "height": 32,
                       "domain": f"domain{index % 2}", "native_split": "train"})
        for _ in range(count):
            annotations.append({"image_id": index, "category_id": 1, "bbox": [0, 0, 2, 2]})
    annotation = train / "_annotations.coco.json"
    annotation.write_text(json.dumps({"images": images, "annotations": annotations,
                                     "categories": [{"id": 1, "name": "wheat_head"}]}))
    return root, tmp_path / "prepared"


def test_wheat_probe_uses_train_density_but_never_puts_answers_in_prompts(wrapper, wheat):
    root, output = wheat
    protocol = wrapper["prepare_wheat_probe"](root, output)
    assert [row["id"] for row in protocol["rows"]] == ["2", "0"]
    assert [row["training_box_count"] for row in protocol["selection_audit"]] == [8, 3]
    assert len({row["prompt"] for row in protocol["rows"]}) == 1
    assert all(set(row) == {"id", "image", "split", "prompt"} for row in protocol["rows"])
    assert wrapper["verify_inputs"](output / "inputs.json") == protocol
    assert protocol["task_contract"]["coordinate_format"] == "xyxy_1000"
    with pytest.raises(ValueError, match="immutable"):
        wrapper["prepare_wheat_probe"](root, output)


@pytest.mark.parametrize("change", [
    "pixels", "prompt", "annotation", "split", "test", "generation", "processor",
])
def test_generic_preparation_rejects_mutation_and_test_scope(wrapper, wheat, change):
    root, output = wheat
    protocol = wrapper["prepare_wheat_probe"](root, output)
    if change == "pixels":
        Image.new("RGB", (32, 32), "blue").save(protocol["rows"][0]["image"])
    elif change == "prompt":
        protocol["rows"][0]["prompt"] += " new instruction"
    elif change == "annotation":
        Path(protocol["annotation"]).write_text("{}")
    elif change == "split":
        protocol["rows"][0]["split"] = "test"
    elif change == "test":
        protocol["phase"] = "test"
    elif change == "generation":
        protocol["generation"]["max_tokens"] = 32
    else:
        protocol["native_processor"] = False
    (output / "inputs.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        wrapper["verify_inputs"](output / "inputs.json")


def test_generic_run_binds_task_and_keeps_geometry_errors_without_accuracy(wrapper, wheat):
    root, prepared = wheat
    protocol = wrapper["prepare_wheat_probe"](root, prepared)
    result_dir = prepared.parent / "result"
    result_dir.mkdir()
    captured = {}

    def fake_run(rows, output, **kwargs):
        captured.update(kwargs)
        return {rows[0]["id"]: {"response": '[{"bbox_2d":[1,2,3,4],"label":"wheat_head"}]'},
                rows[1]["id"]: {"response": '[{"bbox_2d":[8,2,3,4],"label":"wheat_head"}]'}}

    wrapper["run"].__globals__["run_local"] = fake_run
    result = wrapper["run"](SimpleNamespace(
        input=prepared, output=result_dir, model_directory=Path("unused"),
        integrity=Path("unused"), seed=1729, memory_limit_gib=32, resume=False,
    ))
    assert captured["task_contract"]["prepared_protocol_sha256"]
    assert captured["image_root"] == protocol["image_root"]
    assert result["errors"] == 1 and result["accuracy_score_computed"] is False
    assert result["geometry"][0]["valid_detection_count"] == 1
