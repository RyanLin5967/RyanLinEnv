"""Execute released MLX aggregation on CPU fake streams, without importing MLX."""

import argparse
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from vision_rl.local_vlm_execution import collect_stream


def check(package):
    package = Path(package)
    sources = {name: package / "generate" / name for name in ("common.py", "dispatch.py")}
    nodes = []
    for filename, wanted in [("common.py", "GenerationResult"), ("dispatch.py", "generate")]:
        matches = [node for node in ast.parse(sources[filename].read_text()).body
                   if getattr(node, "name", None) == wanted]
        if len(matches) != 1:
            raise ValueError("The installed released aggregation definition differs")
        nodes.extend(matches)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    ), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "dataclass": dataclasses.dataclass,
        "mx": SimpleNamespace(get_peak_memory=lambda: 37_000_000_000),
        "DiffusionOutputHandler": lambda *args: None,
    }
    # Only the native dataclass and generate aggregator execute; no package
    # imports, model loading, tokenizer execution, accelerator or API call.
    exec(compile(module, str(sources["dispatch.py"]), "exec"), namespace)
    result_type = namespace["GenerationResult"]
    cases = {
        "normal_stop_with_final_flush": [
            ("reasoning", 1, None), ("</think>\n", 2, None), ("A", 3, "stop"),
        ],
        "token_limit_with_final_flush": [("reasoning", 1, None), ("tail", 1, "length")],
        "empty_delta_then_stop": [("", 1, None), ("B", 2, "stop")],
        "immediate_eos": [("", 1, "stop")],
        "zero_budget": [("", 0, "length")],
        "empty_stream": [],
    }
    results = []
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1, 2]))
    resets = []
    tokenizer = SimpleNamespace(stopping_criteria=SimpleNamespace(reset=resets.append))
    for name, values in cases.items():
        chunks = [result_type(
            text=text, token=791, prompt_tokens=73, generation_tokens=tokens,
            total_tokens=73 + tokens, prompt_tps=3.25, generation_tps=1.75,
            peak_memory=36.125, cached_tokens=0, finish_reason=finish,
        ) for text, tokens, finish in values]
        namespace["stream_generate"] = lambda *args, **kwargs: iter(chunks)
        reference = namespace["generate"](model, tokenizer, "frozen prompt", verbose=False)
        events = []
        actual = collect_stream(chunks, result_factory=result_type, peak_memory=lambda: 37.0,
                                progress=events.append)
        assert dataclasses.asdict(actual) == dataclasses.asdict(reference), name
        results.append({"case": name, "all_native_fields_equal": True,
                        "finish_reason": actual.finish_reason,
                        "generation_tokens": actual.generation_tokens,
                        "total_tokens": actual.total_tokens,
                        "progress_events": len(events)})
    assert resets == [[1, 2]] * len(cases)
    return {
        "scope": "CPU synthetic stream aggregation parity with installed released generate",
        "native_model_calls": 0, "accelerator_imports": 0,
        "real_model_stream_parity_verified": False,
        "limitations": "Does not prove bitwise real-model sampling or wall-clock timing parity",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources.values()},
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.mlx_package)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"cases": len(report["cases"]), "native_model_calls": 0}))


if __name__ == "__main__":
    main()
