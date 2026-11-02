"""The checked-in tutorials show genuine completed runs when opened."""

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["CREDA", "MIL-CREDA"])
def test_tutorial_keeps_completed_execution_and_visible_results(method):
    path = ROOT / method / "Notebooks" / "Toyset.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert cells
    counts = [c.get("execution_count") for c in cells]
    assert all(isinstance(n, int) and n > 0 for n in counts), counts
    assert counts == sorted(set(counts)), counts
    outputs = [o for c in cells for o in c.get("outputs", [])]
    errors = [o for o in outputs if o["output_type"] == "error"]
    assert errors == [], errors
    images = [o for o in outputs if "image/png" in o.get("data", {})]
    assert images, "The saved run must show its figures without re-execution"
    assert all(o["data"]["image/png"] for o in images)
    visible = "\n".join(
        "".join(o.get("text", [])) +
        "".join(o.get("data", {}).get("text/plain", []))
        for o in outputs
    )
    for family in ("lunas", "toroide", "rollo"):
        assert family in visible, f"No saved result for {family}"
    if method == "CREDA":
        assert "v2" in visible, "Saved CREDA results must use the default v2"


@pytest.mark.parametrize("method", ["CREDA", "MIL-CREDA"])
def test_saved_input_data_plot_is_visible(method):
    path = ROOT / method / "Notebooks" / "Toyset.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"]
             if c["cell_type"] == "code" and "INPUT_FIGURE =" in "".join(c["source"])]
    assert len(cells) == 1
    assert any("image/png" in o.get("data", {})
               for o in cells[0].get("outputs", [])), "The input data overlay must be saved"


@pytest.mark.parametrize("method", ["CREDA", "MIL-CREDA"])
def test_saved_outputs_include_adaptation_and_confidence_gifs_for_each_family(method):
    path = ROOT / method / "Notebooks" / "Toyset.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    gif_cells = [
        [o["data"]["image/gif"] for o in c.get("outputs", [])
         if "image/gif" in o.get("data", {})]
        for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    # Adaptation and confidence each have one GIF per toy family.
    assert [len(gifs) for gifs in gif_cells if gifs] == [3, 3]
    for gifs in gif_cells:
        for encoded in gifs:
            with Image.open(io.BytesIO(base64.b64decode("".join(encoded)))) as gif:
                assert gif.n_frames > 1
                first = gif.convert("RGB").copy()
                gif.seek(gif.n_frames - 1)
                last = gif.convert("RGB")
                assert ImageChops.difference(first, last).getbbox() is not None
