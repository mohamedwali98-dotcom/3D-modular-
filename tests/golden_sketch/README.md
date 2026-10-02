# Golden hand-drawn sheets

One folder per sheet: `image.jpg` (or `.png`) and `expected.json`. No people, no personal data.

```json
{
  "views": ["top", "front", "right"],
  "envelope": {"x": 100, "y": 50, "z": 25},
  "dimensions": [
    {"view": "front", "value": 100, "kind": "linear", "axis": "a", "span_mm": [0, 100]},
    {"view": "top", "value": 12.5, "kind": "diameter"}
  ],
  "holes": [{"axis": "y", "diameter": 12.5, "through": true, "at": [12.5, 12.5]}]
}
```

- `span_mm`: the two positions the value measures, in mm from the lowest position of that view axis.
- `holes[].at`: the hole centre on the two axes across the hole, in the team frame (x, y, z order, hole axis left out; Z has the front face at Z = depth).
- Real sheets need real readers: run with `SKETCH_GOLDEN=1` and `SKETCH_READERS` set.

## Scoring the shipped path

`uv run python scripts/golden_eval.py` reads every folder that has an image through the path the web app runs
(`sheet_read.read_drawing`, then the pipeline) with TrOCR alone, and prints what it got: faces named, values read,
envelope sizes right or silently wrong, and whether the part builds. `--check` fails when a number fell more than 2
points below `baseline.json`; `--write-baseline` records today's numbers after you add samples. Two optional keys
serve it: `"projection"` (`"first"` or `"third"`, what the user would pick; default `"auto"`) and `"envelope"` with
only the sizes you know for sure.

## Real-photo slot

No real phone photos are in the set yet. To add one: make `tests/golden_sketch/<name>/`, put the photo there as
`image.jpg` and write its `expected.json` in the format above. `tests/sketch/test_golden.py` picks up every folder
that has both files (run with `SKETCH_GOLDEN=1`), and `scripts/sketch_accuracy.py` reports on them. The synthetic
bridge block has no image on disk: its sheet is drawn by the test.
