"""Face images -> MultiViewSpec -> STEP, STL and G-code, from the command line.

  uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch --out tmp/mv
  add --set envelope.z_mm=20 when it asks for a dimension; --no-ai skips TrOCR, TripoSR and the vision model."""
import argparse
from pathlib import Path

from s2c.config import load_env
from s2c.multiview.pipeline import ImageInput, MvPipeline, default_pipeline
from s2c.multiview.spec import MvAbstain


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", required=True,
                    help="PATH[@FACE[@KIND]]; FACE front|back|left|right|top|bottom, KIND sketch|photo|drawing")
    ap.add_argument("--reference", help="1 TND, 1 EUR, 2 EUR, card or a4")
    ap.add_argument("--set", action="append", default=[], help="FIELD=VALUE, for example envelope.z_mm=20")
    ap.add_argument("--reject", action="append", default=[], help="canonical face to replace by a rectangle")
    ap.add_argument("--out", default="tmp/mv")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()

    images = []
    for arg in args.image:
        path, face, kind = (arg.split("@") + [None, None])[:3]
        images.append(ImageInput(Path(path).read_bytes(), face, kind))
    values = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    pipe = MvPipeline() if args.no_ai else default_pipeline()

    observed = pipe.observe(images, args.reference)
    if isinstance(observed, MvAbstain):
        raise SystemExit(f"{observed.stage}: {observed.reason}. {observed.remedy}")
    spec = pipe.fuse(observed, values, rejected=args.reject)
    if isinstance(spec, MvAbstain):
        hint = (spec.partial or {}).get("suggested")
        raise SystemExit(f"{spec.stage}: {spec.reason}. {spec.remedy}" + (f" Suggested: {hint}" if hint else ""))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(spec.model_dump_json(indent=2))
    res = pipe.build(spec, out, observed.masks)
    if isinstance(res, MvAbstain):
        raise SystemExit(f"{res.stage}: {res.reason}. {res.remedy}")

    print(f"spec    {out / 'spec.json'}\nSTEP    {res.step}\nSTL     {res.stl}\nG-code  {res.gcode or 'none'}")
    for face in ("front", "top", "right"):
        print(f"{face:<7} {getattr(spec.views, face).source} ({observed.filled_by.get(face, '')})")
    amber = {p: s for p, s in spec.provenance.items() if s in ("scaled", "inferred", "estimated", "default")}
    for path, source in amber.items():
        print(f"check   {path} ({source})")
    for face, score in res.iou.items():
        print(f"IoU     {face} {score:.2f}")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
