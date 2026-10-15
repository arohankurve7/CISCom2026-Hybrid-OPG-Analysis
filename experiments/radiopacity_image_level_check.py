from pathlib import Path
from pipeline.hybrid_inference import HybridOPGInference

TEST_DIR = Path("datasets/fdi_500/test/images")

pipeline = HybridOPGInference()

total = 0
positive = 0
negative = 0

print("=" * 72)
print("IMAGE-LEVEL RESTORATIVE MATERIAL CANDIDATE CHECK")
print("=" * 72)

for image_path in sorted(TEST_DIR.glob("*")):
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue

    result = pipeline.run(image_path)
    components = result["radiopacity"]["components"]

    present = len(components) > 0

    total += 1
    positive += int(present)
    negative += int(not present)

    print(
        f"{image_path.name:12s} | "
        f"{'YES' if present else 'NO ':3s} | "
        f"candidates={len(components)}"
    )

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print("Total images :", total)
print("YES          :", positive)
print("NO           :", negative)
