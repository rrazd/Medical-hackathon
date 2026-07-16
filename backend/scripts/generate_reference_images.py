from pathlib import Path
import csv
import random

from PIL import Image, ImageDraw, ImageFont


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"
CSV_PATH = DATA_ROOT / "reference_cases.csv"
IMAGE_SIZE = (320, 220)
OUTCOME_SEVERITY_DELTA = {
    "responder": 0.65,
    "partial_responder": 0.35,
    "non_responder": 0.08,
}


def _draw_placeholder(path: Path, case_id: str, stage: str, severity: float, seed: int) -> None:
    random.seed(seed)
    width, height = IMAGE_SIZE
    base = int(210 - 80 * severity)
    image = Image.new("RGB", IMAGE_SIZE, (base, base + 8, base + 14))
    draw = ImageDraw.Draw(image)

    for _ in range(90):
        x0 = random.randint(0, width - 30)
        y0 = random.randint(0, height - 20)
        rx = random.randint(12, 42)
        ry = random.randint(8, 30)
        intensity = int(80 + 120 * severity + random.randint(-12, 12))
        color = (
            min(255, intensity + 25),
            max(35, int(130 - 45 * severity) + random.randint(-10, 10)),
            max(40, int(115 - 55 * severity) + random.randint(-10, 10)),
        )
        draw.ellipse((x0, y0, x0 + rx, y0 + ry), fill=color)

    for x in range(0, width, 16):
        shade = int(235 - 45 * severity) if (x // 16) % 2 == 0 else int(205 - 55 * severity)
        draw.line((x, 0, x, height), fill=(shade, shade, min(255, shade + 10)), width=1)

    label = f"SYNTHETIC {stage.upper()}"
    font = ImageFont.load_default()
    draw.rectangle((8, 8, 168, 30), fill=(245, 245, 245), outline=(40, 40, 40))
    draw.text((14, 14), label, fill=(20, 20, 20), font=font)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(30, 30, 30), width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=88)


def main() -> None:
    with CSV_PATH.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for index, row in enumerate(rows, start=1):
        before_severity = max(0.15, min(0.95, float(row["inflammation_score"])))
        after_severity = max(0.05, before_severity - OUTCOME_SEVERITY_DELTA[row["outcome_label"]])
        before_path = DATA_ROOT / row["before_image_path"]
        after_path = DATA_ROOT / row["after_image_path"]
        _draw_placeholder(before_path, row["case_id"], "before", before_severity, index * 100 + 1)
        _draw_placeholder(after_path, row["case_id"], "after", after_severity, index * 100 + 2)
        print(f"generated {before_path.relative_to(DATA_ROOT)}")
        print(f"generated {after_path.relative_to(DATA_ROOT)}")


if __name__ == "__main__":
    main()
