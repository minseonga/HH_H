import argparse
import csv
import html
import os


BLUE = "#2563eb"
ORANGE = "#f97316"
GRAY = "#64748b"
DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def text(x, y, value, size=10, fill=DARK, anchor="start", weight="400", rotate=None):
    value = html.escape(str(value))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{value}</text>\n'
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def rect(x, y, w, h, fill, stroke=None, width=1):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"{stroke_attr}/>\n'


def polyline(points, stroke, width=2.2):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>\n'


def circle(x, y, r, fill, stroke="white", width=0.8):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>\n'


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_svg(rows, output_path):
    width, height = 640, 390
    left, top = 58, 76
    plot_w, plot_h = 472, 226
    metrics = [
        ("system_share", "system prefix", GRAY),
        ("visual_share", "visual", BLUE),
        ("text_share", "text-side", ORANGE),
    ]
    by_layer = {int(float(row["layer"])): row for row in rows}
    n_layers = max(by_layer) + 1 if by_layer else 32

    def sx(layer):
        return left + layer / max(n_layers - 1, 1) * plot_w

    def sy(value):
        return top + plot_h - value * plot_h

    body = []
    body.append(text(width / 2, 28, "Vanilla attention by source region", 17, DARK, "middle", "700"))
    body.append(text(width / 2, 48, "Generated-step attention averaged by layer; dashed lines mark L9-L16.", 9, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 10, y + 3, f"{tick:.2f}", 8, MUTED, "end"))

    for layer in range(n_layers):
        x = sx(layer)
        if layer in [9, 16]:
            body.append(line(x, top, x, top + plot_h, "#334155", 1.15, "5 5"))
        elif layer % 4 == 0:
            body.append(line(x, top, x, top + plot_h, "#edf2f7", 0.8))
        if layer % 4 == 0 or layer in [9, 16]:
            body.append(text(x, top + plot_h + 18, f"L{layer}", 7, DARK, "middle", rotate=35))

    for metric, _, color in metrics:
        pts = [
            (sx(layer), sy(float(by_layer.get(layer, {}).get(metric, 0.0))))
            for layer in range(n_layers)
        ]
        body.append(polyline(pts, color))
        for layer, (x, y) in enumerate(pts):
            value = float(by_layer.get(layer, {}).get(metric, 0.0))
            if value > 0.01:
                body.append(circle(x, y, 2.2, color))

    lx, ly = left + plot_w - 126, top + 18
    for idx, (metric, label, color) in enumerate(metrics):
        y = ly + idx * 20
        body.append(line(lx, y, lx + 24, y, color, 2.4))
        body.append(circle(lx + 12, y, 2.6, color))
        body.append(text(lx + 31, y + 3, label, 9, DARK))

    body.append(text(left + plot_w / 2, height - 22, "layer", 10, DARK, "middle"))
    body.append(text(20, top + plot_h / 2, "attention share", 10, DARK, "middle", rotate=-90))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write("".join(body))
        f.write("</svg>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-csv", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or os.path.join(
        os.path.dirname(args.layer_csv),
        "vanilla_region_attention_layer_lines_compact.svg",
    )
    make_svg(load_rows(args.layer_csv), output)
    print(output)


if __name__ == "__main__":
    main()
