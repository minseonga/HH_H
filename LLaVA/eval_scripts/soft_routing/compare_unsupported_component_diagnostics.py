import argparse
import csv
import os


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_run(value):
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        parent = os.path.dirname(path.rstrip(os.sep))
        name = os.path.basename(parent if os.path.basename(path) == "unsupported_component_diagnostic_summary" else path)
    summary_dir = path
    if os.path.basename(summary_dir.rstrip(os.sep)) != "unsupported_component_diagnostic_summary":
        candidate = os.path.join(summary_dir, "unsupported_component_diagnostic_summary")
        if os.path.isdir(candidate):
            summary_dir = candidate
    return name, summary_dir


def add_method(rows, method):
    output = []
    for row in rows:
        row = dict(row)
        row["method"] = method
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=diagnostic_summary_dir or method_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    layer_rows = []
    step_rows = []
    head_rows = []
    for item in args.run:
        method, summary_dir = parse_run(item)
        layer_rows.extend(add_method(read_csv(os.path.join(summary_dir, "unsupported_component_layer_call_summary.csv")), method))
        step_rows.extend(add_method(read_csv(os.path.join(summary_dir, "unsupported_component_step_summary.csv")), method))
        head_rows.extend(add_method(read_csv(os.path.join(summary_dir, "unsupported_component_selected_head_summary.csv")), method))

    write_csv(os.path.join(args.output_dir, "combined_layer_call_summary.csv"), layer_rows)
    write_csv(os.path.join(args.output_dir, "combined_step_summary.csv"), step_rows)
    write_csv(os.path.join(args.output_dir, "combined_selected_head_summary.csv"), head_rows)

    print("[summary] decode layer calls")
    for row in layer_rows:
        if row.get("group") == "phase:decode":
            print(row)
    print("[summary] decode steps")
    for row in step_rows:
        if row.get("group") == "phase:decode":
            print(row)
    print("[summary] top selected heads")
    for row in head_rows[:40]:
        if row.get("phase") == "all":
            print(row)


if __name__ == "__main__":
    main()
