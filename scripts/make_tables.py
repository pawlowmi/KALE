import json, os, glob, argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Print eval result tables vs openai baseline")
parser.add_argument("--eval_dir", default="/mnt/data/eval_results")
parser.add_argument("--openai_dir", default=None, help="Path to openai baseline results (default: eval_dir/ViT-L-14_openai)")
parser.add_argument("--exp", nargs=2, metavar=("NAME", "PATH"), action="append", default=[],
                    help="Add experiment: --exp paper-baseline /path/to/eval/exp_dir. Repeatable.")
parser.add_argument("--flat_exp", nargs=2, metavar=("NAME", "PATH"), action="append", default=[],
                    help="Add single-checkpoint flat dir as one column: --flat_exp paper-baseline /path/to/flat/dir")
parser.add_argument("--alias", nargs=2, metavar=("STEP", "NAME"), action="append", default=[["step_40000", "kuea-baseline"]],
                    help="Rename a step label within any experiment, e.g. --alias step_40000 kuea-baseline")
args = parser.parse_args()

EVAL_DIR = args.eval_dir
OPENAI_DIR = args.openai_dir or EVAL_DIR + "/ViT-L-14_openai"
PAPER_BASELINE_DIR = EVAL_DIR + "/ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy"
STEP_ALIASES = dict(args.alias)

# Default experiment if none specified
if not args.exp:
    args.exp = [["exp", EVAL_DIR + "/dev/ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy"]]


def load_results(directory):
    results = defaultdict(dict)
    for f in glob.glob(directory + "/*.json"):
        d = json.load(open(f))
        task, dataset, m = d["task"], d["dataset"].split("/")[-1], d["metrics"]
        if task == "zeroshot_classification":
            results["zeroshot"][dataset] = round(m["acc1"] * 100, 2)
        elif task == "linear_probe":
            results["lp"][dataset] = round(m["lp_acc1"] * 100, 2)
        elif task == "zeroshot_retrieval":
            r1 = (m["image_retrieval_recall@1"] + m["text_retrieval_recall@1"]) / 2
            results["retrieval"][dataset] = round(r1 * 100, 2)
    return results


def load_exp_cols(exp_name, exp_path):
    """Load all step_* and final subdirs as separate columns prefixed with exp_name."""
    cols = {}
    for step_dir in sorted(glob.glob(exp_path + "/step_*") + glob.glob(exp_path + "/final")):
        raw = os.path.basename(step_dir)
        label = STEP_ALIASES.get(raw, raw)
        col_name = f"{exp_name}/{label}" if len(args.exp) > 1 else label
        cols[col_name] = load_results(step_dir)
    return cols


def sort_key(col):
    part = col.split("/")[-1]
    if part == "final":
        return (2, float("inf"), "")
    if part.startswith("step_"):
        try:
            return (1, int(part.split("_")[1]), col)
        except ValueError:
            pass
    return (0, 0, col)  # named cols (e.g. paper-baseline) sort first


def print_table(task_name, openai, all_cols):
    col_labels = sorted(all_cols.keys(), key=sort_key)
    all_datasets = sorted(set(
        list(openai.get(task_name, {}).keys()) +
        [ds for c in all_cols.values() for ds in c.get(task_name, {}).keys()]
    ))
    cols = ["openai"] + col_labels

    data = {}
    for ds in all_datasets:
        row = {"openai": openai.get(task_name, {}).get(ds)}
        for lbl in col_labels:
            row[lbl] = all_cols[lbl].get(task_name, {}).get(ds)
        data[ds] = row

    avgs = {}
    for col in cols:
        vals = [data[ds][col] for ds in all_datasets if data[ds].get(col) is not None]
        avgs[col] = sum(vals) / len(vals) if vals else None

    openai_avg = avgs["openai"]
    col_w = 20
    ds_w = 34
    sep = "-" * (ds_w + col_w * len(cols))
    header = ("%-*s" % (ds_w, "Dataset")) + "".join(("%*s" % (col_w, c)) for c in cols)
    print()
    print("  " + task_name.upper())
    print(sep)
    print(header)
    print(sep)
    for ds in all_datasets:
        row_str = "%-*s" % (ds_w, ds)
        for col in cols:
            v = data[ds].get(col)
            row_str += "%*.2f" % (col_w, v) if v is not None else "%*s" % (col_w, "N/A")
        print(row_str)
    print(sep)
    avg_str = "%-*s" % (ds_w, "avg")
    for col in cols:
        v = avgs.get(col)
        if v is None:
            avg_str += "%*s" % (col_w, "N/A")
        elif col == "openai":
            avg_str += "%*.2f" % (col_w, v)
        else:
            delta = v - openai_avg
            sign = "+" if delta >= 0 else ""
            cell = "%.2f(%s%.2f)" % (v, sign, delta)
            avg_str += "%*s" % (col_w, cell)
    print(avg_str)


openai = load_results(OPENAI_DIR)
all_cols = {}
if os.path.isdir(PAPER_BASELINE_DIR):
    all_cols["paper-baseline"] = load_results(PAPER_BASELINE_DIR)
for exp_name, exp_path in args.exp:
    all_cols.update(load_exp_cols(exp_name, exp_path))
for exp_name, exp_path in args.flat_exp:
    all_cols[exp_name] = load_results(exp_path)

for task in ["zeroshot", "lp", "retrieval"]:
    print_table(task, openai, all_cols)
