import json, os, glob
from collections import defaultdict

EVAL_DIR = "/mnt/data/eval_results"
OPENAI_DIR = EVAL_DIR + "/ViT-L-14_openai"
DEV_DIR = EVAL_DIR + "/dev/ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy"

STEP_ALIASES = {
    "step_40000": "kuea-baseline",
}

def load_results(directory):
    results = defaultdict(dict)
    for f in glob.glob(directory + "/*.json"):
        d = json.load(open(f))
        task = d["task"]
        dataset = d["dataset"].split("/")[-1]
        m = d["metrics"]
        if task == "zeroshot_classification":
            results["zeroshot"][dataset] = round(m["acc1"] * 100, 2)
        elif task == "linear_probe":
            results["lp"][dataset] = round(m["lp_acc1"] * 100, 2)
        elif task == "zeroshot_retrieval":
            r1 = (m["image_retrieval_recall@1"] + m["text_retrieval_recall@1"]) / 2
            results["retrieval"][dataset] = round(r1 * 100, 2)
    return results

def load_step_results(base_dir):
    steps = {}
    for step_dir in sorted(glob.glob(base_dir + "/step_*")):
        raw = os.path.basename(step_dir)
        label = STEP_ALIASES.get(raw, raw)
        steps[label] = load_results(step_dir)
    return steps

def print_table(task_name, openai, paper_steps):
    step_labels = sorted(paper_steps.keys(),
                         key=lambda x: int(x.split("_")[1]) if x.startswith("step_") else float("inf"))
    all_datasets = sorted(set(
        list(openai.get(task_name, {}).keys()) +
        [ds for s in paper_steps.values() for ds in s.get(task_name, {}).keys()]
    ))
    cols = ["openai"] + step_labels

    data = {}
    for ds in all_datasets:
        row = {"openai": openai.get(task_name, {}).get(ds)}
        for lbl in step_labels:
            row[lbl] = paper_steps[lbl].get(task_name, {}).get(ds)
        data[ds] = row

    avgs = {}
    for col in cols:
        vals = [data[ds][col] for ds in all_datasets if data[ds].get(col) is not None]
        avgs[col] = sum(vals) / len(vals) if vals else None

    openai_avg = avgs["openai"]
    col_w = 18
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
paper_steps = load_step_results(DEV_DIR)

for task in ["zeroshot", "lp", "retrieval"]:
    print_table(task, openai, paper_steps)
