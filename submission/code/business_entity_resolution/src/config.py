"""Paths and global settings for the entity-resolution pipeline."""
import os

ROOT = os.environ.get(
    "ER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA_DIR = os.environ.get("ER_DATA", os.path.join(ROOT, "dataset"))
WORK_DIR = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))
OUT_DIR = os.environ.get("ER_OUT", os.path.join(ROOT, "output"))
N_JOBS = int(os.environ.get("ER_JOBS", os.cpu_count() or 4))

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def src_path(split, k):
    return os.path.join(DATA_DIR, split, f"{split}_source{k}.tsv")


def gt_path():
    return os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")


def work(name):
    return os.path.join(WORK_DIR, name)
