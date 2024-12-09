import os
import sys
from concurrent.futures import ThreadPoolExecutor


app_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(
    os.path.join(app_path, "../src/")
)  # for server '/home/jonfunk/ProteusAI/src/'

is_zs_running = False
executor = ThreadPoolExecutor()
PAPER_URL = "https://www.biorxiv.org/content/10.1101/2024.10.01.616114v1"

VERSION = "version " + "0.1"
REP_TYPES = [
    "ESM-2",
    "ESM-1v",
    "One-hot",
    "BLOSUM50",
    "BLOSUM62",
]  # Add VAE and MSA-Transformer later
IN_MEMORY = ["BLOSUM62", "BLOSUM50", "One-hot"]
TRAIN_TEST_VAL_SPLITS = ["Random"]
MODEL_TYPES = ["KNN", "Gaussian Process", "Random Forrest", "Ridge", "SVM"]
MODEL_DICT = {
    "Random Forrest": "rf",
    "KNN": "knn",
    "SVM": "svm",
    "VAE": "vae",
    "ESM-2": "esm2",
    "ESM-1v": "esm1v",
    "Gaussian Process": "gp",
    "ESM-Fold": "esm_fold",
    "Ridge": "ridge",
}
REP_DICT = {
    "One-hot": "ohe",
    "BLOSUM50": "blosum50",
    "BLOSUM62": "blosum62",
    "ESM-2": "esm2",
    "ESM-1v": "esm1v",
    "VAE": "vae",
}
INVERTED_REPS = {v: k for k, v in REP_DICT.items()}
DESIGN_MODELS = {"ESM-IF": "esm_if"}
REP_VISUAL = ["UMAP", "t-SNE", "PCA"]
FAST_INTERACT_INTERVAL = 60  # in milliseconds
SIDEBAR_WIDTH = 450
BATCH_SIZE = 1
ZS_MODELS = ["ESM-1v", "ESM-2"]
FOLDING_MODELS = ["ESM-Fold"]
ACQUISITION_FNS = ["Expected Improvement", "Upper Confidence Bound", "Greedy"]
ACQ_DICT = {
    "Expected Improvement": "ei",
    "Upper Confidence Bound": "ucb",
    "Greedy": "greedy",
}
USR_PATH = os.path.join(app_path, "../usrs")
SEARCH_HEURISTICS = ["Diversity"]
OPTIM_DICT = {"Maximize Y-values": "max", "Minimize Y-values": "min"}
MAX_EVAL_DICT = {
    "ohe": 10000,
    "blosum62": 10000,
    "blosum50": 10000,
    "esm2": 200,
    "esm1v": 200,
}
