from shiny import reactive
import pandas as pd

# Initialize reactive values
dummy = pd.DataFrame(
    {
        "Sequence": ["MGVARGTV...G", "AGVARGTV...G", "...", "MGVARGTV...V"],
        "Description": ["wt", "M1A", "...", "G142V"],
        "Activity": ["0.0", "0.32", "...", "-0.21"],
    }
)

MODE = reactive.Value("start")
DATASET = reactive.Value(dummy)
DATASET_PATH = reactive.Value("")
LIBRARY = reactive.Value(None)
REP_PATH = reactive.Value(None)
REPS_AVAIL = reactive.Value([])
PROTEIN = reactive.Value(None)
ZS_RESULTS = reactive.Value([])
CHAINS = reactive.Value(None)
Y_TYPE = reactive.Value(None)
_MODEL_TYPES = reactive.Value([])
DESIGN_OUTPUT = reactive.Value("start")
FIXED_RES = reactive.Value(None)
DESIGN_LIB = reactive.Value(None)
TSNE_DF = reactive.Value(None)
LIBRARY_PLOT = reactive.Value(None)
