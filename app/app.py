# This source code is part of the proteusAI package and is distributed
# under the MIT License.

"""
ProteusAI Shiny App.
"""

__name__ = "ProteusAI"
__author__ = "Jonathan Funk"

import asyncio
import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import tooltips
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData
import shinywidgets as widgets
from proteusAI.io_tools.fasta import hash_sequence
import proteusAI.visual_tools.plots

import proteusAI as pai


app_path = os.path.dirname(os.path.realpath(__file__))
google_analytics = os.path.join(app_path, "google_analytics.html")
google_analytics_string = ""

try:
    with open(google_analytics, "r") as f:
        google_analytics_string = f.read()
except Exception as e:
    print(f"No google analytics file found:{e}")

is_zs_running = False
executor = ThreadPoolExecutor()

VERSION = (
    "Version " + "0.5 (Beta Version: please contact jonfu@dtu.dk in case of bugs). "
)
REP_TYPES = [
    "ESM-2 (650M)",
    "ESM-2 (150M)",
    "ESM-2 (35M)",
    "ESM-2 (8M)",
    "ESM-1v",
    "One-hot",
    "BLOSUM50",
    "BLOSUM62",
]  # Add VAE and MSA-Transformer later
IN_MEMORY = ["BLOSUM62", "BLOSUM50", "One-hot", "VHSE"]
TRAIN_TEST_VAL_SPLITS = ["Random"]
MODEL_TYPES = ["KNN", "Random Forest", "Ridge", "SVM", "Gaussian Process"]
MODEL_DICT = {
    "Random Forest": "rf",
    "KNN": "knn",
    "SVM": "svm",
    "VAE": "vae",
    "VHSE": "vhse",
    "ESM-2 (650M)": "esm2_650M",
    "ESM-2 (150M)": "esm2_150M",
    "ESM-2 (35M)": "esm2_35M",
    "ESM-2 (8M)": "esm2_8M",
    "ESM-1v": "esm1v",
    "Gaussian Process": "gp",
    "ESM-Fold": "esm_fold",
    "Ridge": "ridge",
    "K-means": "k_means",
    "HDBSCAN": "hdbscan",
}
INV_MODEL_DICT = {value: key for key, value in MODEL_DICT.items()}
REP_DICT = {
    "One-hot": "ohe",
    "BLOSUM50": "blosum50",
    "BLOSUM62": "blosum62",
    "VHSE": "vhse",
    "ESM-2 (650M)": "esm2_650M",
    "ESM-2 (150M)": "esm2_150M",
    "ESM-2 (35M)": "esm2_35M",
    "ESM-2 (8M)": "esm2_8M",
    "ESM-1v": "esm1v",
    "VAE": "vae",
}
INVERTED_REPS = {v: k for k, v in REP_DICT.items()}
DESIGN_MODELS = {"ESM-IF": "esm_if"}
REP_VISUAL = ["UMAP", "t-SNE", "PCA"]
REP_VISUAL_DICT = {
    "UMAP": "umap",
    "t-SNE": "tsne",
    "PCA": "pca",
}
FAST_INTERACT_INTERVAL = 60  # in milliseconds
SIDEBAR_WIDTH = 450
BATCH_SIZE_DICT = {
    "ohe": 10000,
    "blosum62": 10000,
    "blosum50": 10000,
    "esm2_650M": 100,
    "esm2_150M": 100,
    "esm2_35M": 100,
    "esm2_8M": 100,
    "esm1v": 200,
    "vhse": 10000,
}
ZS_MODELS = ["ESM-1v", "ESM-2 (650M)", "ESM-2 (150M)", "ESM-2 (35M)", "ESM-2 (8M)"]
FOLDING_MODELS = ["ESM-Fold"]
ACQUISITION_FNS = ["Expected Improvement", "Upper Confidence Bound", "Greedy"]
CLUSTERING_ALG = ["HDBSCAN"]
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
    "esm2_650M": 200,
    "esm2_150M": 200,
    "esm2_35M": 500,
    "esm2_8M": 1000,
    "esm1v": 200,
    "vhse": 10000,
}

PAPER_URL = "https://www.biorxiv.org/content/10.1101/2024.10.01.616114v1"

app_ui = ui.page_fluid(
    ui.tags.head(ui.HTML(google_analytics_string)),
    ui.output_image("image", inline=True),
    VERSION,
    ui.HTML(f'<a href="{PAPER_URL}" target="_blank">Please cite our paper.</a>'),
    # ui.output_text("current_time", inline=True),  # for debugging
    ###############
    ## DATA PAGE ##
    ###############
    ui.navset_card_tab(
        ui.nav_panel(
            "Upload Data",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.row(
                        ui.column(
                            6,
                            ui.input_text(
                                "USER",
                                "Enter user name or proceed as guest",
                                value="Guest",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button("login", "Login"),
                            style="padding:50px;",
                        ),
                        ui.column(
                            6,
                            ui.input_action_button("sign_up", "Create account"),
                        ),
                    ),
                    ui.h4("Upload data using one of these tabs"),
                    ### NAVSET ###
                    ui.navset_tab(
                        ui.nav_panel(
                            "Library",
                            ui.input_checkbox("demo_library_check", "Use Demo Data"),
                            ui.panel_conditional(
                                "!input.demo_library_check",
                                ui.input_file(
                                    id="library_file",
                                    label="Upload Dataset",
                                    accept=[".csv", ".xlsx", ".xls"],
                                    placeholder="None",
                                ),
                            ),
                            ui.panel_conditional(
                                "input.demo_library_check",
                                ui.input_select(
                                    "library_demo_data",
                                    "Select MLDE or Discovery Demo",
                                    ["MLDE", "Discovery"],
                                ),
                            ),
                            ui.h6("Select which columns to extract from the dataset"),
                            ui.row(
                                ui.column(
                                    6, ui.input_select("seq_col", "Sequence column", [])
                                ),
                                ui.column(
                                    6,
                                    ui.input_select(
                                        "description_col", "Description column", []
                                    ),
                                ),
                                ui.column(6, ui.input_select("y_col", "Y-values", [])),
                                ui.column(
                                    6,
                                    ui.input_select(
                                        "y_type",
                                        "Data type",
                                        ["Numeric", "Categorical"],
                                    ),
                                ),
                            ),
                            ui.input_action_button(
                                "confirm_library",
                                "Upload to continue",
                                style="padding:10px; width:400px; height:40px;"
                                "background-color:#00629b; color:white; border:none; border-radius:5px;",
                            ),
                        ),
                        ui.nav_panel(
                            "Sequence",
                            ui.input_checkbox("demo_sequence_check", "Use Demo Data"),
                            ui.panel_conditional(
                                "!input.demo_sequence_check",
                                ui.input_file(
                                    id="protein_file",
                                    label="Upload FASTA",
                                    accept=[".fasta"],
                                    placeholder="None",
                                ),
                            ),
                            ui.input_action_button(
                                "confirm_sequence",
                                "Upload to continue",
                                style="padding:10px; width:400px; height:40px;"
                                "background-color:#00629b; color:white; border:none; border-radius:5px;",
                            ),
                        ),
                        ui.nav_panel(
                            "Structure",
                            ui.input_checkbox("demo_structure_check", "Use Demo Data"),
                            ui.panel_conditional(
                                "!input.demo_structure_check",
                                ui.input_file(
                                    id="structure_file",
                                    label="Upload Structure",
                                    accept=[".pdb"],
                                    placeholder="None",
                                ),
                            ),
                            ui.input_action_button(
                                "confirm_structure",
                                "Upload to continue",
                                style="padding:10px; width:400px; height:40px;"
                                "background-color:#00629b; color:white; border:none; border-radius:5px;",
                            ),
                        ),
                    ),
                    width=SIDEBAR_WIDTH,
                ),
                ### MAIN PANEL ###
                ui.input_switch("tooltips_switch", "How to use ProteusAI", False),
                ui.panel_conditional(
                    "input.tooltips_switch",
                    ui.h4("What is ProteusAI?"),
                    ui.p(
                        tooltips.data_tooltips,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h4("What kind of data can I analyse using ProteusAI?"),
                    ui.h5("1. A csv or excel file with unlabelled sequences"),
                    ui.p(
                        tooltips.data_unlabelled_datasets_tooltip,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h5(
                        "2. A csv or excel file with sequences and a continuous variable"
                    ),
                    ui.p(
                        tooltips.data_continuous_datasets_tooltip,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h5(
                        "3. A csv or excel file with sequences that partially or fully labelled with a catagorical variable"
                    ),
                    ui.p(
                        tooltips.data_catagorical_datasets_tooltip,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h5("4. A fasta file containing a single protein sequence"),
                    ui.p(
                        tooltips.data_fasta_tooltip,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h5(
                        "5. A pdb file containing a protein structure and any associated ligands"
                    ),
                    ui.p(
                        tooltips.data_structure_tooltip,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.p(
                        "Upload experimental data (CSV or Excel file), a single protein (FASTA) or a PDB structure",
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    style="text-align: justify; margin-bottom:10px;",
                ),
                ui.panel_conditional(
                    "typeof output.protein_fasta !== 'string'",
                    ui.output_data_frame("dataset_table"),
                ),
                ui.output_text("protein_fasta"),
                ui.output_text("protein_struc"),
            ),
        ),
        #################
        ## DESIGN PAGE ##
        #################
        ui.nav_panel(
            "Structure Design",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(ui.output_ui("design_ui"), width=SIDEBAR_WIDTH),
                ### MAIN PANEL ###
                ui.panel_conditional(
                    "typeof output.protein_struc === 'string'",
                    ui.output_ui("struc3D_design"),
                    ui.output_text("fixed_res_text"),
                    ui.output_data_frame("design_out"),
                    ui.output_ui("design_download_ui"),
                ),
            ),
        ),
        ####################
        ## ZERO-SHOT PAGE ##
        ####################
        ui.nav_panel(
            "Compute Zero-Shots",
            ui.layout_sidebar(
                ui.sidebar(ui.output_ui("zero_shot_ui"), width=SIDEBAR_WIDTH),
                ui.panel_conditional(
                    "typeof output.protein_fasta === 'string'",
                    ui.output_ui("zs_download_ui"),
                ),
            ),
        ),
        ##########################
        ## REPRESENTATIONS PAGE ##
        ##########################
        ui.nav_panel(
            "Compute Representations",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(ui.output_ui("representations_ui"), width=SIDEBAR_WIDTH),
                ### MAIN PANEL ###
                ui.panel_conditional(
                    "typeof output.protein_struc === 'string'", ui.output_ui("struc3D")
                ),
                widgets.output_widget("tsne_plot"),
            ),
        ),
        ################
        ## MLDE PAGE ##
        ################
        ui.nav_panel(
            "MLDE",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("mlde_ui"),
                    ui.output_ui("mlde_search_ui"),
                    width=SIDEBAR_WIDTH,
                ),
                ### MAIN PANEL ###
                ui.output_ui("mlde_results_ui"),
            ),
        ),
        ####################
        ## DISCOVERY PAGE ##
        ####################
        ui.nav_panel(
            "Sequence Discovery",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.navset_tab(
                        ui.nav_panel("Clustering", ui.output_ui("discovery_ui_clu")),
                        ui.nav_panel(
                            "Classification",
                            ui.output_ui("discovery_ui_class"),
                            ui.output_ui("discovery_search_ui"),
                        ),
                    ),
                    width=SIDEBAR_WIDTH,
                ),
                ### MAIN PANEL ###
                ui.navset_tab(
                    ui.nav_panel(
                        "Model Diagnostics",
                        widgets.output_widget("discovery_plot"),
                        ui.output_data_frame("discovery_table"),
                    ),
                    ui.nav_panel(
                        "Search Results",
                        widgets.output_widget("discovery_search_plot"),
                        ui.output_ui("discovery_download_ui"),
                    ),
                ),
            ),
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    ##############
    ## FRONTEND ##
    ##############

    ##################
    ### DESIGN TAB ###
    ##################
    @output
    @render.ui
    def design_ui():
        if MODE() == "structure":
            return (
                ui.TagList(
                    ui.h4("Structure Based Protein Design"),
                    ui.p(
                        tooltips.design_tooltips,
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.h4("Step 1: Choose a model and the sampling temperature"),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_select(
                                "design_models",
                                "Choose a model",
                                list(DESIGN_MODELS.keys()),
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_numeric(
                                "sampling_temp",
                                "Sampling temperature",
                                min=10e-9,
                                value=0.1,
                            ),
                        ),
                    ),
                    ui.h4(
                        "Step 2: Select which chain to redesign and which positions to fix"
                    ),
                    ui.p(
                        "Postions to fix must be comma separated, e.g. 1, 2, 3",
                        style="font-size:14px; margin-top:10px; text-align: justify;",
                    ),
                    ui.row(
                        ui.column(6, ui.output_ui("design_chains")),
                        ui.column(
                            6,
                            ui.input_text(
                                "design_res",
                                "Positions to fix",
                            ),
                        ),
                    ),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_checkbox(
                                "design_protein_interface",
                                "Fix positions at protein-protein interfaces",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_numeric(
                                "design_protein_interface_distance",
                                "Distance (Å)",
                                value=7,
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_checkbox(
                                "design_ligand_interface",
                                "Fix positions at ligand interfaces",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_numeric(
                                "design_ligand_interface_distance",
                                "Distance (Å)",
                                value=7,
                            ),
                        ),
                    ),
                    ui.h4("Step 3: Design new sequences"),
                    ui.row(
                        ui.column(
                            12,
                            ui.input_numeric(
                                "n_designs",
                                "Select the desired number of new sequences",
                                min=1,
                                value=20,
                            ),
                        ),
                    ),
                    ui.row(
                        ui.column(
                            12,
                            ui.input_task_button(
                                "desgin_button",
                                "Design",
                                style="padding:10px; width:400px; height:40px; "
                                "background-color:#00629b; color:white; border:none; border-radius:5px;",
                            ),
                        ),
                    ),
                    ui.h4("Step 4: Download new sequences"),
                    ui.row(
                        ui.column(
                            12,
                            ui.download_button(
                                "download_designs",
                                "Download",
                                style="padding:10px; width:400px; height:40px; "
                                "background-color:#00629b; color:white; border:none; border-radius:5px;",
                            ),
                        ),
                    ),
                ),
            )
            # DISABLED FOLDING
            # ui.output_ui(
            #        "folding",
            #    ),

        else:
            return ui.TagList(
                "Upload a protein structure in the 'Data' tab to proceed with the Design module."
            )

    ###########################
    ### REPRESENTATIONS TAB ###
    ###########################
    @output
    @render.ui
    def representations_ui():
        if MODE() != "start":
            return ui.TagList(
                ui.h4("Representation Learning"),
                ui.p(
                    tooltips.representations_tooltips,
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.h4("Step 1: Choose a model compute the representations"),
                ui.row(
                    ui.column(
                        7,
                        ui.input_select("dat_rep_type", "Choose model", REP_TYPES),
                    ),
                    ui.column(
                        5,
                        ui.input_task_button(
                            "dat_compute_reps",
                            "Compute",
                            style="padding:10px; width:150px; height:40px; "
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                        style="padding:25px;",
                    ),
                ),
                ui.column(6, ui.output_ui("rep_chain_ui")),
                ui.h4("Step 2: Choose a method to visualize the representations"),
                ui.p(
                    "Plots can be downloaded by clicking the camera icon at the top right of the plot.",
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.panel_conditional(
                    "input.dat_rep_type === 'VAE' || input.dat_rep_type === 'MSA-Transformer'",
                    ui.input_file("MSA_vae_training", "Upload MSA file"),
                ),
                ui.panel_conditional(
                    "input.dat_rep_type === 'VAE'",
                    ui.input_checkbox("custom_vae", "Customize VAE parameters"),
                    ui.input_action_button("train_vae", "Train VAE"),
                ),
                ui.input_select(
                    "vis_method", "Choose a visualization Method", REP_VISUAL
                ),
                ui.row(
                    ui.column(
                        7,
                        ui.input_select(
                            "plot_rep_type", "Choose a representation set", REPS_AVAIL()
                        ),
                    ),
                    ui.column(
                        5,
                        ui.input_task_button(
                            "update_plot",
                            "Plot",
                            style="padding:10px; width:150px; height:40px; "
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                        style="padding:25px;",
                    ),
                ),
            )

        else:
            return ui.TagList(
                "Upload a library in the 'Data' tab or compute a library in the 'Zero-shot' tab to proceed with the Representations module."
            )

    #####################
    ### ZERO-SHOT TAB ###
    #####################
    @output
    @render.ui
    def zero_shot_ui():
        if MODE() == "zero-shot" or MODE() == "structure":
            return ui.TagList(
                ui.h4("Compute a Zero-Shot Library"),
                ui.p(
                    tooltips.zs_tooltips,
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.h4("Step 1: Choose a model to calculate Zero-Shot Scores"),
                ui.row(
                    ui.column(
                        7, ui.input_select("zs_model", "Choose a model", ZS_MODELS)
                    ),
                    ui.column(
                        5,
                        ui.input_task_button(
                            "compute_zs",
                            "Compute",
                            style="padding:10px; width:150px; height:40px; "
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                        style="padding:25px;",
                    ),
                    ui.column(6, ui.output_ui("zs_chain_ui")),
                ),
                ui.h4("Step 2: Choose a Zero-Shot dataset to plot"),
                ui.row(
                    ui.column(
                        7,
                        ui.input_select(
                            "computed_zs_scores",
                            "Choose Zero-Shot Scores",
                            COMP_ZS_SCORES(),
                        ),
                    ),
                    ui.column(
                        5,
                        ui.input_action_button(
                            "plot_all",
                            "Plot",
                            style="padding:10px; width:150px; height:40px; "
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                        style="padding:25px;",
                    ),
                ),
                ui.h4("Step 3: Download tabulated results"),
                ui.p(
                    "Plots can be downloaded by clicking the camera icon at the top right of the plot.",
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.download_button(
                    "download_zs_df",
                    "Download Zero-Shot results table",
                    style="padding:10px; background-color:#00629b; color:white; border:none; border-radius:5px;",
                ),
            )

        else:
            return ui.TagList(
                tooltips.zs_file_type,
            )

    ################
    ### MLDE tab ###
    ################
    @output
    @render.ui
    def mlde_ui():
        if Y_TYPE() == "class":
            return ui.TagList(
                "The MLDE workflow is available for numerical Y-Values. Please use the Discovery workflow for categorical Y-values"
            )

        elif MODE() != "start":
            return ui.TagList(
                ui.h4("Machine Learning Guided Directed Evolution (MLDE)"),
                ui.p(
                    tooltips.mlde_tooltips,
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.h4("Step 1: Train a model to predict your Y-values"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            "model_type", "Surrogate model", _MODEL_TYPES()
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "model_rep_type", "Representation type", REPS_AVAIL()
                        ),
                    ),
                    ui.column(6, ui.output_ui("mlde_chain_ui")),
                    ui.column(6, ui.output_ui("mlde_dynamic_ui")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "random_seed", "Random seed", min=0, max=1024, value=42
                        ),
                    ),
                    ui.column(
                        6,
                        ui.panel_conditional(
                            "input.model_type !== 'Gaussian Process'",
                            ui.input_numeric(
                                "k_folds",
                                "K-Fold cross validation",
                                value=1,
                                min=1,
                            ),
                        ),
                    ),
                    ui.column(12, "Cross-validation split:"),
                ),
                ui.input_checkbox("smart_split", "Use smart data split", True),
                ui.panel_conditional(
                    "!input.smart_split",
                    ui.row(
                        ui.column(
                            4,
                            ui.input_numeric(
                                "n_train", "Training (%)", value=80, min=0, max=100
                            ),
                        ),
                        ui.column(
                            4,
                            ui.input_numeric(
                                "n_test", "Test (%)", value=10, min=0, max=100
                            ),
                        ),
                        ui.column(
                            4,
                            ui.input_numeric(
                                "n_val", "Validation (%)", value=10, min=0, max=100
                            ),
                        ),
                    ),
                ),
                ui.row(
                    ui.column(
                        12,
                        ui.input_task_button(
                            "mlde_train_button",
                            "Train",
                            style="padding:10px; width:400px; height:40px; "
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                    )
                ),
                ui.h4("Step 2: Design new sequences"),
            )

        else:
            return ui.TagList(tooltips.mlde_file_type)

    ### MLDE SEARCH UI ###
    @output
    @render.ui
    def mlde_search_ui(alt=None):
        if MODEL() is not None:
            model_type = INV_MODEL_DICT[MODEL().model_type]
            return ui.TagList(
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            "acquisition_fn", "Acquisition Function", ACQUISITION_FNS
                        ),
                    ),
                    ui.column(
                        6, ui.input_select("search_model", "Model", [model_type])
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "optim_problem",
                            "Optimization problem",
                            ["Maximize Y-values", "Minimize Y-values"],
                        ),
                    ),
                ),
                ui.column(
                    12,
                    ui.input_slider(
                        "mlde_explore",
                        "Exploration vs. Exploitation",
                        value=0.1,
                        min=0,
                        max=1,
                    ),
                ),
                ui.column(
                    12,
                    ui.input_task_button(
                        "mlde_search_btn",
                        "Design",
                        style="padding:10px; width:400px; height:40px; "
                        "background-color:#00629b; color:white; border:none; border-radius:5px;",
                    ),
                ),
                ui.column(
                    12,
                    ui.download_button(
                        "download_mlde_search",
                        "Download",
                        style="padding:10px; width:400px; height:40px; "
                        "background-color:#00629b; color:white; border:none; border-radius:5px;",
                    ),
                ),
            )

    @output
    @render.ui
    def mlde_results_ui(alt=None):
        if MODEL() is not None:
            return (
                ui.TagList(
                    ui.h4(
                        "Model Diagnostics: Predictions vs. True Values in the Validation Set"
                    ),
                    widgets.output_widget("pred_vs_true"),
                    ui.input_switch(
                        "validation_data_switch",
                        "Show validation data and model statistics",
                        False,
                    ),
                    ui.panel_conditional(
                        "input.validation_data_switch",
                        ui.h4("Descriptive Statistics for the Trained Model"),
                        ui.output_data_frame("mlde_statistics"),
                        ui.h4("Validation Data"),
                        ui.output_data_frame("mlde_model_table"),
                    ),
                    ui.h4("Designed Sequences For Testing"),
                    ui.output_ui("mlde_download_ui"),
                ),
            )

    @render.data_frame
    def mlde_statistics(alt=None):
        model = MODEL()
        if model is not None:
            calibration = round(model.calibration, 2)
            test_r2 = round(model.test_r2, 2)
            test_pearson = round(model.test_pearson[0], 2)
            # Add significance stars
            pearson_p = model.test_pearson[1]
            pearson_p_str = f"{pearson_p:.2e}"  # Format p-value in scientific notation

            test_ken_tau = round(
                model.test_ken_tau.statistic,
                2,
            )
            test_ken_tau_p = model.test_ken_tau.pvalue
            kendall_p_str = (
                f"{test_ken_tau_p:.2e}"  # Format p-value in scientific notation
            )

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "Metric": [
                        "Calibration error (90% confidence)",
                        "Validation R-squared value",
                        "Validation Pearson correlation",
                        "Kendall Tau correlation",
                    ],
                    "Value": [
                        f"+/- {calibration}",
                        test_r2,
                        f"{test_pearson} (p-value: {pearson_p_str})",
                        f"{test_ken_tau} (p-value: {kendall_p_str})",
                    ],
                    "Description": [
                        "The calibration error is a measure of model and experimental noise. A value of 0 indicates a perfect model and no experimental noise.",
                        "The R-squared value is a measure of how well the model fits the data. A value of 1 indicates a perfect fit.",
                        "The Pearson correlation measures the strength and direction of the predictions and actual values. A value of 1 indicates a perfect correlation.",
                        "The Kendall Tau correlation is a measure of how well the model ranks the data. A value of 1 indicates a perfect ranking.",
                    ],
                }
            )

            return df

    ###################
    ## DISCOVERY TAB ##
    ###################
    @output
    @render.ui
    def discovery_ui_clu(alt=None):
        if Y_TYPE() != "class":
            return ui.TagList(
                "The Discovery workflow is only available for categorical Y-Values. Please use the MLDE workflow for numerical Y-values"
            )

        elif MODE() == "dataset":
            return ui.TagList(
                ui.h4("Protein Discovery and Annotation"),
                ui.p(
                    tooltips.discovery_tooltips,
                    style="font-size:14px; margin-top:10px; text-align: justify;",
                ),
                ui.h4("Step 1: Train a model to cluster the sequences"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            "clustering_alg", "Clustering algorithm", CLUSTERING_ALG
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "clustering_rep", "Representations", REPS_AVAIL()
                        ),
                    ),
                ),
                ui.panel_conditional(
                    "input.clustering_alg === 'K-means'",
                    ui.input_checkbox("custom_k_means", "Customize K-means"),
                    ui.panel_conditional(
                        "input.custom_k_means",
                        ui.input_numeric(
                            "k_means_k", "K-number of clusters", value=5, min=0
                        ),
                    ),
                ),
                ui.panel_conditional(
                    "input.clustering_alg === 'HDBSCAN'",
                    ui.input_checkbox("custom_hdbscan", "Customize HDBSCAN"),
                    ui.panel_conditional(
                        "input.custom_hdbscan",
                        ui.row(
                            ui.column(
                                6,
                                ui.input_select(
                                    "hdbscan_dr", "Dimensionality Reduction", REP_VISUAL
                                ),
                            ),
                            ui.column(
                                6,
                                ui.input_numeric(
                                    "hdbscan_n_neighbors", "Cluster Density", 70, min=1
                                ),  # n_neighbors UMAP
                            ),
                            ui.column(
                                6,
                                ui.input_numeric(
                                    "hdbscan_min_cluster_size",
                                    "Minimum Cluster Size",
                                    30,
                                    min=1,
                                ),
                            ),
                            ui.column(
                                6,
                                ui.input_numeric(
                                    "hdbscan_min_samples", "Minimum Samples", 50, min=1
                                ),
                            ),
                        ),
                    ),
                ),
                ui.row(
                    ui.input_task_button(
                        "clustering_button",
                        "Cluster",
                        style="padding:10px; width:400px; height:40px;"
                        "background-color:#00629b; color:white; border:none; border-radius:5px;",
                    ),
                    style="padding:25px;",
                ),
                ui.output_ui("clustering_search_ui"),
            )

        else:
            return ui.TagList(tooltips.discovery_file_type)

    @output
    @render.ui
    def discovery_ui_class(alt=None):
        if Y_TYPE() != "class":
            return ui.TagList(
                "The Discovery workflow is only available for categorical Y-Values. Please use the MLDE workflow for numerical Y-values"
            )

        elif MODE() == "dataset":
            return ui.TagList(
                ui.h4("Protein Discovery and Annotation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            "discovery_model_type", "Surrogate model", _MODEL_TYPES()
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "discovery_model_rep_type",
                            "Representaion type",
                            REPS_AVAIL(),
                        ),
                    ),
                ),
                ui.row(
                    ui.panel_conditional(
                        "input.discovery_model_type !== 'Gaussian Process'",
                        ui.column(
                            6,
                            ui.input_numeric(
                                "discovery_k_folds",
                                "K-Fold cross validation",
                                value=5,
                                min=1,
                                max=10,
                            ),
                        ),
                    ),
                    ui.column(6, ui.output_ui("discovery_dynamic_ui")),
                ),
                ui.input_checkbox(
                    "discovery_model_params", "Customize model parameters", value=False
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider(
                            "discovery_random_seed",
                            "Random seed",
                            min=0,
                            max=1024,
                            value=42,
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "discovery_vis_method",
                            "Visualization method",
                            choices=REP_VISUAL,
                        ),
                    ),
                ),
                ui.row(
                    ui.column(12, "Cross-validation split:"),
                ),
                ui.row(
                    ui.column(
                        3,
                        ui.input_numeric(
                            "discovery_n_train",
                            "Training (%)",
                            value=80,
                            min=0,
                            max=100,
                        ),
                    ),
                    ui.column(
                        3,
                        ui.input_numeric(
                            "discovery_n_test", "Test (%)", value=10, min=0, max=100
                        ),
                    ),
                    ui.column(
                        3,
                        ui.input_numeric(
                            "discovery_n_val",
                            "Validation (%)",
                            value=10,
                            min=0,
                            max=100,
                        ),
                    ),
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_task_button(
                            "discovery_train_button",
                            "Train",
                            style="padding:10px; width:400px; height:40px;"
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                    ),
                ),
            )

        else:
            return ui.TagList(tooltips.discovery_file_type)

    ### DISCOVERY SEARCH UI ###
    @output
    @render.ui
    def discovery_search_ui(alt=None):
        if Y_TYPE() != "class":
            return ui.TagList(
                "The Discovery workflow is only available for categorical Y-Values. Please use the MLDE workflow for numerical Y-values"
            )

        elif MODE() == "dataset":
            model = DISCOVERY_MODEL()
            if (
                model is not None
                and INV_MODEL_DICT[model.model_type] not in CLUSTERING_ALG
            ):
                clusters = list(model.library.class_dict.values())
                sample_from = clusters
                model_type = INV_MODEL_DICT[DISCOVERY_MODEL().model_type]

                return (
                    ui.TagList(
                        ui.row(
                            ui.h4("Step 2: Choose sample sequences from clusters"),
                        ),
                        ui.row(
                            ui.column(
                                6,
                                ui.input_select(
                                    "discovery_search_criteria",
                                    "Search heuristic",
                                    SEARCH_HEURISTICS,
                                ),
                            ),
                            ui.column(
                                6,
                                ui.input_select(
                                    "discovery_search_model", "Model", [model_type]
                                ),
                            ),
                        ),
                        ui.row(
                            ui.column(
                                6,
                                ui.input_selectize(
                                    "sample_from",
                                    "Clusters of interest",
                                    sample_from,
                                    multiple=True,
                                ),
                            ),
                            ui.column(
                                6,
                                ui.input_numeric(
                                    "n_samples", "Number of sequences", value=10, min=2
                                ),
                            ),
                        ),
                        ui.column(
                            12,
                            ui.input_task_button("discovery_search", "Search"),
                            style="padding:25px;",
                        ),
                    ),
                )

    ### CLUSTERIN SEARCH UI ###
    @output
    @render.ui
    def clustering_search_ui(alt=None):
        model = DISCOVERY_MODEL()
        if model is not None and INV_MODEL_DICT[model.model_type] in CLUSTERING_ALG:
            clusters = [
                str(i) for i in set([prot.y_pred for prot in model.library.proteins])
            ]
            sample_from = clusters
            model_type = INV_MODEL_DICT[DISCOVERY_MODEL().model_type]

            return ui.TagList(
                ui.h4("Step 2: Choose sample sequences from clusters"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_select(
                            "discovery_search_criteria",
                            "Search heuristic",
                            SEARCH_HEURISTICS,
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "discovery_search_model", "Model", [model_type]
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_selectize(
                            "sample_from",
                            "Clusters of interest",
                            sample_from,
                            multiple=True,
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_numeric(
                            "n_samples", "Number of sequences", value=10, min=2
                        ),
                    ),
                    ui.column(
                        12,
                        ui.input_task_button(
                            "discovery_search",
                            "Search",
                            style="padding:10px; width:400px; height:40px;"
                            "background-color:#00629b; color:white; border:none; border-radius:5px;",
                        ),
                        style="padding:25px;",
                    ),
                ),
            )

    #######################
    ### Reactive Values ###
    #######################

    # DUMMY DATA
    dummy = pd.DataFrame(
        {
            "Sequence": [
                "MGVARGTV...G",
                "AGVARGTV...G",
                "AGVARGTV...G",
                "AGVARGTV...G",
                "...",
                "MGVARGTV...V",
            ],
            "Description": ["wt", "M1A", "D147I", "A176L", "...", "G142V"],
            "Class": ["A", None, "B", "A", "...", "A"],
            "Activity": ["0.0", "0.32", "0.67", "1.2", "...", "-0.21"],
        }
    )

    # INPUT
    MODE = reactive.Value("start")
    DATASET = reactive.Value(dummy)
    DATASET_PATH = reactive.Value(str)
    LIBRARY = reactive.Value(None)
    REP_PATH = reactive.Value(None)
    REPS_AVAIL = reactive.Value(REP_TYPES)
    PROTEIN = reactive.Value(None)
    ZS_RESULTS = reactive.Value([])
    CHAINS = reactive.Value(None)
    Y_TYPE = reactive.Value(None)
    _MODEL_TYPES = reactive.Value(MODEL_TYPES.copy())

    # DESIGN
    PROT_INTERFACE = reactive.Value(None)
    LIG_INTERFACE = reactive.Value(None)
    DESIGN_OUTPUT = reactive.Value("start")
    FIXED_RES = reactive.Value(None)
    DESIGN_LIB = reactive.Value(None)
    # FOLD_LIB = reactive.Value(None)

    # ZER0-SHOT
    ZS_SCORES = reactive.Value(pd.DataFrame())
    COMP_ZS_SCORES = reactive.Value([])

    # REPRESENTATIONS
    TSNE_DF = reactive.Value(None)
    LIBRARY_PLOT = reactive.Value(None)

    # MLDE
    MODEL = reactive.Value(None)
    DISCOVERY_MODEL = reactive.Value(None)
    TEST_DF = reactive.Value(
        pd.DataFrame({"names": [], "y_true": [], "y_pred": [], "y_sigma": []})
    )
    DISCOVERY_TEST_DF = reactive.Value(
        pd.DataFrame({"names": [], "y_true": [], "y_pred": [], "y_sigma": []})
    )
    DISCOVERY_LIB = reactive.Value(None)
    MLDE_SEARCH_DF = reactive.Value(None)

    # Discovery
    DISCOVERY_SEARCH = reactive.Value(None)
    DISCOVERY_DF = reactive.Value(None)
    DISCOVERY_MODEL_PLOT = reactive.Value(None)
    DISCOVERY_SEARCH_PLOT = reactive.Value(None)
    DISCOVERY_DR_DF = reactive.Value(None)

    ###############
    ### BACKEND ###
    ###############

    ### TO TEST ASYNCHRONOUS PROCESSES
    @render.text
    def current_time():
        reactive.invalidate_later(1)
        return datetime.datetime.now().strftime("%H:%M:%S %p")

    ### APP LOGO ###
    @output
    @render.image
    def image():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"), "height": "75px"}
        return img

    ########
    ## IO ##
    ########

    ### READING DATASET ###
    @reactive.Effect
    @reactive.event(input.library_file)
    def _():
        df = DATASET()
        f: list[FileInfo] = input.library_file()
        df = pd.read_csv(f[0]["datapath"])

        # set reactive variables
        DATASET.set(df)
        DATASET_PATH.set(f[0]["datapath"])

    ### READ MLDE DEMO ###
    def read_MLDE_demo():
        data_path = os.path.join(
            app_path, "../demo/demo_data/Nitric_Oxide_Dioxygenase.csv"
        )
        df = pd.read_csv(data_path)
        y_col = "Data"
        seqs_col = "Sequence"
        names_col = "Description"
        DATASET.set(df)
        DATASET_PATH.set(data_path)
        return seqs_col, names_col, y_col

    ### READ DISCOVERY DEMO ###
    def read_discovery_demo():
        data_path = os.path.join(app_path, "../demo/demo_data/methyltransfereases.csv")
        df = pd.read_csv(data_path)
        y_col = "coverage_5"
        seqs_col = "sequence"
        names_col = "uid"
        DATASET.set(df)
        DATASET_PATH.set(data_path)
        return seqs_col, names_col, y_col

    ### READ DEMO LIBRARY
    def demo_library():
        if input.library_demo_data() == "MLDE":
            seqs_col, names_col, y_col = read_MLDE_demo()
        elif input.library_demo_data() == "Discovery":
            seqs_col, names_col, y_col = read_discovery_demo()

    ### RENDER DATASET TABLE ###
    @output
    @render.data_frame
    def dataset_table():
        df = render.DataTable(DATASET(), summary=True)
        return df

    ### UPDATE DATASET
    @reactive.Effect
    @reactive.event(input.library_demo_data)
    def _():
        if input.demo_library_check():
            demo_library()

    ### UPDATE DATASET
    @reactive.Effect
    @reactive.event(input.demo_library_check)
    def _():
        if input.demo_library_check():
            demo_library()

    ### UPDATE COL SELECTION
    def update_col_selection(
        seq_col="Sequences", description_col="Descriptions", y_col="Y-values"
    ):
        cols = list(DATASET().columns)

        # set reactive variables
        ui.update_select("seq_col", label=seq_col, choices=cols, selected=cols[0])

        ui.update_select(
            "description_col", label=description_col, choices=cols, selected=cols[1]
        )

        ui.update_select("y_col", label=y_col, choices=cols, selected=cols[-1])

    ### EXTRACT DATASET COLUMNS ###
    @reactive.Effect
    def _():
        update_col_selection()

    ### CONFIRM LIBRARY ###
    @reactive.Effect
    @reactive.event(input.confirm_library)
    async def _():
        try:
            df = DATASET()
            if input.library_file() is None and not input.demo_library_check():
                with ui.Progress(min=1, max=15) as p:
                    p.set(
                        message="No data uploaded",
                        detail="Upload a library in the 'Data' tab to proceed with the MLDE module.",
                    )
                    time.sleep(2.5)

            else:
                if input.demo_library_check():
                    data_path = DATASET_PATH()
                    file_name = data_path.split("/")[-1]

                else:
                    f: list[FileInfo] = input.library_file()
                    file_name = f[0]["name"]
                    data_path = f[0]["datapath"]

                ys = df[input.y_col()].to_list()
                seqs_col = input.seq_col()
                y_col = input.y_col()
                names_col = input.description_col()

                # Determine if the data is numerical or categorical
                if all(pd.isna(ys)):  # Check if all values are NaN
                    y_type = "class"
                    choice = "Classification"
                    _y_type = "Categorical"
                    _MODEL_TYPES.set(
                        [x for x in MODEL_TYPES if x != "Gaussian Process"]
                    )
                elif is_numerical(ys):
                    y_type = "num"
                    choice = "Regression"
                    _y_type = "Numeric"
                    _MODEL_TYPES.set([x for x in MODEL_TYPES if x != "KNN"])
                else:
                    y_type = "class"
                    choice = "Classification"
                    _y_type = "Categorical"
                    _MODEL_TYPES.set(
                        [x for x in MODEL_TYPES if x != "Gaussian Process"]
                    )

                lib = pai.Library(
                    user=input.USER().lower(),
                    source=data_path,
                    seqs_col=seqs_col,
                    y_col=y_col,
                    y_type=y_type,
                    names_col=names_col,
                    fname=file_name,
                )
                # set reactive variables
                LIBRARY.set(lib)
                ui.update_select(
                    "model_rep_type", choices=[INVERTED_REPS[i] for i in lib.reps]
                )

                Y_TYPE.set(y_type)

                reps = [INVERTED_REPS[i] for i in lib.reps]

                for rep in IN_MEMORY:
                    if rep not in reps:
                        reps.append(rep)

                REPS_AVAIL.set(reps)

                ui.update_select("model_task", choices=[choice])

                ui.update_select("model_type", choices=_MODEL_TYPES())

                ui.update_select("discovery_model_type", choices=_MODEL_TYPES())

                ui.update_select("y_type", choices=[_y_type])

                PROTEIN.set(None)

                REP_PATH.set(None)  # used in train

                MODE.set("dataset")

                LIBRARY_PLOT.set(None)

                with ui.Progress(min=1, max=15) as p:
                    p.set(
                        message="Data loaded",
                    )
                    await asyncio.sleep(0.5)

        except Exception:
            with ui.Progress(min=1, max=15) as p:
                p.set(
                    message="Problem with input file",
                    detail="Please check if there are any problems with the input file.",
                )
                await asyncio.sleep(2.5)

    ### READING PROTEIN FILE ###
    @reactive.Effect
    @reactive.event(input.protein_file)
    def _():
        f: list[FileInfo] = input.protein_file()
        usr_path = os.path.join(USR_PATH, input.USER().lower())
        file_name = f[0]["name"]
        prot = pai.Protein(source=f[0]["datapath"], user=usr_path, fname=file_name)

        PROTEIN.set(prot)
        DATASET_PATH.set(f[0]["datapath"])

    ### CONFIRM PROTEIN ###
    @reactive.Effect
    @reactive.event(input.confirm_sequence)
    async def _():
        if input.protein_file() is None and not input.demo_sequence_check():
            with ui.Progress(min=1, max=15) as p:
                p.set(message="No data uploaded", detail="Upload data to continue")
                time.sleep(2.5)
        else:
            with ui.Progress(min=1, max=15) as p:
                LIBRARY.set(None)
                p.set(
                    message="Searching for available data...",
                    detail="This may take a while...",
                )

                usr_path = os.path.join(USR_PATH, input.USER().lower())

                if input.demo_sequence_check():
                    data_path = os.path.join(
                        app_path, "../demo/demo_data/Nitric_Oxide_Dioxygenase_wt.fasta"
                    )
                    file_name = data_path.split("/")[-1]

                else:
                    f: list[FileInfo] = input.protein_file()
                    data_path = f[0]["datapath"]
                    file_name = f[0]["name"]

                prot = pai.Protein(source=data_path, user=usr_path, fname=file_name)

                # set shiny variables
                PROTEIN.set(prot)

                DATASET_PATH.set(data_path)
                MODE.set("zero-shot")

                # if chains is list and len > 1, select first chain
                if prot.chains == []:
                    chain = prot.chains
                    seq_hash = hash_sequence(prot.seq)
                elif isinstance(prot.chains, list):
                    chain = prot.chains[0]
                    seq_hash = hash_sequence(prot.seq[chain])

                # check for zs-computations # TODO: test if the number of computations match with the number of sequences.
                zs_computed = []
                rep_computed = []
                for model in ZS_MODELS:
                    # check hash existence
                    zs_path = os.path.join(
                        prot.user, f"{prot.name}/zero_shot/results/{MODEL_DICT[model]}"
                    )

                    if os.path.exists(zs_path):
                        if f"{seq_hash}_zs_scores.csv" in os.listdir(zs_path):
                            zs_computed.append(model)

                    for rep in REP_TYPES:
                        rep_path = os.path.join(
                            prot.user, f"{prot.name}/zero_shot/rep/{REP_DICT[rep]}"
                        )
                        if os.path.exists(rep_path):
                            rep_computed.append(REP_DICT[rep])

                # load zs-library if exists # TODO: test if the number of computations match with the number of sequences.
                for model in ZS_MODELS:
                    try:
                        rep_path = os.path.join(
                            prot.user, f"{prot.name}/zero_shot/rep/{REP_DICT[model]}"
                        )
                        print(seq_hash)
                        print(
                            os.listdir(
                                os.path.join(
                                    prot.user,
                                    f"{prot.name}/zero_shot/results/{REP_DICT[model]}/",
                                )
                            )
                        )
                        df_path = os.path.join(
                            prot.user,
                            f"{prot.name}/zero_shot/results/{REP_DICT[model]}/{seq_hash}_zs_scores.csv",
                        )

                        if os.path.exists(df_path):
                            df = pd.read_csv(df_path)  # noqa: F841

                            p.set(
                                message="Loading data...",
                                detail="This may take a while...",
                            )

                            if not df.empty:
                                lib = pai.Library(
                                    user=usr_path,
                                    seqs=df.sequence,
                                    ys=df.mmp,
                                    y_type="num",
                                    names=df.mutant.to_list(),
                                    proteins=[],
                                    rep_path=rep_path,
                                )

                            # set reactive variables
                            LIBRARY.set(lib)
                            DATASET.set(df)
                            ZS_SCORES.set(df)
                        else:
                            print("Warning: DataFrame is empty, skipping...")

                    except Exception as e:
                        print(f"Error processing model {model}: {e}")

                rep_computed = list(set(rep_computed))
                for rep in IN_MEMORY:
                    if rep not in rep_computed:
                        rep_computed.append(REP_DICT[rep])

                # set reactive variables
                ui.update_select(
                    "model_rep_type", choices=[INVERTED_REPS[i] for i in rep_computed]
                )

                COMP_ZS_SCORES.set(zs_computed)

                REPS_AVAIL.set([INVERTED_REPS[i] for i in rep_computed])

                ZS_RESULTS.set(zs_computed)

                LIBRARY_PLOT.set(None)

                ZS_SCORES.set(pd.DataFrame())

                ui.update_select("model_task", choices=["Regression"])

                ui.update_select("zs_scores", choices=zs_computed)

                ui.update_select(
                    "model_type", choices=[x for x in MODEL_TYPES if x != "KNN"]
                )

    ### RENDER FASTA ###
    @output
    @render.text
    def protein_fasta():
        if PROTEIN() is None:
            seq = None
        elif isinstance(PROTEIN().seq, dict):
            seq = (
                "Protein name: "
                + PROTEIN().name
                + " \n".join(
                    [
                        " chain " + chain + ":\n" + PROTEIN().seq[chain]
                        for chain in PROTEIN().chains
                    ]
                )
            )
        else:
            seq = "Protein name: " + PROTEIN().name + "\n" + PROTEIN().seq
        return seq

    ### CONFIRM STRUCTURE ###
    @reactive.Effect
    @reactive.event(input.confirm_structure)
    async def _():
        if input.structure_file() is None and not input.demo_structure_check():
            with ui.Progress(min=1, max=15) as p:
                p.set(message="No data uploaded", detail="Upload data to continue")
                time.sleep(2.5)
        else:
            with ui.Progress(min=1, max=15) as p:
                p.set(
                    message="Searching for available data...",
                    detail="This may take a while...",
                )
                prot = PROTEIN()

                if input.demo_structure_check():
                    data_path = os.path.join(app_path, "../demo/demo_data/1zb6.pdb")
                    file_name = data_path.split("/")[-1]
                else:
                    f: list[FileInfo] = input.structure_file()
                    file_name = f[0]["name"]
                    data_path = f[0]["datapath"]

                usr_path = os.path.join(USR_PATH, input.USER().lower())
                prot = pai.Protein(source=data_path, user=usr_path, fname=file_name)

                name = file_name.split(".")[0]
                prot.name = name

                # load zs-library if exists
                computed_zs = []
                reps = []

                # load reps
                for rep in REP_TYPES:
                    rep_path = os.path.join(
                        prot.user, f"{name}/zero_shot/rep/{REP_DICT[rep]}"
                    )
                    p.set(message="Loading data...", detail="This may take a while...")
                    if os.path.exists(rep_path):
                        reps.append(rep)

                # load zs_scores
                for model in ZS_MODELS:
                    for chain in prot.chains:
                        seq_hash = hash_sequence(prot.seq[chain])
                        df_path = os.path.join(
                            prot.user,
                            f"{name}/zero_shot/results/{chain}/{REP_DICT[model]}/{seq_hash}_zs_scores.csv",
                        )
                        if os.path.exists(df_path):
                            df = pd.read_csv(df_path)  # noqa: F841
                            p.set(
                                message="Loading data...",
                                detail="This may take a while...",
                            )
                            computed_zs.append(model)

                # set reactive variables
                PROTEIN.set(prot)
                DATASET_PATH.set(data_path)
                MODE.set("structure")

                ui.update_select("model_rep_type", choices=reps)

                ui.update_select("computed_zs_scores", choices=computed_zs)

                ui.update_select(
                    "model_type", choices=[x for x in MODEL_TYPES if x != "KNN"]
                )

                for rep in IN_MEMORY:
                    if rep not in reps:
                        reps.append(rep)

                REPS_AVAIL.set(reps)

                COMP_ZS_SCORES.set(computed_zs)

                LIBRARY_PLOT.set(None)

                ZS_SCORES.set(pd.DataFrame())

                CHAINS.set(prot.chains)

    ### STRUCTURE OUTPUT CHECK ###
    @output
    @render.text
    def protein_struc():
        if PROTEIN() is None:
            struc = None
        elif PROTEIN().struc is not None:
            struc = "Structure loaded"
        else:
            struc = None

        return struc

    ############
    ## DESIGN ##
    ############

    ### DESIGN TAB OUTPUT CONTROL ###
    @render.ui
    def struc3D_design():
        out = DESIGN_OUTPUT()  # noqa: F841
        # if type(out) == str  or input.design_sidechains() == None:
        #    sidechains = []
        # else:
        #    sidechains = [int(''.join([char for char in item if char.isdigit()])) for item in input.design_sidechains()]

        highlights = list(set(input.design_res().strip().split(",")))  # + sidechains))

        if PROT_INTERFACE():
            highlights = highlights + [str(i) for i in PROT_INTERFACE()]

        if LIG_INTERFACE():
            highlights = highlights + [str(i) for i in LIG_INTERFACE()]

        highlights_dict = {input.mutlichain_chain(): highlights}

        view = PROTEIN().view_struc(
            color="white", highlight=highlights_dict
        )  # , sticks=sidechains)
        return ui.TagList(ui.HTML(view.write_html()))

    ### SELECT PROTEIN-PROTEIN INTERFACE ###
    @reactive.Effect
    @reactive.event(input.design_protein_interface)
    def _():
        prot = PROTEIN()
        if input.design_protein_interface():
            prot_contacts = prot.get_contacts(
                source_chain=input.mutlichain_chain(),
                dist=input.design_protein_interface_distance(),
                target="protein",
            )
            PROT_INTERFACE.set(
                prot_contacts
            )  # here will be a function that selects the interface values
        else:
            PROT_INTERFACE.set(None)

    ### SELECT PROTEIN LIGAND INTERFACE ###
    @reactive.Effect
    @reactive.event(input.design_ligand_interface)
    def _():
        prot = PROTEIN()
        if input.design_ligand_interface():
            prot_contacts = prot.get_contacts(
                source_chain=input.mutlichain_chain(),
                dist=input.design_protein_interface_distance(),
                target="ligand",
            )
            LIG_INTERFACE.set(
                prot_contacts
            )  # here will be a function that selects the interface values
        else:
            LIG_INTERFACE.set(None)

    ### DESIGN BUTTON LOGIC ###
    IS_DESIGN_RUNNING = reactive.Value(False)

    async def compute_design():
        # Prevent multiple invocations of the task within the same session
        if IS_DESIGN_RUNNING():
            print("Design computation is already in progress for this session.")
            return

        # Set task running state to True for this session
        IS_DESIGN_RUNNING.set(True)

        try:
            n_designs = int(input.n_designs())
            with ui.Progress(min=1, max=n_designs) as p:
                prot = PROTEIN()
                seq = prot.seq[input.mutlichain_chain()]
                p.set(
                    message="Initiating structure based design",
                    detail=f"Computing {n_designs} samples...",
                )
                out = DESIGN_OUTPUT()

                sidechains = []
                # if type(out) == str or input.design_sidechains() == None:
                #    sidechains = []
                # else:
                #    sidechains = [int(''.join([char for char in item if char.isdigit()])) for item in input.design_sidechains()]

                residues_str = list(
                    set(input.design_res().strip().split(",") + sidechains)
                )
                fixed_ids = [
                    int(r)
                    for r in residues_str
                    if r.strip()
                    and (
                        r.strip().isdigit()
                        or (r.strip()[1:].isdigit() if r.strip()[0] == "-" else False)
                    )
                ]

                if PROT_INTERFACE():
                    fixed_ids = fixed_ids + PROT_INTERFACE()

                if LIG_INTERFACE():
                    fixed_ids = fixed_ids + LIG_INTERFACE()

                fixed_ids = [i for i in set(fixed_ids) if not isinstance(i, str)]

                fixed_ids.sort()

                fixed = []
                if len(fixed_ids) > 0:
                    fixed = [seq[i - 1] + str(i) for i in fixed_ids if i < len(seq)]

                # Run the blocking function `prot.zs_prediction` in a separate thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()

                out = await loop.run_in_executor(
                    executor,
                    prot.esm_if,
                    fixed_ids,
                    input.mutlichain_chain(),
                    None,
                    float(input.sampling_temp()),
                    n_designs,
                )

                lib = pai.Library(user=prot.user, source=out)

                # set reactive values
                FIXED_RES.set(fixed)
                DESIGN_OUTPUT.set(out["df"])
                DESIGN_LIB.set(lib)

        except Exception as e:
            print(f"An error occurred in Design: {e}")

        finally:
            # Reset the task running state in the session
            IS_DESIGN_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.desgin_button)
    async def design_btn_click():
        # Launch the expensive computation asynchronously
        asyncio.create_task(compute_design())

    ### RENDER DESIGN DATAFRAME ###
    @output
    @render.data_frame
    def design_out():
        out = DESIGN_OUTPUT()
        if isinstance(out, str):
            return None
        else:
            return render.DataTable(out, summary=True)

    ### CHAIN SELECTION MENU ###
    @output
    @render.ui
    def design_chains():
        return ui.input_select(
            "mutlichain_chain", "Chain to redesign", choices=CHAINS()
        )

    # ### DOWNLOAD DESIGN RESULTS # BUTTON MOVED TO SIDE BAR
    # @output
    # @render.ui
    # def design_download_ui():
    #     out = DESIGN_OUTPUT()
    #     if not isinstance(out, str):
    #         return ui.download_button("download_designs", "Download design results")

    ### DOWNLOAD LOGIC FOR DESIGN RESULTS ###
    @render.download(filename=lambda: f"{PROTEIN().name}_designs.csv")
    def download_designs():
        yield DESIGN_OUTPUT().to_csv(index=False)

    ### OUTPUT TEXED FOR FIXED RESIDUES ###
    @output
    @render.text
    def fixed_res_text(alt=None):
        out = DESIGN_OUTPUT()
        if not isinstance(out, str):
            msg = "Residues that were fixed during design: \n" + ", ".join(FIXED_RES())
            return msg

    ###############
    ## ZERO-SHOT ##
    ###############

    ### COMPUTE ZS-SCORES ###
    IS_ZS_RUNNING = reactive.Value(False)

    async def compute_zs_scores(method, prot, zs_chain, computed_zs):

        if IS_ZS_RUNNING():
            print("ZS score computation is already running, skipping this invocation.")
            return

        # Set task running state to True for this session
        IS_ZS_RUNNING.set(True)

        try:
            if isinstance(prot.seq, dict):
                seq = prot.seq[zs_chain]
                chain = zs_chain
            else:
                seq = prot.seq
                chain = None

            with ui.Progress(min=1, max=len(seq)) as p:
                p.set(
                    message="Initiating structure based design",
                    detail=f"Computing {len(seq)} positions...",
                )

                # Get the model from REP_DICT
                model = REP_DICT[method]

                # Run the blocking function `prot.zs_prediction` in a separate thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                batch_size = BATCH_SIZE_DICT[model]

                data = await loop.run_in_executor(
                    executor,
                    prot.zs_prediction,
                    model,
                    batch_size,
                    None,
                    None,  # device
                    chain,
                )

                # Create a library based on the prediction data
                lib = pai.Library(user=prot.user, source=data)

                if method not in computed_zs:
                    computed_zs.append(method)

                ui.update_select("computed_zs_scores", choices=computed_zs)

                # Handle the computed ZS scores (update UI elements, reactive values, etc.)
                ui.update_select("computed_zs_scores", choices=computed_zs)
                LIBRARY.set(lib)
                DATASET.set(data["df"])
                ZS_SCORES.set(data["df"])

        except Exception as e:
            print(f"An error occurred in ZS prediction: {e}")

        finally:
            # Reset the task running state in the session
            IS_ZS_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.compute_zs)
    async def compute_zs_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_ZS_RUNNING():
            print("ZS score computation is already in progress for this session.")
            return

        prot = PROTEIN()
        if isinstance(prot.seq, dict):
            chain = input.zs_chain()
        else:
            chain = None

        # Launch the expensive computation asynchronously
        asyncio.create_task(
            compute_zs_scores(
                method=input.zs_model(),
                prot=prot,
                zs_chain=chain,
                computed_zs=COMP_ZS_SCORES(),
            )
        )

    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def zs_chain_ui(alt=None):
        if MODE() == "structure":
            return ui.TagList(
                ui.input_select("zs_chain", "Select Protein Chain", choices=CHAINS())
            )

    @reactive.Effect
    @reactive.event(input.zs_chain)
    def _():
        method = input.zs_model()
        chain = input.zs_chain()

        prot = PROTEIN()
        name = prot.name

        if chain is not None and len(prot.chains) >= 1:
            chain = input.zs_chain()
            seq_hash = hash_sequence(prot.seq[chain])
            s = f"{name}/zero_shot/results/{chain}/{REP_DICT[method]}/{seq_hash}_zs_scores.csv"
        else:
            chain = None
            seq_hash = hash_sequence(prot.seq)
            s = f"{name}/zero_shot/results/{REP_DICT[method]}/{seq_hash}_zs_scores.csv"

        df_path = os.path.join(prot.user, s)
        if os.path.exists(df_path):
            COMP_ZS_SCORES.set([method])
        else:
            COMP_ZS_SCORES.set([])

        # dumb but necessary
        ui.update_select("zs_chain", selected=chain)

    ### RENDER ZS-DATAFRAME ###
    @output
    @render.data_frame
    @reactive.event(input.plot_all)
    def zs_df(alt=None):
        prot = PROTEIN()
        method = REP_DICT[input.computed_zs_scores()]
        if prot.chains is not None and len(prot.chains) >= 1:
            seq_hash = hash_sequence(prot.seq[input.zs_chain()])
            path = os.path.join(
                prot.zs_path,
                "results",
                input.zs_chain(),
                method,
                f"{seq_hash}_zs_scores.csv",
            )
        else:
            seq_hash = hash_sequence(prot.seq)
            path = os.path.join(
                prot.zs_path, "results", method, f"{seq_hash}_zs_scores.csv"
            )
        df = pd.read_csv(path)

        try:
            df = df.rename(
                columns={
                    "name": "Mutation",
                    "p": "Mutation Probability",
                    "mmp": "Zero-Shot Score",
                    "entropy": "Entropy",
                }
            )
        except Exception:
            # Legacy, will be phased out
            df = df.rename(
                columns={
                    "mutant": "Mutation",
                    "p": "Mutation Probability",
                    "mmp": "Zero-Shot Score",
                    "entropy": "Entropy",
                }
            )

        ZS_SCORES.set(df)
        df = df.drop("sequence", axis=1)
        return df

    ### DOWNLOAD ZS RESULTS ###
    @output
    @render.ui
    def zs_download_ui():
        out = ZS_SCORES()
        if out is not None:
            return ui.TagList(
                # Section for Entropy Plot
                ui.h4("Plot of Entropy Scores for Each Position"),
                ui.input_switch("entropy_plot_switch", "Show more information", False),
                ui.panel_conditional(
                    "input.entropy_plot_switch",
                    tooltips.zs_entropy_tooltips,
                    style="text-align: justify; margin-bottom:10px;",
                ),
                widgets.output_widget("entropy_plot"),
                # Section for Heatmap Plot
                ui.h4("Heatmap of Zero-Shot Scores"),
                ui.input_switch("heatmap_switch", "Show more information", False),
                ui.panel_conditional(
                    "input.heatmap_switch",
                    tooltips.zs_heatmap_tooltips,
                    style="text-align: justify; margin-bottom:10px;",
                ),
                widgets.output_widget("heatmap_plot"),
                # Descriptors
                ui.h4("Table of Zero-Shot Scores"),
                ui.row(
                    ui.column(3, ui.h5("Mutation:")),
                    ui.column(9, ui.h6("Sequence change from wild-type descriptor")),
                    ui.column(3, ui.h5("Mutation Probability:")),
                    ui.column(
                        9,
                        ui.h6(
                            "Predicted probability of the new amino acid at the position"
                        ),
                    ),
                    ui.column(3, ui.h5("Zero-Shot Score:")),
                    ui.column(
                        9,
                        ui.h6(
                            "Log probability of the wild type vs the mutant (higher is better)."
                        ),
                    ),
                    ui.column(3, ui.h5("Entropy:")),
                    ui.column(
                        9, ui.h6("Measure of the diversity tolerated at the position.")
                    ),
                    ui.column(
                        12,
                        ui.h6("Click column header to sort the table by that column."),
                    ),
                ),
                ui.output_data_frame("zs_df"),
            )

    ### DOWNLOAD LOGIC FOR ZS RESULTS ###
    @render.download(
        filename=lambda: f"{PROTEIN().name}_{input.zs_model()}_zs_predictions.csv"
    )
    def download_zs_df():
        print(ZS_SCORES())
        yield ZS_SCORES().to_csv(index=False)

    ### OUTPUT PROTEIN MODE ###
    @output
    @widgets.render_widget
    @reactive.event(input.plot_all)
    def entropy_plot(alt=None):
        """
        Create the per position entropy plot
        """
        prot = PROTEIN()
        if isinstance(prot.seq, dict):
            chain = input.zs_chain()
            seq = prot.seq[chain]
        else:
            seq = prot.seq
            chain = None

        start = 0
        end = len(seq)

        section = (start, end)
        if section[1] > len(seq):
            assert section[1] > section[0]
            width = section[1] - section[0]
            section = (len(seq) - width, len(seq))

        fig = prot.plot_entropy(
            section=section, model=MODEL_DICT[input.computed_zs_scores()], chain=chain
        )
        return fig

    ### UPDATE SCORES PLOT ###
    @output
    @widgets.render_widget
    @reactive.event(input.plot_all)
    def heatmap_plot(alt=None):
        """
        Create heatmap of zeroshot scores
        """
        prot = PROTEIN()

        if isinstance(prot.seq, dict):
            chain = input.zs_chain()
            seq = prot.seq[chain]
        else:
            seq = prot.seq
            chain = None

        start = 0
        end = len(seq)

        section = (start, end)

        if section[1] > len(seq):
            assert section[1] > section[0]
            width = section[1] - section[0]
            section = (len(seq) - width, len(seq))

        fig = prot.plot_scores(
            section=section,
            color_scheme="rwb",
            model=MODEL_DICT[input.computed_zs_scores()],
            chain=chain,
        )

        return fig

    ### STRUCTURE MODE ###
    @render.ui
    def struc3D():
        view = PROTEIN().view_struc(color="confidence")  # TODO: Add coloring options
        return ui.TagList(ui.HTML(view.write_html()))

    #####################
    ## REPRESENTATIONS ##
    #####################

    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def rep_chain_ui(alt=None):
        if MODE() == "structure":
            return ui.TagList(
                ui.input_select("rep_chain", "Select Protein Chain", choices=CHAINS())
            )

    ### COMPUTE REPRESENTATIONS ###
    IS_REP_COMP_RUNNING = reactive.Value(False)

    async def compute_reps():

        if IS_REP_COMP_RUNNING():
            print(
                "Representation computation is already running, skipping this invocation."
            )
            return

        # Set task running state to True for this session
        IS_REP_COMP_RUNNING.set(True)

        mode = MODE()
        lib = LIBRARY()
        prot = PROTEIN()
        method = MODEL_DICT[input.dat_rep_type()]

        if mode == "structure":
            chain = input.rep_chain()
        else:
            chain = None

        # if no library was loaded one has to be created
        if mode in ["zero-shot", "structure"]:
            data = prot.zs_library(model=method, chain=chain)
            lib = pai.Library(user=prot.user, source=data)

            pbar_max = len(lib)
        else:
            pbar_max = len(lib)

        try:
            with ui.Progress(min=1, max=pbar_max) as p:
                print(f"Computing library: {REP_DICT[input.dat_rep_type()]}")
                p.set(
                    message="Computation in progress",
                    detail=f"Computing representations: {REP_DICT[input.dat_rep_type()]}",
                )
                batch_size = BATCH_SIZE_DICT[method]

                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(
                    executor, lib.compute, method, batch_size
                )

                LIBRARY.set(lib)

                # update representation selection
                ui.update_select(
                    "model_rep_type", choices=[INVERTED_REPS[i] for i in lib.reps]
                )

                reps = [INVERTED_REPS[i] for i in lib.reps]
                for rep in IN_MEMORY:
                    if rep not in reps:
                        reps.append(rep)
                REPS_AVAIL.set(reps)

        except Exception as e:
            print(f"An error occurred when computing reps: {e}")

        finally:
            # Reset the task running state in the session
            IS_REP_COMP_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.dat_compute_reps)
    async def compute_reps_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_REP_COMP_RUNNING():
            print(
                "Representations computation is already in progress for this session."
            )
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(compute_reps())

    ### UPDATE REPRESENTATIONS PLOT ###
    IS_REP_PLOT_RUNNING = reactive.Value(False)

    async def plot_reps():
        """
        Render plot once button is pressed.
        """
        if input.plot_rep_type():
            IS_REP_PLOT_RUNNING.set(True)
            with ui.Progress(min=1, max=15) as p:
                p.set(message="Plotting", detail="This may take a while...")

                lib = LIBRARY()
                mode = MODE()
                prot = PROTEIN()

                # if no library was loaded one has to be created
                if mode in ["zero-shot", "structure"] and lib is None:
                    data = prot.zs_library(model=REP_DICT[input.plot_rep_type()])
                    lib = pai.Library(user=prot.user, source=data)

                names = lib.names
                rep = REP_DICT[input.plot_rep_type()]

                # Update to pass the new parameters
                try:
                    loop = asyncio.get_running_loop()
                    if input.vis_method() == "t-SNE":
                        # rep: str, y_upper=None, y_lower=None, names=None, highlight_mask=None, highlight_label=None
                        fig, ax, df = await loop.run_in_executor(
                            executor, lib.plot_tsne, rep, None, None, names
                        )
                    elif input.vis_method() == "UMAP":
                        fig, ax, df = await loop.run_in_executor(
                            executor, lib.plot_umap, rep, None, None, names
                        )
                    elif input.vis_method() == "PCA":
                        fig, ax, df = await loop.run_in_executor(
                            executor, lib.plot_pca, rep, None, None, names
                        )

                    TSNE_DF.set(df)
                    LIBRARY_PLOT.set(fig)

                except Exception as e:
                    print(f"An error occurred: {e}")

                finally:
                    # Reset the task running state in the session
                    IS_REP_PLOT_RUNNING.set(False)
        else:
            pass

    # Button click event
    @reactive.effect
    @reactive.event(input.update_plot)
    async def plot_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_REP_PLOT_RUNNING():
            print(
                "Representations computation is already in progress for this session."
            )
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(plot_reps())

    ### RENDER REPRESENTATIONS PLOT ###
    @output
    @widgets.render_widget
    def tsne_plot(alt=None):
        if LIBRARY_PLOT():
            fig = LIBRARY_PLOT()
            return fig

    ##########
    ## MLDE ##
    ##########

    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def mlde_chain_ui(alt=None):
        if MODE() == "structure":
            return ui.TagList(
                ui.input_select("mlde_chain", "Select Protein Chain", choices=CHAINS())
            )

    ### MLDE TAB OUTPUT CONTROL ###
    @output
    @render.ui
    def mlde_dynamic_ui():
        if MODE() in ["zero-shot", "structure"]:

            prot = PROTEIN()
            name = prot.name
            chain = input.mlde_chain()

            computed_zs = []
            for model in ZS_MODELS:
                seq_hash = hash_sequence(prot.seq[input.mlde_chain()])
                df_path = os.path.join(
                    prot.user,
                    f"{name}/zero_shot/results/{chain}/{REP_DICT[model]}/{seq_hash}_zs_scores.csv",
                )
                if os.path.exists(df_path):
                    computed_zs.append(model)

            COMP_ZS_SCORES.set(computed_zs)

            return ui.TagList(
                ui.input_select(
                    "mlde_computed_zs_scores",
                    "Choose zero-shot scores",
                    COMP_ZS_SCORES(),
                )
            )

        if MODE() == "dataset":
            return None

    ### N-TRAIN COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.n_train)
    def _():
        n_train = input.n_train()

        n_test_max = 100 - n_train

        new_test = round((n_test_max) / 2, 2)
        ui.update_numeric("n_test", min=0, max=n_test_max, value=new_test)
        new_val_max = 100 - n_train - new_test
        ui.update_numeric("n_val", min=0, max=new_val_max, value=new_val_max)

    ### N-TEST COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.n_test)
    def _():
        n_train = input.n_train()
        n_test = input.n_test()
        n_val_max = 100 - n_train - n_test
        ui.update_numeric("n_val", max=n_val_max, value=n_val_max)

    ### MLDE TRAIN ###
    IS_MLDE_TRAINING_RUNNING = reactive.Value(False)

    async def train_mlde_model():

        IS_MLDE_TRAINING_RUNNING.set(True)

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")

            rep_type = REP_DICT[input.model_rep_type()]
            prot = PROTEIN()

            if MODE() == "structure":
                f: list[FileInfo] = input.structure_file()  # noqa: F841
            else:
                f: list[FileInfo] = input.library_file()  # noqa: F841

            lib = LIBRARY()

            if MODE() == "structure":
                chain = input.mlde_chain()
            else:
                chain = None

            if MODE() in ["zero-shot", "structure"]:
                data = prot.zs_library(
                    model=MODEL_DICT[input.mlde_computed_zs_scores()], chain=chain
                )
                lib = pai.Library(user=prot.user, source=data)

            smart_split = input.smart_split()
            if smart_split:
                split = "smart"
            else:
                split = (input.n_train(), input.n_test(), input.n_val())

            k_folds = input.k_folds()
            if k_folds <= 1:
                k_folds = None

            model = pai.Model(
                model_type=MODEL_DICT[input.model_type()],
                library=lib,
                rep=rep_type,
                split=split,
                seed=input.random_seed(),
                k_folds=k_folds,
            )

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, model.train)
                print("done")
                test_df = pd.DataFrame(
                    {
                        "names": model.test_names,
                        "y_true": model.y_test,
                        "y_pred": model.y_test_pred,
                        "y_sigma": model.y_test_sigma,
                    }
                )

                search_dest = os.path.join(
                    f"{model.library.rep_path}",
                    f"../models/{model.model_type}/{model.rep}/predictions",
                )
                search_file = os.path.join(
                    search_dest, f"{model.model_type}_{model.rep}_predictions.csv"
                )

                if os.path.exists(search_file):
                    MLDE_SEARCH_DF.set(pd.read_csv(search_file))

                # set reactive variables
                MODEL.set(model)
                TEST_DF.set(test_df)

            except Exception as e:
                print(f"An error occurred in training MLDE model: {e}")

            finally:
                # Reset the task running state in the session
                IS_MLDE_TRAINING_RUNNING.set(False)
                with ui.Progress(min=1, max=15) as p:
                    p.set(message="Done...", detail="")

    @reactive.effect
    @reactive.event(input.mlde_train_button)
    async def mlde_train_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_MLDE_TRAINING_RUNNING():
            print("MLDE model training is already in progress for this session.")

        # Launch the expensive computation asynchronously
        asyncio.create_task(train_mlde_model())

    ### PREPARE PREDICTED VERSUS TRUE PLOT ###
    @output
    @render.ui
    def pred_vs_true_ui():
        hover_opts_kwargs = {}
        hover_opts_kwargs["delay"] = FAST_INTERACT_INTERVAL
        hover_opts_kwargs["delay_type"] = "throttle"

        return ui.output_plot("pred_vs_true", hover=ui.hover_opts(**hover_opts_kwargs))

    ### RENDER PREDICTED VERSUS TRUE DATAFRAME ###
    @output
    @widgets.render_widget
    def pred_vs_true(alt=None):
        df = TEST_DF()
        y_true = df.y_true.to_list()
        y_pred = df.y_pred.to_list()
        y_names = df.names.to_list()
        if MODEL() is not None:
            p = MODEL().true_vs_predicted(y_pred=y_pred, y_true=y_true, y_names=y_names)

        # p = proteusAI.visual_tools.plots.plot_predictions_vs_groundtruth_interactive(y_pred=y_pred,y_true=y_true, y_names=y_names)
        else:
            p = None
        return p

    ### RENDER MLDE TABLE ###
    @output
    @render.data_frame
    def mlde_model_table(alt=None):
        model = MODEL()
        if model is not None:
            table = TEST_DF()
            if "y_sigma" in table.columns:
                # if y_sigma column is empty, drop it
                if table["y_sigma"].isnull().all():
                    table = table.drop(["y_sigma"], axis=1)
            return table

    #################
    ## MLDE SEARCH ##
    #################
    IS_MLDE_SEARCH_RUNNING = reactive.Value(False)

    async def mlde_search():
        IS_MLDE_SEARCH_RUNNING.set(True)
        with ui.Progress(min=1, max=15) as p:
            p.set(
                message="Searching for new mutants",
                detail="Preparing genetic algorithm...",
            )

            model = MODEL()
            mlde_explore = input.mlde_explore()
            optim_problem = OPTIM_DICT[input.optim_problem()]
            max_eval = MAX_EVAL_DICT[model.rep]
            acq_fn = ACQ_DICT[input.acquisition_fn()]
            batch_size = BATCH_SIZE_DICT[model.rep]

            try:
                loop = asyncio.get_running_loop()
                out = await loop.run_in_executor(
                    executor,
                    model.search,
                    10,  # top N proteins
                    ["all"],
                    optim_problem,
                    "ga",
                    max_eval,
                    mlde_explore,
                    batch_size,
                    None,
                    acq_fn,
                    False,  # accumulate previous results
                )

                MLDE_SEARCH_DF.set(out)

            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                # Reset the task running state in the session
                IS_MLDE_SEARCH_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.mlde_search_btn)
    async def mlde_search_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_MLDE_SEARCH_RUNNING():
            print("MDLE search is already in progress for this session.")
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(mlde_search())

    ### RENDER MLDE TABLE ###
    @output
    @render.data_frame
    def mlde_search_table(alt=None):
        table = MLDE_SEARCH_DF()
        model = MODEL()  # noqa: F841

        table = table.drop(["sequence"], axis=1)
        if "y_true" in table.columns:
            table = table.drop(["y_true"], axis=1)
        if "y_sigma" in table.columns:
            # if y_sigma column is empty, drop it
            if table["y_sigma"].isnull().all():
                table = table.drop(["y_sigma"], axis=1)

        return table

    ### DOWNLOAD MLDE RESULTS ###
    @output
    @render.ui
    def mlde_download_ui():
        out = MLDE_SEARCH_DF()
        if out is not None:
            return ui.TagList(
                ui.output_data_frame("mlde_search_table"),
                # ui.download_button("download_mlde_search", "Download MLDE results"),
            )

    ### DOWNLOAD LOGIC FOR MLDE RESULTS ###
    @render.download(
        filename=lambda: f"{MODEL().model_type}_{MODEL().x}_predictions.csv"
    )
    def download_mlde_search():
        yield MLDE_SEARCH_DF().to_csv(index=False)

    ###############
    ## DISCOVERY ##
    ###############

    ### N-TRAIN COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.discovery_n_train)
    def _():
        n_train = input.discovery_n_train()

        n_test_max = 100 - n_train

        new_test = round((n_test_max) / 2, 2)

        ui.update_numeric("discovery_n_test", min=0, max=n_test_max, value=new_test)
        new_val_max = 100 - n_train - new_test
        ui.update_numeric("discovery_n_val", min=0, max=new_val_max, value=new_val_max)

    ### N-TEST COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.discovery_n_test)
    def _():
        n_train = input.discovery_n_train()
        n_test = input.discovery_n_test()
        n_val_max = 100 - n_train - n_test

        ui.update_numeric("n_val", max=n_val_max, value=n_val_max)

    ###############
    ## DISCOVERY ##
    ###############

    ### TRAIN DISCOVERY MODEL ###
    IS_DISCOVERY_TRAIN_RUNNING = reactive.Value(False)

    async def discovery_train():
        IS_DISCOVERY_TRAIN_RUNNING.set(True)

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")

            rep_type = REP_DICT[input.discovery_model_rep_type()]
            lib = LIBRARY()

            split = (
                input.discovery_n_train(),
                input.discovery_n_test(),
                input.discovery_n_val(),
            )
            k_folds = input.discovery_k_folds()
            if k_folds <= 1:
                k_folds = None

            vis_method = REP_VISUAL_DICT[input.discovery_vis_method()]

            model = pai.Model(
                model_type=MODEL_DICT[input.discovery_model_type()],
                library=lib,
                rep=rep_type,
                split=split,
                seed=input.discovery_random_seed(),
                k_folds=k_folds,
            )
            try:
                # train model
                loop = asyncio.get_running_loop()
                out = await loop.run_in_executor(executor, model.train)

                model_lib = pai.Library(user=lib.user, source=out)
                test_df = pd.DataFrame(
                    {
                        "names": model.test_names,
                        "y_true": model.y_test,
                        "y_pred": model.y_test_pred,
                        "y_sigma": model.y_test_sigma,
                    }
                )

                # Dimensionality reduction df avoids recomputing the DR
                dr_df = out["dr_df"]

                # Visualize results
                p.set(message="Visualizing results", detail="This may take a while...")

                fig, ax, df = await loop.run_in_executor(
                    executor,
                    model_lib.plot,
                    model.rep,
                    vis_method,
                    None,
                    None,
                    model_lib.names,
                    dr_df,
                )

                # set reactive variables
                DISCOVERY_LIB.set(model_lib)
                DISCOVERY_MODEL.set(model)
                DISCOVERY_TEST_DF.set(test_df)
                DISCOVERY_MODEL_PLOT.set(fig)

            except Exception as e:
                print(f"An error occurred in training the Discovery model: {e}")

            finally:
                # Reset the task running state in the session
                IS_DISCOVERY_TRAIN_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.discovery_train_button)
    async def discovery_train_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_DISCOVERY_TRAIN_RUNNING():
            print("Discovery train is already in progress for this session.")
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(discovery_train())

    IS_CLUSTERING_RUNNING = reactive.Value(False)

    ### CLUSTERING DISCOVERY ###
    async def clustering():
        IS_CLUSTERING_RUNNING.set(True)

        with ui.Progress(min=1, max=15) as p:
            p.set(message="Clustering data", detail="This may take a while...")

            rep_type = REP_DICT[input.clustering_rep()]
            lib = LIBRARY()

            # hdbscan_n_neighbors, hdbscan_min_cluster_size, hdbscan_min_samples
            n_neighbors = input.hdbscan_n_neighbors()
            min_cluster_size = input.hdbscan_min_cluster_size()
            min_samples = input.hdbscan_min_samples()
            dr_method = REP_VISUAL_DICT[input.hdbscan_dr()]

            model = pai.Model(
                model_type=MODEL_DICT[input.clustering_alg()],  #
                library=lib,
                rep=rep_type,
                seed=None,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                dr_method=dr_method,
            )

            try:
                loop = asyncio.get_running_loop()

                out = await loop.run_in_executor(executor, model.train, n_neighbors)

                model_lib = pai.Library(user=lib.user, source=out)

                # Dimensionality reduction df avoids recomputing the DR
                dr_df = out["dr_df"]

                # Visualize results
                p.set(message="Visualizing results", detail="This may take a while...")

                fig, ax, df = await loop.run_in_executor(
                    executor,
                    model_lib.plot,
                    model.rep,
                    dr_method,
                    None,
                    None,
                    model_lib.names,
                    None,
                    None,
                    True,
                    dr_df,
                )

                out_df = pd.DataFrame(
                    {
                        "names": out["df"][out["names_col"]],
                        "y_true": out["df"][out["y_col"]],
                        "y_pred": out["df"][out["y_pred_col"]],
                        "y_sigma": out["df"][out["y_sigma_col"]],
                    }
                )

                # set reactive variables
                DISCOVERY_LIB.set(model_lib)
                DISCOVERY_MODEL.set(model)
                DISCOVERY_TEST_DF.set(out_df)
                DISCOVERY_MODEL_PLOT.set(fig)
                DISCOVERY_DR_DF.set(dr_df)

            except Exception as e:
                print(f"An error occurred in training the Discovery model: {e}")

            finally:
                # Reset the task running state in the session
                IS_CLUSTERING_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.clustering_button)
    async def clustering_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_CLUSTERING_RUNNING():
            print("Discovery train is already in progress for this session.")
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(clustering())

    ### RENDER DISCOVERY PLOT ###
    @output
    @widgets.render_widget
    def discovery_plot(alt=None):
        if DISCOVERY_LIB():
            fig = DISCOVERY_MODEL_PLOT()
            return fig

    ### RENDER PREDICTED VERSUS TURE DATAFRAME ###
    @output
    @render.data_frame
    def discovery_table(alt=None):
        df = DISCOVERY_TEST_DF()
        if DISCOVERY_MODEL() is None:
            return None
        else:
            model = DISCOVERY_MODEL()
            class_dict = model.library.class_dict
            df["y_true"] = [class_dict[i] for i in df["y_true"]]
            try:
                df["y_pred"] = [class_dict[int(i)] for i in df["y_pred"]]
            except Exception:
                pass
            return df

    ### DISCOVERY SEARCH ###
    IS_DISCOVERY_SEARCH_RUNNING = reactive.Value(False)

    async def discovery_search():
        IS_DISCOVERY_SEARCH_RUNNING.set(True)
        with ui.Progress(min=1, max=10000) as p:
            p.set(message="Sampling diverse sequences", detail=f"...")  # noqa: F541

            labels = input.sample_from()
            if labels == ():
                labels = []
            if isinstance(labels, str):
                labels = [labels]

            model = DISCOVERY_MODEL()
            try:
                batch_size = BATCH_SIZE_DICT[model.rep]
                loop = asyncio.get_running_loop()
                out = await loop.run_in_executor(
                    executor,
                    model.search,
                    input.n_samples(),
                    labels,
                    None,
                    "ga",
                    None,
                    None,
                    batch_size,
                    None,
                    None,
                )

                search_results = out["mask"]

                # Visualize results
                p.set(message="Visualizing results", detail="This may take a while...")

                df = DISCOVERY_DR_DF()

                fig, ax, df = await loop.run_in_executor(
                    executor,
                    model.library.plot,
                    model.rep,
                    "umap",
                    None,
                    None,
                    model.library.names,
                    search_results,
                    None,
                    True,
                    df,
                )

                DISCOVERY_SEARCH_PLOT.set(fig)

                DISCOVERY_DF.set(out["df"])

                DISCOVERY_SEARCH.set(search_results)

            except Exception as e:
                print(f"An error occurred in discovery search: {e}")

            finally:
                # Reset the task running state in the session
                IS_DISCOVERY_SEARCH_RUNNING.set(False)

    # Button click event
    @reactive.effect
    @reactive.event(input.discovery_search)
    async def discovery_search_btn_click():
        # Prevent multiple invocations of the task within the same session
        if IS_DISCOVERY_SEARCH_RUNNING():
            print("Discovery search is already in progress for this session.")
            return

        # Launch the expensive computation asynchronously
        asyncio.create_task(discovery_search())

    ### RENDER SEARCH PLOT ###
    @output
    @widgets.render_widget
    def discovery_search_plot(alt=None):
        highlight_mask = DISCOVERY_SEARCH()
        if highlight_mask is not None:
            fig = DISCOVERY_SEARCH_PLOT()
            return fig

    ### RENDER DISCOVERY TABLE ###
    @output
    @render.data_frame
    def discovery_search_table(alt=None):
        df = DISCOVERY_DF()
        model = DISCOVERY_MODEL()
        seq_col = model.library.seq_col
        df = df.drop(seq_col, axis=1)
        class_dict = model.library.class_dict
        try:
            df["y_true"] = [class_dict[i] for i in df["y_true"]]
            df["y_predicted"] = [class_dict[int(i)] for i in df["y_predicted"]]
        except Exception:
            pass
        return df

    ### DOWNLOAD DISCOVERY RESULTS ###
    @output
    @render.ui
    def discovery_download_ui():
        out = DISCOVERY_DF()
        if out is not None:
            return ui.TagList(
                ui.output_data_frame("discovery_search_table"),
                ui.download_button("download_discovery", "Download discovery results"),
            )

    ### DOWNLOAD LOGIC FOR DESIGN RESULTS ###
    @render.download(filename=lambda: f"{LIBRARY().fname}_discovery.csv")
    def download_discovery():
        yield DISCOVERY_DF().to_csv(index=False)

    ###############
    ### HELPERS ###
    ###############
    def is_numerical(data):
        """
        Check if the data is numerical.
        Args:
            data: List of data values.
        Returns:
            Boolean: True if all data is numerical (int, float, complex), False otherwise.
        """
        return all(isinstance(x, (int, float, complex)) for x in data)


app = App(app_ui, server)
