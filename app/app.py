# This source code is part of the proteusAI package and is distributed
# under the MIT License.

"""
ProteusAI Shiny App.
"""

__name__ = "ProteusAI"
__author__ = "Jonathan Funk"

import shiny
from shiny import App, ui, render, Inputs, Outputs, Session, reactive
from shiny.types import FileInfo, ImgData
import pandas as pd
import sys
import os
app_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(app_path, '../src/')) # for server '/home/jonfunk/ProteusAI/src/'
import proteusAI as pai
import os
import time
from pathlib import Path


VERSION = "version " + "0.1"
REP_TYPES = ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"] # Add VAE and MSA-Transformer later
IN_MEMORY = ["BLOSUM62", "BLOSUM50", "One-hot"]
TRAIN_TEST_VAL_SPLITS = ["Random"]
MODEL_TYPES = ["KNN", "Gaussian Process", "Random Forrest", "Ridge","SVM"]
MODEL_DICT = {"Random Forrest":"rf", "KNN":"knn", "SVM":"svm", "VAE":"vae", "ESM-2":"esm2", "ESM-1v":"esm1v", "Gaussian Process":"gp", "ESM-Fold":"esm_fold", "Ridge":"ridge"}
REP_DICT = {"One-hot":"ohe", "BLOSUM50":"blosum50", "BLOSUM62":"blosum62", "ESM-2":"esm2", "ESM-1v":"esm1v", "VAE":"vae"}
INVERTED_REPS = {v: k for k, v in REP_DICT.items()}
DESIGN_MODELS = {"ESM-IF":"esm_if"}
REP_VISUAL = ["UMAP","t-SNE", "PCA"]
FAST_INTERACT_INTERVAL = 60 # in milliseconds
SIDEBAR_WIDTH = 450
BATCH_SIZE = 100
ZS_MODELS = ["ESM-1v", "ESM-2"]
FOLDING_MODELS = ["ESM-Fold"]
ACQUISITION_FNS = ["Expected Improvement", "Upper Confidence Bound", "Greedy"]
USR_PATH = os.path.join(app_path, '../usrs')
SEARCH_HEURISTICS = ['Diversity']
OPTIM_DICT = {"Maximize Y-values":"max", "Minimize Y-values":"min"}
MAX_EVAL_DICT = {"ohe":10000, "blosum62":10000, "blosum50":10000, "esm2":200, "esm1v":200}


app_ui = ui.page_fluid(

    #ui.panel_title("ProteusAI"),
    ui.output_image("image", inline=True),
    VERSION,
    
    ###############
    ## DATA PAGE ##
    ###############
    ui.navset_card_tab(
        ui.nav_panel("Data", 
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.row(
                        ui.column(6,
                            ui.input_text('USER', 'Enter user name or proceed as guest', value='Guest'),
                        ),
                        ui.column(6,
                            ui.input_action_button('login', 'Login'),
                            style='padding:50px;'
                        ),
                        ui.column(6,
                            ui.input_action_button('sign_up', 'Create account'),
                        ),
                    ),

                    ### NAVSET ###
                    ui.navset_tab(
                        ui.nav_panel("Library",
                                ui.input_file(id="dataset_file", label="Select dataset (Default: demo dataset)", accept=['.csv', '.xlsx', '.xls'], placeholder="None"),
                               "Data selection",
                                ui.row(
                                    ui.column(6,
                                        ui.input_select("seq_col", "Sequence column", []),
                                    ),
                                    ui.column(6,
                                        ui.input_select("description_col", "Description column", []),
                                    ),
                                    ui.column(6,
                                        ui.input_select("y_col", "Y-values", []),
                                    ),
                                    ui.column(6,
                                        ui.input_select("y_type", "Data type", ["Numeric", "Categorical"]),
                                    ),
                                ),
                               ui.input_action_button('confirm_dataset', 'Continue'),
                        ),

                        ui.nav_panel("Sequence",
                            ui.input_file(id="protein_file", label="Upload FASTA", accept=['.fasta'], placeholder="None"),
                            ui.input_action_button('confirm_protein', 'Continue'),
                            
                        ),

                        ui.nav_panel("Structure",
                            ui.input_file(id="structure_file", label="Upload Structure", accept=['.pdb'], placeholder="None"),
                            ui.input_action_button('confirm_structure', 'Continue'),
                        ),
                    ),
                width=SIDEBAR_WIDTH
                ),

                ### MAIN PANEL ###
                ui.panel_conditional("typeof output.protein_fasta !== 'string'",
                    "Upload experimental data (CSV or Excel file) or a single protein (FASTA)",
                    ui.output_data_frame("dataset_table"),
                ),

                ui.output_text("protein_fasta"),
                ui.output_text("protein_struc")

            ),
        ),


        #################
        ## DESIGN PAGE ##
        #################
        ui.nav_panel("Design",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("design_ui"),
                width=SIDEBAR_WIDTH
                ),

                ### MAIN PANEL ###
                ui.panel_conditional("typeof output.protein_struc === 'string'",
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
        ui.nav_panel("Zero-Shot",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("zero_shot_ui"),
                    width=SIDEBAR_WIDTH
                ),

                ui.panel_conditional("typeof output.protein_fasta === 'string'",
                    #ui.output_plot("entropy_plot"),
                    #ui.output_plot("scores_plot"),
                    #ui.output_data_frame("zs_df"),
                    ui.output_ui("zs_download_ui"),
                ),
            ),
        ),


        ##########################
        ## REPRESENTATIONS PAGE ##
        ##########################
        ui.nav_panel("Representations", 
                ### SIDEBAR ###
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.output_ui("representations_ui"),
                    width=SIDEBAR_WIDTH,
                ),

                ### MAIN PANEL ###
                ui.panel_conditional(
                    "typeof output.protein_struc === 'string'",
                    ui.output_ui("struc3D"),
                ),

                ui.output_plot('tsne_plot'),                              
            ),
        ),


        ################
        ## MLDE PAGE ##
        ################
        ui.nav_panel("MLDE", 
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("mlde_ui"),
                    ui.output_ui("mlde_search_ui"),
                    width=SIDEBAR_WIDTH,
                ),

                ### MAIN PANEL ###
                ui.navset_tab(
                    ui.nav_panel("Model Diagnostics",
                        ui.output_ui("pred_vs_true_ui"),
                        ui.output_data_frame("mlde_model_table"),
                    ),
                    ui.nav_panel("Search Results",
                        ui.output_ui("mlde_download_ui")
                    ),
                ),
            ),    
        ),


        ####################
        ## DISCOVERY PAGE ##
        ####################
        ui.nav_panel("Discovery",
            ### SIDEBAR ###
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("discovery_ui"),
                    ui.output_ui("discovery_search_ui"),
                    width=SIDEBAR_WIDTH,
                ),

            ### MAIN PANEL ###
            ui.navset_tab(
                    ui.nav_panel("Model Diagnostics",
                        ui.output_plot("discovery_plot"),
                        ui.output_data_frame("discovery_table"),
                    ),
                    ui.nav_panel("Search Results",
                        ui.output_plot("discovery_search_plot"),
                        ui.output_ui("discovery_download_ui")
                    ),
                ),
            ),
        ),


    ),
)

def server(input: Inputs, output: Outputs, session: Session):
    #######################
    ### Reactive Values ###
    #######################

    # DUMMY DATA
    dummy = pd.DataFrame({
        "Sequence":["MGVARGTV...G", "AGVARGTV...G", "...", "MGVARGTV...V"],
        "Description":["wt", "M1A", "...", "G142V"],
        "Activity":["0.0", "0.32", "...", "-0.21"]
    })

    # INPUT
    MODE = reactive.Value('start')
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
    DESIGN_OUTPUT = reactive.Value('start')
    FOLD_LIB = reactive.Value(None)
    FIXED_RES = reactive.Value(None)
    DESIGN_LIB = reactive.Value(None)

    # ZER0-SHOT
    ZS_SCORES = reactive.Value(pd.DataFrame())
    COMP_ZS_SCORES = reactive.Value([])

    # REPRESENTATIONS
    TSNE_DF = reactive.Value(None) 
    LIBRARY_PLOT = reactive.Value(None)

    # MLDE
    MODEL = reactive.Value(None)
    DISCOVERY_MODEL = reactive.Value(None)
    VAL_DF = reactive.Value(pd.DataFrame({'names':[], 'y_true':[], 'y_pred':[], 'y_sigma':[]}))
    DISCOVERY_VAL_DF = reactive.Value(pd.DataFrame({'names':[], 'y_true':[], 'y_pred':[], 'y_sigma':[]}))
    DISCOVERY_LIB = reactive.Value(None)
    MLDE_SEARCH_DF = reactive.Value(None)

    # Discovery
    DISCOVERY_SEARCH = reactive.Value(None)
    DISCOVERY_DF = reactive.Value(None)


    ##############
    ## FRONTEND ##
    ##############

    ##################
    ### DESIGN TAB ###
    ##################
    @output
    @render.ui
    def design_ui():
        if MODE() == 'structure':
            return ui.TagList(
                ui.row(
                    ui.h5("Structure Based Protein Design"),
                    ui.row(
                        ui.column(6,
                            ui.input_select("design_models", "Choose model", list(DESIGN_MODELS.keys())),
                        ),
                        ui.column(6,
                            ui.output_ui("design_chains"),
                        ),
                        ui.column(6,
                            ui.input_numeric("n_designs", "Number of samples", min=1, value=20),
                        ),
                        ui.column(6,
                            ui.input_numeric("sampling_temp", "Sampling temperature", min=10e-9, value=0.1),
                        ),
                    ),

                    ui.input_text("design_res", "Select residues by ID that should remain unchanged during redesign','"),

                    ui.input_checkbox("design_interfaces", "Fix interfaces"),

                    ui.panel_conditional("input.design_interfaces === true",
                        ui.row(
                            ui.column(6,
                                ui.input_checkbox("design_protein_interface", "Protein-protein interfaces"),
                            ),
                            ui.column(6,
                                ui.input_numeric("design_protein_interface_distance", "Distance cut-off (Angstroms)", value=7),
                            ),
                            ui.column(6,
                                ui.input_checkbox("design_ligand_interface", "Ligand interfaces"),
                            ),
                            ui.column(6,
                                ui.input_numeric("design_ligand_interface_distance", "Distance cut-off (Angstroms)", value=7),
                            ),
                        ),

                    ),
                    
                    ui.column(4,
                        ui.input_action_button("desgin_button", "Design"),  
                    ),                
                ),

                # DISABLED FOLDING
                #ui.output_ui(
                #        "folding",
                #    ),
            )

        else:
            return ui.TagList("Upload a protein structure in the 'Data' tab to proceed.")


    ### DYNAMIC DESIGN OUTPUT ###
    #@output
    #@render.ui
    #def folding():
    #    out = DESIGN_OUTPUT()
    #    if type(out) != str:
    #        return ui.TagList(  
    #            ui.h5("Fold designed sequences"),
    #            ui.input_selectize("fold_these", "Select sequences to be folded", out['names'].to_list(), multiple=True),
    #            ui.row(
    #                ui.column(6,
    #                    ui.input_select("folding_model", "Select folding model", FOLDING_MODELS),  
    #                ),
    #                ui.column(6,
    #                    ui.input_slider("num_recycles", "Recycling steps", value=0, min=0, max=8),
    #                ),
    #            
    #            ),
    #            ui.row(
    #                ui.column(6,
    #                    ui.input_action_button("folding_button", "Fold")
    #                ),
    #                ui.column(6,
    #                    ui.input_checkbox("energy_minimization", "Energy minimization"),
    #                    style='padding:10px;',
    #                ),  
    #            ),

    #            ui.h5("Analyze protein structures"),
    #            ui.input_selectize("design_sidechains", "Compare geometries of fixed before and after folding", choices=FIXED_RES(), multiple=True),
    #            ui.column(6,
    #                ui.input_action_button("analyze_designs", "Analyze Strucures"),
    #            ),
    #        )


    ###########################
    ### REPRESENTATIONS TAB ###
    ###########################
    @output
    @render.ui
    def representations_ui():
        if MODE() != "start":
            return ui.TagList(
                ui.h4("Representation Learning"),
                ui.row(
                    ui.column(7,
                        ui.input_select("dat_rep_type", "Compute representations", REP_TYPES),
                    ),
                    ui.column(5,
                        ui.input_action_button("dat_compute_reps", "Compute"),
                            style='padding:25px;',
                        ),
                    ),

                    ui.column(6,
                        ui.output_ui('rep_chain_ui'),
                    ),

                    ui.h4("Visualization"),

                    ui.panel_conditional("input.dat_rep_type === 'VAE' || input.dat_rep_type === 'MSA-Transformer'",
                        ui.input_file("MSA_vae_training", "Upload MSA file"),
                    ),

                    ui.panel_conditional("input.dat_rep_type === 'VAE'",
                        ui.input_checkbox("custom_vae", "Customize VAE parameters"),
                        ui.input_action_button("train_vae", "Train VAE")
                    ),
                
                    ui.input_select("vis_method","Visualization Method", REP_VISUAL),
                    
                    #ui.input_select("color_by", "Color by", ["Y-value", "Site", "Custom"]),
                    
                    # Conditional panel for Site
                    #ui.panel_conditional("input.color_by === 'Site'",
                    #        ui.input_text("color_text","Select sites to color seperated by ';' (e.g. 21;42)"),
                    #    ),
                    
                    # Conditional panel for Y-value with numeric data
                    #ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'Numeric'",
                    #        ui.row(
                    #            ui.column(6,
                    #                    ui.input_numeric("y_upper", "Choose an upper limit for y", value=None),
                    #                ),
                    #            ui.column(6,
                    #                    ui.input_numeric("y_lower", "Choose an lower limit for y", value=None),
                    #                ),
                    #        ),
                    #    ),

                    # Conditional panel fo Y-value with categorical data
                    #ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'Categorical'",
                    #        ui.input_text("selected_classes","Select classes to colorize seperated by ';' (e.g. class1;class2)"),
                    #    ),
                    
                    #ui.input_text("hide_sites", "Hide points based on site seperated by ';' (e.g. 21;42)"),

                    #ui.input_checkbox("hide_by_y", "Hide points based Y-Value", value=False),

                    #ui.panel_conditional("input.hide_by_y === true",
                    #    ui.row(
                    #    # change these to be the min and max values observed in the library
                    #    ui.column(6,
                    #                ui.input_slider("hide_upper_y","hide points above y", min=0, max=100, value=100),
                    #        ),
                    #    ui.column(6,
                    #                ui.input_slider("hide_lower_y","hide points below y", min=0, max=100, value=0),
                    #        ),
                    #    ),
                    #),

                    ui.row(
                        ui.column(12, "Visualize representations"),
                        ui.column(7,
                            ui.input_select("plot_rep_type", "", REPS_AVAIL()),
                        ),
                            ui.column(5,
                            ui.input_action_button("update_plot", "Update plot"),
                        ),
                    ),
            )

        else:
            return ui.TagList(
               "Upload a Library in the 'Data' tab or compute a library in the 'Zero-shot' tab for more visualization options.",
            )
        

    #####################
    ### ZERO-SHOT TAB ###
    #####################
    @output
    @render.ui
    def zero_shot_ui():
        if MODE() == "zero-shot" or MODE() == "structure":
            return ui.TagList(
                ui.h4("Representation Learning"),
                ui.row(
                    ui.h5("Compute a zero-shot Library"),
                    ui.column(7,
                        ui.input_select("zs_model", "Choose a model", ZS_MODELS),
                    ),

                    ui.column(5,
                        ui.input_action_button("compute_zs", "Compute"),
                        style='padding:25px;',
                    ),

                    ui.column(6,
                        ui.output_ui('zs_chain_ui'),
                    ),

                    ui.h4("Visualize"),

                    ui.row(
                        ui.column(6,
                            ui.input_select("computed_zs_scores", "Computed Zero-shot scores", COMP_ZS_SCORES()),
                        ),
                    ),

                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_entropy", "Plot Entropy"),
                        ),
                        ui.column(6,
                            ui.input_checkbox("plot_entropy_section", "Customize plot"),
                            style='padding:10px;'
                        ),

                        ui.panel_conditional("input.plot_entropy_section === true",
                            ui.row(
                                ui.column(6,
                                    ui.input_slider("entropy_slider", "Sliding window", min=0, max=len(PROTEIN().seq), value=0),
                                ),
                                ui.column(6,
                                    ui.input_numeric("entropy_width", "Sliding window width", value=10),
                                ),
                            ),
                        ),
                    ),

                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_scores", "Plot Scores"),
                        ),

                        ui.column(6,
                            ui.input_checkbox("plot_scores_section", "Customize plot"),
                            style='padding:10px;'
                        ),

                        ui.panel_conditional("input.plot_scores_section === true", 
                            ui.row(
                                ui.column(6,
                                    ui.input_slider("scores_slider", "Sliding window", min=0, max=len(PROTEIN().seq), value=0),
                                ),
                                ui.column(6,
                                    ui.input_numeric("scores_width", "Sliding window width", value=10),
                                ),
                            ),
                        ),

                        ui.column(6,
                            ui.input_action_button("zs_table", "Table view"),
                        ),
                    ),
                ),
            )

        else:
            return ui.TagList(
                "The zero-shot mode is available for individual proteins and structures."
            )
    

    ################
    ### MLDE tab ###
    ################
    @output
    @render.ui
    def mlde_ui():
        if Y_TYPE() == 'class':
            return ui.TagList(
                "The MLDE workflow is available for numerical Y-Values. Please use the Discovery workflow for categorical Y-values"
            ) 
    
        elif MODE() != 'start':
            return ui.TagList(
                    ui.row(
                        ui.h5("Machine Learning Guided Directed Evolution"),

                        ui.column(6,
                            ui.input_select("model_type", "Surrogate model", _MODEL_TYPES())
                        ),

                        ui.column(6,
                            ui.input_select("model_rep_type", "Representation type", REPS_AVAIL()),
                        ),

                        ui.column(6,
                            ui.output_ui('mlde_chain_ui'),
                        ),

                        ui.column(6,
                            ui.output_ui("mlde_dynamic_ui"),
                        ),
                        
                    ),
                    
                    ui.input_checkbox("customize_model_params", "Customize model parameters", value=False),
                    
                    ui.row(
                        ui.column(6,
                            ui.input_slider("random_seed", "Random seed", min=0, max=1024, value=42)
                        ),
                        ui.panel_conditional("input.model_type !== 'Gaussian Process'",
                            ui.column(6,
                                ui.input_numeric("k_folds", "K-Fold cross validation", value=5, min=1, max=10),
                            ),
                        ),
                        
                        ui.column(12,
                            "Cross-validation split:",
                        ),
                        ui.column(4,
                            ui.input_numeric("n_train", "Training (%)", value=80, min=0, max=100),
                        ),
                        ui.column(4,
                            ui.input_numeric("n_test", "Test (%)", value=10, min=0, max=100),
                        ),
                        ui.column(4,
                            ui.input_numeric("n_val", "Validation (%)", value=10, min=0, max=100),
                        ),
                        
                        ui.column(4,
                            ui.input_action_button("train_button", "Train"),
                        ),
                    ),
                
            ) 
        
        else:
            return ui.TagList(
                "Upload data in the 'Data' tab to proceed."
            )    


    ### MLDE SEARCH UI ###
    @output
    @render.ui
    def mlde_search_ui(alt=None):
        if MODEL() != None:
            inv_model_dict = {value: key for key, value in MODEL_DICT.items()}
            model_type = inv_model_dict[MODEL().model_type]
            return ui.TagList(
                ui.h5("Search new mutants"),
                ui.row(
                    ui.column(6,
                        ui.input_select("acquisition_fn", "Acquisition Function", ACQUISITION_FNS),
                    ),
                    ui.column(6,
                        ui.input_select("search_model", "Model", [model_type]),
                    ),
                    ui.column(6,
                        ui.input_select("optim_problem", "Optimization problem", ['Maximize Y-values', 'Minimize Y-values']),
                    ),
                ),
                
                ui.column(8,
                    ui.input_slider("mlde_explore", "Exploration vs. Exploitation", value=0.1, min=0, max=1),
                ),

                ui.column(6,
                    ui.input_action_button("mlde_search", "Search"),
                ),
            )


    ###################
    ## DISCOVERY TAB ##
    ###################
    @output
    @render.ui
    def discovery_ui(alt=None):
        if Y_TYPE == 'num':
            return ui.TagList(
                "The Discovery workflow is only available for categorical Y-Values. Please use the MLDE workflow for numerical Y-values"
            )

        elif MODE() != 'start':
            return ui.TagList(
                    ui.row(
                        ui.h5("Protein Discovery and Annotation"),

                        ui.column(6,
                            ui.input_select("discovery_model_type", "Surrogate model", _MODEL_TYPES())
                        ),

                        ui.column(6,
                            ui.input_select("discovery_model_rep_type", "Representaion type", REPS_AVAIL()),
                        ),
                        ui.panel_conditional("input.discovery_model_type !== 'Gaussian Process'",
                            ui.column(6,
                                ui.input_numeric("discovery_k_folds", "K-Fold cross validation", value=5, min=1, max=10),
                            ),
                        ),
                        ui.column(6,
                            ui.output_ui("discovery_dynamic_ui"),
                        ),
                        
                    ),
                    
                    ui.input_checkbox("discovery_model_params", "Customize model parameters", value=False),
                    
                    ui.row(
                        ui.column(6,
                            ui.input_slider("discovery_random_seed", "Random seed", min=0, max=1024, value=42)
                        ),

                        ui.column(6,
                            ui.input_select("discovery_vis_method", "Visualization method", choices=REP_VISUAL), 
                        ),
                        
                        ui.column(12,
                            "Cross-validation split:",
                        ),
                        ui.column(4,
                            ui.input_numeric("discovery_n_train", "Training (%)", value=80, min=0, max=100),
                        ),
                        ui.column(4,
                            ui.input_numeric("discovery_n_test", "Test (%)", value=10, min=0, max=100),
                        ),
                        ui.column(4,
                            ui.input_numeric("discovery_n_val", "Validation (%)", value=10, min=0, max=100),
                        ),
                        
                        ui.column(4,
                            ui.input_action_button("discovery_train_button", "Train"),
                        ),
                    ),
                
            )
        
        else:
            return ui.TagList(
                "Upload data in the 'Data' tab to proceed."
            ) 


    ### Discovery SEARCH UI ###
    @output
    @render.ui
    def discovery_search_ui(alt=None):
        model = DISCOVERY_MODEL()
        if model != None:
            clusters = list(model.library.class_dict.values())
            sample_from = clusters
            inv_model_dict = {value: key for key, value in MODEL_DICT.items()}
            model_type = inv_model_dict[DISCOVERY_MODEL().model_type]

            return ui.TagList(
                ui.h5("Search new mutants"),
                ui.row(
                    ui.column(6,
                        ui.input_select("discovery_search_criteria", "Search heuristic", SEARCH_HEURISTICS),
                    ),
                    ui.column(6,
                        ui.input_select("discovery_search_model", "Model", [model_type]),
                    ),
                    ui.column(6,
                        ui.input_selectize("sample_from", "Sample from Cluster", sample_from, multiple=True),
                    ),

                    ui.column(6,
                        ui.input_numeric("n_samples", "Number of sequences", value=10, min=2),
                    ),

                    ui.column(6,
                        ui.input_action_button("discovery_search", "Search"),
                        style='padding:25px;'
                    )
                )
            )


    ###############
    ### BACKEND ###
    ###############

    ### APP LOGO ###
    @output
    @render.image
    def image():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"),  "height": "75px"}
        return img


    ########
    ## IO ##
    ########

    ### READING DATASET ###
    @reactive.Effect
    @reactive.event(input.dataset_file)
    def _():
        df = DATASET()
        f: list[FileInfo] = input.dataset_file()
        df = pd.read_csv(f[0]["datapath"])

        # set reactive variables
        DATASET.set(df)
        DATASET_PATH.set(f[0]["datapath"])


    ### RENDER DATASET TABLE ###
    @output
    @render.data_frame
    def dataset_table():
        df = render.DataTable(DATASET(), summary=True)
        return df


    ### EXTRACT DATASET COLUMNS ###
    @reactive.Effect()
    def _():
        cols = list(DATASET().columns)

        # set reactive variables
        ui.update_select(
            "seq_col",
            label="Sequences",
            choices=cols,
            selected=cols[0],
        )

        ui.update_select(
            "description_col",
            label="Descriptions",
            choices=cols,
            selected=cols[1],
        )

        ui.update_select(
            "y_col",
            label=f"Y-values",
            choices=cols,
            selected=cols[-1],
        )


    ### CONFIRM DATASET ###
    @reactive.Effect
    @reactive.event(input.confirm_dataset)
    async def _():
        df = DATASET()
        if input.dataset_file() == None:
            with ui.Progress(min=1, max=15) as p:
                p.set(message="No data uploaded", detail="Upload data to continue")
                time.sleep(2.5)                
        else:
            f: list[FileInfo] = input.dataset_file()
            file_name = f[0]['name']
            ys = df[input.y_col()].to_list()

            # Determine if the data is numerical or categorical
            if is_numerical(ys):
                y_type = "num"
                choice = "Regression"
                _y_type = "Numeric"
                _MODEL_TYPES.set([x for x in MODEL_TYPES if x != "KNN"])
            else:
                y_type = "class"
                choice = "Classification"
                _y_type = "Categorical"
                _MODEL_TYPES.set([x for x in MODEL_TYPES if x != "Gaussian Process"])

            lib = pai.Library(user=input.USER().lower(), source=f[0]["datapath"], seqs_col=input.seq_col(), y_col=input.y_col(), 
                            y_type=y_type, names_col=input.description_col(), fname=file_name)


            # set reactive variables
            LIBRARY.set(lib)
            ui.update_select(
                "model_rep_type",
                choices=[INVERTED_REPS[i] for i in lib.reps]
            )

            Y_TYPE.set(y_type)

            reps = [INVERTED_REPS[i] for i in lib.reps]

            for rep in IN_MEMORY:
                    if rep not in reps:
                        reps.append(rep)

            REPS_AVAIL.set(reps)
            
            ui.update_select(
                "model_task",
                choices=[choice]
            )

            ui.update_select(
                "model_type",
                choices = _MODEL_TYPES()
            )

            ui.update_select(
                "discovery_model_type",
                choices = _MODEL_TYPES()
            )

            ui.update_select(
                "y_type",
                choices=[_y_type]
            )

            PROTEIN.set(None)
            
            REP_PATH.set(None) # used in train
            
            MODE.set("dataset")

            LIBRARY_PLOT.set(None)


    ### READING PROTEIN FILE ###
    @reactive.Effect
    @reactive.event(input.protein_file)
    def _():
        f: list[FileInfo] = input.protein_file()
        usr_path = os.path.join(USR_PATH, input.USER().lower())
        file_name = f[0]['name']
        prot = pai.Protein(source=f[0]["datapath"], user=usr_path, fname=file_name)

        PROTEIN.set(prot)
        DATASET_PATH.set(f[0]["datapath"])


    ### CONFIRM PROTEIN ###
    @reactive.Effect
    @reactive.event(input.confirm_protein)
    async def _():
        if input.protein_file() == None:
            with ui.Progress(min=1, max=15) as p:
                p.set(message="No data uploaded", detail="Upload data to continue")
                time.sleep(2.5)
        else:
            with ui.Progress(min=1, max=15) as p:
                LIBRARY.set(None)
                p.set(message="Searching for available data...", detail="This may take a while...")
                # initialize protein
                #prot = protein()
                f: list[FileInfo] = input.protein_file()
                usr_path = os.path.join(USR_PATH, input.USER().lower())
                file_name = f[0]['name']

                prot = pai.Protein(source=f[0]["datapath"], user=usr_path, fname=file_name)

                # set shiny variables
                PROTEIN.set(prot)

                DATASET_PATH.set(f[0]["datapath"])
                MODE.set('zero-shot')

                # check for zs-computations # TODO: test if the number of computations match with the number of sequences.
                zs_computed = []
                rep_computed = []
                for model in ZS_MODELS:
                    # check hash existence
                    zs_path = os.path.join(prot.user, f"{prot.name}/zero_shot/results/{MODEL_DICT[model]}")

                    if os.path.exists(zs_path):
                        if "zs_scores.csv" in os.listdir(zs_path):
                            zs_computed.append(model)
                    
                    for rep in REP_TYPES:
                        rep_path = os.path.join(prot.user, f"{prot.name}/zero_shot/rep/{REP_DICT[rep]}")
                        if os.path.exists(rep_path):
                            rep_computed.append(REP_DICT[rep])

                # load zs-library if exists # TODO: test if the number of computations match with the number of sequences.
                for model in ZS_MODELS:
                    try:
                        rep_path = os.path.join(prot.user, f"{prot.name}/zero_shot/rep/{REP_DICT[model]}")
                        df_path = os.path.join(prot.user, f"{prot.name}/zero_shot/{REP_DICT[model]}/zs_scores.csv")
                        
                        df = pd.read_csv(df_path)
                        p.set(message="Loading data...", detail="This may take a while...")
                        lib = pai.Library(user=usr_path, seqs=df.sequence, ys=df.mmp, y_type="num", names=df.mutant.to_list(), proteins=[], rep_path=rep_path)

                        # set reactive variables
                        LIBRARY.set(lib)
                        DATASET.set(df)
                        ZS_SCORES.set(df)
                        
                    except:
                        pass
                
                rep_computed = list(set(rep_computed))
                for rep in IN_MEMORY:
                    if rep not in rep_computed:
                        rep_computed.append(REP_DICT[rep])

                # set reactive variables
                ui.update_select(
                            "model_rep_type",
                            choices=[INVERTED_REPS[i] for i in rep_computed]
                        )

                COMP_ZS_SCORES.set(zs_computed)

                REPS_AVAIL.set(rep_computed)

                ZS_RESULTS.set(zs_computed)

                LIBRARY_PLOT.set(None)

                ZS_SCORES.set(pd.DataFrame())

                ui.update_select(
                    "model_task",
                    choices=["Regression"]
                )   

                ui.update_select(
                    "zs_scores",
                    choices=zs_computed
                )

                ui.update_select(
                    "model_type",
                    choices = [x for x in MODEL_TYPES if x != "KNN"]
                )


    ### RENDER FASTA ###
    @output
    @render.text
    def protein_fasta():
        if PROTEIN() == None:
            seq = None
        elif type(PROTEIN().seq) == dict:
            seq = 'Protein name: ' + PROTEIN().name + ' \n'.join([" chain " + chain + ':\n' + PROTEIN().seq[chain] for chain in PROTEIN().chains])
        else:
            seq = 'Protein name: ' + PROTEIN().name + '\n' + PROTEIN().seq
        return seq


    ### CONFIRM STRUCTURE ###
    @reactive.Effect
    @reactive.event(input.confirm_structure)
    async def _():
        if input.structure_file() == None:
            with ui.Progress(min=1, max=15) as p:
                p.set(message="No data uploaded", detail="Upload data to continue")
                time.sleep(2.5)
        else:
            with ui.Progress(min=1, max=15) as p:
                p.set(message="Searching for available data...", detail="This may take a while...")
                prot = PROTEIN()
                f: list[FileInfo] = input.structure_file()
                file_name = f[0]['name']

                usr_path = os.path.join(USR_PATH, input.USER().lower())
                prot = pai.Protein(source=f[0]["datapath"], user=usr_path, fname=file_name)

                name = f[0]["name"].split('.')[0]
                prot.name = name

                # load zs-library if exists
                computed_zs = []
                reps = []

                # load reps
                for rep in REP_TYPES:
                    rep_path = os.path.join(prot.user, f"{name}/zero_shot/rep/{REP_DICT[rep]}")
                    p.set(message="Loading data...", detail="This may take a while...")
                    if os.path.exists(rep_path):
                        reps.append(rep)
                
                # load zs_scores
                for model in ZS_MODELS:
                    for chain in prot.chains:
                        df_path = os.path.join(prot.user, f"{name}/zero_shot/results/{chain}/{REP_DICT[model]}/zs_scores.csv")
                        if os.path.exists(df_path):
                            df = pd.read_csv(df_path)
                            p.set(message="Loading data...", detail="This may take a while...")
                            computed_zs.append(model)

                # set reactive variables
                PROTEIN.set(prot)
                DATASET_PATH.set(f[0]["datapath"])
                MODE.set('structure')

                ui.update_select(
                        "model_rep_type",
                        choices=reps
                    )

                ui.update_select(
                    "computed_zs_scores",
                    choices=computed_zs
                )

                ui.update_select(
                    "model_type",
                    choices = [x for x in MODEL_TYPES if x != "KNN"]
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
        if PROTEIN() == None:
            struc = None
        elif PROTEIN().struc != None:
            struc = 'Structure loaded'
        else:
            struc = None

        return struc


    ############
    ## DESIGN ##
    ############

    ### DESIGN TAB OUTPUT CONTROL ###
    @render.ui
    def struc3D_design():
        out = DESIGN_OUTPUT()
        #if type(out) == str  or input.design_sidechains() == None:
        #    sidechains = [] 
        #else:
        #    sidechains = [int(''.join([char for char in item if char.isdigit()])) for item in input.design_sidechains()]
            

        highlights = list(set(input.design_res().strip().split(',')))# + sidechains))

        if PROT_INTERFACE():
            highlights = highlights + PROT_INTERFACE()

        if LIG_INTERFACE():
            highlights = highlights + LIG_INTERFACE()

        highlights = [i for i in set(highlights) if type(i) != str]

        highlights_dict = {input.mutlichain_chain():highlights}

        view = PROTEIN().view_struc(color="white", highlight=highlights_dict) #, sticks=sidechains)
        return ui.TagList(
            ui.HTML(view.write_html())
        )


    ### SELECT PROTEIN-PROTEIN INTERFACE ###
    @reactive.Effect
    @reactive.event(input.design_protein_interface)
    def _():
        prot = PROTEIN()
        if input.design_protein_interface():
            prot_contacts = prot.get_contacts(chain=input.mutlichain_chain(), dist=input.design_protein_interface_distance(), target = 'protein')
            PROT_INTERFACE.set(prot_contacts) # here will be a function that selects the interface values
        else:
            PROT_INTERFACE.set(None)


    ### SELECT PROTEIN LIGAND INTERFACE ###
    @reactive.Effect
    @reactive.event(input.design_ligand_interface)
    def _():
        prot = PROTEIN()
        if input.design_ligand_interface():
            prot_contacts = prot.get_contacts(chain=input.mutlichain_chain(), dist=input.design_protein_interface_distance(), target = 'ligand')
            LIG_INTERFACE.set(prot_contacts) # here will be a function that selects the interface values
        else:
            LIG_INTERFACE.set(None)
        

    ### DESIGN BUTTON LOGIC ###
    @reactive.Effect
    @reactive.event(input.desgin_button)
    def _():
        n_designs = int(input.n_designs())
        with ui.Progress(min=1, max=n_designs) as p:
            prot = PROTEIN()
            seq = prot.seq[input.mutlichain_chain()]
            p.set(message="Initiating structure based design", detail=f"Computing {n_designs} samples...")
            out = DESIGN_OUTPUT()
            
            sidechains = []
            #if type(out) == str or input.design_sidechains() == None:
            #    sidechains = []
            #else:
            #    sidechains = [int(''.join([char for char in item if char.isdigit()])) for item in input.design_sidechains()]

            residues_str = list(set(input.design_res().strip().split(',') + sidechains))
            fixed_ids = [int(r) for r in residues_str if r.strip() and (r.strip().isdigit() or (r.strip()[1:].isdigit() if r.strip()[0] == '-' else False))]

            if PROT_INTERFACE():
                fixed_ids = fixed_ids + PROT_INTERFACE()

            if LIG_INTERFACE():
                fixed_ids = fixed_ids + LIG_INTERFACE()

            fixed_ids = [i for i in set(fixed_ids) if type(i) != str]

            fixed_ids.sort()

            fixed = []
            if len(fixed_ids) > 0:
                fixed = [seq[i-1] + str(i) for i in fixed_ids if i < len(seq)]
            
            out = prot.esm_if(fixed=fixed_ids, num_samples=n_designs, temperature=float(input.sampling_temp()), pbar=p, chain=input.mutlichain_chain())
            lib = pai.Library(user=prot.user, source=out)

            # set reactive values
            FIXED_RES.set(fixed)
            DESIGN_OUTPUT.set(out['df'])
            DESIGN_LIB.set(lib)


    ### RENDER DESIGN DATAFRAME ###
    @output
    @render.data_frame
    def design_out():
        out = DESIGN_OUTPUT()
        if type(out) == str:
            return None
        else:
            return render.DataTable(out, summary=True) 


    ### CHAIN SELECTION MENU ###
    @output
    @render.ui
    def design_chains():
        num_chains = len(CHAINS())
        return ui.input_select("mutlichain_chain", "Design chain", choices=CHAINS())


    ### DOWNLOAD DESIGN RESULTS
    @output
    @render.ui
    def design_download_ui():
        out = DESIGN_OUTPUT()
        if type(out) != str:
            return ui.download_button("download_designs", "Download design results")


    ### DOWNLOAD LOGIC FOR DESIGN RESULTS ###
    @render.download(
        filename=lambda: f"{PROTEIN().name}_designs.csv"
    )
    def download_designs():
        yield DESIGN_OUTPUT().to_csv(index=False)


    ### OUTPUT TEXED FOR FIXED RESIDUES ###
    @output
    @render.text
    def fixed_res_text(alt=None):
        out = DESIGN_OUTPUT()
        if type(out) != str:
            msg = "Residues that were fixed during design: \n"+ ", ".join(FIXED_RES())
            return msg     


    ### FOLDING BUTTON ###
    #@reactive.Effect
    #@reactive.event(input.folding_button)
    #def _():
    #    """
    #    Fold selected proteins
    #    """
    #    lib = DESIGN_LIB()
    #    out = DESIGN_OUTPUT()
    #    prot = PROTEIN()
    #    selection = input.fold_these()
    #    with ui.Progress(min=1, max=len(selection)) as p:
    #        p.set(message="Initiating folding", detail=f"Computing {len(selection)} structures...")
    #        model = MODEL_DICT[input.folding_model()]
    #        num_recycles = input.num_recycles()
    #        to_fold = out[out[lib.names_col].isin(selection)][lib.names_col].to_list()
    #        out = lib.fold(names=to_fold, model=model, num_recycles=num_recycles, relax=input.energy_minimization() ,pbar=p)
    #        fold_lib = pai.Library(user=prot.user, source=out)
    #        FOLD_LIB.set(fold_lib)


    ### ANALYZE DESIGNS ###
    #@reactive.Effect
    #@reactive.event(input.analyze_designs)
    #def _():
    #    if input.design_sidechains() == None:
    #        sidechains = [] 
    #    else:
    #        sidechains = [int(''.join([char for char in item if char.isdigit()])) for item in input.design_sidechains()]
    #    
    #    lib = FOLD_LIB()
    #    prot = PROTEIN()
    #
    #    sidechains_dict = {input.mutlichain_chain():sidechains}
    #    lib.struc_geom(ref=prot, residues=sidechains_dict)


    ###############
    ## ZERO-SHOT ##
    ###############

    ### COMPUTE ZS-SCORES ###
    @reactive.Effect
    @reactive.event(input.compute_zs)
    def _():
        method = input.zs_model()
        prot = PROTEIN()

        if type(prot.seq) == dict:
            seq = prot.seq[input.zs_chain()]
            chain = input.zs_chain()
        else:
            seq = prot.seq
            chain = None

        with ui.Progress(min=1, max=len(seq)) as p:
            p.set(message=f"Computation {method} zero-shot scores", detail="Initializing...")
            
            computed_zs = COMP_ZS_SCORES()

            model = REP_DICT[method]

            data = prot.zs_prediction(model=model, batch_size=BATCH_SIZE, pbar=p, chain=chain)

            lib = pai.Library(user=prot.user, source=data)

            if method not in computed_zs:
                computed_zs.append(method)
            
            # set reactive values
            ui.update_select(
                "computed_zs_scores",
                choices=computed_zs
            )

            LIBRARY.set(lib)
            DATASET.set(data['df'])
            ZS_SCORES.set(data['df'])
            COMP_ZS_SCORES.set(computed_zs)

    
    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def zs_chain_ui(alt=None):
        if MODE() == 'structure':
            return ui.TagList(
                ui.input_select('zs_chain', 'Select Protein Chain', choices=CHAINS()),
            )


    @reactive.Effect
    @reactive.event(input.zs_chain)
    def _():
        method = input.zs_model()
        chain = input.zs_chain()
        
        prot = PROTEIN()
        name = prot.name

        df_path = os.path.join(prot.user, f"{name}/zero_shot/results/{chain}/{REP_DICT[method]}/zs_scores.csv")
        if os.path.exists(df_path):
            COMP_ZS_SCORES.set([method])
        else:
            COMP_ZS_SCORES.set([])

        # dumb but necessary
        ui.update_select(
            'zs_chain',
            selected=chain
        )


    ### RENDER ZS-DATAFRAME ###
    @output
    @render.data_frame
    @reactive.event(input.zs_table)
    def zs_df(alt=None):
        prot = PROTEIN()
        method = REP_DICT[input.computed_zs_scores()]
        if prot.chains != None and len(prot.chains) >= 1:
            path = os.path.join(prot.zs_path, "results", input.zs_chain(), method, "zs_scores.csv")
        else:
            path = os.path.join(prot.zs_path, "results", method, "zs_scores.csv")
        df = pd.read_csv(path)
        df = df.drop('sequence', axis=1)
        return df


    ### DOWNLOAD ZS RESULTS ###
    @output
    @render.ui
    def zs_download_ui():
        out = ZS_SCORES()
        if out is not None:
            return ui.TagList(
                ui.output_plot("entropy_plot"),
                ui.output_plot("scores_plot"),
                ui.output_data_frame("zs_df"),
                ui.download_button("download_zs_df", "Download discovery results"),
            )


    ### DOWNLOAD LOGIC FOR ZS RESULTS ###
    @render.download(
        filename=lambda: f"{PROTEIN().name}_{input.zs_model()}_zs_predictions.csv"
    )
    def download_zs_df():
        yield ZS_SCORES().to_csv(index=False)


    ### OUTPUT PROTEIN MODE ###
    @output
    @render.plot
    @reactive.event(input.plot_entropy)
    def entropy_plot(alt=None):
        """
        Create the per position entropy plot
        """
        prot = PROTEIN()
        if type(prot.seq) == dict:
            chain = input.zs_chain()
            seq = prot.seq[chain]
        else:
            seq = prot.seq
            chain = None

        if input.plot_entropy_section():
            start = input.entropy_slider()
            end = start + input.entropy_width()
        else:
            start = 0
            end = len(seq)

        section = (start, end)
        if section[1] > len(seq):
            assert section[1] > section[0]
            width = section[1] - section[0]
            section = (len(seq) - width, len(seq))
        
        fig = prot.plot_entropy(section=section, model=MODEL_DICT[input.computed_zs_scores()], chain=chain)
        return fig


    ### UPDATE SCORES PLOT ###
    @output
    @render.plot
    @reactive.event(input.plot_scores)
    def scores_plot(alt=None):
        """
        Create the per position entropy plot
        """
        prot = PROTEIN()

        if type(prot.seq) == dict:
            chain = input.zs_chain()
            seq = prot.seq[chain]
        else:
            seq = prot.seq
            chain = None

        if input.plot_scores_section():
            start = input.scores_slider()
            end = start + input.scores_width()
        else:
            start = 0
            end = len(seq)

        section = (start, end)

        if section[1] > len(seq):
            assert section[1] > section[0]
            width = section[1] - section[0]
            section = (len(seq) - width, len(seq))
        
        fig = prot.plot_scores(section=section, color_scheme = "rwb", model=MODEL_DICT[input.computed_zs_scores()], chain=chain)
        return fig


    ### STRUCTURE MODE ###
    @render.ui
    def struc3D():
        view = PROTEIN().view_struc(color="confidence") # TODO: Add coloring options
        return ui.TagList(
            ui.HTML(view.write_html())
        )


    #####################
    ## REPRESENTATIONS ##
    #####################

    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def rep_chain_ui(alt=None):
        if MODE() == 'structure':
            return ui.TagList(
                ui.input_select('rep_chain', 'Select Protein Chain', choices=CHAINS()),
            )

    ### COMPUTE REPRESENTATIONS ###
    @reactive.Effect
    @reactive.event(input.dat_compute_reps)
    async def _():
        mode = MODE()
        lib = LIBRARY()
        prot = PROTEIN()
        method = input.dat_rep_type()

        if mode == 'structure':
            chain = input.rep_chain()
        else:
            chain = None

        # if no library was loaded one has to be created
        if mode in ['zero-shot', 'structure']:
            data = prot.zs_library(model=MODEL_DICT[method], chain=chain)
            lib = pai.Library(user=prot.user, source=data)
            dest = os.path.join(prot.rep_path, MODEL_DICT[method])
            pbar_max = len(lib)
        else:
            pbar_max = len(lib)
        
        with ui.Progress(min=1, max=pbar_max) as p:

            p.set(message="Computation in progress", detail="Initializing...")

            print(f"Computing library: {REP_DICT[input.dat_rep_type()]}")
            
            lib.compute(method=REP_DICT[input.dat_rep_type()], batch_size=BATCH_SIZE, pbar=p)

            LIBRARY.set(lib)
            print("Done!")

            # update representation selection
            ui.update_select(
                "model_rep_type",
                choices=[INVERTED_REPS[i] for i in lib.reps]
            )

            reps = [INVERTED_REPS[i] for i in lib.reps]
            for rep in IN_MEMORY:
                    if rep not in reps:
                        reps.append(rep)
            REPS_AVAIL.set(reps)


    ### UPDATE REPRESENTATIONS PLOT ###
    @reactive.Effect
    @reactive.event(input.update_plot)
    def _():
        """
        Render plot once button is pressed.
        """
        if input.plot_rep_type():
            with ui.Progress(min=1, max=15) as p:
                
                #if MODE() == "dataset":
                p.set(message="Plotting", detail="This may take a while...")

                lib = LIBRARY()
                mode = MODE()
                prot = PROTEIN()

                # if no library was loaded one has to be created
                if mode in ['zero-shot', 'structure'] and lib == None:
                    data = prot.zs_library(model = REP_DICT[input.plot_rep_type()])
                    lib = pai.Library(user=prot.user, source=data)

                names = lib.names
                #y_upper = input.y_upper()
                #y_lower = input.y_lower()
                rep = REP_DICT[input.plot_rep_type()]
                
                # Update to pass the new parameters
                if input.vis_method() == 't-SNE':
                    fig, ax, df = lib.plot_tsne(rep=rep, names=names)
                elif input.vis_method() == 'UMAP':
                    fig, ax, df = lib.plot_umap(rep=rep, names=names)
                elif input.vis_method() == 'PCA':
                    fig, ax, df = lib.plot_pca(rep=rep, names=names)

                TSNE_DF.set(df)
                LIBRARY_PLOT.set((fig, ax))
        else:
            pass


    ### RENDER REPRESENTATIONS PLOT ###
    @output
    @render.plot
    def tsne_plot(alt=None):
        if LIBRARY_PLOT():
            fig, ax = LIBRARY_PLOT()
            return fig, ax

    ##########
    ## MLDE ##
    ##########

    ### ZERO-SHOT CHAIN UI ###
    @output
    @render.ui
    def mlde_chain_ui(alt=None):
        if MODE() == 'structure':
            return ui.TagList(
                ui.input_select('mlde_chain', 'Select Protein Chain', choices=CHAINS()),
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
                df_path = os.path.join(prot.user, f"{name}/zero_shot/results/{chain}/{REP_DICT[model]}/zs_scores.csv")
                if os.path.exists(df_path):
                    computed_zs.append(model)

            COMP_ZS_SCORES.set(computed_zs)
            
            return ui.TagList(
                ui.input_select("mlde_computed_zs_scores", "Choose zero-shot scores", COMP_ZS_SCORES())
            )

        if MODE() == "dataset":
            return None
        

    ### N-TRAIN COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.n_train)
    def _():
        n_train = input.n_train()
        
        n_test_max = 100 - n_train
            
        new_test = round((n_test_max)/2, 2)
        ui.update_numeric(
            "n_test",
            min=0,
            max = n_test_max,
            value = new_test
        )
        new_val_max = 100 - n_train - new_test
        ui.update_numeric(
            "n_val",
            min=0,
            max = new_val_max,
            value = new_val_max
        )


    ### N-TEST COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.n_test)
    def _():
        n_train = input.n_train()
        n_test = input.n_test()
        n_val_max = 100 - n_train - n_test
        ui.update_numeric(
            "n_val",
            max = n_val_max,
            value = n_val_max
        )


    ### TRAIN MODEL ###
    @reactive.Effect
    @reactive.event(input.train_button)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")

            rep_type = REP_DICT[input.model_rep_type()]
            prot = PROTEIN()

            if MODE() == 'structure':
                f: list[FileInfo] = input.structure_file()
            else:
                f: list[FileInfo] = input.dataset_file()

            lib = LIBRARY()

            if MODE() == 'structure':
                chain = input.mlde_chain()
            else:
                chain = None

            if MODE() in ['zero-shot', 'structure']:
                data = prot.zs_library(model=MODEL_DICT[input.mlde_computed_zs_scores()], chain=chain)
                lib = pai.Library(user=prot.user, source=data)

            split = (input.n_train(), input.n_test(), input.n_val())
            k_folds = input.k_folds()
            if k_folds <= 1:
                k_folds = None

            m = pai.Model(model_type=MODEL_DICT[input.model_type()])
            m.train(library=lib, x=rep_type, split=split, seed=input.random_seed(), model_type=MODEL_DICT[input.model_type()], k_folds=k_folds)
            
            val_df = pd.DataFrame({'names':m.val_names, 'y_true':m.y_val, 'y_pred':m.y_val_pred, 'y_sigma':m.y_val_sigma})

            search_dest = os.path.join(f"{m.library.rep_path}", f"../models/{m.model_type}/{m.x}/predictions")
            search_file = os.path.join(search_dest, f"{m.model_type}_{m.x}_predictions.csv")

            if os.path.exists(search_file):
                MLDE_SEARCH_DF.set(pd.read_csv(search_file))

            # set reactive variables
            MODEL.set(m)
            VAL_DF.set(val_df)


    ### PREPARE PREDICTED VERSUS TRUE PLOT ###
    @output
    @render.ui
    def pred_vs_true_ui():
        hover_opts_kwargs = {}
        hover_opts_kwargs["delay"] = FAST_INTERACT_INTERVAL
        hover_opts_kwargs["delay_type"] = "throttle"

        return ui.output_plot(
            "pred_vs_true",
            hover=ui.hover_opts(**hover_opts_kwargs),
        )


    ### RENDER PREDICTED VERSUS TRUE DATAFRAME ###
    @output
    @render.plot
    def pred_vs_true(alt=None):
        df = VAL_DF()
        if MODEL() != None:
            p = MODEL().true_vs_predicted(y_true=df.y_true.to_list(), y_pred=df.y_pred.to_list())
        else:
            p = None
        return p
    

    ### RENDER DISCOVERY TABLE ###
    @output
    @render.data_frame
    def mlde_model_table(alt=None):
        model = MODEL()
        if model is not None:
            table = VAL_DF()
            return table
    

    #################
    ## MLDE SEARCH ##
    #################
    @reactive.Effect
    @reactive.event(input.mlde_search)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Searching for new mutants", detail="Preparing genetic algorithm...")

            model = MODEL()
            mlde_explore = input.mlde_explore()
            optim_problem = OPTIM_DICT[input.optim_problem()]
            max_eval = MAX_EVAL_DICT[model.x]

            out = model.search(optim_problem=optim_problem, method='ga', max_eval=max_eval, explore=mlde_explore, batch_size=BATCH_SIZE, pbar=p)

            MLDE_SEARCH_DF.set(out)

    
    ### RENDER MLDE TABLE ###
    @output
    @render.data_frame
    def mlde_search_table(alt=None):
        table = MLDE_SEARCH_DF()
        model = MODEL()

        table = table.drop(['sequence'], axis=1)
        if 'y_true' in table.columns:
            table = table.drop(['y_true'], axis=1)
            
        return table


    ### DOWNLOAD MLDE RESULTS ###
    @output
    @render.ui
    def mlde_download_ui():
        out = MLDE_SEARCH_DF()
        if out is not None:
            return ui.TagList(
                ui.output_data_frame("mlde_search_table"),
                ui.download_button("download_mlde_search", "Download discovery results"),
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
            
        new_test = round((n_test_max)/2, 2)

        ui.update_numeric(
            "discovery_n_test",
            min=0,
            max = n_test_max,
            value = new_test
        )
        new_val_max = 100 - n_train - new_test
        ui.update_numeric(
            "discovery_n_val",
            min=0,
            max = new_val_max,
            value = new_val_max
        )


    ### N-TEST COMPUTATION ###
    @reactive.Effect
    @reactive.event(input.discovery_n_test)
    def _():
        n_train = input.discovery_n_train()
        n_test = input.discovery_n_test()
        n_val_max = 100 - n_train - n_test

        ui.update_numeric(
            "n_val",
            max = n_val_max,
            value = n_val_max
        )


    ################
    ##  DISCOVERY ##
    ################

    ### TRAIN DISCOVERY MODEL ###
    @reactive.Effect
    @reactive.event(input.discovery_train_button)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")
            
            rep_type = REP_DICT[input.discovery_model_rep_type()]
            lib = LIBRARY()

            split = (input.discovery_n_train(), input.discovery_n_test(), input.discovery_n_val())
            k_folds = input.discovery_k_folds()
            if k_folds <= 1:
                k_folds = None

            m = pai.Model(model_type=MODEL_DICT[input.discovery_model_type()])
            out = m.train(library=lib, x=rep_type, split=split, seed=input.discovery_random_seed(), model_type=MODEL_DICT[input.discovery_model_type()], k_folds=k_folds, pbar=p)
            model_lib = pai.Library(user=lib.user, source=out)
            val_df = pd.DataFrame({'names':m.val_names, 'y_true':m.y_val, 'y_pred':m.y_val_pred, 'y_sigma':m.y_val_sigma})

            # set reactive variables
            DISCOVERY_LIB.set(model_lib)
            DISCOVERY_MODEL.set(m)
            DISCOVERY_VAL_DF.set(val_df)


    ### RENDER DISCOVERY PLOT ###
    @output
    @render.plot
    def discovery_plot(alt=None):
        if DISCOVERY_LIB():
            lib = DISCOVERY_LIB()
            model = DISCOVERY_MODEL()
            vis_method = input.discovery_vis_method()
            with ui.Progress(min=1, max=15) as p:
                p.set(message="Visualizing results", detail="This may take a while...")
                names = lib.names
                
                # Update to pass the new parameters
                if vis_method == 't-SNE':
                    fig, ax, df = lib.plot_tsne(rep=model.x, names=names)
                elif vis_method == 'UMAP':
                    fig, ax, df = lib.plot_umap(rep=model.x, names=names)
                elif vis_method == 'PCA':
                    fig, ax, df = lib.plot_pca(rep=model.x, names=names)
                return fig, ax


    ### RENDER PREDICTED VERSUS TURE DATAFRAME ###
    @output
    @render.data_frame
    def discovery_table(alt=None):
        df = DISCOVERY_VAL_DF()
        if DISCOVERY_MODEL() == None:
            return None
        else:
            model = DISCOVERY_MODEL()
            class_dict = model.library.class_dict
            df['y_true'] = [class_dict[i] for i in df['y_true']]
            df['y_pred'] = [class_dict[i] for i in df['y_pred']]
            return df


    ### DISCOVERY SEARCH ###    
    @reactive.Effect
    @reactive.event(input.discovery_search)
    def _():
        with ui.Progress(min=1, max=10000) as p:
            p.set(message="Sampling diverse sequences", detail=f"...")
            
            labels = input.sample_from()
            if labels == ():
                labels = ['all']
            if type(labels) == str:
                labels = [labels]

            
            model = DISCOVERY_MODEL()
            out, search_results = model.search(N=input.n_samples(), labels=labels, method='ga', pbar=p)

            DISCOVERY_DF.set(out['df'])

            DISCOVERY_SEARCH.set(search_results)


    ### RENDER SEARCH PLOT ###
    @output
    @render.plot
    def discovery_search_plot(alt=None):
        highlight_mask = DISCOVERY_SEARCH()
        if highlight_mask is not None:
            with ui.Progress() as p:
                p.set(message="Visualizing search results", detail="...")
                model = DISCOVERY_MODEL()
                lib = DISCOVERY_LIB()
                vis_method = input.discovery_vis_method()
                names = lib.names
                
                # Update to pass the new parameters
                if vis_method == 't-SNE':
                    fig, ax, df = lib.plot_tsne(rep=model.x, names=names, highlight_mask=highlight_mask, highlight_label="Sampled")
                elif vis_method == 'UMAP':
                    fig, ax, df = lib.plot_umap(rep=model.x, names=names, highlight_mask=highlight_mask, highlight_label="Sampled")
                elif vis_method == 'PCA':
                    fig, ax, df = lib.plot_pca(rep=model.x, names=names, highlight_mask=highlight_mask, highlight_label="Sampled")
                return fig, ax


    ### RENDER DISCOVERY TABLE ###
    @output
    @render.data_frame
    def discovery_search_table(alt=None):
        table = DISCOVERY_DF()
        model = DISCOVERY_MODEL()
        seq_col = model.library.seq_col
        table = table.drop(seq_col, axis=1)
        return table


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
    @render.download(
        filename=lambda: f"{LIBRARY().fname}_discovery.csv"
    )
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