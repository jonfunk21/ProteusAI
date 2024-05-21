# This source code is part of the proteusAI package and is distributed
# under the MIT License.

"""
The proteusAI shiny app.
"""

__name__ = "proteusAI"
__author__ = "Jonathan Funk"

import shiny
from shiny import App, ui, render, Inputs, Outputs, Session, reactive
from shiny.types import FileInfo, ImgData
from shiny.plotutils import brushed_points, near_points
import pandas as pd
import sys
import os
app_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(app_path, '../src/'))
import proteusAI as pai
import os
import matplotlib.pyplot as plt

VERSION = "version " + "0.1"
representation_types = ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"] # Add VAE and MSA-Transformer later
train_test_val_splits = ["Random"]
model_types = ["Gaussian Process", "Random Forrest", "KNN", "SVM"]
model_dict = {"Random Forrest":"rf", "KNN":"knn", "SVM":"svm", "VAE":"vae", "ESM-2":"esm2", "ESM-1v":"esm1v", "Gaussian Process":"gp"}
representation_dict = {"One-hot":"ohe", "BLOSUM50":"blosum50", "BLOSUM62":"blosum62", "ESM-2":"esm2", "ESM-1v":"esm1v", "VAE":"vae"}
FAST_INTERACT_INTERVAL = 60 # in milliseconds
SIDEBAR_WIDTH = 450
BATCH_SIZE = 10
ZS_MODELS = ["ESM-2", "ESM-1v"]

# TODO: check if Project path exists, create one if not
# TODO: if no project path is provided create a temporary file system on the server - which can then be downloaded

app_ui = ui.page_fluid(
    
    #ui.panel_title("ProteusAI"),
    ui.output_image("image", inline=True),
    VERSION,
    
    ui.navset_card_tab(

        ###############
        ## DATA PAGE ##
        ###############
        ui.nav_panel(
            "Data", 
            
            ui.layout_sidebar(
                ui.sidebar(
                    
                    ui.navset_tab(
                        ui.nav_panel("Library",

                                ui.input_file(id="dataset_file", label="Select dataset (Default: demo dataset)", accept=['.csv', '.xlsx', '.xls'], placeholder="None"),

                               
                               # CHANGE THIS TO EMPTY STRING LATER
                               ui.input_text(id="project_path", label="Project Path (Default: demo path)", value="demo/example_project"),
                               
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
                                       ui.input_select("y_type", "Data type", ["Numeric", "Categorical"])
                                   ),
                               ),
                    
                               ui.input_action_button('confirm_dataset', 'Continue'),
                               
                               
                        ),
                        ui.nav_panel("Sequence",
                            ui.input_file(id="protein_file", label="Upload FASTA", accept=['.fasta'], placeholder="None"),
                            ui.input_text(id="protein_path", label="Project Path (Default: demo path)", value="demo/example_project"),
                            ui.input_action_button('confirm_protein', 'Continue'),
                            
                        ),
                        ui.nav_panel("Structure",
                            ui.input_file(id="structure_file", label="Upload Structure", accept=['.pdb'], placeholder="None"),
                            ui.input_text(id="structure_path", label="Project Path (Default: demo path)", value="demo/example_project"),
                            ui.input_action_button('confirm_structure', 'Continue'),
                            
                        ),
                        
                    ),
                width=SIDEBAR_WIDTH
                ),
                # Main panel
                ui.panel_conditional("typeof output.protein_fasta !== 'string'",
                    "Upload experimental data (CSV or Excel file) or a single protein (FASTA)",
                    ui.output_data_frame("dataset_table"),
                ),
                
                ui.output_text("protein_fasta"),
                ui.output_text("protein_struc")

            ),
        ),

        ####################
        ## Zero-shot PAGE ##
        ####################
        ui.nav_panel(
            "Zero-Shot",

            ui.layout_sidebar(
                ui.sidebar(

                    ui.output_ui("zero_shot_ui"),
                    width=SIDEBAR_WIDTH
                ),

                ui.panel_conditional("typeof output.protein_fasta === 'string'",
            
                    ui.output_plot("entropy_plot"),

                    ui.output_plot("scores_plot")
                ),
            ),
        ),

        ##########################
        ## Representations PAGE ##
        ##########################
        ui.nav_panel("Representations", 
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.output_ui("representations_ui"),
                    width=SIDEBAR_WIDTH
                ),

                ui.panel_conditional(
                    "typeof output.protein_struc === 'string'",
                    ui.output_ui("struc3D"),
                ),

                #ui.output_ui("struc3D"),
                ui.output_plot('tsne_plot'),
                
                #ui.panel_conditional("typeof output.protein_fasta !== 'string'",
                #                     
                #    ui.output_plot('tsne_plot')
                #),               
            )
        ),

        ################
        ## MLDE PAGE ##
        ################
        ui.nav_panel("MLDE", 
            ui.layout_sidebar(
                ui.sidebar(
                    ui.row(
                        ui.column(12,
                            ui.output_ui("mlde_dynamic_ui")
                        ),
                        ui.column(6,
                            ui.input_select("model_type", "Surrogate model", model_types)
                        ),
                        ui.column(6,
                            ui.input_select("model_task", "Model task", ["Regression", "Classification"])
                        ),
                        # TODO: Only show the computed representation types
                        ui.column(6,
                            ui.input_select("model_rep_type", "Representaion type", representation_types),
                        )
                        
                    ),
                    
                    ui.input_checkbox("customize_model_params", "Customize model parameters", value=False),
                    

                    
                    ui.row(
                        ui.column(6,
                            ui.input_select("train_split","Train, test, validation split method", train_test_val_splits)
                        ),
                        ui.column(6,
                            ui.input_slider("random_seed", "Random seed", min=0, max=1024, value=42)
                        ),
                        ui.column(12,
                            "Cross-validation split:"
                        ),
                        ui.column(4,
                            ui.input_numeric("n_train", "Training (%)", value=80, min=0, max=100)
                        ),
                        ui.column(4,
                            ui.input_numeric("n_test", "Test (%)", value=10, min=0, max=100)
                        ),
                        ui.column(4,
                            ui.input_numeric("n_val", "Validation (%)", value=10, min=0, max=100)
                        ),
                        ui.column(8,
                            ui.input_action_button("review_data", "Review data")
                        ),
                        ui.column(4,
                            ui.input_action_button("train_button", "Train")
                        )
                    ),

                width=SIDEBAR_WIDTH
                ),
                ui.output_ui("pred_vs_true_ui"),

                ui.output_data_frame("model_table")
            )
        ),

        #################
        ## Design PAGE ##
        #################
        
        ui.nav_panel(
            "Design",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.navset_tab(
                        ui.nav_panel(
                            "Single sequence",
                            ui.row(
                                ui.column(6,
                                    ui.input_text("new_seq", "Predict Y-value for new sequence"),
                                ),
                                ui.column(6,
                                    "Your prediction here"
                                )
                            ),
                            "It would also be nice to attach measures of confidence here, e.g. model confidence or similarity to training data."
                        ),
                        ui.nav_panel(
                            "Upload new library",
                            ui.row(
                                ui.input_file("new_seqs_file", "Upload sequences")
                            )
                        ),
                    ),
                width=SIDEBAR_WIDTH),
                #ui.panel_main(

                #)
            )
        ),

        ui.nav_panel(
            "Download",
        ),

        ui.nav_panel(
            "About us",
        ),
    )
)

def server(input: Inputs, output: Outputs, session: Session):

    # Operation mode
    MODE = reactive.Value('start')

    ### Homepage ###
    # App logo
    @output
    @render.image
    def image():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"),  "height": "75px"}
        return img
    
    # dummy data set for demo purposes
    dummy = pd.DataFrame({
        "Sequence":["MGVARGTV...G", "AGVARGTV...G", "...", "MGVARGTV...V"],
        "Description":["wt", "M1A", "...", "G142V"],
        "Activity":["0.5", "0.32", "...", "0.7"]
    })

    # dummy dataset until real dataset is entere
    #dataset = reactive.Value(pd.read_csv("app/demo_data.csv"))
    dataset = reactive.Value(dummy)
    dataset_path = reactive.Value(str)
    library = reactive.Value(None)
        
    # Reading dataset
    @reactive.Effect
    @reactive.event(input.dataset_file)
    def _():
        df = dataset()
        f: list[FileInfo] = input.dataset_file()
        df = pd.read_csv(f[0]["datapath"])
        dataset.set(df)
        dataset_path.set(f[0]["datapath"])

    @output
    @render.data_frame
    def dataset_table():
        df = render.DataTable(dataset(), summary=True)
        return df
    
    @reactive.Effect()
    def _():
        cols = list(dataset().columns)

        ui.update_select(
            "seq_col",
            label="Select Sequences",
            choices=cols,
            selected=cols[0],
        )
        ui.update_select(
            "description_col",
            label="Select Descriptions",
            choices=cols,
            selected=cols[1],
        )
        ui.update_select(
            "y_col",
            label=f"Select Y-values",
            choices=cols,
            selected=cols[-1],
        )

    # check if data is numerical
    def is_numerical(data):
        """
        Check if the data is numerical.
        Args:
            data: List of data values.
        Returns:
            Boolean: True if all data is numerical (int, float, complex), False otherwise.
        """
        return all(isinstance(x, (int, float, complex)) for x in data)
    
    # Loading library data
    representation_path = reactive.Value(None)
    available_reps = reactive.Value(representation_types)

    @reactive.Effect
    @reactive.event(input.confirm_dataset)
    def _():
        ys = dataset()[input.y_col()].to_list()

        # Determine if the data is numerical or categorical
        if is_numerical(ys):
            y_type = "num"
            choice = "Regression"
            _y_type = "Numeric"
        else:
            y_type = "class"
            choice = "Classification"
            _y_type = "Categorical"
    
        # TRY TO FIND A MORE STABLE SOLUTION HERE
        seqs = dataset()[input.seq_col()].to_list()
        names = dataset()[input.description_col()].to_list()

        lib = pai.Library(project=input.project_path(), seqs=seqs, ys=ys, y_type=y_type, names=names, proteins=[])
        
        library.set(lib)

        # update representation selection
        # TODO: Make sure these have to be 100% computed
        inverted_reps = {v: k for k, v in representation_dict.items()}
        ui.update_select(
            "model_rep_type",
            choices=[inverted_reps[i] for i in lib.reps]
        )

        #ui.update_select(
        #    "plot_rep_type",
        #    choices=[inverted_reps[i] for i in lib.reps]
        #)
        available_reps.set([inverted_reps[i] for i in lib.reps])
        
        # update available model tasks
        ui.update_select(
            "model_task",
            choices=[choice]
        )

        ui.update_select(
            "y_type",
            choices=[_y_type]
        )

        protein.set(None)
        
        representation_path.set(None) # used in train
        
        MODE.set("dataset")

    
    protein = reactive.Value(None)
    zs_results = reactive.Value([])

    # reading protein fasta
    @reactive.Effect
    @reactive.event(input.protein_file)
    def _():
        #prot = protein()
        f: list[FileInfo] = input.protein_file()
        prot = pai.Protein(fasta=f[0]["datapath"], project=input.protein_path())
        prot.name = f[0]["name"].split('.')[0]
        protein.set(prot)
        dataset_path.set(f[0]["datapath"])

    # Confirm protein
    @reactive.Effect
    @reactive.event(input.confirm_protein)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Searching for available data...", detail="This may take a while...")
            # initialize protein
            #prot = protein()
            f: list[FileInfo] = input.protein_file()
            prot = pai.Protein(fasta=f[0]["datapath"], project=input.protein_path())
            name = f[0]["name"].split('.')[0]
            prot.name = name
            
            # set shiny variables
            protein.set(prot)
            print(protein())
            dataset_path.set(f[0]["datapath"])
            MODE.set('zero-shot')

            # check for zs-computations
            zs_computed = []
            rep_computed = []
            for model in ZS_MODELS:
                # check hash existence
                zs_path = os.path.join(prot.project, f"zero_shot/{name}/{model_dict[model]}")

                if os.path.exists(zs_path):
                    if "zs_scores.csv" in os.listdir(zs_path):
                        zs_computed.append(model)
                
                for rep in representation_types:
                    rep_path = os.path.join(zs_path, "rep", rep)
                    if os.path.exists(rep_path):
                        rep_computed.append(rep)

                
            zs_results.set(zs_computed)

            # load zs-library if exists
            for model in ZS_MODELS:
                try:
                    rep_path = os.path.join(prot.project, f"zero_shot/{prot.name}/rep/{representation_dict[model]}")
                    df_path = os.path.join(prot.project, f"zero_shot/{prot.name}/{representation_dict[model]}/zs_scores.csv")
                    
                    df = pd.read_csv(df_path)
                    p.set(message="Loading data...", detail="This may take a while...")
                    lib = pai.Library(project=input.protein_path(), seqs=df.sequence, ys=df.mmp, y_type="num", names=df.mutant.to_list(), proteins=[], rep_path=rep_path)

                    library.set(lib)
                    dataset.set(df)
                    zs_scores.set(df)
                    
                    ui.update_select(
                        "model_rep_type",
                        choices=[inverted_reps[i] for i in lib.reps]
                    )

                    #ui.update_select(
                    #    "plot_rep_type",
                    #    choices=[inverted_reps[i] for i in lib.reps]
                    #)
                    available_reps.set([inverted_reps[i] for i in lib.reps])
                except:
                    pass
        

        # update uis
        ui.update_select(
            "model_task",
            choices=["Regression"]
        )   

        # update representation selection
        ui.update_select(
            "zs_scores",
            choices=zs_computed
        ) 

    @output
    @render.text
    def protein_fasta():
        if protein() == None:
            seq = None
        else:
            seq = 'Protein name: ' + protein().name + '\n' + protein().seq
        return seq
    
    # Confirm structure
    @reactive.Effect
    @reactive.event(input.confirm_structure)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Searching for available data...", detail="This may take a while...")
            prot = protein()
            f: list[FileInfo] = input.structure_file()
            prot = pai.Protein(struc=f[0]["datapath"], project=input.project_path())

            name = f[0]["name"].split('.')[0]
            prot.name = name
            
            # set shiny variables
            protein.set(prot)
            dataset_path.set(f[0]["datapath"])
            MODE.set('structure')

            # load zs-library if exists
            for model in ZS_MODELS:
                try:
                    rep_path = os.path.join(prot.project, f"zero_shot/{prot.name}/rep/{representation_dict[model]}")
                    df_path = os.path.join(prot.project, f"zero_shot/{prot.name}/{representation_dict[model]}/zs_scores.csv")
                    
                    df = pd.read_csv(df_path)
                    p.set(message="Loading data...", detail="This may take a while...")
                    lib = pai.Library(project=input.protein_path(), seqs=df.sequence, ys=df.mmp, y_type="num", names=df.mutant.to_list(), proteins=[], rep_path=rep_path)

                    library.set(lib)
                    dataset.set(df)
                    zs_scores.set(df)
                    inverted_reps = {v: k for k, v in representation_dict.items()}

                    ui.update_select(
                        "model_rep_type",
                        choices=[inverted_reps[i] for i in lib.reps]
                    )

                    #ui.update_select(
                    #    "plot_rep_type",
                    #    choices=[inverted_reps[i] for i in lib.reps]
                    #)
                    available_reps.set([inverted_reps[i] for i in lib.reps])
                except:
                    pass
        

    @output
    @render.text
    def protein_struc():
        if protein() == None:
            struc = None

        elif protein().struc != None:
            struc = 'Structure loaded'

        else:
            struc = None
        return struc

    #########################
    ## Representations TAB ##
    #########################
    
    # Dataset case
    # Visualizations
    tsne_df = reactive.Value()   

    # Dynamic sidebar tab for Zero-shot tab
    @output
    @render.ui
    def representations_ui():
        if MODE() != "dataset" or MODE() == "":
            return ui.TagList(
                ui.h4("Dataset mode"),
                ui.row(
                    ui.column(7,
                        ui.input_select("dat_rep_type", "Compute representations", representation_types),
                    ),
                    ui.column(5,
                        ui.input_action_button("dat_compute_reps", "Compute"),
                            #f"Representations 100 % computed",
                            style='padding:25px;'
                        )
                    ),

                    ui.h4("Visualization"),

                    ui.panel_conditional("input.dat_rep_type === 'VAE' || input.dat_rep_type === 'MSA-Transformer'",
                        ui.input_file("MSA_vae_training", "Upload MSA file")
                    ),

                    ui.panel_conditional("input.dat_rep_type === 'VAE'",
                        ui.input_checkbox("custom_vae", "Customize VAE parameters"),
                        ui.input_action_button("train_vae", "Train VAE")
                    ),
                
                    ui.input_select("vis_method","Visualization Method",["t-SNE", "PCA"]),
                    
                    ui.input_select("color_by", "Color by", ["Y-value", "Site", "Custom"]),
                    
                    # Conditional panel for Site
                    ui.panel_conditional("input.color_by === 'Site'",
                            ui.input_text("color_text","Select sites to color seperated by ';' (e.g. 21;42)")
                        ),
                    
                    # Conditional panel for Y-value with numeric data
                    ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'Numeric'",
                            ui.row(
                                ui.column(6,
                                        ui.input_numeric("y_upper", "Choose an upper limit for y", value=None)  
                                    ),
                                ui.column(6,
                                        ui.input_numeric("y_lower", "Choose an lower limit for y", value=None)  
                                    ),
                            )
                        ),
                    # Conditional panel fo Y-value with categorical data
                    ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'Categorical'",
                            ui.input_text("selected_classes","Select classes to colorize seperated by ';' (e.g. class1;class2)")
                        ),
                    
                    ui.input_text("hide_sites", "Hide points based on site seperated by ';' (e.g. 21;42)"),

                    ui.input_checkbox("hide_by_y", "Hide points based Y-Value", value=False),
                    ui.panel_conditional("input.hide_by_y === true",
                        ui.row(
                        # change these to be the min and max values observed in the library
                        ui.column(6,
                                    ui.input_slider("hide_upper_y","hide points above y", min=0, max=100, value=100)
                            ),
                        ui.column(6,
                                    ui.input_slider("hide_lower_y","hide points below y", min=0, max=100, value=0)
                            )
                    ),

                    ),
                    ui.row(
                        ui.column(12, "Visualize representations"),
                        ui.column(7,
                            ui.input_select("plot_rep_type", "", available_reps()),
                        ),
                            ui.column(5,
                            ui.input_action_button("update_plot", "Update plot")
                        )
                    )
            )
        else:
            return ui.TagList(
                h4("Upload a Librariy in the 'Data' tab or compute a library in the 'Zero-shot' tab for more visualization options.")
            )
        

    # Dynamic sidebar tab for Zero-shot tab
    @output
    @render.ui
    def zero_shot_ui():
        if MODE() == "zero-shot" or MODE() == "structure":
            return ui.TagList(
                ui.h4("Zero-shot modeling"),
                ui.row(
                    ui.h5("Compute a zero-shot Library"),
                    ui.column(7,
                        # add MSA-Transformer and VAE later
                        ui.input_select("zs_model", "Choose a model", ZS_MODELS)
                    ),

                    ui.column(5,
                        ui.input_action_button("compute_zs", "Compute"),
                        style='padding:25px;'
                    ),
                    
                    ui.h4("Visualize"),
                    
                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_entropy", "Plot Entropy")
                        ),
                        ui.column(6,
                            ui.input_checkbox("plot_entropy_section", "Customize plot"),
                            style='padding:10px;'
                        ),
                        ui.panel_conditional("input.plot_entropy_section === true",
                            ui.row(
                                ui.column(6,
                                    ui.input_slider("entropy_slider", "Sliding window", min=0, max=len(protein().seq), value=0)
                                ),
                                ui.column(6,
                                    ui.input_numeric("entropy_width", "Sliding window width", value=10)
                                )
                            )

                        )
                    ),

                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_scores", "Plot Scores")
                        ),

                        ui.column(6,
                            ui.input_checkbox("plot_scores_section", "Customize plot"),
                            style='padding:10px;'
                        ),

                        ui.panel_conditional("input.plot_scores_section === true", 
                            ui.row(
                                ui.column(6,
                                    ui.input_slider("scores_slider", "Sliding window", min=0, max=len(protein().seq), value=0)
                                ),
                                ui.column(6,
                                    ui.input_numeric("scores_width", "Sliding window width", value=10)
                                ),
                            )
                        ),
                    ),
                    #ui.row(
                    #    ui.column(7,
                    #        ui.input_select('zs_rep_type', "Compute representations", representation_types)
                    #    ),
                    #    ui.column(5,
                    #    ui.input_action_button("zs_compute_reps", "Compute"),
                    #        #f"Representations 100 % computed",
                    #        style='padding:25px;'
                    #    )
                    #)
                ),
            )
        else:
            return ui.TagList(
                "The zero-shot mode is available for individual proteins and structures."
            )

    # Compute representations
    @reactive.Effect
    @reactive.event(input.dat_compute_reps)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Computation in progress", detail="This may take a while...")

            print(f"Computing library: {representation_dict[input.dat_rep_type()]}")
            
            lib = library()
            
            lib.compute(method=representation_dict[input.dat_rep_type()], batch_size=BATCH_SIZE)

            library.set(lib)
            print("Done!")

            # update representation selection
            inverted_reps = {v: k for k, v in representation_dict.items()}
            ui.update_select(
                "model_rep_type",
                choices=[inverted_reps[i] for i in lib.reps]
            )

            #ui.update_select(
            #    "plot_rep_type",
            #    choices=[inverted_reps[i] for i in lib.reps]
            #)
            available_reps.set([inverted_reps[i] for i in lib.reps])

    # Output dataset mode
    @output
    @render.plot
    @reactive.event(input.update_plot)
    def tsne_plot():
        """
        Render plot once button is pressed.
        """
        with ui.Progress(min=1, max=15) as p:
            #if MODE() == "dataset":
            p.set(message="Plotting", detail="This may take a while...")
            lib = library()
            names = lib.names
            y_upper = input.y_upper()
            y_lower = input.y_lower()
            rep = representation_dict[input.plot_rep_type()]
            
            # Update to pass the new parameters
            fig, ax, df = lib.plot_tsne(rep=rep, y_upper=y_upper, y_lower=y_lower, names=names)

            tsne_df.set(df)
            return fig, ax
    
        
    # Zero-shot modeling
    zs_scores = reactive.Value(pd.DataFrame())

    # compute zero-shot scores
    @reactive.Effect
    @reactive.event(input.compute_zs)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Computing", detail="This may take several minutes...")
            prot = protein()
            model = representation_dict[input.zs_model()]

            print(f"computing zero shot scores using {model}")

            df = prot.zs_prediction(model=model, batch_size=BATCH_SIZE)
            
            rep_path = os.path.join(prot.project, f"zero_shot/{prot.name}/rep/", representation_dict[input.model_rep_type()])

            lib = pai.Library(project=input.protein_path(), seqs=df.sequence, ys=df.mmp, y_type="num", names=df.mutant.to_list(), proteins=[], rep_path=rep_path)
            
            library.set(lib)
            dataset.set(df)
            zs_scores.set(df)
            print(df)
    

    # Output protein mode
    @output
    @render.plot
    @reactive.event(input.plot_entropy)
    def entropy_plot():
        """
        Create the per position entropy plot
        """
        prot = protein()
        seq = prot.seq

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

        prot = protein()
        fig = prot.plot_entropy(section=section)
        return fig
    
    @reactive.Effect
    @reactive.event(input.plot_entropy_section)
    def _():
        if input.plot_entropy_section():
            ui.update_action_button(
                "plot_entropy",
                label="Update Entropy"
            )

    @output
    @render.plot
    @reactive.event(input.plot_scores)
    def scores_plot():
        """
        Create the per position entropy plot
        """
        prot = protein()
        seq = prot.seq

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
        
        fig = prot.plot_scores(section=section, color_scheme = "rwb")
        return fig
    
    @reactive.Effect
    @reactive.event(input.plot_scores_section)
    def _():
        if input.plot_scores_section():
            ui.update_action_button(
                "plot_scores",
                label="Update Scores"
            )

    # compute representations for Zero-Shot
    @reactive.Effect
    @reactive.event(input.zs_compute_reps)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Computation in progress", detail="This may take a while...")

            print(f"Computing library: {representation_dict[input.zs_rep_type()]}")
            
            prot = protein()

            wt_seq = prot.seq
            canonical_aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            mutants, sequences = [], []
            for pos in range(len(wt_seq)):
                for aa in canonical_aas:
                    if wt_seq[pos] != aa:
                        mutants.append(wt_seq[pos] + str(pos+1) + aa)
                        sequences.append(wt_seq[:pos] + aa + wt_seq[pos+1:])
            
            dest = os.path.join(prot.project, f"zero_shot/{prot.name}/rep/{representation_dict[input.zs_rep_type()]}")

            lib = pai.Library(project=input.protein_path(), seqs=sequences, names=mutants, proteins=[])
            lib.compute(method=representation_dict[input.zs_rep_type()], dest=dest, batch_size=BATCH_SIZE)
            
            print("Done!")

            # update representation selection
            inverted_reps = {v: k for k, v in representation_dict.items()}
            
            ui.update_select(
                "model_rep_type",
                choices=[inverted_reps[i] for i in lib.reps]
            )

            #ui.update_select(
            #    "plot_rep_type",
            #    choices=[inverted_reps[i] for i in lib.reps]
            #)
            available_reps.set([inverted_reps[i] for i in lib.reps])

    ### structure mode 
    @render.ui
    def struc3D():
        view = protein().view_struc()
        return ui.TagList(
            ui.HTML(view.write_html())
        )
    
    ### MLDE tab ###
    @output
    @render.ui
    def mlde_dynamic_ui():        
        if MODE() == "zero-shot":
            return ui.TagList(
                ui.h4("MLDE from zero-shot data")
            )
        if MODE() == "dataset":
            return ui.TagList(
                ui.h4("MLDE from experimental data")
            )
        
    model = reactive.Value(None)
    
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

    # Reviewing data
    data_reviewed = reactive.value(None)

    @reactive.Effect
    @reactive.event(input.review_data)
    def _():
        print("button pressed")
        print(dataset())
        print(zs_results())
        data_reviewed.set('reviewed')

    # Training models
    val_df = reactive.Value(pd.DataFrame({'names':[], 'y_true':[], 'y_pred':[]}))

    # Train model
    @reactive.Effect
    @reactive.event(input.train_button)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")

            if input.train_split() == "Random":
                split = "random"
            else:
                print(f"{input.train_split()} is not implemented yet - choosing random split")
                split = "random"

            rep_type = representation_dict[input.model_rep_type()]
            if MODE() in ["zero-shot", "structure"]:
                prot = protein()
                name = prot.name
                rep_path = os.path.join(prot.project, f"zero_shot/{prot.name}/rep/", representation_dict[input.model_rep_type()])
            else:
                rep_path = None

            lib = library()

            print(f"training {model_dict[input.model_type()]}")

            m = pai.Model(model_type=model_dict[input.model_type()], seed=input.random_seed())
            m.train(library=lib, x=rep_type, split=split, seed=input.random_seed(), model_type=model_dict[input.model_type()], rep_path=rep_path)

            print("training done!")

            model.set(m)
            val_df.set(pd.DataFrame({'names':m.val_names, 'y_true':m.y_val, 'y_pred':m.y_val_pred}))

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
    
    @output
    @render.plot
    def pred_vs_true():
        df = val_df()
        if model() != None:
            p = model().true_vs_predicted(y_true=df.y_true.to_list(), y_pred=df.y_pred.to_list())
        else:
            p = None
        return p

    @output
    @render.data_frame()
    def model_table():
        df = val_df()
        if len(df) == 0:
            return None
        else:
            return df

app = App(app_ui, server)
