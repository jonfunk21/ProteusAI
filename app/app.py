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
sys.path.append('src/')
import proteusAI as pai
import os
import matplotlib.pyplot as plt
import plotnine
import hashlib

VERSION = "version " + "0.1"
representation_types = ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"] # Add VAE and MSA-Transformer later
train_test_val_splits = ["Random"]
model_types = ["Random Forrest", "KNN", "SVM"] # Add VAEs later
model_dict = {"Random Forrest":"rf", "KNN":"knn", "SVM":"svm", "VAE":"vae", "ESM-2":"esm2", "ESM-1v":"esm1v"}
representation_dict = {"One-hot":"ohe", "BLOSUM50":"blosum50", "BLOSUM62":"blosum62", "ESM-2":"esm2", "ESM-1v":"esm1v", "VAE":"vae"}
FAST_INTERACT_INTERVAL = 60 # in milliseconds
SIDEBAR_WIDTH = 450
BATCH_SIZE = 100
ZS_MODELS = ["ESM-2", "ESM-1v"]
print(plotnine.__version__)

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
                        ui.nav_panel("Load Library",

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
                    
                               ui.input_action_button('confirm_dataset', 'Confirm Selection'),
                               
                               
                        ),
                        ui.nav_panel("Single Protein",
                            ui.input_file(id="protein_file", label="Upload FASTA", accept=['.fasta'], placeholder="None"),
                            ui.input_text(id="protein_path", label="Project Path (Default: demo path)", value="demo/example_project"),
                            ui.input_action_button('confirm_protein', 'Confirm Selection'),
                            
                        ),
                        
                    ),
                width=SIDEBAR_WIDTH
                ),
                # Main panel
                ui.panel_conditional("typeof output.protein_fasta !== 'string'",
                    "Upload experimental data (CSV or Excel file) or a single protein (FASTA)",
                    ui.output_data_frame("dataset_table"),
                ),
                
                ui.output_text("protein_fasta")

            ),
        ),

        ##################
        ## Analyze PAGE ##
        ##################
        ui.nav_panel("Analyze", 
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.output_ui("analyze_ui"),
                    width=SIDEBAR_WIDTH
                ),

                ui.panel_conditional("typeof output.protein_fasta === 'string'",
            
                    ui.output_plot("entropy_plot"),

                    ui.output_plot("scores_plot")
                ),
                
                ui.panel_conditional("typeof output.protein_fasta !== 'string'",
                                     
                    ui.output_plot('tsne_plot')

                ),

               
            )
        ),

        #################
        ## MODELS PAGE ##
        #################
        ui.nav_menu("Learn",
                    
            ui.nav_panel("Supervised", 
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.row(
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
                        
                        ui.panel_conditional("input.customize_model_params === true",
                                "Not implemented yet LOL",
                                ui.input_text("frustrations", "Draft your angry tweet here (How do we call tweets on X now?):")            
                            ),

                        
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
                    #ui.panel_main(
                    ui.output_ui("pred_vs_true_ui"),
                    #ui.tags.b("Points near cursor"),
                    #ui.output_table("near_hover"),
                    #ui.output_table("in_brush")
                #)
                )
            ),

            ui.nav_panel(
                "Load model",
                "Under construction...",            
            )
                    
        ),

        ##################
        ## PREDICT PAGE ##
        ##################
        ui.nav_menu(
            "Design",
            ui.nav_panel(
                "New sequences",
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
                "New library",
                "Under construction (Lauras M.Sc. Thesis)..."  
            ),
        ),
        

        ui.nav_panel(
            "Download Results",

        ),
    )
)

def server(input: Inputs, output: Outputs, session: Session):

    # Operation mode
    MODE = reactive.Value(None)

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
        ui.update_select(
            "plot_rep_type",
            choices=[inverted_reps[i] for i in lib.reps]
        )
        
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
        MODE.set("dataset")

    
    protein = reactive.Value(None)
    zs_results = reactive.Value([])

    #reading protein fasta
    @reactive.Effect
    @reactive.event(input.protein_file)
    def _():
        prot = protein()
        f: list[FileInfo] = input.protein_file()
        prot = pai.Protein(file=f[0]["datapath"], project=input.protein_path())
        protein.set(prot)
        dataset_path.set(f[0]["datapath"])

    @reactive.Effect
    @reactive.event(input.confirm_protein)
    def _():
        # initialize protein
        prot = protein()
        f: list[FileInfo] = input.protein_file()
        prot = pai.Protein(file=f[0]["datapath"], project=input.protein_path())

        # set shiny variables
        protein.set(prot)
        dataset_path.set(f[0]["datapath"])
        MODE.set('protein')

        # check for zs-computations
        seq = prot.seq
        computed = []
        for model in ZS_MODELS:
            # compute hash
            zs_hash = hashlib.md5((seq+model_dict[model]).encode()).hexdigest()

            # check hash existence
            zs_path = os.path.join(prot.project, f"zero_shot/{model_dict[model]}/{zs_hash}")

            if os.path.exists(zs_path):
                computed.append(model)
            
        zs_results.set(computed)

        print(zs_results())

    @output
    @render.text
    def protein_fasta():
        if protein() == None:
            seq = None
        else:
            seq = 'Protein name: ' + protein().name + '\n' + protein().seq
        return seq
        
    #################
    ## Analyze TAB ##
    #################
    
    # Dataset case
    # Visualizations
    tsne_df = reactive.Value()   

    # Dynamic sidebar tab for Analyze tab
    @output
    @render.ui
    def analyze_ui():
        if MODE() == "dataset":
            return ui.TagList(
                ui.h4("Dataset mode"),
                ui.row(
                    ui.column(6,
                        ui.input_select("dat_rep_type", "Compute representation", representation_types),
                    ),
                    ui.column(6,
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
                                        ui.input_numeric("y_upper", "Choose upper limit for y", value=None)  
                                    ),
                                ui.column(6,
                                        ui.input_numeric("y_lower", "Choose lower limit for y", value=None)  
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
                        ui.column(6,
                            ui.input_select("plot_rep_type", "", representation_types),
                        ),
                            ui.column(6,
                            ui.input_action_button("update_plot", "Update plot")
                        )
                    )
            )
        if MODE() == "protein":
            return ui.TagList(
                ui.h4("Zero-shot modeling"),
                ui.row(
                    ui.column(6,
                        # add MSA-Transformer and VAE later
                        ui.input_select("zs_model", "Choose model", ZS_MODELS)
                    ),

                    ui.column(6,
                        ui.input_action_button("compute_zs", "Compute"),
                        style='padding:25px;'
                    ),
                    
                    ui.column(6,
                        ui.panel_conditional("input.zs_model === 'VAE' || input.zs_model === 'MSA-Transformer'",
                            ui.input_file("input_msa", "MSA")
                        ),
                    ),
                    
                    

                    ui.h4("Visualize"),

                    ui.column(6,
                        ui.input_select("zs_data", "Zero-shot data", choices=zs_results())
                    ),

                    ui.h4("Plot Entropy"),
                    
                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_entropy", "Plot")
                        ),
                        ui.column(6,
                            ui.input_checkbox("plot_entropy_section", "Plot subsection"),
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

                    ui.h4("Plot Scores"),

                    ui.row(
                        ui.column(6,
                            ui.input_action_button("plot_scores", "Plot")
                        ),

                        ui.column(6,
                            ui.input_checkbox("plot_scores_section", "Plot subsection"),
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
                                #ui.column(6,
                                #    ui.input_select("scores_color", "Customize color scheme", choices=["rwb", "b"])
                                #)
                            )
                        ),
                    ),
                ),
            )
        else:
            return ui.TagList(
                "To proceed either upload a Dataset or a Protein and click proceed in the 'Data' tab."
            )

    # Compute representations
    @reactive.Effect
    @reactive.event(input.dat_compute_reps)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")

            print(f"Computing library: {representation_dict[input.dat_rep_type()]}")
            
            lib = library()
            
            lib.compute(method=representation_dict[input.dat_rep_type()])

            library.set(lib)
            print("Done!")

            # update representation selection
            inverted_reps = {v: k for k, v in representation_dict.items()}
            ui.update_select(
                "model_rep_type",
                choices=[inverted_reps[i] for i in lib.reps]
            )

    # Output dataset mode
    @output
    @render.plot
    @reactive.event(input.update_plot)
    def tsne_plot():
        """
        Render plot once button is pressed.
        """
        with ui.Progress(min=1, max=15) as p:
            if MODE() == "dataset":
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

    @reactive.Effect
    @reactive.event(input.compute_zs)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Computing", detail="This may take several minutes...")
            prot = protein()
            model = representation_dict[input.zs_model()]

            print(f"computing zero shot scores using {model}")

            df = prot.zs_prediction(model=model, batch_size=BATCH_SIZE)
            zs_scores.set(df)
    
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
                label="Update plot"
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
                label="Update plot"
            )

    ### Model tab ###

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

    # Training models
    val_df = reactive.Value(pd.DataFrame({'names':[], 'y_true':[], 'y_pred':[]}))

    # Train model
    @reactive.Effect
    @reactive.event(input.train_button)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Training model", detail="This may take a while...")
            # model type
            # splits not implemented yet
            if input.train_split() == "Random":
                split = "random"
            else:
                print(f"{input.train_split()} is not implemented yet - choosing random split")
                split = "random"

            # representations types only esm-2 for now
            if input.model_rep_type() == "ESM-2":
                rep_type = "esm2"
            else:
                rep_type = "esm2"

            lib = library()

            print(f"training {model_dict[input.model_type()]}")

            m = pai.Model(model_type=model_dict[input.model_type()], seed=input.random_seed())
            m.train(library=lib, x=rep_type, split=split, seed=input.random_seed(), model_type=model_dict[input.model_type()])

            print("training done!")

            model.set(m)
            val_df.set(pd.DataFrame({'names':m.val_names, 'y_true':m.y_val, 'y_pred':m.y_val_pred}))

    @output
    @render.ui
    def pred_vs_true_ui():
        hover_opts_kwargs = {}
        hover_opts_kwargs["delay"] = FAST_INTERACT_INTERVAL
        hover_opts_kwargs["delay_type"] = "throttle"
        #brush_opts_kwargs = {}
        #brush_opts_kwargs["direction"] = "xy"
        #brush_opts_kwargs["delay"] = FAST_INTERACT_INTERVAL
        #brush_opts_kwargs["delay_type"] = "throttle"

        return ui.output_plot(
            "pred_vs_true",
            hover=ui.hover_opts(**hover_opts_kwargs),
            #brush=ui.brush_opts(**brush_opts_kwargs),
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
    @render.table()
    def near_hover():
        df = val_df()
        return near_points(
            df,
            input.pred_vs_true_hover(),
            threshold=15,
            xvar="y_true",
            yvar="y_pred",
            add_dist=True,
            all_rows=False,
        )
    
    #@output
    #@render.table()
    #def in_brush():
    #    df = val_df()
    #    return brushed_points(
    #        df,
    #        input.pred_vs_true_brush(),
    #        all_rows=False,
    #    )


app = App(app_ui, server)
