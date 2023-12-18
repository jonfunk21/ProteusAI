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
import asyncio


representation_types = ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"]
train_test_val_splits = ["Random"]
model_types = ["Random Forrest", "KNN", "SVM"]
model_dict = {"Random Forrest":"rf", "KNN":"knn", "SVM":"svm"}
representation_dict = {"One-hot":"ohe", "BLOSUM50":"blosum50", "BLOSUM62":"blosum62", "ESM-2":"esm2", "ESM-1v":"esm1v"}
FAST_INTERACT_INTERVAL = 60 # in milliseconds

app_ui = ui.page_fluid(
    
    #ui.panel_title("ProteusAI"),
    ui.output_image("image", inline=True),
    
    ui.navset_tab_card(

        ui.nav(
            "Data", 
            
            ui.layout_sidebar(
                ui.panel_sidebar(
                    
                    ui.navset_tab(
                        ui.nav("Load Data",
                               ui.input_file(id="dataset_file", label="Select dataset (Default: demo dataset)", accept=['.csv', '.xlsx', '.xls', '.fasta'], placeholder="None"),
                               
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
                                       ui.input_select("y_type", "Data type", ["numeric", "categorical"])
                                   ),
                               ),
                    
                               ui.input_action_button('confirm_selection', 'Confirm Selection'),
                               
                               
                        ),
                        ui.nav("Custom Protein", "Under construction..."),
                    )
                ),
                ui.panel_main(
                    
                    "Raw data view",
                    ui.output_data_frame("dataset_table")
                    
                ),
                
            ),
        
        ),
        
        ui.nav("Library", 
               
               ui.layout_sidebar(
                   ui.panel_sidebar(
                       ui.navset_tab(
                           ui.nav("Visualize",
                                  ui.row(
                                      ui.column(6,
                                          ui.input_select("vis_rep_type", "Representation type", representation_types),
                                      ),
                                      ui.column(6,
                                          ui.input_action_button("vis_compute_reps", "Compute"),
                                            f"Representations 100 % computed",
                                            style='padding:25px;'
                                      )
                                  ),

                                
                                ui.input_select("vis_method","Visualization Method",["t-SNE", "PCA"]),
                                
                                ui.input_select("color_by", "Color by", ["Y-value", "Site", "Custom"]),
                                
                                # Conditional panel for Site
                                ui.panel_conditional("input.color_by === 'Site'",
                                        ui.input_text("color_text","Select sites to color seperated by ';' (e.g. 21;42)")
                                    ),
                                
                                # Conditional panel for Y-value with numeric data
                                ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'numeric'",
                                        ui.row(
                                            ui.column(6,
                                                    ui.input_text("upper_y", "Choose upper limit for y")  
                                                ),
                                            ui.column(6,
                                                    ui.input_text("lower_y", "Choose lower limit for y")  
                                                ),
                                        )
                                    ),
                                # Conditional panel fo Y-value with categorical data
                                ui.panel_conditional("input.color_by === 'Y-value' && input.y_type === 'categorical'",
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

                                ui.input_action_button("update_plot", "Update plot")
                                
                            
                           ),
                           ui.nav("Edit",
                               
                           )
                       )
                   ),
                   ui.panel_main(
                       "Visualizations under construction..."
                   )
               )
        ),

        ui.nav("Model", 
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.navset_tab(
                            ui.nav("Supervised",
                                   ui.row(
                                       ui.column(6,
                                            ui.input_select("model_type", "Model type", model_types)
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
                                            ui.input_slider("split_seed", "Random seed", min=0, max=1024, value=42)
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
                                    
                            ),
                            ui.nav("Zero-shot",
                                   "Model Customization",
                                   ui.row(
                                        ui.column(6,
                                            ui.input_select("zs_model", "Choose zero-shot model", ["ESM-2", "ESM-1v", "MSA-Transformer"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("add_input", "Aditional input (e.g., MSA)", ["None", "MSA"])
                                        ),
                                        ui.input_action_button("compute_zs", "Compute")
                                   )
                                   
                            ),
                            ui.nav(
                                "Load model",
                                "Under construction...",
                                
                            )
                        )
                    ),
                ui.panel_main(
                    ui.output_ui("pred_vs_true_ui"),
                    #ui.output_plot("pred_vs_true"),
                    ui.tags.b("Points near cursor"),
                    ui.output_table("near_hover"),
                    #ui.output_table("in_brush")
                )
                )
        ),

        
        ui.nav_menu(
            "Predict",
            ui.nav(
                "New sequences",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.navset_tab(
                            ui.nav(
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
                            ui.nav(
                                "Upload new library",
                                ui.row(
                                    ui.input_file("new_seqs_file", "Upload sequences")
                                )
                            ),
                        )
                    ),
                    ui.panel_main(

                    )
                )
            ),
            ui.nav(
                "New library",
                "Under construction (Lauras M.Sc. Thesis)..."  
            ),
        ),
        

        ui.nav(
            "Download Results",

        ),
    )
)

def server(input: Inputs, output: Outputs, session: Session):

    ### Homepage ###
    # App logo
    @output
    @render.image
    def image():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"),  "height": "75px"}
        return img

    # dummy dataset until real dataset is entere
    dataset = reactive.Value(pd.read_csv("app/demo_data.csv"))
    
    dataset_path = reactive.Value(str)

    library = reactive.Value(None)
        
    # Reading data
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

    # loading existing library
    # Reading data

    # Checking library
    @reactive.Effect
    @reactive.event(input.confirm_selection)
    def _():
        if input.y_type() == "numeric":
            y_type = "num"
        else:
            y_type = "class"
    
        # TRY TO FIND A MORE STABLE SOLUTION HERE
        seqs = dataset()[input.seq_col()].to_list()
        ys = dataset()[input.y_col()].to_list()
        names = dataset()[input.description_col()].to_list()
    
        lib = pai.Library(project=input.project_path(), seqs=seqs, ys=ys, y_type=y_type, names=names)
        
        library.set(lib)

        # update representation selection
        # TODO: Make sure these have to be 100% computed
        inverted_reps = {v: k for k, v in representation_dict.items()}
        ui.update_select(
            "model_rep_type",
            choices=[inverted_reps[i] for i in lib.reps]
        )
    
    ### Library tab ###
    
    # Compute representations
    # compute representations buttons: 'vis_compute_reps' and 'model_compute_reps' will trigger computations
    # representation types are set by vis_rep_type
    @reactive.Effect
    @reactive.event(input.vis_compute_reps)
    async def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")

            print(f"Computing library: {representation_dict[input.vis_rep_type()]}")
            
            lib = library()
            
            lib.compute(method=representation_dict[input.vis_rep_type()])

            library.set(lib)
            print("Done!")

            # update representation selection
            inverted_reps = {v: k for k, v in representation_dict.items()}
            ui.update_select(
                "model_rep_type",
                choices=[inverted_reps[i] for i in lib.reps]
            )

    ### Model tab ###

    model = reactive.Value(pai.Model())
    
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

            m = pai.Model(model_type=model_dict[input.model_type()], seed=input.split_seed())
            m.train(library=lib, x=rep_type, split=split, seed=input.split_seed(), model_type=model_dict[input.model_type()])

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
        p = model().true_vs_predicted_ggplot(df)
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