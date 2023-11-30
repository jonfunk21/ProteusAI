import shiny
from shiny import App, ui, render, Inputs, Outputs, Session, reactive
from shiny.types import FileInfo, ImgData
import pandas as pd
import sys
sys.path.append('src/')
import proteusAI as pai
import os

representation_types = ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"]

app_ui = ui.page_fluid(
    
    #ui.panel_title("ProteusAI"),
    ui.output_image("image", inline=True, ),
    
    ui.navset_tab_card(

        ui.nav(
            "Data", 
            
            ui.layout_sidebar(
                ui.panel_sidebar(
                    
                    ui.navset_tab(
                        ui.nav("Load Data",
                               ui.input_file(id="dataset_file", label="Select dataset", accept=['.csv', '.xlsx', '.xls', '.fasta'], placeholder="None"),
                               
                               # CHANGE THIS TO EMPTY STRING LATER
                               ui.input_text(id="project_path", label="Project Path", value="demo/example_project"),
                               
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
                    
                               ui.input_action_button('compute_library', 'Confirm Selection'),
                               
                               
                        ),
                        ui.nav("Custom Protein", "Under construction..."),
                    )
                ),
                ui.panel_main(
                    
                    "Raw data view",
                    #ui.panel_conditional(
                    #    "input.dataset_table === null",
                    #    ui.output_table("dataset_table")
                    #),
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
                                            ui.input_select("model_type", "Model type", ["Random Forrest", "KNN", "SVM", "FFNN"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("model_task", "Model task", ["Regression", "Classification"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("model_rep_type", "Representaion type", representation_types),
                                        ),

                                        ui.column(6,
                                            ui.input_action_button("model_compute_reps", "Compute"),
                                            f"Representations 100 % computed",
                                            style='padding:25px;'
                                        ),
                                        
                                   ),
                                    
                                    ui.input_checkbox("customize_model_params", "Customize model parameters", value=False),
                                    
                                    ui.panel_conditional("input.customize_model_params === true",
                                            "Not implemented yet LOL",
                                            ui.input_text("frustrations", "Draft your angry tweet here (How do we call tweets on X now?):")
                                                         
                                        ),

                                    ui.input_action_button("train_button", "Train Model")
                                   
                            ),
                            ui.nav("Zero-shot",
                                   "Model Customization",
                                   ui.row(
                                        ui.column(6,
                                            ui.input_select("zs_model", "Choose zero-shot model", ["ESM-2", "ESM-1v", "MSA-Transformer"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("add_input", "Aditional input (e.g., MSA)", ["None", "MSA"])
                                        )
                                   )
                                   
                            ),
                            ui.nav(
                                "Load model"
                            )
                        )
                    ),
                ui.panel_main(
                    "Model training and results under construction..."
                )
                )
        ),

        ui.nav_menu(
            "Results",

            ui.nav("Option 1", "Option 1 content"),
            
            align="right",
        ),

        ui.nav_menu(
            "Predict",

            ui.nav("Option 1", "Option 1 content"),
            
            align="right",
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
    dataset = reactive.Value(pd.DataFrame(data={
                'Sequence':['MAGLVQR','VAGLVQR', 'MTGLVQR'],
                'Description':['wt','M1V', 'A2T'],
                '...':['...', '...', '...'],
                'Y-value':[0.2,-0.1, 0.9]
            })
    )
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

    # Computing library
    @reactive.Effect
    @reactive.event(input.compute_library)
    def _():
        print(input.project_path())
        lib = pai.Library(project=input.project_path())

        if input.y_type() == "numeric":
            y_type = "num"
        else:
            y_type = "class"

        # TRY TO FIND A MORE STABLE SOLUTION HERE
        try:
            lib.read_data(data=dataset_path(), seqs=input.seq_col(), y=input.y_col(), y_type=y_type, names='Description')
        except:
            pass

        library.set(lib)
    
    ### Library tab ###


app = App(app_ui, server)