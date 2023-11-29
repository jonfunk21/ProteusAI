import shiny
from shiny import App, ui, render, Inputs, Outputs, Session, reactive
from shiny.types import FileInfo, ImgData
import pandas as pd
import sys
sys.path.append('src/')
import proteusAI as pai
import os

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
                                   ui.column(12,
                                       ui.input_select("seq_col", "Sequence column", []),
                                   ),
                                   ui.column(6,
                                        ui.input_select("y_col", "Y-values", []),
                                   ),
                                   ui.column(6,
                                       ui.input_select("y_type", "Data type", ["numeric", "categorical"])
                                   ),
                               ),
                    
                               ui.input_action_button('compute_library', 'Confirm Selection', class_="btn-success"),
                               
                               
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
                    ui.output_table("dataset_table")
                    
                ),
            ),
        
        ),

        ui.nav("Library", 
               
               ui.layout_sidebar(
                   ui.panel_sidebar(
                       ui.navset_tab(
                           ui.nav("Visualize",
                                ui.input_select("vis_method","Visualization Method",["t-SNE", "PCA"]),
                                ui.input_select("color_by", "Color by", ["Y-value", "Site", "Custom"])
                           ),
                           ui.nav("Edit",
                               
                           )
                       )
                   ),
                   ui.panel_main(
                       
                   )
               )
        ),

        ui.nav("Model", 
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.navset_tab(
                            ui.nav("Supervised",
                                   "Model Customization",
                                   ui.row(
                                       ui.column(6,
                                            ui.input_select("model_type", "Model type", ["Random Forrest", "KNN", "SVM", "FFNN"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("model_task", "Model task", ["Regression", "Classification"])
                                        ),
                                        ui.column(6,
                                            ui.input_select("model_rep_type", "Representaion type", ["ESM-2", "ESM-1v", "One-hot", "BLOSUM50", "BLOSUM62"]),
                                            "100 % computed"
                                        ),

                                        ui.column(6,
                                            ui.input_action_button("compute_reps", "Compute", class_="btn-success"),
                                            style='padding:25px;'
                                        )
                                   ),
                                   "Choose model parameters"
                                   
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
                    "Here you will be able to inspect your model results..."
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
                'Sequence':['Protein sequences'],
                'Description':['Descriptions, e.g. M1V'],
                '...':['...'],
                'Y-value':['Experimental values']
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
    @render.table
    def dataset_table():
        return dataset()
    

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
        

app = App(app_ui, server)