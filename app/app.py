import shiny
from shiny import App, ui, render, Inputs, Outputs, Session, reactive
from shiny.types import FileInfo
import pandas as pd
import sys
sys.path.append('src/')
import proteusAI as pai
import os

app_ui = ui.page_fluid(
    ui.panel_title("ProteusAI"),

    ui.navset_tab_card(

        
        ui.nav(
            "Data", 
            
            ui.layout_sidebar(
                ui.panel_sidebar(
                    # CHANGE THIS TO EMPTY STRING LATER
                    ui.navset_tab(
                        ui.nav("Load Data",
                               ui.input_file(id="dataset_file", label="Select dataset", accept=['.csv', '.xlsx', '.xls', '.fasta']),
                               
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
                    
                               ui.input_action_button(id='compute_library', label='Confirm Selection', class_="btn-success"),
                               
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

        ui.nav("Library", "tab a content"),

        ui.nav("Model", "tab a content"),

        ui.nav_menu(
            "Results",

            ui.nav("Option 1", "Option 1 content"),
            
            align="right",
        ),

        ui.nav_menu(
            "Visualization",

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
            label=f"Select Sequences)",
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

        lib.read_data(data=dataset_path(), seqs=input.seq_col(), y=input.y_col(), y_type=y_type, names='Description')
        library.set(lib)
        

app = App(app_ui, server)