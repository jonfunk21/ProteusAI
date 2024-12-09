from shiny import ui, render, reactive
from data import MODE, DATASET
from globals import *

def design_ui():
    return ui.nav_panel(
        "Design",
        ui.layout_sidebar(
            ui.sidebar(ui.output_ui("design_ui_placeholder"), width=3),
            ui.panel_conditional(
                "typeof output.protein_struc === 'string'",
                ui.output_ui("struc3D_design"),
                ui.output_text("fixed_res_text"),
                ui.output_data_frame("design_out"),
                ui.output_ui("design_download_ui"),
            ),
        ),
    )


def design_server(input, output, session):
    @output
    @render.ui
    def design_ui_rendered():
        if MODE() == "structure":
            return ui.TagList(
                ui.row(
                    ui.h5("Structure Based Protein Design"),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_select(
                                "design_models",
                                "Choose model",
                                ["Model1", "Model2"],  # Example options
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_numeric(
                                "n_designs", "Number of samples", min=1, value=20
                            ),
                        ),
                    ),
                    ui.input_text(
                        "design_res",
                        "Select residues by ID that should remain unchanged during redesign",
                    ),
                    ui.column(
                        4,
                        ui.input_task_button("design_button", "Design"),
                    ),
                )
            )
        else:
            return ui.TagList(
                "Upload a protein structure in the 'Data' tab to proceed."
            )
