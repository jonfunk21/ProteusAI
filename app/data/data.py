from shared_state import *
from globals import *
from shiny import ui, reactive, render
import pandas as pd

def data_ui():
    return ui.nav_panel(
        "Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.row(
                    ui.column(
                        6,
                        ui.input_text("USER", "Enter user name or proceed as guest", value="Guest"),
                    ),
                    ui.column(6, ui.input_action_button("login", "Login"), style="padding:50px;"),
                    ui.column(6, ui.input_action_button("sign_up", "Create account")),
                ),
                ui.navset_tab(
                    ui.nav_panel(
                        "Library",
                        ui.row(
                            ui.column(
                                8,
                                ui.input_file(
                                    id="dataset_file",
                                    label="Upload Dataset",
                                    accept=[".csv", ".xlsx", ".xls"],
                                    placeholder="None",
                                ),
                            ),
                            ui.column(
                                4,
                                ui.input_checkbox("demo_library_check", "Use Demo Data"),
                                style="padding:27px;",
                            ),
                        ),
                        ui.input_action_button("confirm_dataset", "Continue"),
                    ),
                    # Add other tabs here
                ),
                width=3,
            ),
            ui.panel_conditional(
                "input.data_switch",
                "Show more data information",
                ui.output_data_frame("dataset_table"),
            ),
        ),
    )

def data_server(input, output, session):
    @reactive.Effect
    @reactive.event(input.dataset_file)
    def _():
        df = DATASET()
        f: list[FileInfo] = input.dataset_file()
        df = pd.read_csv(f[0]["datapath"])

        # set reactive variables
        DATASET.set(df)
        DATASET_PATH.set(f[0]["datapath"])


    @reactive.Effect
    def update_dataset():
        if input.confirm_dataset():
            uploaded_file = input.dataset_file()
            if uploaded_file:
                # Update the shared DATASET reactive value
                DATASET(pd.read_csv(uploaded_file[0]["datapath"]))
                MODE("library")  # Example: Update mode to "library" once data is uploaded
