# This source code is part of the proteusAI package and is distributed
# under the MIT License.

"""
ProteusAI Shiny App.
"""

__name__ = "ProteusAI"
__author__ = "Jonathan Funk"

from shiny import App, ui
from shared_state import * 
from globals import *
from data import data_ui, data_server
from design import design_ui, design_server  # Import the design tab UI
from pathlib import Path
from shiny.types import FileInfo, ImgData


app_ui = ui.page_fluid(
    ui.output_image("image", inline=True),
    VERSION,
    ui.HTML(f'<a href="{PAPER_URL}" target="_blank">Please cite our paper.</a>'),
    ui.navset_card_tab(
        data_ui(),  # Integrate Data UI
        design_ui(),  # Integrate Design UI
    ),
)

def server(input, output, session):
    # App logo
    @output
    @render.image
    def image():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"), "height": "75px"}
        return img

    # Module servers
    data_server(input, output, session)
    design_server(input, output, session)

app = App(app_ui, server)
