"""
Read station-specific configuration.
"""

import json
from pathlib import Path


class Station:

    def __init__(self, station_name):

        self.station_name = station_name

        self.station_folder = (
            Path(__file__).resolve().parent.parent
            / "stations"
            / station_name
        )

        self.data = self.station_folder / "data"

        self.outputs = self.station_folder / "outputs"

        self.figures = self.outputs / "Figures"

        self.tables = self.outputs / "Tables"

        self.reports = self.outputs / "Reports"

        self.logs = self.outputs / "Logs"

        with open(
            self.station_folder / "station_config.json",
            "r",
            encoding="utf-8",
        ) as f:

            cfg = json.load(f)

        self.selected_threshold = cfg.get(
            "selected_threshold",
            None
        )

        self.run_length = cfg.get(
            "run_length",
            None
        )

        self.input_file = self.data / "Tmax.xlsx"