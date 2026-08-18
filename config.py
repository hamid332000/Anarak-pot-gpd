
"""
Configuration file for POT-GPD Analyzer
"""

from pathlib import Path
from src.station import Station

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# Project
# =====================================================

PROJECT_NAME = "POT-GPD Analyzer"
# =====================================================
# Analysis stage
# =====================================================

# "threshold"  -> threshold determination only
# "full"       -> complete POT-GPD analysis

ANALYSIS_STAGE = "full"

# -----------------------------------------------------
# Change ONLY this line to analyse another station
# -----------------------------------------------------

STATION_NAME = "Jandagh"
# =====================================================
# Station folders
# =====================================================

STATIONS = BASE_DIR / "stations"

STATION = STATIONS / STATION_NAME
station = Station(STATION_NAME)
 


DATA = STATION / "data"

OUTPUT = STATION / "outputs"

# =====================================================
# Input file
# =====================================================

INPUT_FILE = DATA / "Tmax.xlsx"
# =====================================================
# Input
# =====================================================

DATE_COLUMN = 0
VALUE_COLUMN = 1

DATE_NAME = "Date"
VALUE_NAME = "Tmax"

# =====================================================
# Threshold search
# =====================================================

LOWER_QUANTILE = 0.90
UPPER_QUANTILE = 0.995

MIN_EXCEEDANCES = 30
THRESHOLD_STEP = 0.1

 

# =====================================================
# Bootstrap settings
# =====================================================
 
BOOTSTRAP_AD = 200          # Anderson–Darling test
BOOTSTRAP_QQ = 200           # QQ simulation envelope
BOOTSTRAP_RETURN = 500      # Return-level confidence intervals
BOOTSTRAP_MRL = 500         # Mean residual life confidence intervals
#BOOTSTRAP_AD = 5000          # Anderson–Darling goodness-of-fit test
#BOOTSTRAP_QQ = 2000          # QQ simulation envelope
#BOOTSTRAP_RETURN = 5000      # Return-level confidence intervals
#BOOTSTRAP_MRL = 2000         # Mean residual life confidence intervals
RANDOM_SEED = 12345

# =====================================================
# Declustering
# =====================================================
 

ALPHA = 0.05

# =====================================================
# Return periods
# =====================================================

RETURN_PERIODS = [10, 20, 50]

# =====================================================
# Output folders
# =====================================================

FIGURES = OUTPUT / "Figures"

TABLES = OUTPUT / "Tables"

REPORTS = OUTPUT / "Reports"

LOGS = OUTPUT / "Logs"

# =====================================================
# Create folders automatically
# =====================================================

for folder in [
    STATIONS,
    STATION,
    DATA,
    OUTPUT,
    FIGURES,
    TABLES,
    REPORTS,
    LOGS,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )