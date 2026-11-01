"""Local heavy artifacts are configured outside the published materials."""
from pathlib import Path
import os

def experiment_directory():
    value = os.environ.get('IONOSPHERIC_EXPERIMENT')
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2] / 'local_data' / 'experiment'

def baseline_directory():
    value = os.environ.get('IONOSPHERIC_BASELINE')
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2] / 'local_data' / 'baseline'
