import pandas as pd
from typing import List, Callable, Dict


def prepare_harmbench() -> pd.DataFrame:
    """Loads and returns the HarmBench dataset."""
    dataset = pd.read_csv("prompts/harmbench/harmbench_behaviors_text_all.csv")
    return dataset

def prepare_cs333() -> pd.DataFrame:
    pass

def prepare_custome_jailbreaking() -> pd.DataFrame:
    pass

KNOWN_DATASETS: Dict[str, Callable[[], pd.DataFrame]] = {
    "HarmBench": prepare_harmbench,
}


class DatasetManager:
    def __init__(self, datasets: List[str]) -> None:
        """
        Initialize the DatasetManager.

        Args:
            datasets: A list of dataset string identifiers to load.
        """
        for dataset in datasets:
            if dataset not in KNOWN_DATASETS:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. Available options: {list(KNOWN_DATASETS.keys())}"
                )

        self.datasets = datasets

    def prepare_datasets(self) -> List[pd.DataFrame]:
        """
        Load and return the requested datasets.

        Returns:
            A list of pandas DataFrames corresponding to the initialized datasets.
        """
        prepared_datasets = []
        
        for dataset in self.datasets:
            prepared_datasets.append(KNOWN_DATASETS[dataset]())

        return prepared_datasets