import pandas as pd

from .. import BASE_DIR


def get_hospitals() -> pd.DataFrame:
    """
    Get the cities DataFrame from the resources.

    Returns
    -------
    pd.DataFrame
        DataFrame containing French cities.
    """
    return pd.read_csv(BASE_DIR / "resources" / "hospitals.csv")['HOPITAL']
