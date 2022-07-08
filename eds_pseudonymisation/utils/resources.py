import pandas as pd

from eds_pseudonymisation import BASE_DIR


def get_cities() -> pd.DataFrame:
    """
    Get the cities DataFrame from the resources.

    Returns
    -------
    pd.DataFrame
        DataFrame containing French cities.
    """
    return pd.read_csv(BASE_DIR / "resources" / "cities.csv.gz")
