import pandas as pd
from typing import Tuple

class DatasetValidator:
    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates the fundamental integrity of the CSV DataFrame.
        """
        if df.empty:
            return False, "CSV file is empty"
            
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
            return False, "All columns in the CSV must be numeric EEG data"
            
        if df.isnull().values.any():
            return False, "CSV file contains missing (null) values"
            
        if len(df.columns) == 0:
            return False, "CSV file contains no columns"
            
        return True, "Valid"
