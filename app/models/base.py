# app/models/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict_proba(self, X) -> pd.Series: ...
