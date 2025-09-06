from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd

def data_preparation(df):
    y = df['target']
    X = df.drop('target', axis=1)

    y = y.replace({'False': 0, 'True': 1}).astype('int')
    X = X.astype('float32')

    return X, y




def get_subtrain_splits(X, y, percentages):

    splits = {}
    for p in percentages:
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=p,       
            stratify=y,         
            random_state=42    
        )
        splits[p] = (X_sub, y_sub)
    return splits






class FeatureEngineerer(BaseEstimator, TransformerMixin):


    def __init__(self):
        self.freq_dicts = {}

    def fit(self, X, y=None):
        # Berechne Frequenzen pro Spalte nur einmal
        for col in X.columns:
            self.freq_dicts[col] = X[col].value_counts()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        features = X.columns
        
        # Liste zur Speicherung der neuen Spalten
        new_features = {}

        for col in features:
            
            # Flag erstellen ob der Wert einzigartig ist
            flag = (X[col].map(self.freq_dicts[col]).fillna(0) <= 1).astype(int)
            new_features[f'flag_is_unique_{col}'] = flag
            
        
        # Füge alle Flags in einem Schritt hinzu
        X_transformed = pd.concat([X_transformed, pd.DataFrame(new_features)], axis=1)

        return X_transformed