from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class ProcessWrapper(BaseEstimator, TransformerMixin):
    def __init__(self,processor):
        self.preprocessor = processor()
        self.is_train = True
        
    def fit(self, X, y=None, **params):
        print(self.preprocessor, "called")
        args = {"X":X, "y":y}
        if hasattr(self.preprocessor,'fit_resample'):
            self.preprocessor.fit_resample(X,y, **params)
        else:
            self.preprocessor.fit(X,y, **params)
        return self
    
    def transform(self, X, y=None):
        args = {"X":X, "y":y}
        if hasattr(self.preprocessor,'fit_resample'):
            if self.is_train:
                X, y = self.preprocessor.fit_resample(X,y)
        else:
            processed = self.preprocessor.transform(**{k: v for k, v in args.items() if k in self.preprocessor.transform.__code__.co_varnames})
            if isinstance(processed, tuple):
                X, y = processed
            else:
                X = processed
        return X, y

class RemoveEmptyColumn(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.remain_cols = []

    def fit(self, X, y=None):
        self.remain_cols = list(X.dropna(axis=1, how='all').columns)

    def transform(self, X, y=None):
        X = X[self.remain_cols]
        return X, y

class RemoveOneValueColumn(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = []

    def fit(self, X, y=None):
        for col in [x for x in X.columns if 'X_' in x]:
            if len(X[col].value_counts())==1:
                self.drop_cols.append(col)

    def transform(self, X, y=None):
        X = X.drop(self.drop_cols, axis=1)
        return X, y

class ConcatProdLineWithOneHot(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.uniq_cols = []
        self.target_col = 'PROD_LINE'
        self.is_train = True
    
    def fit(self, X, y=None):
        #self.uniq_cols = list(X[self.target_col].unique())
        pass

    def transform(self, X, y=None):
        X[self.target_col] = X['PRODUCT_CODE']+'_'+X['LINE']
        if self.is_train:
            self.uniq_cols = list(X[self.target_col].unique())
            self.is_train=False
            
        for col in self.uniq_cols:
            X[col] = (X[self.target_col]==col).astype(int)
        X = X.drop(['PRODUCT_CODE','LINE', self.target_col],axis=1)
        return X, y

class DropDuplicatedColumns():
    def __init__(self) -> None:
        self.only_cols = None
        self.is_train = True

    def fit(self, X, y=None):
        self.only_cols = X.loc[:,~X.T.duplicated(keep='first')].columns
        
    def transform(self, X, y=None):
        X = X[self.only_cols]
        return X, y

class RobustScaling():
    def __init__(self, target_cols=None) -> None:
        self.target_cols = target_cols
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        X[self.target_cols] = self.scaler.transform(X[self.target_cols])
        return X, y
    
class Balance(BaseEstimator, TransformerMixin):
    def __init__(self, sampler) -> None:
        self.sampler = sampler
        self.is_train = True

    def fit(self, X,y=None):
        return self
    
    def transform(self, X, y):
        if "over_sampling" in self.sampler.__module__:
            X, y = self.sampler.fit_resample(X,y)
        return X,y