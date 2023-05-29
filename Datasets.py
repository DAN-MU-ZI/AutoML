import pandas as pd

class Dataset():
    def __init__(self,
                data:pd.DataFrame,
                target_col:str=None,
                drop_cols:list=None) -> None:
        self.x, self.y = self._splitLabel(data, target_col, drop_cols)
        self.categorical_features = list(self.x.select_dtypes(exclude=['number']).columns)

    def _splitLabel(self, data:pd.DataFrame, target_col:str, drop_cols:list):
        if drop_cols:
            data = data.drop(drop_cols, axis=1)
            
        try:
            return data.drop(target_col, axis=1), data[target_col]
        except:
            return data, None
    
    def updateCatfeat(self):
        self.categorical_features = list(self.x.select_dtypes(exclude=['number']).columns)

class TrainTestDataset():
    def __init__(self, train_dataset:Dataset, test_dataset:Dataset=None, baseProc=None) -> None:
        self.train_dataset = train_dataset
        self.train_dataset = test_dataset
        if baseProc:
            baseProc.fit(self.train_dataset.x,self.train_dataset.y)
            self.train_dataset.x,self.train_dataset.y = baseProc.transform(self.train_dataset.x, self.train_dataset.y)
            self.test_dataset.x,self.test_dataset.y = baseProc.transform(self.test_dataset.x, self.test_dataset.y)
    