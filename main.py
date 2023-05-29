import warnings
warnings.filterwarnings('always')
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from ModelGroup import ModelGroup
from PreprocessGroup import PreprocessPipeline
from Datasets import *
from Processor import *
from Trainer import *

class AutoML():
    def __init__(self) -> None:
        self.pipeline = PreprocessPipeline()
        self.modelGroup = ModelGroup()
        self.trainer = Trainer()
        self.train_dataset = None
        self.test_dataset = None

    def train(self):
        models = []
        total_scores = pd.DataFrame()
        for name, pipe in self.pipeline:
            train_X = self.train_dataset.x.copy()
            train_y = self.train_dataset.y.copy()
            test = self.train_dataset.x.copy()
            train_attrs = [x for x in self.train_dataset.__dir__() if '__' not in x]
            test_attrs = [x for x in self.test_dataset.__dir__() if '__' not in x]
            print(pipe)
            for idx, preprocess in enumerate(pipe):
                preprocess.fit(train_X, train_y)
                train_X, train_y = preprocess.transform(train_X, train_y)
                test, _ = preprocess.transform(test)
            
            scores = pd.DataFrame()
            for model in self.modelGroup:
                print(model)
                model, score = self.trainer.train(train_X, train_y,model)
                models.append(model)
                scores = scores.append(score)
            
            for n, p in zip(name, pipe):
                scores[n]=p.processor
            
            total_scores = total_scores.append(scores)
        return models, total_scores
    def test(self):
        pass

automl = AutoML()
automl.modelGroup.add([RandomForestClassifier, GradientBoostingClassifier, XGBClassifier])

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_dataset = Dataset(train, "Y_Class", ['PRODUCT_ID', 'Y_Quality'])
test_dataset = Dataset(test, "Y_Class", ['PRODUCT_ID'])
pipe = [RemoveEmptyColumn(),
        DropDuplicatedColumns(),
        RemoveOneValueColumn(),
        ConcatProdLineWithOneHot()]

for p in pipe:
    p.fit(train_dataset.x,train_dataset.y)
    train_dataset.x,train_dataset.y = p.transform(train_dataset.x,train_dataset.y)
    test_dataset.x,test_dataset.y = p.transform(test_dataset.x,test_dataset.y)
train_dataset.updateCatfeat()
test_dataset.updateCatfeat()

automl.train_dataset = train_dataset
automl.test_dataset = test_dataset
#print([x for x in train_dataset.__dir__() if '__' not in x])
models, scores = automl.train()
print(scores)
scores.to_csv("result.csv", index=False)