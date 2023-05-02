import numpy as np

from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def generate_lists(k, current_list):
    if 0 == k:
        return current_list
    else:
        # n번째 원소를 1로 설정하는 경우
        res = np.array([[]])
        res = np.append(res, generate_lists(k-1, np.append(current_list, [True])))
        res = np.append(res, generate_lists(k-1, np.append(current_list, [False])))
        return res

class PreprocessPipeline():
    def __init__(self, preprocessors:list = []) -> None:
        #oversampler, unsersampler, decomposition, feature_selection, impute,model_selection, sacler
        self.preprocessors = np.array(preprocessors)
        pass
    
    def add(self, preprocessors):
        # 해당 전처리기가 fit transform 함수를 가지고있는지 검사해야함
        for p in preprocessors:
            if not (hasattr(p, 'fit') and callable(getattr(p, 'fit'))):
                raise TypeError(
                    "해당 전처리는 fit함수를 가지고 있지 않습니다."
                    "'%s'" % (p, type(p))
                )
            elif not (hasattr(p, 'transform') and callable(getattr(p, 'transform'))):
                raise TypeError(
                    "해당 전처리는 transform함수를 가지고 있지 않습니다."
                    "'%s'" % (p, type(p))
                )

        self.preprocessors = np.append(self.preprocessors, preprocessors)

    def __iter__(self):
        numCases = len(self.preprocessors)
        for flag in generate_lists(numCases, np.array([[]])).reshape((-1, numCases)).astype(bool):
            pipe = self.preprocessors[flag]
            if len(pipe):
                yield pipe
            
    
