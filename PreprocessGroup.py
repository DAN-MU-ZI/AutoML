import numpy as np
import itertools
from Processor import ProcessWrapper

from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
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
    def __init__(self) -> None:
        #oversampler, unsersampler, decomposition, feature_selection, impute,model_selection, sacler
        self.preprocessors = {"impute":[SimpleImputer],#, KNNImputer],
                            "sampler":[AllKNN,
                                        CondensedNearestNeighbour,
                                        EditedNearestNeighbours,
                                        InstanceHardnessThreshold,
                                        NearMiss,
                                        NeighbourhoodCleaningRule,
                                        OneSidedSelection,
                                        RandomUnderSampler,
                                        RepeatedEditedNearestNeighbours,
                                        TomekLinks,
                                        ADASYN,
                                        SMOTE,
                                        SMOTENC,
                                        SVMSMOTE,
                                        BorderlineSMOTE,
                                        KMeansSMOTE,
                                        RandomOverSampler],
                            "composition":[PCA, IncrementalPCA, KernelPCA],
                            #"feat_selection":[SelectFromModel,SelectKBest,VarianceThreshold],
                            "scaler":[MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler],
                            }
        self.processGrp = []
    
    def add(self, preprocessors):
        for preocess_type, processor in preprocessors:
            if preocess_type in self.preprocessors.keys():
                self.preprocessors[preocess_type].append(processor)
            else:
                if isinstance(processor, list):
                    self.preprocessors[preocess_type] = processor
                else:
                    self.preprocessors[preocess_type] = [processor]
        self._updateGrp()

    def _updateGrp(self):
        self.processGrp = []
        for pipe in self:
            self.processGrp.append(pipe)

    def __iter__(self):
        baseGrp = {x[0]:x[1] for x in self.preprocessors.items() if x[0].startswith("base")}
        processGrp = {x[0]:x[1] for x in self.preprocessors.items() if not x[0].startswith("base")}
        
        if baseGrp:
            for k,v in baseGrp.items():
                for values in itertools.product(*processGrp.values()):
                    res = []
                    keys = list(processGrp.keys())
                    res.extend(values[i] for i in range(len(keys)))
                    yield keys, res
        else:
            for values in itertools.product(*processGrp.values()):
                res = []
                keys = list(processGrp.keys())
                res = [ProcessWrapper(values[i]) for i in range(len(keys))]
                # res = [values[i] for i in range(len(keys))]
                yield keys, res

    def __getitem__(self, idx):
        return self.processGrp[idx]
    