import numpy as np


class ModelGroup():
    def __init__(self, models:list=[]) -> None:
        self.models = np.array(models)
        pass
    
    def add(self, models):
        # 해당 전처리기가 fit transform 함수를 가지고있는지 검사해야함
        for model in models:
            if not (hasattr(model, 'fit') and callable(getattr(model, 'fit'))):
                raise TypeError(
                    "해당 모델은 fit함수를 가지고 있지 않습니다."
                    "'%s'" % (model, type(model))
                )

        self.models = np.append(self.models, models)

    def __iter__(self):
        for model in self.models:
            yield model
