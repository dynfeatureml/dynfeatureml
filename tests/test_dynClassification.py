# Module: Tests.py
# Author: Prashant Nair <prashant.nair2050@gmail.com>
# License: MIT

#compare_models_test
import pytest
from dynfeatureml import DynFeatureClassifier
from dynfeatureml import datasets


def test():
    # loading dataset
    data = datasets.get_data('iris')
    
    #Creating features and label
    features = data.iloc[:,[0,1,2,3]].values
    label = data.iloc[:,-1].values
    
    #Create Model
    modelTest = DynFeatureClassifier()
    modelTest.fit(estimator="lr", X=features, y=label)
    
    print("ModelList : ", modelTest.modelList)

    #
    assert 1 == 1
    
if __name__ == "__main__":
    test()