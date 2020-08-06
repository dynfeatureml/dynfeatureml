import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score


class DynFeatureClassifier(BaseEstimator, ClassifierMixin):
    '''
    Doc: DynFeatureClassifier 
    
    Author: Prashant Nair
    
    Syntax: DynFeatureClassifier(modelList)
    Expected existing model to be passed in the modelList
    
    '''
    
    global modelList, noFeaturesPerModel
    
    
    
    def __init__(self, modelList=None):
        self.modelList = []
        self.noFeaturesPerModel = []
        if modelList:
            self.modelList.append(modelList)
        
       
    def fit(self, estimator=None, X=None, y=None , **kwargs):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        #self.X_ = X.copy()
        #self.y_ = y.copy()
        #
        #Creating Model
        
        if estimator == 'lr':
            print("LR Running")
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, **kwargs)
            full_name = 'Logistic Regression'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            model
            
        elif estimator == 'knn':
            print("KNN Running")
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(**kwargs)
            full_name = 'K Neighbors Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'nb':
            print("NB Running")
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**kwargs)
            full_name = 'Naive Bayes'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'dt':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**kwargs)
            full_name = 'Decision Tree Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'svm':
            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(max_iter=1000, tol=0.001, **kwargs)
            full_name = 'SVM - Linear Kernel'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'rbfsvm':
            from sklearn.svm import SVC
            model = SVC(gamma='auto', C=1, probability=True, kernel='rbf', **kwargs)
            full_name = 'SVM - Radial Kernel'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
        
        elif estimator == 'mlp':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(max_iter=500, **kwargs)
            full_name = 'MLP Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
        
        elif estimator == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, **kwargs)
            full_name = 'Random Forest Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
        
        elif estimator == 'ada':
            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(**kwargs)
            full_name = 'Ada Boost Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'gbc':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**kwargs)
            full_name = 'Gradient Boosting Classifier'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
            
        elif estimator == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(random_state=seed, verbosity=0, **kwargs)
            full_name = 'Extreme Gradient Boosting'
            self.modelList.append(model.fit(X,y))
            self.noFeaturesPerModel.append(model.n_features_in_)
        
        else:
            print("{} Not Available".format(estimator))
        
        
        # Return the classifier
        return self
    
    def predict(self, X=None):
        from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
        from sklearn.utils.multiclass import unique_labels
        from sklearn.metrics import accuracy_score
        import pandas as pd
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        # Logic
        predictValues = []
        
        #Finding number of features
        if type(X) == list:
            numFeatures = np.array(X).shape[1]
        else:
            numFeatures = X.shape[1]
        
        #Predict from all models with relevant features
        #for modelCount in range(0,len(self.modelList)):
        #    for featureCount in range(1,len(self.noFeaturesPerModel) + 1):
        #        if X.shape[0] in self.noFeaturesPerModel:
        #            try:
        #                predictValues.append(self.modelList[modelCount].predict(X[:,X.shape[0]]))
        #            except:
        #                pass

            
            
        #Strategy: I want to predict from all models.. Models expecting less features will get less data. Model Expecting more
        #          features will get the imputed values
        
        for i in range(1,numFeatures + 1):
            if numFeatures == i:
                #print(modelList[0].predict(np.array([feat[0][0:2]])))
                #counter = 0
                for modelCountLoop in range(0,len(self.modelList)):
                    for featureCountLoop in range(0,X.shape[0]):
                        predictValues.append(self.modelList[modelCountLoop].predict(np.array([X[featureCountLoop][0:self.modelList[modelCountLoop].n_features_in_]])))
                        #print(self.modelList[modelCountLoop].predict(np.array([X[featureCountLoop][0:self.modelList[modelCountLoop].n_features_in_]])))
                        #counter = counter + 1
                        #finalPrediction = pd.DataFrame(predictValues).mode()[0][0]
                        
        
        #print("Log: ",predictValues)
        
        #for modelCountLoop in range(0,len(self.modelList)):
        #        for featureCountLoop in range(0,X.shape[0]):
        #            predictValues.append(self.modelList[modelCountLoop].predict(np.array([X[featureCountLoop][0:self.modelList[modelCountLoop].coef_.shape[1]]])))
        #            finalPrediction = pd.DataFrame(predictValues).mode()[0][0]
        #print("Log: ",predictValues)
        
        #print("Count: ",counter)
        finalPrediction = pd.DataFrame(predictValues).mode()[0][0]
        return np.array([[finalPrediction]])
    
    def saveModel(self, modelObject, fileName):
        import dill
        dill.dump(modelObject , open(fileName, 'wb'))
        
    def loadModel(self, fileName):
        import dill
        dill.load(open(fileName, 'rb'))