import numpy as np
import pandas as pd
import sklearn as sk
from time import clock
from itertools import *
from sklearn.metrics import roc_auc_score
class TestingFramework(object):
    def __init__(self,
                 model        = ClusteringFramework,
                 folder       = 'datasets',
                 filenames    = ['gas_clustered_hc_easy_0.005_0.005.csv','gas_clustered_hc_easy_0.005_0.005.csv'],
                 parameters   = {'outlier_scoring':['dist2center','cluster_size','dist2neighbor'],
                                 'k':[2,10,20,30],
                                 'norm_type_column01':['none','standard','0-1'],
                                 'norm_type_column02':['none','standard','0-1'],
                                 'norm_type_row01':['none','l1_norm','l2_norm','Inf_norm'],
                                 'outlier_scaling':['0-1','standard'],
                                 'outlier_final':['max','pca']},
                 comb_params  = ['outlier_scoring'],
                 score_column = 'outlier_score',
                 score_method = roc_auc_score
                ):
        self.model        = model
        self.folder       = folder
        self.filenames    = filenames
        self.parameters   = self.__create_parameter_tuples(parameters,comb_params)
        self.score_column = score_column
        self.score_method = score_method
        
    def fit(self,data=None,label=None):
        parameter_dicts = self.parameters
        model_results   = []
        best_overall    = (-np.Inf,None,dict())
        best_per_file   = {}
        if data is not None:
            self.filenames = ['placeholder']
        for filename in self.filenames:
            best_per_file[filename] = (-np.Inf,None,dict())
            if data is not None:
                X   = data
                y   = label
            else:
                X,y = self.__load_file(filename)
            for parameter_set in parameter_dicts:
                model      = self.model(**parameter_set)
                time01     = clock()
                df         = model.predict(X)
                time02     = clock()
                total_time = time02 - time01
                try:
                    param01 = model.param01
                except:
                    param01 = 0
                score      = self.__score(y,df[self.score_column])
                tmp_tuple  = (score,total_time,param01,filename,model,parameter_set)
                model_results.append(tmp_tuple)
                if score > best_overall[0]:
                    best_overall = tmp_tuple
                if score > best_per_file[filename][0]:
                    best_per_file[filename] = tmp_tuple
        self.model_results = model_results
        self.best_per_file = best_per_file
        self.best_overall  = best_overall
        return self
        
    def __load_file(self,filename,sep=',',header=None,label_location='end'):
        filename = self.folder + '\\' + filename
        data     = pd.read_csv(filename, sep=sep, header=header).values
        if label_location == 'end':
            X = data[:,:-1]
            y = data[:,-1]
        elif label_location == 'start':
            X = data[:,1:]
            y = data[:,0]
        elif label_location == 'none':
            X = data
            y = None
        else:
            print("'label_location' input is not recognized/supported!")
            print("Assuming there is no label column!")
            X = data
            y = None
        return X,y
        
    def __create_parameter_tuples(self,parameters,comb_params):
        keys        = list(parameters.keys())
        for key in keys:
            if key in comb_params:
                tmp_list = [list(combinations(parameters[key],i)) for i in range(1,len(parameters[key])+1)]
                options  = []
                for val in tmp_list:
                    options = options + val
                options = [list(val) for val in options]
                parameters[key] = options
        parameter_sets = list(product(*(parameters.values())))
        parameters     = []
        for i in range(len(parameter_sets)):
            tmp_dict = {}
            for j in range(len(keys)):
                key           = keys[j]
                tmp_dict[key] = parameter_sets[i][j]
            parameters.append(tmp_dict)
        return parameters
        
    def __score(self,y,y_hat):
        if y is not None:
            return self.score_method(y,y_hat)
        else:
            return 0
    
    def best_model(self):
        return self.best_overall
    
    def ranked_model(self):
        scores        = [self.model_results[i][0] for i in range(len(self.model_results))]
        score_indices = np.argsort(-1*np.array(scores))
        ranked_models = [self.model_results[i] for i in score_indices]
        return ranked_models
    
    def save_model(self,filename='model_results.csv',param01='param'):
        values = [self.model_results[i][:4] for i in range(len(self.model_results))]
        dicts  = [self.model_results[i][5:][0] for i in range(len(self.model_results))]
        df01   = pd.DataFrame(values,columns=['score','total time',param01,'filename'])
        df02   = pd.DataFrame(dicts)
        df     = df01.join(df02)
        df.to_csv(filename)
        return df
