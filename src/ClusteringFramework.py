import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
class ClusteringFramework(object):
    def __init__(self,
                 algorithm='kmeans',
                 norm_type_column01='none',
                 norm_type_row01='none',
                 norm_type_column02='none',
                 k=2,
                 k_range=[],
                 outlier_scoring=['dist2center','dist2neighbor','cluster_size'],
                 outlier_scaling='0-1',
                 outlier_final='max',
                 outlier_final_scaling='0-1'
                ):
        self.algorithm             = algorithm
        self.norm_type_column01    = norm_type_column01
        self.norm_type_row01       = norm_type_row01
        self.norm_type_column02    = norm_type_column02
        self.k                     = k
        self.k_range               = k_range
        self.outlier_scoring       = outlier_scoring
        self.outlier_scaling       = outlier_scaling
        self.outlier_final         = outlier_final
        self.outlier_final_scaling = outlier_final_scaling
        
    def predict(self,X):
        self.__normalize_column(X,self.norm_type_column01)
        self.__normalize_row(X,self.norm_type_row01)
        self.__normalize_column(X,self.norm_type_column02)
        
        if self.algorithm.lower() == 'kmeans':
            if len(self.k_range) > 0:
                model = self.__optimal_clusters(X)
            else:
                self.optimal_k = self.k
                model = KMeans(self.k).fit(X)
        self.model   = model
        df           = self.__score_outliers(X)
        df           = self.__scale_outliers(df)
        df           = self.__finalize_outliers(df,self.outlier_final,X)
        self.param01 = self.optimal_k
        return df
        
    def __normalize_row(self,X,technique):
        if technique.lower() == 'l1_norm':
            denom = np.array(np.sum(np.abs(X), axis=1),ndmin=2).T
            X     = np.divide(X, denom)
        elif technique.lower() == 'l2_norm':
            denom = np.array(np.sqrt(np.sum(X**2, axis=1)),ndmin=2).T
            X     = np.divide(X, denom)
        elif technique.lower() == 'inf_norm':
            denom = np.array(np.max(X, axis=1),ndmin=2).T
            X     = np.divide(X, denom)
        elif technique.lower() != 'none':
            print("The input row normalization, is not recognized/supported!",technique)
            print("Using 'none' row normalization")
        return X
        
    def __normalize_column(self,X,technique):
        if technique.lower() == '0-1':
            mins  = np.array(np.min(X, axis=0),ndmin=2)
            maxs  = np.array(np.max(X, axis=0),ndmin=2)
            X     = np.divide(np.subtract(X,mins), maxs - mins)
        elif technique.lower() == 'standard':
            means = np.array(np.mean(X, axis=0),ndmin=2)
            stds  = np.array(np.std(X, axis=0),ndmin=2)
            X     = np.divide(np.subtract(X,means), stds)
        elif technique.lower() != 'none':
            print("The input column normalization is not recognized/supported!",technique)
            print("Using 'none' column normalization")
            print('--')
        return X
    
    def __optimal_clusters(self,X):
        N,D            = X.shape
        k_range        = list(self.k_range)
        if np.max(k_range) >= D:
            k_range = [val for val in k_range if val < D]
        self.k_range   = k_range
        inertia_list   = [(KMeans(k).fit(X)).inertia_ for k in self.k_range]
        diff01_array   = np.diff(inertia_list)
        diff02_array   = np.diff(diff01_array)
        ratio_array    = np.divide(diff02_array, diff01_array[1:] + 1E-10)
        self.optimal_k = np.argmax(ratio_array)
        return KMeans(self.optimal_k).fit(X)
    
    def __score_outliers(self,X):
        n,d = X.shape
        df  = pd.DataFrame(np.array(range(n),ndmin=2).T,columns=['entity'])
        for score in self.outlier_scoring:
            if score.lower()         == 'cluster_size':
                df['cluster_size']   = self.__score_cluster_size(X)
            elif score.lower()       == 'dist2center':
                df['dist2center']    = self.__score_dist2center(X)
            elif score.lower()       == 'dist2neighbor':
                df['dist2neighbor'] = self.__score_dist2neighbor(X)
        return df
                
    def __score_cluster_size(self,X):
        df_clusters       = pd.DataFrame(self.model.labels_,columns=['label'])
        df_sizes          = df_clusters.groupby(['label']).size().reset_index(name='count').set_index(keys='label')
        df_sizes['count'] = 1.0 / df_sizes['count']
        scores            = df_clusters.join(df_sizes,on='label',how='left')['count'].values
        scores            = (scores - np.mean(scores)) / (np.std(scores) + 1E-10)
        return scores
    
    def __score_dist2center(self,X):
        scores = np.min(pairwise_distances(X,self.model.cluster_centers_),axis=1)
        return scores
    
    def __score_dist2neighbor(self,X,k=10):
        k = np.min([k,X.shape[1]])
        scores = np.mean(pairwise_distances(X,self.model.cluster_centers_)[:,:k],axis=1)
        return scores
    
    def __scale_outliers(self,df):
        columns     = list(df.columns)
        columns.remove('entity')
        Y           = df[columns].values
        Y           = self.__normalize_column(Y,self.outlier_scaling)
        df[columns] = Y
        return df
    
    def __finalize_outliers(self,df,technique,X):
        columns     = list(df.columns)
        columns.remove('entity')
        Y           = np.array(df[columns].values,ndmin=2)
        if technique.lower() == 'max':
            df['outlier_score'] = np.max(Y,axis=1)
        elif technique.lower() == 'pca':
            df['outlier_score'] = PCA(n_components=1).fit_transform(Y)
        elif technique.lower() == 'density':
            df['outlier_score'] = self.__score_density(Y,X)
        else:
            print("The input finalization technique is not recognized/supported!")
            print("Using 'max' finalization")
            df['outlier_score'] = np.max(Y,axis=1)
        Y                   = np.array(df['outlier_score'].values,ndmin=2).T
        df['outlier_score'] = self.__normalize_column(Y,self.outlier_final_scaling)
        return df[['entity','outlier_score']]
    
    def __score_density(self,Y,X,k=10,samples=25):
        N,D          = X.shape
        N,d          = Y.shape
        k            = np.min([D-1,k])
        labels       = np.array(self.model.labels_)
        sample_index = []
        for i in set(labels):
            sample_index.append(np.random.choice(np.where(labels==i)[0],size=(samples,1),replace=True))
        sample_index     = np.vstack(sample_index)
        sample_X         = np.squeeze(X[sample_index,:])
        sample_Y         = np.squeeze(Y[sample_index,:])
        sample_distances = pairwise_distances(sample_X)
        sample_density   = np.mean(np.sort(sample_distances,axis=1)[:,1:k+1], axis=1)
        sample_Y         = np.hstack((np.ones((len(sample_index),1),dtype=float),sample_Y,sample_Y**2))
        model            = LinearRegression(fit_intercept=False,normalize=False,copy_X=False).fit(sample_Y,sample_density)
        coefficients     = model.coef_
        density          = np.dot(np.hstack((np.ones((N,1),dtype=float),Y,Y**2)),coefficients)
        return density
        
