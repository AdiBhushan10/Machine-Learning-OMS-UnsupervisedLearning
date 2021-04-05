import warnings
warnings.filterwarnings('ignore')
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score 
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import scale, StandardScaler
from sklearn import mixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
import statistics as sts
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import scipy

randomstate= 333
# Dataset Prep
MyFrame = pd.read_csv('C:/Users/User2/Desktop/GATECH/Machine Learning/Datasets/Dataset1 - Assignment 1/ExoPlanetData.csv')
MyFrame['Detected_ExoPlanet'] = MyFrame['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
MyFrame.drop(columns=['Object_Name','DispositionUsingKeplerData'], inplace=True)  # Dropping columns that are not required for classification such as name of the observed celestial object 
MyFrame.fillna(MyFrame.median(), inplace=True)  # Handling missing values via median()
print(MyFrame.shape)

# Independent features and target feature
train_set = MyFrame.drop(columns=['Detected_ExoPlanet'])
labels = MyFrame['Detected_ExoPlanet']
lbl = labels
# Shuffle to avoid underlying distributions
x, labels = shuffle(train_set, labels, random_state=randomstate)
print(x.shape, labels.shape)
data = scale(x.values)
#data = data[9000,]
print(data.shape)
sample_size = 300


def K_Means_graphs():
    n_clusters = np.arange(2, 25, 1)
    kmeans_models = [KMeans(init='k-means++', n_clusters=n)
                        for n in n_clusters]
    Cost_SSE = np.zeros(n_clusters.size)
    
    #for model in kmeans_models:
    for idx, model in enumerate(kmeans_models):
        model.fit(data)
        Cost_SSE[idx] = model.inertia_
    plt.plot(n_clusters, Cost_SSE,'x-')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('Cost (SSE)')
    plt.show()
    
    #global data
    plt.figure()
    kmeans_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in kmeans_models]
    kmeans_ami_meas = [metrics.adjusted_mutual_info_score(labels, model.fit(data).predict(data)) for model in kmeans_models]#.astype(np.uint8)
    plt.plot(n_clusters, kmeans_v_meas,'x-', label='V-measure') 
    plt.plot(n_clusters, kmeans_ami_meas,'x-', label='AMI')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('Score')
    plt.show()

def EM_graphs():
    n_clusters = np.arange(2, 25, 1)
    spherical_models = [mixture.GaussianMixture(n, covariance_type='spherical', max_iter=20, random_state=randomstate)
              for n in n_clusters]
    diag_models = [mixture.GaussianMixture(n, covariance_type='diag', max_iter=20, random_state=randomstate)
              for n in n_clusters]
    tied_models = [mixture.GaussianMixture(n, covariance_type='tied', max_iter=20, random_state=randomstate)
              for n in n_clusters]
    full_models = [mixture.GaussianMixture(n, covariance_type='full', max_iter=20, random_state=randomstate)
              for n in n_clusters]
    spherical_bics = [model.fit(data).bic(data) for model in spherical_models]
    diag_bics = [model.fit(data).bic(data) for model in diag_models]
    tied_bics = [model.fit(data).bic(data) for model in tied_models]
    full_bics = [model.fit(data).bic(data) for model in full_models]

    plt.figure()
    EM_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in full_models]
    EM_ami_meas = [metrics.adjusted_mutual_info_score(labels, model.fit(data).predict(data)) for model in full_models]
    plt.plot(n_clusters, EM_v_meas,'o-', label='V-measure - Diagonal Covariance')
    plt.plot(n_clusters, EM_ami_meas,'x-', label='AMI')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('Score')
    plt.show()

    return full_bics

def EM_vs_Kmeans():
    plt.figure()
    n_clusters = np.arange(2, 25, 1)
    kmeans_models = [KMeans(init='k-means++', n_clusters=n)
                        for n in n_clusters]
    diag_models = [mixture.GaussianMixture(n, covariance_type='diag', max_iter=20, random_state=randomstate)
              for n in n_clusters]
    kmeans_ami_meas = [metrics.adjusted_mutual_info_score(labels, model.fit(data).predict(data)) for model in kmeans_models]
    EM_ami_meas = [metrics.adjusted_mutual_info_score(labels, model.fit(data).predict(data)) for model in diag_models]
    plt.plot(n_clusters, kmeans_ami_meas,'o-', label='k-means')
    plt.plot(n_clusters, EM_ami_meas,'x-', label='EM')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('Adjusted MI Score')
    plt.show()

def PCA_graphs():
    pca = PCA()
    pca.fit_transform(data)
    plt.plot(range(1,19,1), np.round(100*pca.explained_variance_ratio_), '--',c='r')
    plt.axvline(10, linestyle='dotted', label='Optimal n_components', linewidth = 1)
    #plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label = labels)
    plt.ylabel('Explained Variance (in %)')
    plt.xlabel('Number of Components (Attributes)')
    plt.title('PCA Explained Variance Plot')
    plt.show()

    pca = PCA()
    pca.fit_transform(data)
    y = np.round(100*pca.explained_variance_ratio_.cumsum())
    x = ['PC' + str(x) for x in range(1,len(y)+1)]
    plt.plot(range(1,19,1),y, '--',c='r')
    plt.axvline(10, linestyle='dotted', label='Optimal n_components', linewidth = 10)
    plt.bar(x=range(1,len(y)+1), height = y, color='y', tick_label = x)
    plt.ylabel('Cumulative Explained Variance (in %)')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    plt.show()

def Kurt(X, ica_, n_components):    
    components = ica_.components_
    ica_.components_ = components[:n_components]
    transformed = ica_.transform(X)
    ica_.components_ = components
    kurtosis = scipy.stats.kurtosis(transformed)
    return sorted(kurtosis, reverse = True)

def ICA_graphs():
    
    reconstruction_error = []#dict{}
    n_components = range(1,19,1)
    for n in n_components:
        ica = FastICA(n_components = n, random_state= randomstate)
        data_tfm = ica.fit_transform(data)
        data_proj = ica.inverse_transform(data_tfm) 
        reconstruction_error.append(((data - data_proj) ** 2).mean())  # This is mean of squared errors
    
    plt.plot(n_components, reconstruction_error, '--',c='r')
    plt.axvline(12, linestyle='dotted', label='Optimal n_components', linewidth = 1)
    plt.legend(loc='best')
    plt.xlabel('Number of Components (Attributes)')
    plt.ylabel('Reconstruction Error')
    plt.title('ExoPlanet Identification - Error Plot')
    plt.show()
    
    print("Calculating kurtosis...")
    decisiontree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, min_samples_split = 3)
    ica = FastICA()
    pipe = Pipeline(steps=[('ica', ica), ('decisionTree', decisiontree)])
    X,y = data, labels
    ica.fit(X)
    fig, ax = plt.subplots()
    ax.bar(list(range(1,19)),Kurt(X,ica, 18) , linewidth=2, color = 'blue')
    plt.axvline(12, linestyle='dotted', label='Optimal n_components', linewidth = 1)
    plt.title('Kurtosis for ICA - ExoPlanet Dataset')
    plt.xlabel('n_components')
    ax.set_ylabel('kurtosis')
    plt.show()


def Show_Clusters(x, y, mdl, alg):
        # Reducing the data with given algorithm
        if alg == 'pca':
            pca = PCA(n_components=2, random_state=randomstate)
        if alg == 'ica':
            pca = FastICA(n_components=2, random_state=randomstate)
        if alg == 'rp':
            pca = GaussianRandomProjection(n_components=2, random_state=randomstate)
        x_pca = pca.fit_transform(x)
        n_classes = len(np.unique(y))

        # Create dataframe
        df = pd.DataFrame(x_pca, columns=['pca1', 'pca2'])
        df['labels'] = y
        if mdl == 'KMeans':
            name=mdl
            mdl = KMeans(n_clusters=2, init='k-means++', max_iter=1000, random_state=randomstate)
        else:
            name = mdl
            mdl = mixture.GaussianMixture(2, covariance_type='diag', max_iter=20, random_state=randomstate)
        df['classes'] = mdl.fit_predict(x)

        # Generate plot
        fig, ax = plt.subplots(1,2 , figsize=(15, 8))
        Generate_Clusters(ax, 'pca1', 'pca2', df, name, alg)
        

def Generate_Clusters(ax, component1, component2, df, cluster,alg):
    # Plot input data onto first two components at given axes
    pre_colors = sns.color_palette('hls', len(np.unique(df['labels'])))
    post_colors = sns.color_palette('hls', len(np.unique(df['classes'])))
    sns.scatterplot(x=component1, y=component2, hue='labels', palette=pre_colors, data=df, legend='full', alpha=1.0, ax=ax[0])
    sns.scatterplot(x=component1, y=component2, hue='classes', palette=post_colors, data=df, legend='full', alpha=1.0, ax=ax[1])

    # Set titles
    ax[0].set_title('Before clustering with {}'.format(alg.upper()))
    ax[1].set_title('After {} Clustering with {}'.format(cluster.upper(), alg.upper()))
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)
    plt.show()

def RP_graphs():
    reconstruction_error = []#dict{}
    n_components = range(1,19,1)
    for n in n_components:
        grp = GaussianRandomProjection(n_components = n, random_state= randomstate)
        data_tfm = grp.fit_transform(data)
        data_proj = data_tfm.dot(grp.components_) + np.mean(data, axis = 0)
        reconstruction_error.append(((data - data_proj) ** 2).mean())  # This is mean of squared errors
    plt.plot(n_components, reconstruction_error, '--',c='r')
    plt.axvline(10, linestyle='dotted', label='Optimal n_components', linewidth = 1)
    plt.legend(loc='best')
    plt.xlabel('Number of Components (Attributes)')
    plt.ylabel('Root Mean Squared Error')
    plt.title('ExoPlanet Identification - Error Plot')
    plt.show()

def FA_graphs():
    reconstruction_error = []#dict{}
    egv = []
    n_components = range(1,19,1)
    for n in n_components:
        fa =  FactorAnalysis(n_components = n, random_state= randomstate)
        fa.fit(data)
        #egv.append(fa.eigenvalues_)
        data_tfm = fa.fit_transform(data)
        data_proj = data_tfm.dot(fa.components_) + np.mean(data, axis = 0) 
        reconstruction_error.append(((data - data_proj) ** 2).mean())  # This is mean of squared errors
    plt.plot(n_components, reconstruction_error, '--',c='r')
    plt.axvline(11, linestyle='dotted', label='Optimal n_components', linewidth = 1)
    plt.legend(loc='best')
    plt.xlabel('Number of Components (Attributes)')
    plt.ylabel('Root Mean Squared Error')
    plt.title('ExoPlanet Identification - Error Plot')
    plt.show()

def Dim_Red_Time_graphs():
    from time import time
    plt.figure()
    n_components = [i for i in range(1, 19)]
    PCA_algorithms = [PCA(n_components=n) for n in n_components]
    ICA_algorithms = [FastICA(n_components=n) for n in n_components]
    RP_algorithms = [GaussianRandomProjection(n_components=n) for n in n_components]
    FA_algorithms = [FactorAnalysis(n_components=1) for n in n_components]
    PCA_times = []
    for algo in PCA_algorithms:
        t0 = time()
        algo.fit_transform(data)
        PCA_times.append(time() - t0)
    ICA_times = []
    for algo in ICA_algorithms:
        t0 = time()
        algo.fit_transform(data)
        ICA_times.append(time() - t0)
    RP_times = []
    for algo in RP_algorithms:
        t0 = time()
        algo.fit_transform(data)
        RP_times.append(time() - t0)
    FA_times = []
    for algo in FA_algorithms:
        t0 = time()
        algo.fit_transform(data, labels)
        FA_times.append(time() - t0)
    plt.plot(n_components, PCA_times, label='PCA')
    plt.plot(n_components, ICA_times, label='ICA')
    plt.plot(n_components, RP_times, label='RP')
    plt.plot(n_components, FA_times, label='FA')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('Time Taken (in seconds)')
    plt.title('Fit Time Comparison - ExoPlanet Identification Dataset')
    plt.show()

def KMeans_PCA_Score():
    reduced_data = PCA(n_components = 10, random_state= randomstate).fit_transform(data, labels)
    km = KMeans(init='k-means++', n_clusters=15, random_state= randomstate)
    km.fit(reduced_data)
    clusters = km.predict(reduced_data)
    print('KMeans with PCA - Accuracy:',max(f1_score(labels, clusters,average=None)))

def KMeans_ICA_Score():
    reduced_data = FastICA(n_components = 12, random_state= randomstate).fit_transform(data, labels)
    km = KMeans(init='k-means++', n_clusters=15, random_state= randomstate)
    km.fit(reduced_data)
    clusters = km.predict(reduced_data)
    print('KMeans with ICA - Accuracy:',max(f1_score(labels, clusters,average=None)))

def KMeans_RP_Score():
    reduced_data = GaussianRandomProjection(n_components = 10, random_state= randomstate).fit_transform(data, labels)
    km = KMeans(init='k-means++', n_clusters=15,random_state= randomstate)
    km.fit(reduced_data)
    clusters = km.predict(reduced_data)
    print('KMeans with GaussianRandomProjection - Accuracy:',max(f1_score(labels, clusters,average=None)))

def KMeans_FA_Score():
    reduced_data = FactorAnalysis(n_components = 11, random_state= randomstate).fit_transform(data, labels)
    km = KMeans(init='k-means++', n_clusters=15, random_state= randomstate)
    km.fit(reduced_data)
    clusters = km.predict(reduced_data)
    print('KMeans with FactorAnalysis - Accuracy:',max(f1_score(labels, clusters,average=None)))

def EM_PCA_Score():
    reduced_data = PCA(n_components=10, random_state=randomstate).fit_transform(data)
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='diag', max_iter=20, random_state=randomstate)
    gmm.fit(reduced_data)
    clusters = gmm.predict(reduced_data)
    print('EM with PCA - Accuracy:',max(f1_score(labels, clusters,average=None)))

def EM_ICA_Score():
    reduced_data = FastICA(n_components=12,random_state=randomstate).fit_transform(data)
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='diag', max_iter=20, random_state=randomstate)
    gmm.fit(reduced_data)
    clusters = gmm.predict(reduced_data)
    print('EM with ICA - Accuracy:',max(f1_score(labels, clusters,average=None))) # 70.35

def EM_RP_Score():
    reduced_data = GaussianRandomProjection(n_components=10,random_state=randomstate).fit_transform(data)
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='diag', max_iter=20, random_state=randomstate)
    gmm.fit(reduced_data)
    clusters = gmm.predict(reduced_data)
    print('EM with GaussianRandomProjection - Accuracy:',max(f1_score(labels, clusters,average=None)))   # 43.36

def EM_FA_Score():
    reduced_data = FactorAnalysis(n_components = 11, random_state= randomstate).fit_transform(data, labels)
    gm = mixture.GaussianMixture(n_components=5, covariance_type='diag', max_iter=20, random_state=randomstate)
    gm.fit(reduced_data)
    clusters = gm.predict(reduced_data)
    print('EM with FactorAnalysis - Accuracy:',max(f1_score(labels, clusters,average=None)))   #76.5 #82.48

def NeuralNet_Basic():
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(data, labels, test_size=0.2, random_state=randomstate)
    NeuralNet = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')
    t0= time.time()
    NeuralNet.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet.predict(TstData)
    t1 = time.time() - t0
    print('Fit Model Timespan:',t1)   # 13.19
    print(f1_score(TstLabels, MyPrediction))  # 89.37
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - NN with PCA")
    plt.plot(NeuralNet.loss_curve_)
    plt.show()

def NeuralNet_PCA():
    reduced_data = PCA(n_components=10).fit_transform(data)
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(reduced_data, labels, test_size=0.2, random_state=randomstate)
    NeuralNet = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')
    t0= time.time()
    NeuralNet.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet.predict(TstData)
    t1 = time.time() - t0
    print('Fit Model Timespan:',t1)   # 13.19
    print(f1_score(TstLabels, MyPrediction))  # 89.37
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - NN with PCA")
    plt.plot(NeuralNet.loss_curve_)
    plt.show()

def NeuralNet_ICA():
    reduced_data = FastICA(n_components=12).fit_transform(data)   # N components received from ICA graph
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(reduced_data, labels, test_size=0.2, random_state=randomstate)
    NeuralNet = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')
    t0 = time.time()
    NeuralNet.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet.predict(TstData)
    t1 = time.time() - t0
    print('Fit Model Timespan:',t1)  # 12.79
    print(f1_score(TstLabels, MyPrediction))  # 90.4
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - NN with ICA")
    plt.plot(NeuralNet.loss_curve_)
    plt.show()


def NeuralNet_RP():
    reduced_data = GaussianRandomProjection(n_components=10).fit_transform(data)
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(reduced_data, labels, test_size=0.2, random_state=randomstate)
    NeuralNet = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')
    t0 = time.time()
    NeuralNet.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet.predict(TstData)
    t1 = time.time() - t0
    print('Fit Model Timespan:',t1)   # 13.20
    print(f1_score(TstLabels, MyPrediction)) # 87.20
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - NN with RP")
    plt.plot(NeuralNet.loss_curve_)
    plt.show()

def NeuralNet_FA():
    reduced_data = FactorAnalysis(n_components=11).fit_transform(data)
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(reduced_data, labels, test_size=0.2, random_state=randomstate)
    NeuralNet = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')
    t0= time.time()
    NeuralNet.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet.predict(TstData)
    t1 = time.time() - t0
    print('Fit Model Timespan:',t1)  # 4.54
    print(f1_score(TstLabels, MyPrediction)) # 89.87
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - NN with FA")
    plt.plot(NeuralNet.loss_curve_)
    plt.show()

def NeuralNet_Clustering():
    clusters1 = KMeans(init='k-means++', n_clusters=15, random_state=randomstate).fit_predict(data)
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(np.c_[data, clusters1], labels, test_size=0.25,
                                                                        random_state=randomstate)
    NeuralNet1 = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')  
    t0 = time.time()
    NeuralNet1.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet1.predict(TstData)
    t1 = time.time() - t0
    print('Fit Time - NN using KMeans:',t1) 
    print('Accuracy - NN using KMeans:',f1_score(TstLabels, MyPrediction))  

    clusters2 = mixture.GaussianMixture(n_components= 15, covariance_type='diag', max_iter=20, random_state=randomstate).fit_predict(data)
    #clusters2 = mixture.GaussianMixture(n_components= 5, covariance_type='diag', max_iter=20, random_state=randomstate).fit_predict(data)
    TrnData, TstData, TrnLabels, TstLabels = train_test_split(np.c_[data, clusters2], labels, test_size=0.25,
                                                                        random_state=randomstate)
    NeuralNet2 = MLPClassifier(hidden_layer_sizes=70, activation='logistic', solver = 'adam')  
    t0 = time.time()
    NeuralNet2.fit(TrnData, TrnLabels)
    MyPrediction = NeuralNet2.predict(TstData)
    t1 = time.time() - t0
    print('Fit Time - NN using EM:',t1) 
    print('Accuracy - NN using EM:',f1_score(TstLabels, MyPrediction))  

    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("ExoPlanet Identification - Neural Network with Clustering")
    plt.plot(NeuralNet1.loss_curve_)
    plt.plot(NeuralNet2.loss_curve_)
    plt.show()

#K_Means_graphs()
#EM_graphs()
#EM_vs_Kmeans()
#PCA_graphs()
#ICA_graphs()
#RP_graphs()
#FA_graphs()
#Dim_Red_Time_graphs()

#Show_Clusters(data, labels, 'KMeans','pca')
#Show_Clusters(data, labels,'EM','pca')
#Show_Clusters(data, labels, 'KMeans','ica')
#Show_Clusters(data, labels,'EM','ica')
#Show_Clusters(data, labels, 'KMeans','rp')
#Show_Clusters(data, labels,'EM','rp')

#KMeans_PCA_Score()
#EM_PCA_Score()
#KMeans_ICA_Score()
#EM_ICA_Score()
#KMeans_RP_Score()
#EM_RP_Score()
#KMeans_FA_Score()
#EM_FA_Score()

#NeuralNet_PCA()
#NeuralNet_ICA()
#NeuralNet_RP()
#NeuralNet_LDA()
#NeuralNet_FA()
#NeuralNet_Basic()
#NeuralNet_Kmeans()
#NeuralNet_EM_GMM()
#NeuralNet_Clustering()
'''
Best Output:
KMeans with PCA - Accuracy: 0.07371741656981572
EM with PCA - Accuracy: 0.2564690468391746
KMeans with ICA - Accuracy: 0.10977979274611398
EM with ICA - Accuracy: 0.7035977105478333
KMeans with GaussianRandomProjection - Accuracy: 0.12432086928731224
EM with GaussianRandomProjection - Accuracy: 0.433625730994152
KMeans with FactorAnalysis - Accuracy: 0.5463728191000918
EM with FactorAnalysis - Accuracy: 0.7650188140651356
'''