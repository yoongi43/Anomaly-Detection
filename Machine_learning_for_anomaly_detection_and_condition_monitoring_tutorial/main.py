from model_PCA import *
from model_Autoencoder import *


def main():

    # data_dir = './data/1st_test'
    data_dir = './data/2nd_test'
    X_train, X_test = load(data_dir, showflag=False, normalizeflag=True)
    """PCA model"""
    pca = model_PCA(n_components=2, svd_solver='full')
    pca.fit_transform_train(X_train)
    pca.transform_test(X_test)

    # pca.show_dist_figure(train=True)
    pca.show_dist_figure(square=False)
    pca.anomaly_detection(showflag=True)

    """Autoencoder"""
    AE = Autoencoder(X_train)
    pred_model(AE, X_train, X_test)


if __name__=="__main__":
    main()