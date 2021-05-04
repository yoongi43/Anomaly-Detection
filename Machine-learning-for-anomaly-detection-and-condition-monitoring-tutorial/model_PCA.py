from sklearn.decomposition import PCA
from load import *



def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


class model_PCA(PCA):

    def __init__(self, n_components, svd_solver, extreme=True):
        super().__init__(n_components=n_components, svd_solver=svd_solver)
        self.X_train = None
        self.X_train_PCA = None
        self.data_train = None

        self.cov_mtx, self.inv_cov_mtx = None, None
        self.mean_distr = None
        self.dist_train = None
        self.threshold = None

        self.X_test = None
        self.X_train_PCA = None
        self.data_test = None
        self.dist_test = None

    def fit_transform_train(self, X_train, extreme=True):
        self.X_train = X_train
        self.X_train_PCA = pd.DataFrame(self.fit_transform(X_train))
        print("PCA shape", self.X_train_PCA.shape)
        self.X_train_PCA.index = X_train.index
        self.data_train = np.asarray(self.X_train_PCA.values)

        self.cov_mtx, self.inv_cov_mtx = self.cov_matrix()
        self.mean_distr = self.data_train.mean(axis=0)
        self.dist_train = self.MahalanobisDist(self.data_train, verbose=False)
        self.threshold = self.MD_threshold(self.dist_train, extreme=extreme)

    def cov_matrix(self, verbose=False):
        cov_mtx = np.cov(self.data_train, rowvar=False)
        if is_pos_def(cov_mtx):
            inv_cov_mtx = np.linalg.inv(cov_mtx)
            if is_pos_def(inv_cov_mtx):
                return cov_mtx, inv_cov_mtx
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")

    def MahalanobisDist(self, data, verbose=False):
        vars_mean = self.mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i] @ self.inv_cov_mtx @ diff[i].T))
        return md

    def MD_threshold(self, dist, extreme=False, verbose=False):
        k=3. if extreme else 2.
        threshold = np.mean(dist) * k
        return threshold

    def MD_detectOutliers(self, dist, extreme=False, verbose=False):
        threshold = self.MD_threshold(dist, extreme)
        outliers_bool = dist >= threshold

        return np.where(outliers_bool)[0]

    def transform_test(self, X_test):
        self.X_test = X_test
        self.X_test_PCA = pd.DataFrame(self.transform(X_test))
        self.X_test_PCA.index = X_test.index
        self.data_test = np.asarray(self.X_test_PCA.values)
        self.dist_test = self.MahalanobisDist(self.data_test, verbose=False)

    def show_dist_figure(self, square=True):
        if square:
            data = np.square(self.dist_train)
            color = 'blue'
        else:
            data = self.dist_train
            color = 'green'
        sns.displot(data,
                    bins=10,
                    kde=not square,
                    color=color)

        plt.xlim([0.0, 15 if square else 5])
        plt.xlabel('Mahalanobis dist')
        # plt.figure(figsize=(8,4))
        # plt.hist(np.square(data),
        #          bins=10,
        #          color=color)
        plt.show()

    def anomaly_detection(self, showflag=False):
        anomaly_train = pd.DataFrame()
        anomaly_train['Mob dist'] = self.dist_train
        anomaly_train['Thresh'] = self.threshold

        anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
        anomaly_train.index = self.X_train_PCA.index

        anomaly_test = pd.DataFrame()
        anomaly_test['Mob dist'] = self.dist_test
        anomaly_test['Thresh'] = self.threshold

        anomaly_test['Anomaly'] = anomaly_test['Mob dist'] > anomaly_test['Thresh']
        anomaly_test.index = self.X_test_PCA.index

        anomaly_alldata = pd.concat([anomaly_train, anomaly_test])
        anomaly_alldata.to_csv("./data/Anomaly_distance.csv")

        if showflag:
            anomaly_alldata.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=['green', 'red'])
            plt.show()

        return anomaly_train, anomaly_test, anomaly_alldata


# if __name__=="__main__":
