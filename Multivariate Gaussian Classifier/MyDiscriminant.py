import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance (S1=S2)
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance (S1!=S2)
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        """
        Format of mean   [[0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 0. 0. 0. 0. 0. 0. 0.]]
        Approach to iterate through each value of ytrain and += corresponding Xtrain row to self.mean                      
        """
        filter_arr1 = ytrain == 1
        filter_arr2 = ytrain == 2
        self.mean[0] = np.mean(Xtrain[filter_arr1], axis=0)
        self.mean[1] = np.mean(Xtrain[filter_arr2], axis = 0)

        print("self.mean")
        print(self.mean)

        if self.shared_cov:
            # compute the class-independent covariance
            """
            S1 = S2 cov( all Xtrain - mean all ytrain)
            """
            self.S = np.cov(Xtrain.T, ddof=0)
            print("S1")
            print(self.S)
            return self.S
            # pass
        else:
            """
            This is where we have kdd
            """
            # compute the class-dependent covariance
            filter_arr1 = ytrain == 1
            filter_arr2 = ytrain == 2

            self.S[0] = np.cov(Xtrain[filter_arr1].T, ddof = 0)
            self.S[1] = np.cov(Xtrain[filter_arr2].T, ddof = 0)

            print("S1/S2")
            print(self.S)
            return self.S
            # S1, S2 = np.cov(self.mean[0].T), np.cov(self.mean[1].T)
            # return S1, S2
            

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            gx = np.zeros(self.k)
            for c in np.arange(self.k):
                if self.shared_cov:
                    # gx[c] = -0.5 * np.log(np.linalg.det(np.matrix(self.S))) - (0.5 * np.matrix(Xtest[i] - self.mean[c]) * (np.matrix(self.S)**-1)) * np.matrix(Xtest[i] - self.mean[c]).T + self.p[c]
                    gx[c] = -0.5 * np.log(np.linalg.det(np.matrix(self.S))) - 0.5 * (np.matrix(Xtest[i] - self.mean[c])) * (np.matrix(self.S))**-1 * (np.matrix(Xtest[i] - self.mean[c])).T + np.log(self.p[c])
                else:
                    gx[c] = -0.5 * np.log(np.linalg.det(np.matrix(self.S[c]))) - 0.5 * (np.matrix(Xtest[i] - self.mean[c])) * (np.matrix(self.S[c]))**-1 * (np.matrix(Xtest[i] - self.mean[c])).T + np.log(self.p[c])
            # determine the predicted class based on the values of discriminant function
            print(gx)
            if gx[0] > gx[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
        
        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        filter_arr1 = ytrain == 1
        filter_arr2 = ytrain == 2
        self.mean[0] = np.mean(Xtrain[filter_arr1], axis=0)
        self.mean[1] = np.mean(Xtrain[filter_arr2], axis = 0)

        # compute the variance of different features
        self.S = np.var(Xtrain, axis = 0)

        return self.S

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        gx = np.zeros(self.k)

        cov = np.zeros((Xtest.shape[1],Xtest.shape[1]))
        for n in range(Xtest.shape[1]):
            for m in range(Xtest.shape[1]):
                if n == m:
                    cov[n,m] = self.S[n]
                else:
                    continue
        self.S = cov

        for i in np.arange(Xtest.shape[0]): # for each test set example

            for c in np.arange(self.k):
                    gx[c] = -0.5 * np.log(np.linalg.det(np.matrix(self.S))) - 0.5 * (np.matrix(Xtest[i] - self.mean[c])) * (np.matrix(self.S))**-1 * (np.matrix(Xtest[i] - self.mean[c])).T + np.log(self.p[c])
        # print(predicted_class)

            # determine the predicted class based on the values of discriminant function
            if gx[0] > gx[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
