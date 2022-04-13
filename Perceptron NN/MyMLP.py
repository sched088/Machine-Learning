import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples (i.e., means and std)
        self.mean = x.mean(0)
        self.std = x.std(0) 
        pass # placeholder

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = x - self.mean
        x = x / (self.std + 1e-15)
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for row in range(0, len(label)):
        one_hot[row][label[row]] = 1
    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line
    x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    f_x = x # placeholder

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x,axis=1,keepdims=True))
    f_x = x / np.sum(x, axis=1,keepdims=True)
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid]) # column for each hidden unit
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y): # Train_y Shape: (1000,10) Train_x Shape: (1000,64)
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            wtx = np.transpose(self.weight_1).dot(np.transpose(train_x)) + np.transpose(self.bias_1)
            zh = tanh(wtx)
            vtx = np.transpose(np.transpose(self.weight_2).dot(zh) + np.transpose(self.bias_2))
            yi = softmax(vtx)



            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters

            # output layer
            dv = (yi - train_y) # * np.transpose(zh)
            dweights_2 = lr * (np.transpose(dv).dot(np.transpose(zh)))
            dbias_2 = lr * (np.sum(dv))
            
            # hidden layer
            dw = self.weight_2.dot(np.transpose(dv)) * (1-np.square(tanh(np.transpose(self.weight_1).dot(np.transpose(train_x)) + np.transpose(self.bias_1)))) # derivative of tanh == 1-tanh**2 
            dweights_1 = lr * (dw.dot(train_x))
            dbias_1 = lr * (np.sum(dw))  

            #update the parameters based on sum of gradients for all training samples
            self.weight_1 = self.weight_1 - np.transpose(dweights_1)
            self.weight_2 = self.weight_2 - np.transpose(dweights_2)
            
            self.bias_1 = self.bias_1 - dbias_1
            self.bias_2 = self.bias_2 - dbias_2

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        wtx = np.transpose(self.weight_1).dot(np.transpose(x)) + np.transpose(self.bias_1)
        zh = tanh(wtx)
        vtx = np.transpose(np.transpose(self.weight_2).dot(zh) + np.transpose(self.bias_2))
        yi = softmax(vtx)

        # convert class probability to predicted labels
        y = np.zeros([len(x),]).astype('int') # placeholder
        y = np.argmax(yi,axis = 1)
        # print('here')

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        # current shape (1867, 64) gradescope calling for (1867,4)?
        z = tanh(np.transpose(self.weight_1).dot(np.transpose(x)) + np.transpose(self.bias_1))
        z = np.transpose(z)
        # print('x')
        # print(np.shape(x))
        # print(x)
        # print('z')
        # print(np.shape(z))
        # print(z)
        # print("here")


        # z = x # placeholder

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
