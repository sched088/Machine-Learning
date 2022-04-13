"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package


# Header
import numpy as np

# Implement the Perceptron algorithm
from numpy import ndarray

### Problem is that it still goes to 100 iterations. Error rate is still 1.00 for perceptron and least square ###
### Goes to 100 iterations because w and w_prev are not counting as equal at loop
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w = w0
    w_prev = [0.0,0.0]
    error_rate = 1.00
    error_count = 0
    max_iterations = 100
    print("test:" + str(np.dot(X, w)))
    # loop until convergence (w does not change at all over one pass)
    # or until max iterations are reached
    for i in range(max_iterations):
        print(k)
        if np.all(w == w_prev):
            print(np.all(w-w_prev))
            error_rate = 1/len(y) * error_count
            break
        else:
            error_count = 0
            w_prev = w
            for xi, yi in zip(X, y):
                sum = np.dot(xi, w) * yi
                print("sum=" + str(sum))
                if sum > 0:
                    # activation = 1
                    continue
                else:
                    # activation = -1
                    print("w init " + str(w))
                    w = w + yi*xi
                    print("w post " + str(w))
                    error_count += 1
        #print("w init ool " + str(w))
        #print("w post ool " + str(w_prev))
        k+=1
    #        features = X.shape[1]
    #        misclass = 0
    #
    #        for training_example in range(X.shape[0]):
    #            print(training_example)
    #            if y != np.dot(X, w):
    #                print("Not Equal")
    #            else:
    #                print("Equal")



    # (current pass w ! = previous pass w), then do:
    #

        # for each training sample (x,y):
            # if actual target y does not match the predicted target value, update the weights


            # calculate the number of iterations as the number of updates


    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0


    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples


    return (w, k, error_rate)
