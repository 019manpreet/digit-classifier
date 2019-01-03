
import data
import numpy as np
#import pyplot  
import matplotlib.pyplot as plt
import math
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score

def compute_mean_mles(train_data, train_labels):
    
    # This function computes and returns the mean.

    ordered_digit_train_data = np.zeros((10,700,64)) # A 3-D array that holds all the values in arranged in the order of their labels
    for i in range(0,10):
        ordered_digit_train_data[i,:,:] = data.get_digits_by_label(train_data, train_labels, i)
    
	# means are stored in an array of size 10 by 64.
	# the ith row corresponds to the mean estimate for digit class i.
    means = np.zeros((10, 64))
	
    #for i in range(0,10):
        #temp_digit = data.get_digits_by_label(train_data, train_labels, i)
        #means[i] = np.mean(temp_digit,axis =0)
        
	# below is the vectorized implementation of the for loop above.
    means = np.mean(ordered_digit_train_data, axis=1)
    return means


def compute_sigma_mles(train_data, train_labels, means):
    
    # This function computes the covariance estimate for each digit class
	
	
    ordered_digit_train_data = np.zeros((10,700,64))
    for i in range(0,10):
        ordered_digit_train_data[i,:,:] = data.get_digits_by_label(train_data, train_labels, i)

    # covariances are stored in  3-D array 
	covariances = np.zeros((10, 64, 64))
    for i in range(0,10):
        temp_difference = np.subtract(ordered_digit_train_data[i,:,:],means[i,:])
        covariances[i,:,:] = np.dot(np.transpose(temp_difference), temp_difference) /np.size(ordered_digit_train_data,1)
        
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    This function computes the generative log-likelihood:
        log p(x|y,mu,Sigma)

    and returns an n x 10 array 
    '''
    first_term = 32*math.log(2*math.pi) # first term in the formulae, the factor of 32 comes from 1/2* number of elements in a vector

    log_px_given_y = np.zeros((np.size(digits,0),10))

    for i in range(0,10):
        covariance_k = covariances[i,:,:] + 0.01*(np.identity(64)) #covariance corresponding to each label with stabilising factor added to it
        det_covariance_k = np.linalg.det(covariance_k)
        digits_minus_mean_k = np.subtract(digits,means[i])     # corresponds to (x - mu)
        second_term = (math.log(det_covariance_k))/2     # second term in the formulae
        inverse_covariance_k = np.linalg.pinv(covariance_k)
        third_term = (np.einsum('ij,ij->i', np.dot(digits_minus_mean_k, inverse_covariance_k), digits_minus_mean_k))/2
        log_px_given_y[:,i] = -1*(first_term + second_term + third_term)     # value for a particular coloumn
                
    return log_px_given_y


def conditional_likelihood(digits, means, covariances):
    '''
    Computes the conditional likelihood:

        log p(y|x, mu, Sigma)

    and returns an array of shape (n, 10)
    '''
    log_py = math.log(0.1)
    log_px_given_y = generative_likelihood (digits,means,covariances)
    log_px_given_y_plus_log_py = log_px_given_y + log_py
    log_px = np.zeros((np.size(digits,0),1))
    for i in range(0,np.size(digits,0)):
        # summing over all values of k to get px
        log_px[i,:] = logsumexp(log_px_given_y_plus_log_py[i,:]) 
    
    # implementing the formulae
    log_py_given_x = np.subtract(log_px_given_y_plus_log_py,log_px)    
    
    return log_py_given_x

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Computes the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    
    log_py_given_x = conditional_likelihood(digits,means,covariances)
    logp_yi_given_xi = np.zeros(np.size(digits,0))
    
    for i in range (0, np.size(digits,0)):
        # getting the values (of log likelihood)corresponding to actual label
        logp_yi_given_xi[i] = log_py_given_x[i,int(labels[i])]
    
    avg = np.mean(logp_yi_given_xi)
    
    return avg



def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    predicted_labels = np.argmax(cond_likelihood, axis= 1)
    
    return predicted_labels

    
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('hw5digits.zip','hw5')
    
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels, means)
    
    print("############## For train_data #############")
    predicted_train_labels = classify_data(train_data, means, covariances)
    print("Accuracy = ", accuracy_score(train_labels, predicted_train_labels))
    print("Average conditional likelihood = ", avg_conditional_likelihood(train_data, train_labels, means, covariances))
    
    print("############## For test_data #############")
    predicted_test_labels = classify_data(test_data, means, covariances)
    print("Accuracy = ", accuracy_score(test_labels, predicted_test_labels))
    print("Average conditional likelihood = ", avg_conditional_likelihood(test_data, test_labels, means, covariances))
    
    fig=plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    for i in range(0,10):
        cov = covariances[i,:,:] 
        w, v = np.linalg.eig(cov)
        eig_vals_sorted = np.sort(w)
        eig_vecs_sorted = v[:, w.argsort()]
        #Computing the leading eigenvectors (largest eigenvalue) for each class covariance matrix
        temp_image = eig_vecs_sorted[:,63].reshape(8,8)
        #print("temp_image:", temp_image)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(temp_image)
    plt.show()
    

if __name__ == '__main__':
    main()

