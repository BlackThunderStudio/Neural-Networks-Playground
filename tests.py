from nn import NN

import scipy.io as sio

if __name__ == '__main__':

    # Create Training data
    train_data = sio.loadmat('ex4data1.mat')
    first_network = NN([400, 25, 10])

    first_network.back_propagate(train_data=train_data)