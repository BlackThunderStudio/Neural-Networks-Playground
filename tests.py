from nn import NN

import scipy.io as sio

if __name__ == '__main__':

    # Create Training data
    train_data = sio.loadmat('ex4data1.mat')
    first_network = NN([400, 25, 10])
    input_size = len(train_data['X']) ** -1
    train_parameters = {'train_data':train_data,
                        'step_size':0.02,
                        'input_size':input_size,
                        'lambda':0.012,
                        'num_of_iterations': 100}

    # first_network._forward_propagate(train_data['X'][1])
    # a,b =first_network.back_propagate(train_data=train_data)
    first_network.train(**train_parameters)

