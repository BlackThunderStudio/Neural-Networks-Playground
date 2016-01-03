from nn import NN
import scipy.io as sio

# Create Training data
train_data = sio.loadmat('ex4data1.mat')

if __name__ == '__main__':
    first_network = NN([400, 25, 10])
    first_network.back_propagate(data=train_data['X'][1], y=train_data['y'][1])
    print first_network.h_theta