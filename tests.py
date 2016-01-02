from nn import NN
import scipy.io as sio

# Create Training data
train_data = sio.loadmat('ex4data1.mat')

if __name__ == '__main__':
    first_network = NN([3, 5, 2])
    first_network.forward_propagate()

