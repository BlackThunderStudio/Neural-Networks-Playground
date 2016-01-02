import math
import numpy as np


class Neuron(object):
    """
    Hold the neuron data

    Value represents the z valu of a neuron

    """

    def __init__(self, value=0):
        self.value = value

    def calculate_sigmoid(self):
        """
        Calculates the sigmoid value of the z value of the net

        """
        return 1 / (1 + math.exp(-self.value))

    def __repr__(self):
        return str(self.value)


class Layer(object):
    """
    Holds the layer data as follows you need to give the neuron number as an integer and remmaing part takes the magic

    """
# TODO assure that the layer names only have 3 values

    bias_array = np.ones((1,1))

    def __init__(self, neuron_number, layer_type=None, next_layer=None, mapping_matrix=None, value_vector=None):
        self.layer_type = layer_type
        self.next_layer = next_layer
        self.mapping_matrix = mapping_matrix
        self.value_vector = value_vector
        self.neuron_list = []
        for i in range(0, neuron_number):
            neuron = Neuron()
            self.neuron_list.append(neuron)

    def _create_random_mapping_matrix(self):
        """
        Let the Layer has l_in units itself and the next layer has l_out units therefore we need
        a transformation matrix from current layer to next layer which has a dimension of l_out x (l_in + 1)
        and, please notice that bias units are not count as neurons

        """
        if self.layer_type == 'output':  # Output layer cannot have mapping
            pass
        else:
            l_in = len(self.neuron_list)  # the number of units in current layer
            l_out = len(self.next_layer.neuron_list)  # the number of units in output layer
            weight_matrix = np.random.random((l_out, l_in + 1))  # initialized weight matrix in between 0 and 1
            weight_matrix = weight_matrix * 2 * .12  # normalize it through the epsilon value in which .12 for my choice
            weight_matrix -= 0.12  # shift them left by 0.12 so that the mean still remains zero
            self.mapping_matrix = weight_matrix

    def connect_layer_to_next(self, layer_after_current_layer):
        """
        Connects the current layer to the next layer

        """
        self.next_layer = layer_after_current_layer
        self._create_random_mapping_matrix()

    def _generate_value_vector(self, contains_bias=True):
        """
        Return a vector that contains all the values of the current layer's neurons plus 1 as a bias.

        """
        if contains_bias:
            result_vector = np.zeros(shape=(len(self.neuron_list) + 1, 1))  # initialize the vector
            result_vector[0] = 1  # add bias unit
            for i in range(1, len(self.neuron_list) + 1):
                result_vector[i] = self.neuron_list[i - 1].value
            self.value_vector = result_vector

    def calculate_next_layer_values(self):
        """
        We are given a mapping matrix (W) from layer l to layer l + 1, and required to find each layer's neuron's values
        Only thing is the operation operate this is matrix multiplication M x n where n is bias and the number of values
        of neuron in current layer

        """
        if self.layer_type == 'input':  # If it is the input layer then we do not need to calculate sigmoid of values
            pass




    def __repr__(self):
        return '%s and number of neurons %s' % (self.layer_type, len(self.neuron_list))


class NN(object):
    """
    Define neurons and neural nets for work

    layer_neuron_list is a special kind of input for instance : [3 5 2] list means that we are going to have
    3 layers and by default input layer is the first entry and output layer is the last entry of layer_neuron_list
    and the numbers in layer_neuron_list represents the number of units in each layer

    """

    def __init__(self, layer_neuron_list):
        self.layer_list = []

        for number in layer_neuron_list:
            if layer_neuron_list.index(number) == 0:
                layer = Layer(number, layer_type='input')
                self.layer_list.append(layer)
            elif layer_neuron_list.index(number) == len(layer_neuron_list) - 1:
                layer = Layer(number, layer_type='output')
                layer.is_output_layer = True
                self.layer_list.append(layer)
            else:
                layer = Layer(number, layer_type='hidden')
                layer.is_hidden_layer = True
                self.layer_list.append(layer)

    def connect_layers(self):
        for i in range(0, len(self.layer_list)):
            try:
                self.layer_list[i].connect_layer_to_next(self.layer_list[i + 1])
            except IndexError:
                pass

    def __repr__(self):
        return 'Number of Layers: %s ' % len(self.layer_list)


if __name__ == '__main__':
    first_network = NN([3, 5, 2])
    first_network.connect_layers()

