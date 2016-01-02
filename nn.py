import numpy as np


class Neuron(object):
    """
    Hold the neuron data

    Value represents the z value of a neuron

    """

    def __init__(self, value=0):
        self.value = value

    def __repr__(self):
        return str(self.value)


class Layer(object):
    """
    Holds the layer data as follows you need to give the neuron number as an integer and remaining part takes the magic

    """
# TODO assure that the layer names only have 3 values

    _bias_array = np.ones(1)


    def __init__(self, neuron_number,
                 layer_type=None,
                 next_layer=None,
                 mapping_matrix=None,
                 value_vector=None,
                 a_vector=None):

        self.layer_type = layer_type
        self.next_layer = next_layer
        self.mapping_matrix = mapping_matrix
        self.value_vector = value_vector
        self.a_vector = a_vector
        self.neuron_list = []
        for i in range(0, neuron_number):
            neuron = Neuron()
            self.neuron_list.append(neuron)

    def __repr__(self):
        return '%s and number of neurons %s' % (self.layer_type, len(self.neuron_list))

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

    def generate_value_vector(self):
        """
        Return a vector that contains all the values of the input layer's neurons plus 1 as a bias.

        """

        if self.layer_type != 'input':
            raise NotImplementedError('This method is only available for input layer')
        result_vector = np.zeros(shape=(len(self.neuron_list) + 1, 1))  # initialize the vector
        result_vector[0] = 1  # add bias unit
        for i in range(1, len(self.neuron_list) + 1):
            result_vector[i] = self.neuron_list[i - 1].value
        self.value_vector = result_vector

    def add_bias_to_hidden_layer(self):
        """
        Adds bias unit to the sigmoid of value_vector if the layer type is hidden and also sets the a_value for
        forward propagation

        """
        if self.layer_type != 'hidden':
            raise NotImplementedError('This method is only available for hidden layers')
        self.a_vector = np.concatenate((Layer._bias_array, self._calculate_sigmoid()))

    def _add_bias_to_input_layer(self):
        """
        Add bias to the input layer without considering the sigmoid values since it is input layer

        """
        if self.layer_type != 'input':
            raise NotImplementedError('This method is only available for input layers')
        self.value_vector = np.concatenate((Layer._bias_array, self.value_vector))

    def _calculate_sigmoid(self):
        """
        Calculates the sigmoid of a value_vector in current layer which is denoted as a values

        """
        if self.layer_type == 'input':
            raise AssertionError('Input layer does not contain sigmoid version')
        return 1.0 / (1.0 + np.exp(-1.0 * self.value_vector))

    def calculate_next_layer_values(self):
        """
        We are given a mapping matrix (W) from layer l to layer l + 1, and required to find each layer's neuron's values
        Only thing is the operation operate this is matrix multiplication M x n where n is bias and the number of values
        of neuron in current layer


        Returns the next_layer's value_vector

        """
        if self.layer_type == 'input':
            return np.matmul(self.mapping_matrix, self.value_vector)
        elif self.layer_type == 'hidden':
            return np.matmul(self.mapping_matrix, self.a_vector)
        else:
            raise AssertionError('Output Layer cannot have next_layer')

    def update_value_vector(self, new_value_vector):
        """
        Designed for updating the value vector after forward propagation

        :param new_value_vector coming from the previous layer

        """
        if self.layer_type == 'input':
            raise AssertionError("Input layer's values cannot be updated")
        self.value_vector = new_value_vector

    def calculate_htheta(self):
        """
        Returns the sigmoid function of the output_layer's value_vector

        """
        if self.layer_type != 'output':
            raise NotImplementedError('calculate_output method is only available for output layer')
        self.a_vector = self._calculate_sigmoid()

    def update_mapping_matrix(self, matrix):
        """
        Updates the mapping matrix of a layer based on back-propagation algorithm
        :param matrix is the updated version of mapping_matrix

        """
        self.mapping_matrix = matrix

    def feed_input_layer(self, data_point):
        """


        """
        self.value_vector = data_point
        self._add_bias_to_input_layer()


class NN(object):
    """
    Define neurons and neural nets for work

    layer_neuron_list is a special kind of input for instance : [3 5 2] list means that we are going to have
    3 layers and by default input layer is the first entry and output layer is the last entry of layer_neuron_list
    and the numbers in layer_neuron_list represents the number of units in each layer

    """

    def __init__(self, layer_neuron_list):
        self.layer_list = []
        assert (layer_neuron_list is not None), 'Empty list is not valid parameter for NN object'

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
        self._connect_layers()

    def __repr__(self):
        return 'Number of Layers: %s ' % len(self.layer_list)

    def _connect_layers(self):
        """
        Layers are automatically connected as soon as user initialized the neural net(NN)

        """
        assert (self.layer_list is not None), 'Layers have to contain neurons'
        for i in range(0, len(self.layer_list)):
            try:
                self.layer_list[i].connect_layer_to_next(self.layer_list[i + 1])
            except IndexError:
                pass

    def forward_propagate(self, data):
        """
        Implements the forward propagation algorithm

        """
        for layer in self.layer_list:
            if layer.layer_type == 'input':
                layer.feed_input_layer(data)
                hidden_layer_value_vector = layer.calculate_next_layer_values()  # Calculate z vector of hidden layer
                layer.next_layer.value_vector = hidden_layer_value_vector  # Assign z vector to next layer
            elif layer.layer_type == 'output':
                layer.calculate_htheta()  # Corresponds to h_theta value of neural net
                print layer.a_vector
            else:
                layer.add_bias_to_hidden_layer()
                next_layer_value_vector = layer.calculate_next_layer_values()
                layer.next_layer.value_vector = next_layer_value_vector



