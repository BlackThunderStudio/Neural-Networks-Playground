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
                 a_vector=None,
                 delta_vector=None):

        self.layer_type = layer_type
        self.next_layer = next_layer
        self.mapping_matrix = mapping_matrix
        self.value_vector = value_vector
        self.a_vector = a_vector
        self.delta_vector = delta_vector
        self.neuron_list = []
        for i in range(0, neuron_number):
            neuron = Neuron()
            self.neuron_list.append(neuron)

    def __repr__(self):
        return '%s and number of neurons %s' % (self.layer_type, len(self.neuron_list))

    @staticmethod
    def _correct_numpy(array):
        """
        When the data is read from .mat file numpy does not create default columns which makes impossible the matrix
        calculation, therefore it is need to correct them
        :param array:

        """
        array.shape = (array.shape[0],1)
        return array

    @property
    def neuron_number(self):
        """
        Returns the neuron number

        """
        return len(self.neuron_list)

    @property
    def transpose_mapping_matrix(self):
        """
        In order to implement back propagation we need to calculate the theta_transpose times a_vector
        this method calculates the transpose operation

        """
        return np.transpose(self.mapping_matrix)

    @property
    def big_delta_hidden(self):
        """
        Used to calculate big delta from the equation of DELTA = delta^(l+1) * a_vector transpose

        """
        self.next_layer.delta_vector = Layer._correct_numpy(self.next_layer.delta_vector)
        self.a_vector = Layer._correct_numpy(self.a_vector)

        if self.layer_type != 'hidden':
            raise AssertionError('Big delta is only available for input and hidden layer')
        return np.matmul(self.next_layer.delta_vector, np.transpose(self.a_vector))

    @property
    def big_delta_input(self):
        if self.layer_type != 'input':
            raise AssertionError('This property is only available for input layer')

        self.a_vector = Layer._correct_numpy(self.a_vector)
        self.next_layer.delta_vector = Layer._correct_numpy(self.next_layer.delta_vector)
        return np.matmul(self.next_layer.delta_vector, np.transpose(self.a_vector))

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

    def add_bias_to_hidden_layer(self):
        """
        Adds bias unit to the sigmoid of value_vector if the layer type is hidden and also
        sets the a_value for forward propagation

        """
        if self.layer_type != 'hidden':
            raise AssertionError('This method is only available for hidden layers')
        self.a_vector = np.concatenate((Layer._bias_array, self._calculate_sigmoid()))

    def _add_bias_to_input_layer(self):
        """
        Add bias to the input layer without considering the sigmoid values since it is input layer

        """
        if self.layer_type != 'input':
            raise AssertionError('This method is only available for input layers')
        self.a_vector = np.concatenate((Layer._bias_array, self.value_vector))

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
        if self.layer_type == 'hidden' or self.layer_type == 'input':
            return np.matmul(self.mapping_matrix, self.a_vector)
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
            raise AssertionError('calculate_output method is only available for output layer')
        self.a_vector = self._calculate_sigmoid()

    def update_mapping_matrix(self, matrix):
        """
        Updates the mapping matrix of a layer based on back-propagation algorithm
        :param matrix is the updated version of mapping_matrix

        """
        self.mapping_matrix = matrix

    def feed_input_layer(self, data_point):
        """
        The data point containing the future set is equaled to the layer_s value vector


        """
        if self.layer_type != 'input':
            raise AssertionError('Only input layer can be feed')
        self.value_vector = data_point
        self._add_bias_to_input_layer()

    def calculate_sigmoid_gradient(self):
        """
        This method is the implementation of the gradient of the sigmoid function which is simply:
         a_vector * (1-a_vector)

        """
        if self.layer_type != 'hidden':
            raise AssertionError('This method is only available for hidden layer')
        size = self.a_vector.shape[0]
        ones = np.ones(size)  # create ones vector for subtraction
        return np.multiply(self.a_vector, ones - self.a_vector)

    def calculate_output_layer_delta(self, y_vector):
        """
        This method simply calculates the vectorized version of the last layer's delta value

        """
        if self.layer_type != 'output':
            raise AssertionError('This method is only available for output layer')
        self.delta_vector = self.a_vector - y_vector

    def calculate_hidden_layer_delta(self):
        """
        To calculate the hidden layer's delta values

        """
        if self.layer_type != 'hidden':
            raise AssertionError('This method is only available for hidden layer')
        theta_transpose_times_delta = np.matmul(self.transpose_mapping_matrix, self.next_layer.delta_vector)
        self.delta_vector = np.delete(np.multiply(theta_transpose_times_delta, self.calculate_sigmoid_gradient()), 0, 0)


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
                self.layer_list.append(layer)
            else:
                layer = Layer(number, layer_type='hidden')
                self.layer_list.append(layer)
        self._connect_layers()

    def __repr__(self):
        return 'Number of Layers: %s ' % len(self.layer_list)

    @property
    def output_layer(self):
        """

        :return: output layer of the given neural net

        """
        return self.layer_list[-1]

    @property
    def hidden_layer_list(self):
        """

        :return: hidden layer list of a network
        """
        ret = list()
        for layer in self.layer_list:
            if layer.layer_type == 'hidden':
                ret.append(layer)
            else:
                pass
        return ret

    @property
    def input_layer(self):
        """

        :return: input layer of a network
        """
        return self.layer_list[0]

    @property
    def h_theta(self):
        """
        This property works correctly only if it works after forward propagation, which makes sense because it is only
        used for

        :return:
        """
        for layer in self.layer_list:
            if layer.layer_type != 'output':
                pass
            else:
                return layer.a_vector

    @staticmethod
    def _create_y_mapping(y_value):
        """
        Codes the y vector sparsely
        :param y_value: integer in between 0-9
        :return: a numpy array containing zeros and only one 1 in the true label
        """
        ret = np.zeros(10)
        if y_value == 10:
            ret[0] = 1
        else:
            ret[y_value] = 1
        return ret

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

    def _forward_propagate(self, data):
        """
        Implements the forward propagation algorithm

        """
        self.input_layer.feed_input_layer(data)
        self.input_layer.next_layer.value_vector = self.input_layer.calculate_next_layer_values()  # Calculate z vector of hidden layer
        for layer in self.hidden_layer_list:
            layer.add_bias_to_hidden_layer()
            next_layer_value_vector = layer.calculate_next_layer_values()
            layer.next_layer.value_vector = next_layer_value_vector
        self.output_layer.calculate_htheta()  # Corresponds to h_theta value of neural net

    def back_propagate(self, **kwargs):
        """
        Implements the back propagation algorithm for neural net


        """
        cumulative_delta_input_layer = np.zeros(self.input_layer.mapping_matrix.shape)
        for hidden_layer in self.hidden_layer_list:
            cumulative_delta_hidden_layer = np.zeros(hidden_layer.mapping_matrix.shape)
        for i in range(0, len(kwargs['train_data']['X'])):
            self._forward_propagate(kwargs['train_data']['X'][i])
            self.output_layer.calculate_output_layer_delta(NN._create_y_mapping(kwargs['train_data']['y'][i]))
            for hidden_layer in self.hidden_layer_list:
                hidden_layer.calculate_hidden_layer_delta()
            cumulative_delta_input_layer += self.input_layer.big_delta_input
            cumulative_delta_hidden_layer += self.hidden_layer_list[0].big_delta_hidden

        return cumulative_delta_input_layer, cumulative_delta_hidden_layer

    def update_weights(self, big_delta_input, big_delta_hidden_list, **kwargs):
        """

        :param big_delta_input: mapping matrix from input layer to hidden layer
        :param big_delta_hidden_list: mapping matrices in between hidden layers and output layer
        :return:
        """
        constant = kwargs['step_size'] * (kwargs['input_size'] ** -1)
        old_theta_first_column = self.input_layer.mapping_matrix[:,0]
        old_theta_first_column.shape = (old_theta_first_column.shape[0], 1)
        new_theta = self.input_layer.mapping_matrix - (constant * (big_delta_input + kwargs['lambda'] * self.input_layer.mapping_matrix))
        new_theta += old_theta_first_column * kwargs['lambda'] * constant
        self.input_layer.update_mapping_matrix(new_theta)
        for i in range(0, len(big_delta_hidden_list)):
            layer = self.hidden_layer_list[i]
            old_theta_first_column = layer.mapping_matrix[:,0]
            old_theta_first_column.shape = (old_theta_first_column.shape[0], 1)
            new_theta = layer.mapping_matrix - (constant * (big_delta_hidden_list[i] + kwargs['lambda'] * layer.mapping_matrix))
            new_theta += old_theta_first_column * kwargs['lambda'] * constant
            layer.update_mapping_matrix(new_theta)

    def train(self, **kwargs):
        """
        Combines all the logic


        """
        for i in range(0, kwargs['num_of_iterations']):
            big_delta_input, big_delta_hidden = self.back_propagate(**kwargs)
            self.update_weights(big_delta_input, [big_delta_hidden], **kwargs)

    def predict(self, **kwargs):
        """
        Tests the accuracy of the model build during the training
        :return:
        """
        true_number = 0
        for i in range(0, len(kwargs['test_data']['y'])):

            self._forward_propagate(kwargs['test_data']['X'][i])
            predicted_value = np.where(self.h_theta == self.h_theta.max())[0][0]

            # If the value of the y is 10 it is represented as zero
            if kwargs['test_data']['y'][i][0] == 10:
                if predicted_value == 0:
                    true_number += 1
                else:
                    pass
            else:
                if kwargs['test_data']['y'][i][0] == predicted_value:
                    true_number += 1
                else:
                    pass

        return true_number








