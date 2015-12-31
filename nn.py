import math


class Neuron(object):
    """
    Hold the neuron data

    Value represents the z valu of a neuron

    """

    def __init__(self, is_bias=False, value=0):
        self.is_bias = is_bias
        if is_bias:
            value = 1
        self.value = value

    def calculate_sigmoid(self):
        """
        Calculates the sigmoid value of the z value of the net

        """
        return 1 / (1 + math.exp(-self.value))

    def __repr__(self):
        return self.value


class Layer(object):
    """
    Holds the layer data

    """
#TODO assure that the neural names only have 3 values

    def __init__(self, neuron_number, layer_type=None, next_layer=None):
        self.layer_type = layer_type
        self.next_layer = next_layer
        self.neuron_list = []
        for i in range(0, neuron_number):
            neuron = Neuron()
            self.neuron_list.append(neuron)

    def add_bias_neuron(self):
        neuron = Neuron(is_bias=True)
        self.neuron_list.append(neuron)

    def create_mapping_matrix(self):
        """
        Let the Layer has n units itself and the next layer has m units therefore we need

        a transformation matrix from current layer to next layer which has a dimension of m x (n+1)
        but be careful that bias units are added automatically so we only neen m

        """
        if self.layer_type == 'output': # Output layer cannot have mapping
            return None
        else:
            current_unit_number = len(self.neuron_list)


    def connect_layer_to_next(self, layer_after_current_layer):
        """
        Connects the current layer to the next layer

        """
        self.next_layer = layer_after_current_layer

    def __repr__(self):
        return '%s and number of neurons %s' % (self.layer_type, len(self.neuron_list))


class NN(object):
    """
    Define neurons and neural nets for work


    """

    def __init__(self, layer_neuron_list):
        self.layer_list = []

        for number in layer_neuron_list:
            if layer_neuron_list.index(number) == 0:
                layer = Layer(number, layer_type='input')
                layer.add_bias_neuron()
                self.layer_list.append(layer)
            elif layer_neuron_list.index(number) == len(layer_neuron_list) - 1:
                layer = Layer(number, layer_type='output')
                layer.is_output_layer = True
                self.layer_list.append(layer)
            else:
                layer = Layer(number, layer_type='hidden')
                layer.add_bias_neuron()
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
    print first_network
