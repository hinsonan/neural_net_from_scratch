import numpy as np

# class to house the neural net and its functions
class NeuralNet():

    def __init__(self):
        # seed for rng
        np.random.seed(1)
        
        # converting weights to a 3 by 1 matrix with values from 0 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1))
    
    # converts data from strings to ints
    def convert_data_to_ints(self, np_array_of_data):
        for i in range(len(np_array_of_data)):
            for j in range(len(np_array_of_data[i])):
                if (np_array_of_data[i][j] == "red"):
                    np_array_of_data[i][j] = 0
                else:
                    np_array_of_data[i][j] = 1
                    
        return np_array_of_data
    
    # tanh is the activation function we will use
    def tanh(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1
    
    # we will need the derivative in order to adjust the weights
    def tanh_derivative(self, x):
        return 1 - x**2
    
    def feed_forward(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        inputs = inputs.astype(float)
        output = self.tanh(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_input, training_output, training_iterations):
        # convert training inputs and outputs to ints so it is easier to work with
        noramlized_inputs = self.convert_data_to_ints(training_input)
        normalized_outputs = self.convert_data_to_ints(training_output)
        # train model for number of iterations
        for i in range(training_iterations):
            # call the method to push the data through the nueron
            output = self.feed_forward(noramlized_inputs)

            # calculate the error that will be used for back propagation
            error = normalized_outputs.astype(float) - output

            #adjust weights
            adjustments = np.dot(noramlized_inputs.astype(float).T, error * self.tanh_derivative(output))

            self.synaptic_weights += adjustments

if __name__ == "__main__":
    # training inputs
    training_inputs = np.array([["red", "blue", "red"],
              ["blue", "blue", "red"],
              ["red", "blue", "blue"],
              ["red", "red", "blue"],
              ["blue", "red", "blue"],
              ["blue", "blue", "blue"],
              ["red", "red", "red"]])

    # training outputs
    training_outputs = np.array([["red", "blue", "blue", "red", "blue", "blue", "red"]]).T
    
    # instantiate neural net
    neural_net = NeuralNet()
    
    # train our model
    neural_net.train(training_inputs, training_outputs, 200000)
    
    # test our model with this new input and see if it gets close to predicting the correct answer
    new_input = np.array([["red", "blue", "red"]])

    print(neural_net.feed_forward(neural_net.convert_data_to_ints(new_input)))
    