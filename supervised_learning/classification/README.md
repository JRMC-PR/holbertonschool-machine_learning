# Classification

## General
## Machine Learning Glossary

- **Model**: In machine learning, a model is a mathematical representation of a real-world process. It is created through a process of training on a dataset.

- **Supervised Learning**: Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The model learns to predict the label from the features of the data.

- **Prediction**: A prediction is the output of a machine learning model when provided with an input instance. It is the model's best guess of the target variable.

- **Node**: In the context of neural networks, a node (or neuron) is a basic unit of a neural network that receives inputs and produces an output.

- **Weight**: Weights are the coefficients that the neural network learns during training. They determine the importance of the input features for the prediction task.

- **Bias**: Bias is an additional parameter in the neural network which is used to adjust the output along with the weighted sum of the inputs to the neuron.

- **Activation Functions**: Activation functions are mathematical equations that determine the output of a neural network. They introduce non-linear properties to the network.

    - **Sigmoid**: The sigmoid function is an activation function that outputs a value between 0 and 1. It is often used for binary classification problems.

    - **Tanh**: The tanh (hyperbolic tangent) function is an activation function that outputs a value between -1 and 1. It is similar to the sigmoid function but can handle negative input values.

    - **Relu**: The ReLU (Rectified Linear Unit) function is an activation function that outputs the input directly if it is positive, otherwise, it outputs zero.

    - **Softmax**: The softmax function is an activation function that turns numbers aka logits into probabilities that sum to one. It is often used in multi-class classification problems.

- **Layer**: A layer in a neural network is a collection of neurons which process a set of input features and produce an output.

- **Hidden Layer**: Hidden layers in a neural network are layers that are not directly connected to the input or output. They perform complex computations on the inputs received from the previous layers.

- **Logistic Regression**: Logistic Regression is a statistical model used for binary classification problems. It uses the logistic sigmoid function as its activation function.

- **Loss Function**: A loss function measures the disparity between the actual and predicted values in machine learning. It is used to update the weights during training.

- **Cost Function**: A cost function is the average of the loss functions of the entire dataset. It is a measure of how well the neural network is performing.

- **Forward Propagation**: Forward propagation is the process of passing the input data through the neural network to get the predicted output.

- **Gradient Descent**: Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively adjusting the model's parameters.

- **Back Propagation**: Backpropagation is the method used to calculate the gradient of the loss function with respect to the weights and biases in a neural network.

- **Computation Graph**: A computation graph is a way to represent a math function in the context of machine learning. It is used in the backpropagation process to compute the gradients.

- **Initializing Weights/Biases**: Weights and biases can be initialized in several ways, such as zero initialization, random initialization, and Xavier/Glorot initialization.

- **Importance of Vectorization**: Vectorization is the process of converting an algorithm from operating on a single value at a time to operating on a set of values (vector) at one time. It is important for computational efficiency.

- **Splitting Data**: Data in machine learning is typically split into a training set, a validation set, and a test set. This is done to evaluate the model's performance and prevent overfitting.

- **Multiclass Classification**: Multiclass classification is a classification task where the output variable can take on more than two values.

- **One-Hot Vector**: A one-hot vector is a vector in which all of the elements are zero, except for one, which is one. It is often used to represent categorical variables.

- **Encoding/Decoding One-Hot Vectors**: One-hot vectors can be encoded using functions like `pandas.get_dummies()` or `sklearn.preprocessing.OneHotEncoder()`. Decoding can be done by finding the index of the maximum value.

- **Softmax Function**: The softmax function is used in the output layer of a neural network, it turns logits into probabilities that sum to one. It is used in multi-class classification problems.

- **Cross-Entropy Loss**: Cross-entropy loss is a loss function that measures the performance of a classification model whose output is a probability value between 0 and 1.

- **Pickling in Python**: Pickling in Python is the process of serializing and deserializing Python object structures. It converts Python objects into a format that can be saved to disk or transmitted over a network.

