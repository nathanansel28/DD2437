from math import ceil
from typing import List, Tuple

from numpy import ndarray
from tqdm import tqdm
from util import *


class RestrictedBoltzmannMachine:
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    def __init__(
        self,
        ndim_visible,
        ndim_hidden,
        is_bottom=False,
        image_size=[28, 28],
        is_top=False,
        n_labels=10,
        batch_size=30000,
        show_histograms=False,
    ):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom:
            self.image_size = image_size

        self.is_top = is_top

        if is_top:
            self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)
        )

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.show_histograms = show_histograms

        # self.print_period = 5000
        self.print_period = 50

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            # "period" : 5000, # iteration period to visualize
            "period": 50,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(
                0, self.ndim_hidden, 25
            ),  # pick some random hidden units
        }

        self.history = {"reconstruction_loss": []}

        return

    def cd1(self, visible_trainset, n_iterations=10000):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]

        # number of mini batch in each iteration
        batches_number = ceil(n_samples / self.batch_size)
        iterations = tqdm(range(n_iterations + 1))

        for it in iterations:
            np.random.shuffle(visible_trainset)

            # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
            for batch in tqdm(
                range(batches_number), desc=f"iteration {it}", leave=False
            ):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, n_samples)

                v0_activation = visible_trainset[start:end, :]
                h0_prob, h0_activation = self.get_h_given_v(v0_activation)
                v1_prob, v1_activation = self.get_v_given_h(h0_activation)
                h1_prob, h1_activation = self.get_h_given_v(v1_activation)

                # [TODO TASK 4.1] update the parameters using function 'update_params'
                self.update_params(
                    v0_activation, h0_activation, v1_activation, h1_activation
                )

            # visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:

                viz_rf(
                    weights=self.weight_vh[:, self.rf["ids"]].reshape(
                        (self.image_size[0], self.image_size[1], -1)
                    ),
                    it=it,
                    grid=self.rf["grid"],
                )

            # print progress
            _, h = self.get_h_given_v(visible_trainset)
            _, reconstruction = self.get_v_given_h(h)
            loss = (
                self.compute_reconstruction_loss(visible_trainset, reconstruction)
                / self.batch_size
            )

            if it % self.print_period == 0:
                if self.show_histograms:
                    self.plot_histograms()

            self.history["reconstruction_loss"].append(loss)

            iterations.set_description(
                "iteration=%7d recon_loss=%4.4f" % (it, loss), refresh=True
            )

        return

    def update_params(self, v_0, h_0, v_k, h_k) -> None:
        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        self.delta_bias_v = self.learning_rate * (
            np.sum(v_0 - v_k, axis=0)
        )  # /v_0.shape[0]
        self.delta_weight_vh = self.learning_rate * ((v_0.T @ h_0) - (v_k.T @ h_k))
        self.delta_bias_h = self.learning_rate * (
            np.sum(h_0 - h_k, axis=0)
        )  # /h_0.shape[0]

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        # Apply momentum (if needed)
        # self.weight_vh += self.momentum * (1 - self.momentum) * self.delta_weight_vh
        # self.bias_v += self.momentum * (1 - self.momentum) * self.delta_bias_v
        # self.bias_h += self.momentum * (1 - self.momentum) * self.delta_bias_h

        return

    def get_h_given_v(
        self, visible_minibatch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)

        prob = sigmoid(self.bias_h + visible_minibatch @ self.weight_vh)
        h_sampled = (np.random.rand(*prob.shape) < prob).astype(np.float32)

        return prob, h_sampled

    def get_v_given_h(
        self, hidden_minibatch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            support = self.bias_v + hidden_minibatch @ self.weight_vh.T
            data, labels = support[:, : -self.n_labels], support[:, -self.n_labels :]

            data = sigmoid(data)
            data_sample = sample_binary(data)

            labels = softmax(labels)
            labels_sample = sample_categorical(labels)

            prob = np.concatenate((data, labels), axis=1)
            v_sampled = np.concatenate((data_sample, labels_sample), axis=1)

        else:

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            prob = sigmoid(self.bias_v + hidden_minibatch @ self.weight_vh.T)
            v_sampled = (np.random.rand(*prob.shape) < prob).astype(np.float32)

        return prob, v_sampled

    def plot_histograms(self):
        """Plot histograms of weights, visible biases, and hidden biases."""
        plt.figure(figsize=(15, 4))

        # Histogram of weights
        plt.subplot(1, 3, 1)
        plt.hist(self.weight_vh.flatten(), bins=50, color="blue", alpha=0.7)
        plt.title("Histogram of Weights")

        # Histogram of visible biases
        plt.subplot(1, 3, 2)
        plt.hist(self.bias_v, bins=50, color="red", alpha=0.7)
        plt.title("Histogram of Visible Biases")

        # Histogram of hidden biases
        plt.subplot(1, 3, 3)
        plt.hist(self.bias_h, bins=50, color="green", alpha=0.7)
        plt.title("Histogram of Hidden Biases")

        plt.show()

    def compute_reconstruction_loss(
        self, v_actual: np.ndarray, v_reconstructed: np.ndarray
    ) -> float:
        return np.linalg.norm(v_actual - v_reconstructed)

    def fetch_reconstruction_loss(self) -> List[float]:
        return self.history["reconstruction_loss"]

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(
        self, visible_minibatch: np.ndarray
    ) -> Tuple[ndarray[float], ndarray[float]]:
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        probs = sigmoid(self.bias_h + visible_minibatch @ self.weight_v_to_h)
        sample = sample_binary(probs)

        return probs, sample

    def get_v_given_h_dir(
        self, hidden_minibatch: np.ndarray
    ) -> Tuple[ndarray[float], ndarray[float]]:
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            raise ValueError(
                f"An RBM at the top cannot have directed connections. Cause: self.top = True"
            )

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)
            support = self.bias_v + hidden_minibatch @ self.weight_h_to_v
            probs = sigmoid(support)
            activ = sample_binary(probs)

        return probs, activ

    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):
        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
