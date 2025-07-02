import tensorflow as tf
import keras

class AdversarialAttack():
    def __init__(self, model, epsilon=0.1, iterations=10):
        """
        Initialize the adversarial attack class with the model and its components.

        :param model: The overall model that includes encoder, bridge, and decoder.
        :param epsilon: The limit for the perturbation magnitude.
        :param iterations: The number of iterations for the attack.
        """
        self.model = model
        self._epsilon = epsilon
        self._iterations = iterations
        self.gradient_history = []  # Store gradient history for plotting

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def iterations(self):
        return self._iterations

    def run_output(self, uk, x0, yk_true):
        """
        Returns the crafted adversarial input to deceive the output of the
        autoencoder.

        Parameters:
        - uk: The original input to the network f.
        - x0: Latent representation of the current state.
        - yk_true: True output of the real dynamical system.

        Returns:
        - uk_adv: Crafted adversarial input.
        """
        uk = tf.convert_to_tensor(uk, dtype=tf.float32)
        yk_true = tf.convert_to_tensor(yk_true, dtype=tf.float32)
        yk_true = tf.squeeze(yk_true)
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)

        uk_adv = uk

        for _ in range(self._iterations):
            with tf.GradientTape() as tape:
                tape.watch(uk_adv)
                next_state_pred = self.model.bridge_network([uk_adv, x0])[0]
                yk_pred = self.model.decoder_network([next_state_pred])[0]
                yk_pred = yk_pred[0][-2]
                yk_pred = tf.expand_dims(yk_pred, axis=0)
                yk_true = tf.expand_dims(yk_true, axis=0)

                loss = keras.losses.mean_absolute_error(yk_pred, yk_true)

            # Calculate gradients of loss w.r.t. adversarial_u
            gradients = tape.gradient(loss, uk_adv)

            # Update adversarial_u by moving in the direction that maximizes the loss
            uk_adv = uk_adv + self.epsilon * tf.sign(gradients)
            #print(f"Loss: {loss}, Gradient: {gradients}, uk_adv: {uk_adv}")

        return uk_adv, yk_pred