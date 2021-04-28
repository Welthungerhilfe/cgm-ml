import glob
import itertools
import os
import time
import logging
import logging.config

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from PIL import Image
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


class Autoencoder(tf.keras.Model):

    def __init__(self, family, input_shape, filters, latent_dim, size):
        """ Creates an instance of the model.

        Args:
            family (string): Autoencoder family. Either "ae" or "vae".
            input_shape (tuple): Input and output shape of the model.
            filters (list): The convolution filters.
            latent_dim (int): Size of the latent space.
            size (str): Size of the model.
        """
        super().__init__()

        assert family in ["ae", "vae"], family
        assert size in ["tiny", "small", "big", "huge"]

        # Save some parameters.
        self.family = family
        self.filters = []
        self.latent_dim = latent_dim
        self.size = size

        # Shape for bridging dense and convolutional layers in the decoder.
        bridge_shape = (input_shape[0] // 2**len(filters), input_shape[1] // 2**len(filters), filters[-1])

        # Create encoder and decoder.
        if self.family == "ae":
            output_size = latent_dim
        elif self.family == "vae":
            output_size = 2 * latent_dim

        if self.size == "tiny":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(output_size),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        elif self.size == "small":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(output_size),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        elif self.size == "big":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[6], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(output_size),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        elif self.size == "huge":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[6], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[7], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[8], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(output_size),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[7], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[6], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        # Should not happen.
        else:
            assert False, self.size

        # Prepare losses.
        self.loss_names = []
        if self.family == "ae":
            self.loss_names = ["reconstruction"]
        elif self.family == "vae":
            self.loss_names = ["total", "reconstruction", "divergence"]
        self.number_of_losses = len(self.loss_names)

    def call(self, x):
        """ Calls the model on some input.

        Args:
            x (ndarray or tensor): A sample.

        Returns:
            ndarray or tensor: The result.
        """

        # Autoencoder.
        if self.family == "ae":
            # Encode. Compute z.
            z = self.encode(x)

            # Decode.
            y = self.decode(z, apply_sigmoid=True)

            return y

        # Variational autoencoder.
        elif self.family == "vae":

            # Encode. Compute mean and variance.
            mean, logvar = self.encode(x)

            # Get latent vector.
            z = self.reparameterize(mean, logvar)

            # Decode.
            y = self.decode(z, apply_sigmoid=True)

            return y

    def embed(self, x):
        """ Embeds samples into latent space.

        Args:
            x (ndarray or tensor): A sample.

        Returns:
            ndarray or tensor: The result.
        """

        # Autoencoder.
        if self.family == "ae":
            # Encode. Compute z.
            z = self.encode(x)

            return z

        # Variational autoencoder.
        elif self.family == "vae":

            # Encode. Compute mean and variance.
            mean, logvar = self.encode(x)

            # Get latent vector.
            z = self.reparameterize(mean, logvar)

            return z

    @tf.function
    def sample(self, eps=None):
        """Decodes some samples from latent-space.

        Args:
            eps (ndarray or tensor, optional): Latent vectors. Defaults to None.

        Returns:
            ndarray or tensor: The samples decoded from latent space.
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        """Encodes some samples into latent space.

        Args:
            x (ndarray or tensor): Some samples to encode.

        Returns:
            ndarray or tensor: Mean and logvar of the samples.
        """

        # Autoencoder.
        if self.family == "ae":
            z = self.encoder(x)
            return z

        # Variational autoencoder.
        elif self.family == "vae":
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar

    def reparameterize(self, mean, logvar):
        """Reparametrization trick. Computes mean and logvar and
        then samples some latent vectors from that distribution.

        Args:
            mean (ndarray or tensor): Mean.
            logvar (ndarray or tensor): Logvar.

        Returns:
            ndarray or tensor: Latent space vectors.
        """

        # Only in VAE.
        assert self.family == "vae"

        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """Decodes some latent vectors.

        Args:
            z (ndarray or tensor): Some latent vectors.
            apply_sigmoid (bool, optional): Determines if sigmoid should be applied in the end.. Defaults to False.

        Returns:
            ndarray or tensor: Samples.
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def train(self, dataset_train, dataset_validate, dataset_anomaly, epochs, batch_size, shuffle_buffer_size, render=False, render_every=1, callbacks=[], outputs_path="."):
        """Trains the model.

        Args:
            dataset_train (dataset): The training set.
            dataset_validate (dataset): The validation set.
            dataset_anomaly (dataset): The anomaly set.
            epochs (int): Epochs to train.
            batch_size (int): Batch size.
            shuffle_buffer_size (int): Size of the shuffle buffer.
            render (bool, optional): Triggers rendering of statistics. Defaults to False.
            render_every (int, optional): How often to render statistics. Defaults to 1.
            callbacks (list, optional): Callbacks to be called back.
            outputs_path (string, optional): Path where outputs should be written. Default ".".

        Returns:
            dict: History dictionary.
        """
        print("Starting training...")

        # Create optimizer.
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # Create history object.
        dataset_names = ["train", "validate", "anomaly"]
        keys = [f"{loss_name}_{dataset_name}" for dataset_name, loss_name in itertools.product(dataset_names, self.loss_names)]
        history = {key: [] for key in keys}
        best_validation_loss = 1000000.0
        del dataset_names
        del keys

        # Pick some samples from each set.
        logging.info("Picking some samples...")

        def pick_samples(dataset, number):
            for batch in dataset.batch(number).take(1):
                return batch[0:number]
        dataset_train_samples = pick_samples(dataset_train, 100)
        dataset_validate_samples = pick_samples(dataset_validate, 100)
        dataset_anomaly_samples = pick_samples(dataset_anomaly, 100)

        # Prepare datasets for training.
        # TODO Do we need prefetch?
        logging.info("Preparing datasets...")
        dataset_train = dataset_train.cache()
        #dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_train = dataset_train.shuffle(shuffle_buffer_size)
        dataset_train = dataset_train.batch(batch_size)
        dataset_validate = dataset_validate.cache()
        #dataset_validate = dataset_validate.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_validate = dataset_validate.batch(batch_size)
        dataset_anomaly = dataset_anomaly.cache()
        #dataset_anomaly = dataset_anomaly.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_anomaly = dataset_anomaly.batch(batch_size)

        # Render reconstructions and individual losses before training.
        if render:
            render_reconstructions(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename="reconstruction-0000.png")
            #render_individual_losses(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename="losses-0000.png")
            render_embeddings(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename="embeddings-0000.png")

        # Train.
        logging.info("Train...")
        for epoch in range(1, epochs + 1):

            start_time = time.time()

            # Train with training set and compute mean losses.
            mean_losses_train = [tf.keras.metrics.Mean() for _ in range(self.number_of_losses)]
            batch_index = 1
            for train_x in dataset_train:
                logging.info("Batch %d", batch_index)
                batch_index += 1
                losses = train_step(self, train_x, optimizer)
                assert len(losses) == self.number_of_losses
                for loss, mean_loss in zip(losses, mean_losses_train):
                    mean_loss(loss)
            mean_losses_train = [mean_loss.result() for mean_loss in mean_losses_train]

            # Compute loss for validate and anomaly.
            mean_losses_validate = compute_mean_losses(self, dataset_validate)
            mean_losses_anomaly = compute_mean_losses(self, dataset_anomaly)

            # Convert mean losses to float.
            mean_losses_train = [float(mean_loss) for mean_loss in mean_losses_train]
            mean_losses_validate = [float(mean_loss) for mean_loss in mean_losses_validate]
            mean_losses_anomaly = [float(mean_loss) for mean_loss in mean_losses_anomaly]

            # Save the best model.
            if mean_losses_validate[0] < best_validation_loss:
                logging.info('Found new best model with validation loss %d.', mean_losses_validate[0])
                self.save_weights(outputs_path=outputs_path, filename="model_best")
                best_validation_loss = mean_losses_validate[0]

            end_time = time.time()

            # Update the history.
            logs = {}
            for loss_name, mean_loss in zip(self.loss_names, mean_losses_train):
                logs[loss_name + "_train"] = mean_loss
            for loss_name, mean_loss in zip(self.loss_names, mean_losses_validate):
                logs[loss_name + "_validate"] = mean_loss
            for loss_name, mean_loss in zip(self.loss_names, mean_losses_anomaly):
                logs[loss_name + "_anomaly"] = mean_loss
            for loss_key, loss_value in logs.items():
                history[loss_key] += [loss_value]

            # Call back the callbacks.
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

            # Print status.
            logging.info('Epoch: %d, validate set loss: %d, time elapse for current epoch: %d', epoch, mean_losses_validate[0], end_time - start_time)
            # Render reconstructions after every xth epoch.
            if render and (epoch % render_every) == 0:
                render_reconstructions(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"reconstruction-{epoch:04d}.png")
                #render_individual_losses(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"losses-{epoch:04d}.png")
                render_embeddings(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"embeddings-{epoch:04d}.png")

        # Merge reconstructions into an animation.
        if render:
            render_reconstructions(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"reconstruction-{epoch:04d}.png")
            #render_individual_losses(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"losses-{epoch:04d}.png")
            render_embeddings(self, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path=outputs_path, filename=f"embeddings-{epoch:04d}.png")
            create_animation("reconstruction-*", outputs_path=outputs_path, filename="reconstruction-animation.gif", delete_originals=True)
            #create_animation("losses-*", outputs_path=outputs_path, filename="losses-animation.gif", delete_originals=True)
            create_animation("embeddings-*", outputs_path=outputs_path, filename="embeddings-animation.gif", delete_originals=True)

        # Render the history.
        render_history(history, self.loss_names, outputs_path=outputs_path, filename="history.png")

        # Done.
        return history

    def save_weights(self, outputs_path, filename):
        """Saves the weights of the encoder and the decoder.

        Args:
            name (str): Name of the files.
        """
        self.encoder.save_weights(os.path.join(outputs_path, filename + "_encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(outputs_path, filename + "_decoder_weights.h5"))


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )


@tf.function
def train_step(model, x, optimizer):
    # Get all losses.
    with tf.GradientTape() as tape:
        if model.family == "ae":
            losses = compute_losses_ae(model, x)
        elif model.family == "vae":
            losses = compute_losses_vae(model, x)

    # Get gradient for the total loss and optimize.
    gradients = tape.gradient(losses[0], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Done.
    return losses


def compute_mean_losses(model, dataset):
    """Computes the mean losses.

    Args:
        model (model): A model.
        dataset (dataset): A dataset.

    Returns:
        float: The mean loss.
    """
    mean_losses = [tf.keras.metrics.Mean() for _ in range(model.number_of_losses)]

    # Go through the dataset.
    for validate_x in dataset:
        if model.family == "ae":
            losses = compute_losses_ae(model, validate_x)
        if model.family == "vae":
            losses = compute_losses_vae(model, validate_x)
        for loss, mean_loss in zip(losses, mean_losses):
            mean_loss(loss)

    # Get the losses as numbers.
    mean_losses = [mean_loss.result() for mean_loss in mean_losses]

    # Done.
    return mean_losses


def compute_individual_losses(model, dataset):
    """Computes the individual losses of samples in a dataset.

    Args:
        model (model): A model.
        dataset (dataset): A dataset.

    Returns:
        list: A list of losses.
    """

    # Set up the losses.
    all_losses = [[] for _ in range(model.number_of_losses)]

    # Go through each sample.
    for x in dataset:
        if model.family == "ae":
            losses = compute_losses_ae(model, np.array([x]))
        elif model.family == "vae":
            losses = compute_losses_vae(model, np.array([x]))
        for loss, loss_list in zip(losses, all_losses):
            loss_list += [float(loss)]

    # Done.
    return all_losses


def compute_losses_ae(model, x):
    """Computes the loss of a sample.

    Args:
        model (model): A model.
        x (ndarray or tensor): A sample.

    Returns:
        float: The loss of the sample.
    """
    z = model.encode(x)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    reconstruction_loss = logpx_z
    total_loss = -tf.reduce_mean(reconstruction_loss)

    return [total_loss]


def compute_losses_vae(model, x):
    """Computes the loss of a sample.

    Args:
        model (model): A model.
        x (ndarray or tensor): A sample.

    Returns:
        float: The loss of the sample.
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)

    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    reconstruction_loss = logpx_z
    divergence_loss = logpz - logqz_x
    total_loss = -tf.reduce_mean(reconstruction_loss + divergence_loss)

    return total_loss, -reconstruction_loss, -divergence_loss


def render_reconstructions(model, samples_train, samples_validate, samples_anomaly, outputs_path, filename, steps=10):
    """Renders reconstructions of training set, validation set, and anomaly set.

    Args:
        model (model): A model.
        samples_train (ndarray): Some training samples.
        samples_validate (ndarray): Some validation samples.
        samples_anomaly (ndarray): Some anomaly samples.
        filename (str): Filename where to store the image.
        steps (int, optional): How many samples to reconstruct. Defaults to 10.
    """

    logging.info("Rendering reconstructions...")

    # Reconstruct all samples.
    reconstructions_train = model.predict(samples_train[:steps], steps=steps)
    reconstructions_validate = model.predict(samples_validate[:steps], steps=steps)
    reconstructions_anomaly = model.predict(samples_anomaly[:steps], steps=steps)

    # This will be the result image.
    image = np.zeros((6 * samples_train.shape[1], steps * samples_train.shape[1], 3))

    # Render all samples and their reconstructions.
    def render(samples, reconstructions, offset):
        for sample_index, (sample, reconstruction) in enumerate(zip(samples, reconstructions)):
            s1 = (offset + 0) * sample.shape[1]
            e1 = (offset + 1) * sample.shape[1]
            s2 = sample_index * sample.shape[0]
            e2 = (sample_index + 1) * sample.shape[0]
            image[s1:e1, s2:e2] = sample
            s1 = (offset + 1) * sample.shape[1]
            e1 = (offset + 2) * sample.shape[1]
            s2 = sample_index * sample.shape[0]
            e2 = (sample_index + 1) * sample.shape[0]
            image[s1:e1, s2:e2] = reconstruction
    render(samples_train, reconstructions_train, 0)
    render(samples_validate, reconstructions_validate, 2)
    render(samples_anomaly, reconstructions_anomaly, 4)

    # Convert and save the image.
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(os.path.join(outputs_path, filename))


def render_individual_losses(model, samples_train, samples_validate, samples_anomaly, outputs_path, filename):
    """Render the individual losses as a single histogram.

    Args:
        model (model): A model.
        samples_train (ndarray): Some training samples.
        samples_validate (ndarray): Some validation samples.
        samples_anomaly (ndarray): Some anomaly samples.
        filename (str): Filename of the image.
    """

    logging.info("Rendering individual losses...")

    losses_train = compute_individual_losses(model, samples_train)
    losses_validate = compute_individual_losses(model, samples_validate)
    losses_anomaly = compute_individual_losses(model, samples_anomaly)

    alpha = 0.5
    bins = 20
    plt.hist(losses_train, label="losses_train", alpha=alpha, bins=bins)
    plt.hist(losses_validate, label="losses_validate", alpha=alpha, bins=bins)
    plt.hist(losses_anomaly, label="losses_anomaly", alpha=alpha, bins=bins)
    plt.legend()
    plt.savefig(os.path.join(outputs_path, filename))
    plt.close()


def create_animation(glob_search_path, outputs_path, filename, delete_originals=False):
    """Finds some images and merges them as a GIF.

    Args:
        glob_search_path (str): Glob search path to find the images.
        filename (str): Filename of the animation.
        delete_originals (bool, optional): If the originals should be erased. Defaults to False.
    """
    with imageio.get_writer(os.path.join(outputs_path, filename), mode="I") as writer:
        paths = glob.glob(os.path.join(outputs_path, glob_search_path))
        paths = [path for path in paths if path.endswith(".png")]
        paths = sorted(paths)
        for path in paths:
            image = imageio.imread(path)
            writer.append_data(image)
        image = imageio.imread(paths[-1])
        writer.append_data(image)
        if delete_originals:
            for path in paths:
                os.remove(path)


def render_history(history, loss_names, outputs_path, filename):
    """Renders the training history.

    Args:
        history (dict): History dictionary.
        filename (str): Filename of the image.
    """

    logging.info("Rendering history...")

    _, axes = plt.subplots(len(loss_names), figsize=(8, 12))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for loss_name, axis in zip(loss_names, axes):
        for key, value in history.items():
            if key.startswith(loss_name):
                axis.plot(value, label=key)
        axis.legend()
    plt.savefig(os.path.join(outputs_path, filename))
    plt.close()


def render_embeddings(model, dataset_train_samples, dataset_validate_samples, dataset_anomaly_samples, outputs_path, filename):
    """Renders the embeddings.

    Args:
        model (model): A model to embed the samples.
        dataset_train_samples (list or ndarray): Training samples.
        dataset_validate_samples (list or ndarray): Validation samples.
        dataset_anomaly_samples (list or ndarray): Anomaly samples.
        outputs_path (string): Path where to store the rendering.
        filename (string): Filename of the rendering.
    """

    logging.info("Rendering embeddings...")

    # Embed the samples.
    embeddings_train_samples = model.embed(dataset_train_samples)
    embeddings_validate_samples = model.embed(dataset_validate_samples)
    embeddings_anomaly_samples = model.embed(dataset_anomaly_samples)

    # Apply tsne.
    all_latent_points = []
    all_latent_points += list(embeddings_train_samples)
    all_latent_points += list(embeddings_validate_samples)
    all_latent_points += list(embeddings_anomaly_samples)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(all_latent_points)

    # Class names.
    names = ["train", "validate", "anomaly"]

    # Get the base colors.
    cmap = cm.get_cmap("inferno")
    color_lookup = cmap(np.linspace(0, 1, 1 + len(set(names))))
    color_lookup = [rgb2hex(rgb) for rgb in color_lookup]

    # Get the colors for the samples.
    colors = []
    colors += [color_lookup[0] for _ in embeddings_train_samples]
    colors += [color_lookup[1] for _ in embeddings_validate_samples]
    colors += [color_lookup[2] for _ in embeddings_anomaly_samples]

    # Get the sizes.
    sizes = []
    sizes += [20 for _ in embeddings_train_samples]
    sizes += [20 for _ in embeddings_validate_samples]
    sizes += [20 for _ in embeddings_anomaly_samples]

    # Render the TSNE results.
    plt.figure(figsize=(12, 12))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=sizes, c=colors)

    # Render the legend
    legend_elements = []
    for name, color in zip(set(names), color_lookup):
        legend_elements.append(Line2D([0], [0], marker='o', color=color, label=name, markerfacecolor=color, markersize=15))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the figure.
    plt.savefig(os.path.join(outputs_path, filename))
    plt.close()
