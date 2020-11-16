import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from PIL import Image
import os


class VariationalAutoencoder(tf.keras.Model):

    def __init__(self, input_shape, filters, latent_dim, size):
        """ Creates an instance of the model.

        Args:
            input_shape (tuple): Input and output shape of the model.
            filters (list): The convolution filters.
            latent_dim (int): Size of the latent space.
            size (str): Size of the model.
        """

        super().__init__()

        assert size in ["tiny", "small", "big"]

        # Save some parameters.
        self.filters = []
        self.latent_dim = latent_dim
        self.size = size

        # Shape for bridging dense and convolutional layers in the decoder.
        bridge_shape = (input_shape[0] // 2**len(filters), input_shape[1] // 2**len(filters), filters[-1])

        # Create encoder and decoder.
        if self.size == "tiny":
           
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
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
                tf.keras.layers.Dense(latent_dim + latent_dim),
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
                tf.keras.layers.Dense(latent_dim + latent_dim),
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
        
        # Should not happen.
        else:
            assert False, self.size


    def call(self, x):
        """ Calls the model on some input.

        Args:
            x (ndarray or tensor): A sample.

        Returns:
            ndarray or tensor: The result.
        """

        # Encode. Compute mean and variance.
        mean, logvar = self.encode(x)

        # Get latent vector.
        z = self.reparameterize(mean, logvar)

        # Decode.
        y = self.decode(z, apply_sigmoid=True)

        return y

    
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

    
    def train(self, dataset_train, dataset_validate, dataset_anomaly, epochs, batch_size, shuffle_buffer_size, render=False, render_every=1):
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

        Returns:
            dict: History dictionary.
        """
        print("Starting training...")

        # Create optimizer.
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # Create history object.
        history = { key: [] for key in ["loss_train", "loss_validate", "loss_anomaly"]}
        best_validation_loss = 1000000.0

        # Pick some samples from each set.
        def pick_samples(dataset, number):
            for batch in dataset.batch(number).take(1):
                return  batch[0:number]
        dataset_train_samples = pick_samples(dataset_train, 100)
        dataset_validate_samples = pick_samples(dataset_validate, 100)
        dataset_anomaly_samples = pick_samples(dataset_anomaly, 100)
        
        # Prepare datasets for training.
        dataset_train = dataset_train.cache().prefetch(tf.data.experimental.AUTOTUNE).shuffle(shuffle_buffer_size).batch(batch_size)
        dataset_validate = dataset_validate.cache().prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)
        dataset_anomaly = dataset_anomaly.cache().prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)

        # Render reconstructions and individual losses before training.
        if render:
            render_reconstructions(self, dataset_train_samples, dataset_validate_samples,  dataset_anomaly_samples, filename=f"reconstruction-0000.png")
            render_individual_losses(self, dataset_train_samples, dataset_validate_samples,  dataset_anomaly_samples, filename=f"losses-0000.png")

        # Train.
        for epoch in range(1, epochs + 1):

            start_time = time.time()

            # Train with training set and compute mean loss.
            loss = tf.keras.metrics.Mean()
            for train_x in dataset_train:
                loss(train_step(self, train_x, optimizer))
            loss_train = loss.result()

            # Compute loss for validate and anomaly.
            loss_validate = compute_mean_loss(self, dataset_validate)
            loss_anomaly = compute_mean_loss(self, dataset_anomaly)

            # Update the history.            
            history["loss_train"].append(loss_train)
            history["loss_validate"].append(loss_validate)
            history["loss_anomaly"].append(loss_anomaly)
            
            # Save the best model.
            if loss_validate < best_validation_loss:
                print(f"Found new best model with validation loss {loss_validate}.")
                self.save_weights("model_best")
                best_validation_loss = loss_validate

            end_time = time.time()

            print('Epoch: {}, validate set loss: {}, time elapse for current epoch: {}'
                    .format(epoch, loss_validate, end_time - start_time))

            # Render reconstructions after every xth epoch.
            if render and (epoch % render_every) == 0:
                render_reconstructions(self, dataset_train_samples, dataset_validate_samples,  dataset_anomaly_samples, filename=f"reconstruction-{epoch:04d}.png")
                render_individual_losses(self, dataset_train_samples, dataset_validate_samples,  dataset_anomaly_samples, filename=f"losses-{epoch:04d}.png")
        
        # Merge reconstructions into an animation.
        if render:
            create_animation("reconstruction-*", "reconstruction-animation.gif", delete_originals=True)
            create_animation("losses-*", "losses-animation.gif", delete_originals=True)

        # Render the history.
        render_history(history, "history.png")

        # Done.
        return history


    def save_weights(self, name):
        """Saves the weights of the encoder and the decoder.

        Args:
            name (str): Name of the files.
        """
        self.encoder.save_weights(name + "_encoder_weights.h5")
        self.decoder.save_weights(name + "_decoder_weights.h5")


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def compute_mean_loss(model, dataset):
    """Computes the mean loss.

    Args:
        model (model): A model.
        dataset (dataset): A dataset.

    Returns:
        float: The mean loss.
    """
    loss = tf.keras.metrics.Mean()
    for validate_x in dataset:
        loss(compute_loss(model, validate_x))
    #elbo = -loss.result()
    return loss.result()


def compute_individual_losses(model, dataset):
    """Computes the individual losses of samples in a dataset.

    Args:
        model (model): A model.
        dataset (dataset): A dataset.

    Returns:
        list: A list of losses.
    """
    return [compute_loss(model, np.array([x])) for x in dataset]


def compute_loss(model, x):
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
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def render_reconstructions(model, samples_train, samples_validate, samples_anomaly, filename, steps=10):
    """Renders reconstructions of training set, validation set, and anomaly set.

    Args:
        model (model): A model.
        samples_train (ndarray): Some training samples.
        samples_validate (ndarray): Some validation samples.
        samples_anomaly (ndarray): Some anomaly samples.
        filename (str): Filename where to store the image.
        steps (int, optional): How many samples to reconstruct. Defaults to 10.
    """

    # Reconstruct all samples.
    reconstructions_train = model.predict(samples_train[:steps], steps=steps)
    reconstructions_validate = model.predict(samples_validate[:steps], steps=steps)
    reconstructions_anomaly = model.predict(samples_anomaly[:steps], steps=steps)
    
    # This will be the result image.
    image = np.zeros((6 * samples_train.shape[1], steps  * samples_train.shape[1], 3))
    
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
    image.save(filename)


def render_individual_losses(model, samples_train, samples_validate, samples_anomaly, filename):
    """Render the individual losses as a single histogram.

    Args:
        model (model): A model.
        samples_train (ndarray): Some training samples.
        samples_validate (ndarray): Some validation samples.
        samples_anomaly (ndarray): Some anomaly samples.
        filename (str): Filename of the image.
    """
    losses_train = compute_individual_losses(model, samples_train)
    losses_validate = compute_individual_losses(model, samples_validate)
    losses_anomaly = compute_individual_losses(model, samples_anomaly)

    alpha = 0.5
    bins = 20
    plt.hist(losses_train, label="losses_train", alpha=alpha, bins=bins)
    plt.hist(losses_validate, label="losses_validate", alpha=alpha, bins=bins)
    plt.hist(losses_anomaly, label="losses_anomaly", alpha=alpha, bins=bins)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def create_animation(glob_search_path, filename, delete_originals=False):
    """Finds some images and merges them as a GIF.

    Args:
        glob_search_path (str): Glob search path to find the images.
        filename (str): Filename of the animation.
        delete_originals (bool, optional): If the originals should be erased. Defaults to False.
    """
    with imageio.get_writer(filename, mode="I") as writer:
        paths = glob.glob(glob_search_path)
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


def render_history(history, filename):
    """Renders the training history.

    Args:
        history (dict): History dictionary.
        filename (str): Filename of the image.
    """
    for key, value in history.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(filename)
    plt.close()