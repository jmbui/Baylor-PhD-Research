import tensorflow as tf
from tensorflow.keras import activations, datasets, layers, models, utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def image_rmse(image1, image2):
    """
    This function calculates the RMSE value between two images.

    :param image1: First image in the comparison
    :param image2: Second image in the comparison
    :return: Root Mean Squared Error
    """
    error = np.sqrt(np.sum((image1.astype('float'))
                           - image2.astype('float')) ** 2)
    return error


def load_training_dataset():
    """
    This function loads the training portion of the MNIST dataset. The training dataset
    is further divided in to training and validation images.

    :return: MNIST training and validation data
    """
    # Load the MNIST training subset
    (x_train, y_train), (_, _) = datasets.mnist.load_data()

    x_train = x_train / 255.0  # Normalize the images
    y_train = utils.to_categorical(y_train)  # One-hot encode labels

    # Split between training and validation data
    x_val = x_train[-10000:]
    x_train = x_train[:-10000]
    y_val = y_train[-10000:]
    y_train = y_train[:-10000]

    return x_train, x_val, y_train, y_val


def load_evaluation_datasets():
    """
    This function creates an in-distribution, out-of-distribution,
    and a mixed distribution dataset. These are used to evaluate
    the watchdog functionality.

    :return: The in-distribution, out-of-distribution and mixed
    distribution evaluation sets.
    """
    # Load the MNIST and Fashion MNIST test subsets
    (_, _), (in_imgs, in_labs) = datasets.mnist.load_data()
    (_, _), (out_imgs, out_labs) = datasets.fashion_mnist.load_data()

    # Normalize Images
    in_imgs = in_imgs / 255.0
    out_imgs = out_imgs / 255.0

    # One-hot encode the labels
    in_labs = utils.to_categorical(in_labs)
    out_labs = utils.to_categorical(out_labs)

    # create the mixed distribution_dataset
    mixed_imgs = np.append(in_imgs, out_imgs, axis=0)
    mixed_labs = np.append(in_labs, out_labs, axis=0)

    return (in_imgs, in_labs), (out_imgs, out_labs), (mixed_imgs, mixed_labs)


def build_classifier(input_shape, filters, ks, dropout, num_labels):
    """
    This function builds a CNN using TensorFlow's Sequential API.

    :param input_shape: Input shape for the classifier
    :param filters: Number of filters in the 1st convolutional layer
    :param ks: Kernel size of the convolutional layers
    :param dropout: Dropout
    :param num_labels: Number of labels in the classifier output
    :return: The CNN based classifier model
    """

    classifier = models.Sequential()
    classifier.add(layers.Conv2D(input_shape=input_shape,
                                 filters=filters,
                                 kernel_size=ks,
                                 activation=activations.relu()
                                 ))
    classifier.add(layers.MaxPooling2D((2, 2)))

    classifier.add(layers.Conv2D(filters=filters * 2,
                                 kernel_size=ks,
                                 activation=activations.relu()
                                 ))
    classifier.add(layers.MaxPooling2D((2, 2)))

    classifier.add(layers.Conv2D(filters=filters * 4,
                                 kernel_size=ks,
                                 activation=activations.relu()
                                 ))
    classifier.add(layers.MaxPooling2D((2, 2)))

    classifier.add(layers.Flatten)
    classifier.add(layers.Dropout(dropout))
    classifier.add(layers.Dense(num_labels,
                                activation=activations.softmax()
                                ))

    return classifier


def build_autoencoder(input_shape, layer_filters, ks, latent_dim):
    """
    This function builds an autoencoder which is used to reconstruct the input data.

    :param input_shape: Shape of the input data
    :param layer_filters: Array containing number of filters
    :param ks: Kernel size for the convolutions
    :param latent_dim: Dimension of the latent space
    :return:
    """

    #  Define the encoder layers
    x = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    for filters in layer_filters:
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=ks,
                                   activation='relu', strides=2,
                                   padding='same')(x)
    shape = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(latent_dim, name='latent_vector')(x)

    encoder = tf.keras.Model(input, latent, name='encoder')

    # Build the Decoder
    latent_input = tf.keras.layers.Input(shape=(latent_dim,),
                                         name='decoder_input')

    x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3])(latent_input)
    x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

    for filters in layer_filters[::-1]:
        x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                            kernel_size=ks,
                                            activation='relu', strides=2,
                                            padding='same')(x)

    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=ks,
                                              activation='sigmoid', padding='same',
                                              name='decoder_output')(x)

    decoder = tf.keras.Model(latent_input, outputs, name='decoder')

    autoencoder = tf.keras.Model(encoder, decoder, name='autoencoder')

    return autoencoder


def train_network(network, epochs, batch_size, train_x, train_y, early_stopping):
    """
    This function trains a neural network based on the provided parameters.

    :param network: The network to be trained
    :param epochs: The number of epochs to be trained
    :param batch_size: The batch size for training
    :param train_x: The input data
    :param train_y: The output data
    :param early_stopping: Whether to implement early_stopping criteria
    :return: None
    """

    if early_stopping:
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    else:
        callbacks = None

    network.fit(train_x,
                train_y,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[callbacks],
                )


def apply_watchdog(autoencoder_model, dataset, threshold=1.0):
    """
    This function applies an autoencoder watchdog to the provided dataset with a configurable threshold.
    :param autoencoder_model: The pre-trained autoencoder used to regenerate input data.
    :param dataset: The dataset used in coordination with the applied watchdog (images, labels).
    :param threshold: The threshold value for the watchdog.
    :return: Returns a dataset containing the values permitted by the watchdog (images, labels).
    """

    images, labels = dataset  # split out images and labels
    guarded_images, guarded_labels = [], []  # Create empty dataset arrays

    regenerated_images = autoencoder_model.predict(images)

    for i in range(len(images)):
        real = np.squeeze(images[i])
        pred = np.squeeze(regenerated_images[i])
        if image_rmse(real, pred) < threshold:
            guarded_images.append(images[i])
            guarded_labels.append(labels[i])

    return np.array(guarded_images), np.array(guarded_labels)


def evaluate_network(network, dataset):
    """
    This function evaluates a network using the supplied dataset. This function produces accuracy and loss values, as
    well as produce the ROC curves for each class.
    :param network: The network being evaluated.
    :param dataset: The dataset used for network evaluation.
    :return: Accuracy Percentage and Loss Value
    """

    fpr = dict()
    tpr = dict()
    images, labels = dataset
    label_predictions = network.predict(images)
    [loss, accuracy] = network.evaluate(images, labels)

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], label_predictions[:, i])
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), label_predictions.ravel())

    plt.figure()
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label='Class: %.0f' % i)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='Aggregate Performance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
