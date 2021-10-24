import Watchdog

# Load the datasets
train_x, train_y = Watchdog.load_training_dataset()
in_dist_img, in_dist_lab, _, _, mix_dist_img, mix_dist_lab = Watchdog.load_evaluation_datasets()

# Define the network and training hyperparameters
IMG_SHAPE = train_x.shape[1]
KERNEL = 2
LATENT_DIM = 32
FILTERS = 16
DROPOUT = 0.15
NUM_LABELS = 10
LAYER_FILTERS = [16, 32, 64]
EPOCHS = 20
BATCH_SIZE = 64
E_STOP = True

# Define the networks using the hyperparameters above
classifier = Watchdog.build_classifier(input_shape=IMG_SHAPE,
                                       filters=FILTERS,
                                       ks=KERNEL,
                                       dropout=DROPOUT,
                                       num_labels=NUM_LABELS
                                       )

ae = Watchdog.build_autoencoder(input_shape=IMG_SHAPE,
                                layer_filters=LAYER_FILTERS,
                                ks=KERNEL,
                                latent_dim=LATENT_DIM
                                )

# Train the networks
Watchdog.train_network(network=classifier,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       train_x=train_x,
                       train_y=train_y,
                       early_stopping=E_STOP
                       )

Watchdog.train_network(network=ae,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       train_x=train_x,
                       train_y=train_y,
                       early_stopping=E_STOP
                       )

(images, labels) = Watchdog.apply_watchdog(ae, (mix_dist_img, mix_dist_lab), threshold=4.5)
(loss, accuracy) = Watchdog.evaluate_network(classifier, (images, labels))
print("Evaluated Loss: ", loss)
print("Evaluated Accuracy: ", accuracy)
