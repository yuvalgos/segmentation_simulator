import wandb
import random
import keras
from keras import layers, models


# Define the model
def build_pose_estimation_model(num_poses):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output for Dense layers
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_poses, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="segmentation_probs",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",  # "data/mug_segment.png",
            "epochs": 10,
        }
    )

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # Example for model usage
    num_poses = 5  # Change this to the number of poses you have
    model = build_pose_estimation_model(num_poses)

    model.fit(train_data, train_labels, epochs=10)
    predictions = model.predict(new_image)
    # The 'predictions' variable now contains the probability distribution over the poses.

