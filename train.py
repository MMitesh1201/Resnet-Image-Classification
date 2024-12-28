import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

# Create model with best practices
def create_model():
    # Initialize ResNet50V2 with imagenet weights
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(64, 64, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create model
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    # Add preprocessing layer to handle the size difference
    x = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(inputs)
    
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


import wandb
from wandb.integration.keras import WandbCallback
wandb.login()

wandb.init(project="Facerecofnition")

def train_model():
    x_train, y_train, x_test, y_test = prepare_cifar10()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and compile model
    model, base_model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        WandbCallback()
    ]
    
    # Phase 1: Train only the top layers
    print("Phase 1: Training top layers...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=16),
        validation_data=(x_test, y_test),
        epochs=20,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune the last few layers of ResNet
    print("Phase 2: Fine-tuning ResNet layers...")
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    model.fit(
        datagen.flow(x_train, y_train, batch_size=16),
        validation_data=(x_test, y_test),
        epochs=30,
        callbacks=callbacks
    )
    

    # Save the model
    model.save('resnet50v2_cifar10.h5')
    
    # Finish wandb run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    # Login to wandb first
    wandb.login()
    
    # Then train the model
    model = train_model()
