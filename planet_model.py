import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_planet():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(38, activation='softmax')  # 38 classes for PlantVillage dataset
    model = Model(inputs, outputs, name='PlaNet')
    return model

# Compile the model
planet_model = build_planet()
planet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=30,  # Random rotation up to 30 degrees
    width_shift_range=0.2,  # Randomly shift image width by 20%
    height_shift_range=0.2,  # Randomly shift image height by 20%
    shear_range=0.2,  # Apply shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flipping
    fill_mode='nearest'  # Fill in missing pixels with nearest values
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    'dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'dataset/test', target_size=(224, 224), batch_size=32, class_mode='categorical')


planet_model.fit(train_generator, validation_data=test_generator, epochs=20)


planet_model.save('planet_model.h5')


test_loss, test_acc = planet_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
