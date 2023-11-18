import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential

data_dir = 'C:/Users/Stefan/Documents/GTSRB'
train_path = 'C:/Users/Stefan/Documents/GTSRB/Train'
test_path = 'C:/Users/Stefan/Documents/GTSRB'
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
NUM_CATEGORIES = len(os.listdir(train_path))

# Label Overview
classes = { 0: 'Limita de viteza (20km/h)',
            1: 'Limita de viteza (30km/h)',
            2: 'Limita de viteza (50km/h)',
            3: 'Limita de viteza (60km/h)',
            4: 'Limita de viteza (70km/h)',
            5: 'Limita de viteza (80km/h)',
            6: 'Sfarsitul limitei de viteza (80km/h)',
            7: 'Limita de viteza (100km/h)',
            8: 'Limita de viteza (120km/h)',
            9: 'Nu se depaseste',
            10: 'Vehiculele peste 3.5 tone nu au voie',
            11: 'Prioritate dreapta la intersectie',
            12: 'Drum cu prioritate',
            13: 'Cedeaza trecerea',
            14: 'Stop',
            15: 'Vehiculele nu au voie',
            16: 'Vehiculele peste 3.5 tone nu au voie',
            17: 'Nu e voie sa intri',
            18: 'Atentie',
            19: 'Curba stanga periculoasa',
            20: 'Curba dreapta periculoasa',
            21: 'Curba dubla',
            22: 'Drum accidentat',
            23: 'Drum alunecos',
            24: 'Drumul se ingusteaza spre dreapta',
            25: 'Se lucreaza',
            26: 'Indicatoare rutiere',
            27: 'Trecere pietoni',
            28: 'Trecere copii',
            29: 'Trecere biciclete',
            30: 'Atentie la gheata/zapada',
            31: 'Trecere animale salbatice',
            32: 'End speed + passing limits',
            33: 'Vireaza dreapta',
            34: 'Vireaza stanga',
            35: 'Sens unic',
            36: 'Mergi in fata sau dreapta',
            37: 'Mergi in fata sau stanga',
            38: 'Pastreaza partea dreapta',
            39: 'Pastreaza partea stanga',
            40: 'Sens giratoriu mandatoriu',
            41: 'De acum se poate depasii',
            42: 'De acum pot depasii vehiculele de peste 3.5 tone'}

folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])

# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [list(tuple) for tuple in tuples]


def load_data(data_dir):
    images = list()
    labels = list()
    for category in range(NUM_CATEGORIES):
        categories = os.path.join(data_dir, str(category))
        for img in os.listdir(categories):
            img = load_img(os.path.join(categories, img), target_size=(30, 30))
            image = img_to_array(img)
            images.append(image)
            labels.append(category)

    return images, labels


images, labels = load_data(train_path)

# One hot encoding the labels
labels = to_categorical(labels)

# Splitting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(
                                                    np.array(images),
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    shuffle=True
                                                    )

x_train = x_train/255
x_test = x_test/255
print("X_train.shape", x_train.shape)
print("X_valid.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_test.shape)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation="softmax"))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

aug_validation = ImageDataGenerator()

model_save_path = 'C:/Users/Stefan/Documents/GTSRB/trained_model.h5'

EPOCHS = 30
history = model.fit(
    aug.flow(x_train, y_train, batch_size=32),
    validation_data=aug_validation.flow(x_test, y_test),
    epochs=EPOCHS
)

model.save(model_save_path)
print(f"Model saved to {model_save_path}")

loss, accuracy = model.evaluate(x_test, y_test)

print('test set accuracy: ', accuracy * 100)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
