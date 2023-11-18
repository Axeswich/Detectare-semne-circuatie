from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import traceback
import tensorflow as tf

# Creates a Flask app.
# Specifies the template folder as 'template'.

app = Flask(__name__, template_folder='template')

# Defines constants like the path to the pre-trained model, target image size, and normalization factor.

MODEL_PATH = 'C:/Users/Stefan/Documents/GTSRB/trained_model.h5'
TARGET_SIZE = (30, 30)
NORMALIZATION_FACTOR = 255.0

# Defines a custom session and graph for TensorFlow to manage the model loading and prediction.

model_session = tf.compat.v1.Session()
model_graph = tf.compat.v1.get_default_graph()

# Loads the pre-trained traffic sign recognition model within the custom TensorFlow session and graph.

def load_traffic_sign_model():
    global model_session
    global model_graph

    with model_graph.as_default():
        with model_session.as_default():
            model = load_model(MODEL_PATH)

    return model

# Defines a function to be executed before the first request.
# Clears the default session and sets the custom session as default.
# Loads the traffic sign model and stores it in the Flask app object.

@app.before_first_request
def before_first_request():
    global model_session
    global model_graph

    # Clear the default session and set the custom session as default
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.keras.backend.set_session(model_session)

    # Load the traffic sign model and store it in the app object
    app.model = load_traffic_sign_model()

# Defines a function to preprocess an image.
# Resizes the image, converts it to RGB if not in 'RGB' mode, converts it to grayscale if not in 'L' mode,
# and applies colorization if the image is in 'L' mode.


def preprocess_image(image):
    # Resize image to target size
    image = image.resize(TARGET_SIZE)

    # Convert to RGB if not in 'RGB' mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to grayscale if not in 'L' mode
    if image.mode != 'L':
        image = ImageOps.grayscale(image)

    # Apply colorize only if the image is in 'L' mode
    if image.mode == 'L':
        image = ImageOps.colorize(image, 'black', 'white').convert('RGB')

    return img_to_array(image)

# Defines a function to predict the traffic sign class given an image array.
# Uses the pre-trained model for prediction.


def predict_traffic_sign(image_array):
    global model_session
    global model_graph

    with model_graph.as_default():
        with model_session.as_default():
            image_array = np.expand_dims(image_array, axis=0)
            image_array /= NORMALIZATION_FACTOR

            # Use the loaded model for prediction
            prediction = app.model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)

    return predicted_class[0]

# Defines a function to format the predicted class index into a human-readable traffic sign description.


def format_prediction(class_index):
    classes = {0: 'Limita de viteza (20km/h)',
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
    return classes[class_index]

# '/': Renders the upload HTML page (index.html).
# '/predict': Handles POST requests for traffic sign prediction.
# Accepts an uploaded image file.
# Preprocesses the image, predicts the traffic sign class, formats the prediction, and returns a JSON response.

@app.route('/', methods=['GET'])
def index():
    # Render the upload HTML page
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file)
        print(f"Original image mode: {image.mode}, shape: {np.array(image).shape}")

        # Preprocess the image
        image_array = preprocess_image(image)
        print(f"Processed image shape: {image_array.shape}")

        # Predict the traffic sign
        predicted_class = predict_traffic_sign(image_array)

        # Format the prediction result
        prediction_result = format_prediction(predicted_class)

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})

# Runs the Flask app in debug mode if the script is executed directly.

if __name__ == '__main__':
    app.run(debug=True)
