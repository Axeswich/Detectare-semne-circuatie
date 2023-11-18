from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import traceback

app = Flask(__name__, template_folder='template')

MODEL_PATH = 'C:/Users/Stefan/Documents/GTSRB/trained_model.h5'


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
        # Load the model inside the Flask request context
        model = load_model(MODEL_PATH)

        # The image file seems valid! Detect the traffic sign and return the result.
        image = Image.open(file)
        print(f"Original image mode: {image.mode}, shape: {np.array(image).shape}")
        image = image.resize((30, 30))  # Resize to your target size
        print(f"Resized image mode: {image.mode}, shape: {np.array(image).shape}")

        # Convert to RGB if the image is not in 'RGB' mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"Converted to RGB mode: {image.mode}, shape: {np.array(image).shape}")

        # Convert to grayscale if the image is not in 'L' mode
        if image.mode != 'L':
            image = ImageOps.grayscale(image)
            print(f"Converted to 'L' mode: {image.mode}, shape: {np.array(image).shape}")

        # Apply colorize only if the image is in 'L' mode
        if image.mode == 'L':
            image = ImageOps.colorize(image, 'black', 'white').convert('RGB')
            print(f"Converted to RGB mode (colorized): {image.mode}, shape: {np.array(image).shape}")

        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)

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

        return jsonify({'prediction': classes[predicted_class[0]]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
