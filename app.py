from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('Custom_model.h5')

# Define the class names
class_names = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

IMAGE_SIZE = 64

def preprocess_image(image):
    # Preprocess the image to match the input format of your model
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to match model's input size
    image = np.array(image) / 255.0  # Normalize the image
    if image.shape[-1] == 4:  # Check if the image has an alpha channel
        image = image[..., :3]  # Discard the alpha channel if present
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def upload_file():
    prediction = None
    actual_class = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        actual_class = file.filename
        print("actual class: ", actual_class)
        if file.filename == "":
            return redirect(request.url)
        if file:
            image = Image.open(io.BytesIO(file.read()))
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_index]
            print("Predictions array:", predictions)
            print("Predicted class index:", predicted_class_index)
            print("Predicted class name:", predicted_class)
            prediction = predicted_class
    return render_template("index.html", prediction=prediction, actual_class=actual_class, class_names=class_names)

if __name__ == "__main__":
    app.run(debug=True)
