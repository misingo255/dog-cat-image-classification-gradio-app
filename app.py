import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import gradio as gr



def classify(image):

    IMAGE_SHAPE = (224, 224)

    buffer = np.array(image)/255.0

    buffer = buffer[np.newaxis, ...]

    classifier = tf.keras.Sequential([hub.KerasLayer("models/", input_shape=IMAGE_SHAPE+(3,))])

    results = classifier.predict(buffer)

    predicted_image_index = np.argmax(results)

    image_labels = []
    with open("./static/labels.txt", "r") as f:
        image_labels = f.read().splitlines()

    class_name = image_labels[predicted_image_index]

    if class_name == "dog":
        return "This is an image of a dog"
    elif class_name == "cat":
        return "This is an image of a cat"
    else:
        return "This image is neither or a dog nor of a cat"


app = gr.Interface(fn=classify, inputs=gr.Image(shape=(224,224)), outputs="text")

app.launch()