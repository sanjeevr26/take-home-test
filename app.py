from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import requests # to get image from the web
import shutil # to save it locally

app = Flask(__name__)
model = ResNet50(weights='imagenet')


def get_image(image_url, filename):
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', filename)
    else:
        print('Image Couldnt be retreived')

    return r.status_code


@app.route('/')
def hello():
    return jsonify({'message': 'request is invalid'})


@app.route('/prediction', methods=['POST'])
def predict_image():
    try:
        content = request.json
        image_url = content['image_url']
        filename = image_url.split("/")[-1]
        download_image_status = get_image(image_url, filename)

        if download_image_status == 200:
            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            predictions = decode_predictions(preds, top=3)[0]

            prediction_arr = []
            for value in predictions:
                predictions_dict = {'id': str(value[0]), 'Elephant_name': str(value[1]), 'prediction_value': str(value[2])}
                prediction_arr.append(predictions_dict)

            print('Predicted:', decode_predictions(preds, top=3)[0])
            os.remove(filename)
            return jsonify({"output": prediction_arr})
        else:
            print("Not_downloaded")
            return jsonify({'message': 'Unable to download image - Please recheck the URL'}), 400

    except Exception as e:
        return jsonify({'message': str(e)}), 500


# if __name__ == "__main__":
#    app.run(port=8080)