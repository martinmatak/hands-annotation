import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
import cv2
import numpy as np
import time
import imageio

import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def annotate_image(image):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.1:
            continue
        b = detections[0, idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)

    return draw

files = [
# "1_2015-10-03_20-23-27",
"1_2015-10-03_20-38-29",
"1_2015-10-03_20-53-30",
"1_2015-10-03_21-08-31",
"1_2015-10-03_21-23-32",
"1_2015-10-03_21-38-33",
"1_2015-10-03_21-53-34",
"1_2015-10-03_22-08-35",
"1_2015-10-03_22-23-36",
"1_2015-10-03_22-38-37",
"1_2015-10-03_22-53-39",
"1_2015-10-03_23-08-39",
"1_2015-10-03_23-23-40",
"1_2015-10-03_23-38-42",
"1_2015-10-03_23-53-42",
"1_2015-10-04_00-08-43",
"1_2015-10-04_00-23-45",
"1_2015-10-04_00-38-46"
]

if __name__ == '__main__':
    keras.backend.tensorflow_backend.set_session(get_session())
    print("set tensorflow")

    # load the model
    print('Loading the model, this may take a second...')
    model = keras.models.load_model('/snapshots/resnet50_csv_17.h5', custom_objects=custom_objects)
    print('Model created')
    # print model summary
    print(model.summary())

    for filename in files:
        reader = imageio.get_reader('/data/' + filename + ".mp4")
        print("Loaded file: " + filename)
        fps = reader.get_meta_data()['fps']

        writer = imageio.get_writer('./annotated_videos/' + filename + '-annotated.mp4', fps=fps)

        for im in reader:
            writer.append_data(annotate_image(im))
        writer.close()


