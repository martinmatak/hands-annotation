import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
import cv2
import sys
import numpy as np
import time
import imageio

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def annotate_image(image, model):
    # copy to draw on
    draw = image.copy()

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
        if score < 0.3:
            continue
        b = detections[0, idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)

    return draw


def main(argv):
    if len(argv) != 3:
        print("Usage for image annotation: -i <path_input> <path_output>")
        print("Usage for video annotation: -v <path_input> <path_output>")
        sys.exit(1)
    image_or_video, path_input, path_output = argv
    keras.backend.tensorflow_backend.set_session(get_session())
    print("set tensorflow")

    # load the model
    print('Loading the model, this may take a second...')
    model = keras.models.load_model('./snapshots/resnet50_csv_17.h5', custom_objects=custom_objects)
    print('Model created')
    # print model summary
    print(model.summary())

    if "-v" in image_or_video:
        reader = imageio.get_reader(path_input)
        print("Loaded file: " + path_input)
        fps = reader.get_meta_data()['fps']

        writer = imageio.get_writer(path_output, fps=fps)

        for im in reader:
            writer.append_data(annotate_image(im, model))
        writer.close()
    elif "-i" in image_or_video:
        im = imageio.imread(path_input)
        print("Loaded file: " + path_input)
        imageio.imwrite(path_output, annotate_image(im, model))


if __name__ == "__main__":
    main(sys.argv[1:])
