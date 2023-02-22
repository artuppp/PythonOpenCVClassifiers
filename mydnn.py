from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model_mobile_hand.h5", compile=False)

# Load the labels
class_names = open("labels_mobile_hand.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, frame = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    if confidence_score > 0.90:
        cv2.putText(frame, "Class: " + class_name[2:] + " Confidence " + str(np.round(confidence_score * 100))[:-2] + "%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    else:
        cv2.putText(frame, "Class: Unknown" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    # Show the image in a window
    cv2.imshow("Webcam Image", frame)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()