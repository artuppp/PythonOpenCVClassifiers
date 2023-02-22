import numpy as np
import cv2 as cv

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                         thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':

    detector = cv.FaceDetectorYN.create(
        "models/face_detection_yunet_2022mar.onnx",
        "",
        (640, 480),
        0.9,
        0.3,
        5000
    )

    tm = cv.TickMeter()
    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * 1)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * 1)
    print('Frame size: {}x{}'.format(frameWidth, frameHeight))
    detector.setInputSize([frameWidth, frameHeight])
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        frame = cv.resize(frame, (frameWidth, frameHeight))
        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())
        # Visualize results
        cv.imshow('Live', frame)
    cv.destroyAllWindows()