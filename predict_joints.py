import cv2
from .hand_tracker import HandTracker
import copy
import numpy as np

PALM_MODEL_PATH = "/home/mahdi/HVR/git_repos/hand_tracking/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "/home/mahdi/HVR/git_repos/hand_tracking/hand_landmark_3d.tflite"
ANCHORS_PATH = "/home/mahdi/HVR/git_repos/hand_tracking/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2


#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)



def google_predict(args):
    # image = (np.loadtxt(args.iRGB).reshape(480, 640, 3)).astype(np.uint8)
    # image = (np.loadtxt(args.iRGB).reshape(480, 640, 3))
    from PIL import Image
    image = np.array(Image.open(args.iRGB))

    jointsUV, _, joints_full_estimation = detector(image)

    if args.vis == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        WINDOW = "Hand Tracking"
        cv2.namedWindow(WINDOW)
        if jointsUV is not None:
            k = 0
            for point in jointsUV:
                x, y = point
                cv2.circle(image, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS*5)

                TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
                TEXT_SCALE = 0.5
                TEXT_THICKNESS = 1
                TEXT = str(k)
                text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
                text_origin = (int(x - text_size[0] / 2), int(y + text_size[1] / 2))
                cv2.putText(image, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
                k += 1
                print(x, y)
            for connection in connections:
                x0, y0 = jointsUV[connection[0]]
                x1, y1 = jointsUV[connection[1]]
                cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        cv2.imshow(WINDOW, image)
        cv2.waitKey(2000)

    if jointsUV.shape != (21,2):
        raise ValueError("google hand tracker could not predict hand joints!")

    return jointsUV

if __name__ == '__main__':
    colorUV = np.load('/home/mahdi/HVR/git_repos/hand_tracking/colorUV.npy')

    google_predict(None, colorUV)