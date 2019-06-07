import cv2
import numpy as np
import pickle
from keras.models import load_model
import util

def main():
    # Load model
    model = load_model('trained_model.h5')
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.namedWindow("Draw")
    global drawing, curr_x, curr_y
    drawing = False

    def mouse_callback(event, x, y, flags, param):
        global drawing, curr_x, curr_y
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            curr_x, curr_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(image, (curr_x, curr_y), (x, y), (255, 255, 255), 5)
                curr_x = x
                curr_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(image, (curr_x, curr_y), (x, y), (255, 255, 255), 5)
            curr_x = x
            curr_y = y
        return x, y

    cv2.setMouseCallback('Draw', mouse_callback)
    while (1):
        cv2.imshow('Draw', 255 - image)
        key = cv2.waitKey(10)
        if key == ord(" "):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            y, x = np.nonzero(image)
            mn = min(np.min(y), np.min(x))
            mx = max(np.max(y), np.max(x))
            image = image[mn:mx, mn:mx]

            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            image = np.array(image, dtype=np.float32)[None, :, :, None]
            pred_class = model.predict_classes(image)
            classes = pickle.load(open("classes.p", "rb"))
            util.plotImage(image, classes[pred_class[0]])
            print('Is it ' + classes[pred_class[0]] + '? ')
            image = np.zeros((600, 600, 3), dtype=np.uint8)
            curr_x = -1
            curr_y = -1

if __name__ == '__main__':
    main()
