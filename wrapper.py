from lane import *
from lanet import *
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture("project_video.mp4")
    # _, img = cap.read()
    standard_width = int((1280 / 720.0) * 300.0)
    standard_height = 300
    while (cap.isOpened()):
        _, img = cap.read()
        img = cv2.resize(img, (standard_width, standard_height))
        try:
            processed = process_image(img)
        except:
            processed = img
        cv2.imshow('video input result', processed)
        
        keyCode = cv2.waitKey(30) & 0xff
        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pass
