import numpy as np
import cv2

def get_lines_from_hough(frame, lines):
    left = []
    right = []
    for l in lines:
        x1, y1, x2, y2 = l.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        m = parameters[0]
        b = parameters[1]
        if m < 0:
            left.append([m, b])
        else:
            right.append([m, b])
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    h = frame.shape[0]
    left_line = calculate_coordinates(h, left_avg)
    right_line = calculate_coordinates(h, right_avg)

    return np.array([left_line, right_line])

def calculate_coordinates(h, parameters):
    print(parameters)
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = int(h * 0.9)
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame_in, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    # lines_visualize = np.zeros_like(frame)
    frame = np.array(frame_in)
    # Checks if any lines are detected
    if lines is not None:
        lx1, ly1, lx2, ly2 = lines[0]
        rx1, ry1, rx2, ry2 = lines[1]

        fills = np.array([ [(lx2, ly2), (lx1, ly1), (rx1, ry1), (rx2, ry2)] ])
        cv2.fillPoly(frame, fills, (0, 0, 255))
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return frame

def detect_lane(frame):

    print("inside main function!")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grey, 50, 150)
    height, width = canny.shape
    polys = np.array([ [(30, int(height * 0.9)), (int(width * 0.7), int(height * 0.9)), (int(0.5 * width), int(0.5 * height))] ])
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polys, 255)
    segment = cv2.bitwise_and(canny, mask)
    hough = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

    lines = get_lines_from_hough(frame, hough)
    res = visualize_lines(frame, lines)

    f = cv2.addWeighted(frame, 0.5, res, 0.5, 1)
    return segment, f