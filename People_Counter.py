from sklearn.metrics import mean_absolute_error
import imutils
import cv2

width = 800
counted = 0



"""
  Function that checks if center point of bounding box has crossed the line
   x and y are coordinates of center point of bounding rect
"""
def detect_cross(x, y):
    # equation of a line for two points
    # calculated on http://alas.matf.bg.ac.rs/~ml09185/strana2.html#2.6
    eq = y - 250
    if abs(eq) <= 1:
       # print(str(eq))
        return True
    return False

""" 
def detect_cross(x, y):
    if (x < 900 and x > 100 and y == 151):
        return True
    return False
"""

def MAE():
    y_true = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
    y_pred = [3, 17, 9, 19, 10, 20, 15, 15, 3, 9]
    error = mean_absolute_error(y_true, y_pred)
    print("MAE:", error)


if __name__ == "__main__":

    # load video from dataset:
    video = cv2.VideoCapture("dataset/video10.mp4")
    startingFrame = None
    while video:
        (read, frame) = video.read()
        if not read:
            break

        frame = imutils.resize(frame, 800)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (21, 21), 0)


        if startingFrame is None:
            startingFrame = grayscale
            continue

        distance = cv2.absdiff(startingFrame, grayscale)
        thresh = cv2.threshold(distance, 17, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in contours:
            if cv2.contourArea(c) < 150:
              continue

            (x, y, w, h) = cv2.boundingRect(c)
            rectStartPoint = (x, y)
            rectEndPoint = (x + w, y + h)
            cv2.rectangle(frame, rectStartPoint, rectEndPoint, (0, 255, 0), 2)

            # calculate center points of rectangle
            centerPointX = (x + x + w) / 2
            centerPointY = (y + y + h) / 2

            centerPointXint = (x + x + w) // 2
            centerPointYint = (y + y + h) // 2


            centerPoint = (centerPointXint, centerPointYint)
            cv2.circle(frame, centerPoint, 1, (0, 0, 255), 5)


            cv2.line(frame, (0, 250), (800, 250), (0, 0, 255), 2)

            if (detect_cross(centerPointX, centerPointY)):
                counted += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.putText(frame, "Counted: {}".format(str(counted)), (10, 70),
                    cv2.QT_FONT_NORMAL, 0.8, (0, 255, 0), 2)
        cv2.imshow("People counter", frame)


    video.release()
    cv2.destroyAllWindows()
    MAE()




