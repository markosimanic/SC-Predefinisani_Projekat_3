from sklearn.metrics import mean_absolute_error
import imutils
import cv2

width = 800
counted = 0

"""
  Function that checks if center point of bounding box has crossed the line
"""
def crossed(x, y ):
    if(x < 900 and x > 150 and y == 151):
        return True
    return False


if __name__ == "__main__":

    # load video from dataset:
    video = cv2.VideoCapture("dataset/video1.mp4")
    startingFrame = None
    while True:
        (read, frame) = video.read()
        if not read:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=width)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (21, 21), 0)


        if startingFrame is None:
            startingFrame = grayscale
            continue

        distance = cv2.absdiff(startingFrame, grayscale)
        thresh = cv2.threshold(distance, 21, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=7)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


        for c in contours:
            if cv2.contourArea(c) < 120:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (width // 4, 150), (600, 120), (0, 0, 255), 2)
            # calculate center points of rectangle
            centerPointX = (x + x + w) // 2
            centerPointY = (y + y + h) // 2

            centerPoint = (centerPointX,  centerPointY)

            cv2.circle(frame, centerPoint, 1, (0, 0, 255), 5)

            if (crossed(centerPointX, centerPointY)):
                counted += 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.putText(frame, "Counted: {}".format(str(counted)), (10, 70),
                    cv2.QT_FONT_NORMAL, 0.8, (0, 255, 0), 2)
        cv2.imshow("People counter", frame)


    video.release()
    cv2.destroyAllWindows()

    y_true = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
    y_pred = [3, 6, 3, 5, 5, 7, 3, 6, 4, 6]
    error = mean_absolute_error(y_true, y_pred)
    print(error)


