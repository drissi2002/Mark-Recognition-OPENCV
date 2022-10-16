import cv2
import numpy as np

import utilities

# ------------parameters------------------
webCamFeed = True
path = "test.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 600
widthImg  = 600
questions=5
choices=5
ans= [1,2,0,2,4]
# ----------------------------------------
count=0
while True:

#    if webCamFeed:
#        success, img = cap.read()
#    else:
    img = cv2.imread(path)
    img = cv2.resize(img, (widthImg, heightImg))
    ## image finale
    imgFinal = img.copy()
    ## transform image into a gray one
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## use a blur effect on image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    ## transform image into a black and white one by focusing on the main features
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        ## COPY IMAGE FOR DISPLAY PURPOSES
        imgContours = img.copy()
        ## COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy()
        ## detecting the countours of image
        countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 5)
        ## FILTER FOR RECTANGLE CONTOURS
        rectCon = utilities.rectContour(countours)
        ## GET CORNER POINTS OF THE BIGGEST RECTANGLE
        biggestContour = utilities.getCornerPoints(rectCon[0])
        ## GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE
        gradePoints = utilities.getCornerPoints(rectCon[1])
        # print((biggestContour))
        if biggestContour.size != 0 and gradePoints.size != 0:

            # BIGGEST RECTANGLE WARPING

            ## REORDER FOR WARPING
            biggestPoints = utilities.reorder(biggestContour)
            ## DRAW THE BIGGEST CONTOUR
            cv2.drawContours(imgBigContour, biggestPoints, -1, (255, 0, 0), 5)
            ## PREPARE POINTS FOR WARP
            pts1 = np.float32(biggestPoints)
            ## PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            ## GET TRANSFORMATION MATRIX
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            ## APPLY WARP PERSPECTIVE
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # SECOND BIGGEST RECTANGLE WARPING
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # DRAW THE BIGGEST CONTOUR
            gradePoints = utilities.reorder(gradePoints)  # REORDER FOR WARPING
            ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE
            # cv2.imshow("grade",imgGradeDisplay)

            # APPLY THRESHOLD
            ## CONVERT TO GRAYSCALE
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            ## APPLY THRESHOLD AND INVERSE
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            ## GET INDIVIDUAL BOXES
            boxes = utilities.splitBoxes(imgThresh)
            # cv2.imshow("Split Test ", boxes[3])
            countR = 0
            countC = 0

            # TO STORE THE NON ZERO VALUES OF EACH BOX
            myPixelVal = np.zeros((questions, choices))
            for image in boxes:
                ## cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices): countC = 0;countR += 1

            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print("USER ANSWERS", myIndex)

            # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            print("GRADING", grading)
            score = (sum(grading) / questions) * 100  # FINAL GRADE
            print("SCORE", score)

            # DISPLAYING ANSWERS
            ## DRAW DETECTED ANSWERS
            utilities.showAnswers(imgWarpColored, myIndex, grading, ans)
            ##DRAW GRID
            utilities.drawGrid(imgWarpColored)
            ## NEW BLANK IMAGE WITH WARP IMAGE SIZE
            imgRawDrawings = np.zeros_like(imgWarpColored)
            ## DRAW ON NEW IMAGE
            utilities.showAnswers(imgRawDrawings, myIndex, grading, ans)
            ## INVERSE TRANSFORMATION MATRIX
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            ## INV IMAGE WARP
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            # DISPLAY GRADE
            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
            cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)  # ADD THE GRADE TO NEW IMAGE
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # INV IMAGE WARP

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            # IMAGE ARRAY FOR DISPLAY
            imageArray = ([img, imgGray, imgCanny, imgContours],
                          [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
        imgBlank = np.zeros((heightImg, widthImg, 3),
                            np.uint8)
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Edges", "Contours"],
              ["Biggest Contour", "Threshold", "Warpped", "Final"]]
    imgSatcked = utilities.stackImages(imageArray, 0.5 ,lables)
    cv2.imshow("Stacked Images", imgSatcked)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgFinal)
        cv2.rectangle(imgSatcked, ((int(imgSatcked.shape[1] / 2) - 230), int(imgSatcked.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgSatcked, "Scan Saved", (int(imgSatcked.shape[1] / 2) - 200, int(imgSatcked.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', imgSatcked)
        cv2.waitKey(300)
        count += 1
