import cv2, os
import numpy as np
from matplotlib import pyplot as plt



class classification:

    def __init__(self):
        global contourEstimateCount
        contourEstimateCount = 0
        pass

    def contouringOld(self, filename):
        im = cv2.imread(filename)
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

        # cv2.imshow('test', opening)

        # filer = cv2.bilateralFilter(im,1,175,175)
        # cv2.imshow('bilat', filer)


        canny = cv2.Canny(opening, 50, 50)

        _, contours, hierachy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contourList = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 1) * (len(approx) < 2) & (area > 30)):
                contourList.append(contour)

        contourCount = 0
        for item in contourList:
            contourCount += 1

        cv2.drawContours(im, contourList, -1, (0, 255, 255), 2)
        cv2.imshow('Contour', im)
        type = ""

        if (contourCount < 15):
            type = "Elliptical"
        else:
            type = "Spiral"

        filenameShort = os.path.splitext(os.path.split(filename)[1])[0]

        returnString = filenameShort, ": Type: ", type, ": Countours: ", contourCount


        return returnString

    def cannyEdge(self, filename):
        im = cv2.imread(filename)
        im = cv2.medianBlur(im, 5)
        im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)
        edge = cv2.Canny(im, 50, 50)
        cv2.imshow('Canny Edge Detection', edge)
        cv2.waitKey(0)

    def cannyEdgeArg(self,im):
        im = cv2.medianBlur(im, 5)
        im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)
        edge = cv2.Canny(im, 50, 50)
        return edge

    def contouring(self, im):
        _, contours, hierachy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contourList = []
        for contour in contours:
            #approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            #area = cv2.contourArea(contour)
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(contour)
            contourList.append(contour)

        contourCount = 0
        for item in contourList:
            contourCount += 1

        return contourList, contourCount

    def selectTemplate(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        template = cv2.imread("template2.png", 0)
        template2 = cv2.imread("template3.png", 0)
        template3 = cv2.imread("template4.png", 0)
        w, h = template.shape[::-1]
        w2, h2 = template2.shape[::-1]
        w3, h3 = template3.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(gray, template2, cv2.TM_CCOEFF_NORMED)
        res3 = cv2.matchTemplate(gray, template3, cv2.TM_CCOEFF_NORMED)

        threshold = 0.9
        loc = np.where(res >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        x = 0
        y = 0
        x2 = 0
        y2 = 0
        x3 = 0
        y3 = 0

        isGalaxy = 0
        for pt in zip(*loc[::-1]):
            isGalaxy = 1
            x = pt[0]
            y = pt[1]

        for pt2 in zip(*loc2[::-1]):
            isGalaxy = 2
            x2 = pt2[0]
            y2 = pt2[1]

        for pt3 in zip(*loc3[::-1]):
            isGalaxy = 3
            x3 = pt3[0]
            y3 = pt3[1]

        if isGalaxy == 0:
            print "No matching template, Assumption made that galaxy takes centre frame"
            return im[96:130 + h, 90:90 + w]

        if isGalaxy == 1:
            print "Galaxy template 1"
            return im[96:130 + h, 90:90 + w]

            #return im[y:y + h, x:x + w]

        if isGalaxy == 2:
            print "Galaxy template 2"
            return im[96:130 + h, 90:90 + w]

            #return im[y2:y2 + h, x2:x2 + w]

        if isGalaxy == 3:
            print "Galaxy template 3"
            return im[96:130 + h, 90:90 + w]

            #return im[y3:y3 + h, x3:x3 + w]

    def printList(self,fileList):
        for file in fileList:
            print file

    def cannyInTemplate(self, filename):
        im = cv2.imread(filename)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)
        cv2.imshow('Template Canny', edge)

    def contourInCanny(self,filename):
        im = cv2.imread(filename)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = perim / contourCount * 100

        if total > 825:
            print total, "= Spiral"
        else:
            print total, "= Elliptical"

        # print contourCount
        cv2.drawContours(newIm, contourList, -1, (0, 255, 255), 2)
        cv2.imshow('Contour', newIm)

    def fillCanny(self,filename):
        im = cv2.imread(filename)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        newIm = cv2.medianBlur(newIm, 5)
        edge = cv2.Canny(newIm, 50, 50)
        kernel = np.ones((20, 20), np.uint8)
        edge = cv2.dilate(edge, kernel, 1)
        cv2.imshow('filling test', edge)
        cv2.waitKey(0)

    def templateMatch(self,filename):
        im = cv2.imread(filename)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        template = cv2.imread("template2.png", 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        rectangleCount = 0
        for pt in zip(*loc[::-1]):
            cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
            rectangleCount += 1

        if rectangleCount != 0:
            print "Galaxy Detected"

        cv2.imshow("Test", im)
        cv2.waitKey(0)

    def totalContours(self, file):
        im = cv2.imread(file)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = perim / contourCount * 100

        return total

    def kAvgChecker(self, file):
        im = cv2.imread(file)
        newIm = self.selectTemplate(im)
        kIm = self.kMeans(newIm)
        avgColour = self.kAvgColour(kIm)
        galType = 0
        if avgColour < 16:
            galType = 2
        else:
            galType = 1

        return galType

    def returnClassifier(self,file):
        im = cv2.imread(file)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = perim / contourCount * 100
        galType = 0
        if total > 850:
            galType = 1
            # print total, "= Spiral"
        else:
            galType = 2
            # print total, "= Elliptical"

        # print contourCount
        # cv2.drawContours(newIm, contourList, -1, (0, 255, 255), 2)
        # cv2.imshow('Contour', newIm)
        return galType

    def checkSpiral(self, fileList):
        total = 0.0
        correctType = 0.0
        negativeFiles = []
        for file in fileList:
            total += 1.0
            isSpiral = self.returnClassifier(file)
            if isSpiral == 1:
                print "Spiral"
                correctType += 1.0
            else:
                negativeFiles.append(file)

        print "Total:   ", total
        print "Spiral:  ", correctType
        accuracy = correctType / total * 100
        print "========================================================================="
        print "Accuracy of spiral galaxy detection in a dataset of ", total, " All Spiral Images"
        print "Accuracy:    ", accuracy, "%"
        print "========================================================================="
        print "Negative results are in the following images:"
        for item in negativeFiles:
            print item
        printString = "Accuracy:    ", accuracy, "%"
        return printString

    def checkSpiralK(self, fileList):
        total = 0.0
        correctType = 0.0
        negativeFiles = []
        for file in fileList:
            total += 1.0
            isSpiral = self.kAvgChecker(file)
            if isSpiral == 1:
                print "Spiral"
                correctType += 1.0
            else:
                negativeFiles.append(file)

        print "Total:   ", total
        print "Spiral:  ", correctType
        accuracy = correctType / total * 100
        print "========================================================================="
        print "Accuracy of spiral galaxy detection in a dataset of ", total, " All Spiral Images"
        print "Accuracy:    ", accuracy, "%"
        print "========================================================================="
        print "Negative results are in the following images:"
        for item in negativeFiles:
            print item
        printString = "Accuracy:    ", accuracy, "%"
        return printString

    def checkElliptical(self, fileList):
        total = 0.0
        correctType = 0.0
        negativeFiles = []

        for file in fileList:
            total += 1.0
            isEllip = self.returnClassifier(file)
            if isEllip == 2:
                print "Elliptical"
                correctType += 1.0
            else:
                negativeFiles.append(file)
        print "Total:   ", total
        print "Spiral:  ", correctType
        accuracy = correctType / total * 100
        print "========================================================================="
        print "Accuracy of Elliptical galaxy detection in a dataset of ", total, " All Elliptical Images"
        print "Accuracy:    ", accuracy, "%"
        print "Negative results are in the following images:"
        for item in negativeFiles:
            print item
        printString = "Accuracy:    ", accuracy, "%"
        return printString

    def checkEllipticalK(self, fileList):
        total = 0.0
        correctType = 0.0
        negativeFiles = []

        for file in fileList:
            total += 1.0
            isEllip = self.kAvgChecker(file)
            if isEllip == 2:
                print "Elliptical"
                correctType += 1.0
            else:
                negativeFiles.append(file)
        print "Total:   ", total
        print "Spiral:  ", correctType
        accuracy = correctType / total * 100
        print "========================================================================="
        print "Accuracy of Elliptical galaxy detection in a dataset of ", total, " All Elliptical Images"
        print "Accuracy:    ", accuracy, "%"
        print "Negative results are in the following images:"
        for item in negativeFiles:
            print item
        printString = "Accuracy:    ", accuracy, "%"
        return printString

    def contourEstimate(self, fileList):
        total = 0
        imCount = 0
        for file in fileList:
            imCount += 1
            total += self.totalContours(file)
        global contourEstimateCount
        contourEstimateCount = total / imCount
        print contourEstimateCount

        return contourEstimateCount

    def classifyOverAverage(self, filename):
        im = cv2.imread(filename)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = perim / contourCount * 100
        estimateCount = contourEstimateCount
        if estimateCount == 0:
            estimateCount = self.contourEstimate()

        if (total > estimateCount - (estimateCount * 0.05)):
            print total, "= Spiral"
        else:
            print total, "= Elliptical"

    def kMeans(self,im):

        z = im.reshape((-1,3))

        z = np.float32(z)

        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        ret, label, center = cv2.kmeans(z,k,None,crit,10,cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((im.shape))
        #cv2.imshow('K-Means', res2)
        #cv2.waitKey(0)

        return res2

    def contourKMeans(self,im):

        edge = self.cannyEdgeArg(im)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = contourCount

        print total

        if total < 225:
            print (total , ": Spiral")
        else:
            print (total , ": Elliptical")

        # print contourCount
        cv2.drawContours(im, contourList, -1, (0, 255, 255), 2)
        cv2.imshow('Contour', im)

    def kAvgColour(self, im):
        avgPerRow = np.average(im, axis=0)
        avg = np.average(avgPerRow, axis=0)
        print avg

        if avg[0] < 17:
            print "Elliptical"
        else:
            print "Spiral"

        return avg[0]





class histograms:

    def __init__(self):
        pass

    def colourHist(self, im):

        colour = ('b', 'g', 'r')
        for i, col in enumerate(colour):
            hist = cv2.calcHist([im], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()


class redundantClassification:

    def __init__(self):
        pass

    def houghCirlce(self, filename):
        im = cv2.imread(filename, 0)
        im = cv2.medianBlur(im, 5)
        cim = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=29, minRadius=0, maxRadius=0)

        count = 0
        # circles = np.around(circles)
        if circles is not None:
            for i in circles[0, :]:
                count += 1
                # draw the outer circle
                cv2.circle(cim, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(cim, (i[0], i[1]), 2, (0, 0, 255), 3)

        print count

        cv2.imshow('detected circles', cim)
        cv2.waitKey(0)

    def hog(self, filename):
        im = cv2.imread(filename)
        im = self.selectTemplate(im)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.float32(s) / 255.0
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag = cv2.cvtColor(mag, cv2.COLOR_HSV2BGR)

        cv2.imshow('mag', mag)