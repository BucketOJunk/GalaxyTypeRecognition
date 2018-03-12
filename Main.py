from Tkinter import *
from ttk import Style
import cv2, tkFileDialog, py_compile, os
import numpy as np
import pylab
import time
from PIL import Image,ImageTk
import Tools as Tool



py_compile.compile("Main.py")

size = (850,400)






class App(Frame):
    #-------------------------------------------------------------------------------------------------------------------
    #Initialisation of the image for classification and feature extraction
    #Creates an OpenCV image, and sets the pixel count to NX, the number of lines to ny & and the number of channels to nc
    #it then creates a histogram of the image and plots it.
    #-------------------------------------------------------------------------------------------------------------------
    def init(self):


        image = cv2.imread(filename)

        global ny, nx, nc
        ny, nx, nc = image.shape
        x,h = self.hist(image)
        self.plotHist(x,h,filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Sets the currently selected file
    # -------------------------------------------------------------------------------------------------------------------
    def currentFile(self,file):
        global filename
        filename = file

    # -------------------------------------------------------------------------------------------------------------------
    # Returns curently selected file
    # -------------------------------------------------------------------------------------------------------------------
    def getCurrentFile(self):
        return filename

    # -------------------------------------------------------------------------------------------------------------------
    # Plots grey levels of image by searching in a range of 0 - 256
    # It adds the total value of all colours and divides by 3 to get the one, grey channel
    # Converts to int and then returns the range bracket and the grey levels of the image
    # -------------------------------------------------------------------------------------------------------------------
    def hist(self, im):

        maxgrey = 256

        # ab = numpy.ndarray(maxgrey)
        # for i in range(0, maxgrey):
        #   ab[i] = i
        ab = range(0, 256)

        h = np.zeros(maxgrey)
        for y in range(0, ny):
            for x in range(0, nx):
                greyLevel = 0.0
                for c in range(0, nc):
                    # v = im[y, x, c]
                    greyLevel += im[y, x, c]
                    # h[v] += 1
                greyLevel /= 3
                h[int(greyLevel)] += 1

        return ab, h

    # -------------------------------------------------------------------------------------------------------------------
    # Variety of different methods used for classification purposes
    #TODO:
    #- Create methods which evaluate the accuracy of the different classification methods
    #
    #
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    # Uses contour tracing on current image, if amount of contours is over 15, the galaxy is said to be spiral
    # If the count is under 15, then the galaxy is said to be Elliptical
    # Very lite, very in-accurate
    # -------------------------------------------------------------------------------------------------------------------
    def contourColour(self):
        return classification.contouringOld(filename)


    # -------------------------------------------------------------------------------------------------------------------
    # Standard canny edge detection without any classification logic (Debugging)
    # -------------------------------------------------------------------------------------------------------------------
    def cannyEdge(self):
        classification.cannyEdge(filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Next 2 functions are used to crop the image down to the size of the template match and then perform canny edge detection
    # on the template area. This is to be used for debug purposes to allow the user to see where the application will
    # be performing edge detection on the cropped image
    # -------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------
    # Canny edge arg is used to return and canny edge image of the provided argument
    # -------------------------------------------------------------------------------------------------------------------

    def cannyEdgeArg(self, im):
        return classification.cannyEdgeArg(im)

    def selectTemplate(self, im):
        return classification.selectTemplate(im)

    # -------------------------------------------------------------------------------------------------------------------
    # This is used to provide a contour count on the provided image argument, can be used for any type of image
    # -------------------------------------------------------------------------------------------------------------------

    def contouring(self, im):
        contourList, contourCount = classification.contouring(im)
        return contourList, contourCount

    # -------------------------------------------------------------------------------------------------------------------
    # Performs canny edge detection inside the template match
    # -------------------------------------------------------------------------------------------------------------------
    def cannyInsideTemplate(self):
        classification.cannyInTemplate(filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Used to detect circles in the image
    # - Todo
    # - Increase accuracy on circle detection as results are either too broad or to weak
    # -------------------------------------------------------------------------------------------------------------------
    def houghCircle(self):
        redundant.houghCirlce(filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Never called
    # -------------------------------------------------------------------------------------------------------------------
    def houghLine(self):
        im = cv2.imread(filename, 0)
        im = cv2.medianBlur(im, 5)
        cim = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------------------------------------------------------------------------
    # Converts the image into gradients which can be used on canny edge detection to filter out most if not all noise
    # -todo
    # -Increase Brightness of image returned
    # -------------------------------------------------------------------------------------------------------------------
    def hog(self):
        redundant.hog(filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Plots colour histogram showing colour distribution of each channel
    # -------------------------------------------------------------------------------------------------------------------
    def colourHist(self):
        im = cv2.imread(filename)
        im = classification.selectTemplate(im)
        hist.colourHist(im)

    def contourEstimate(self):
        return classification.contourEstimate(fileList)


    def classifyOverAverage(self):
        classification.classifyOverAverage(filename)

    def printFileList(self):
        classification.printList(fileList)

    def checkSpiral(self):
        label3.config(text="")
        printString = classification.checkSpiral(fileList)
        label3.config(text=printString)

    def checkSpiralK(self):
        label3.config(text="")
        printString = classification.checkSpiralK(fileList)
        label3.config(text=printString)

    def checkElliptical(self):
        label3.config(text="")
        printString = classification.checkElliptical(fileList)
        label3.config(text=printString)

    def checkEllipticalK(self):
        label3.config(text="")
        printString = classification.checkEllipticalK(fileList)
        label3.config(text=printString)



    def returnClassifier(self, file):
        return classification.returnClassifier(file)

    def totalContour(self, file):
        return classification.totalContours(file)

    def kmeans(self):
        im = cv2.imread(filename)
        im = classification.selectTemplate(im)
        kImg = classification.kMeans(im)
        return kImg

    def contourK(self):
        kIm = self.kmeans()
        return classification.contourKMeans(kIm)

    def kColourMean(self):
        kIm = self.kmeans()
        return classification.kAvgColour(kIm)


    # -------------------------------------------------------------------------------------------------------------------
    # Next function is used to count the amount of contours that have been found by using canny edge detection
    # -------------------------------------------------------------------------------------------------------------------

    def contourCountTemplateArea(self):
        classification.contourInCanny(filename)




    # -------------------------------------------------------------------------------------------------------------------
    # Uses template matching with smaller images of galaxies, good for detecting the location of the galaxy
    # but as of yet provides no way to differentiate between spiral and elliptical
    # TODO:
    # -Implement way to differentiate the different types of galaxies, (Maybe using contour tracing inside the template
    # matched area
    # -------------------------------------------------------------------------------------------------------------------

    def templateMatch(self):
        classification.templateMatch(filename)

    # -------------------------------------------------------------------------------------------------------------------
    # Fills the canny to try and remove lines between the detected edges to try and predict shape
    # -todo
    # -Requires value tweaking to improve accuracy
    # -------------------------------------------------------------------------------------------------------------------
    def fillCanny(self):
        classification.fillCanny(filename)




    # -------------------------------------------------------------------------------------------------------------------
    # Default detect method run when the user loads new images or if a new image is selected from the list
    # Will be set to most accurate method for final release
    # Currently set to: Counter average perimeter length
    # -------------------------------------------------------------------------------------------------------------------
    def detect(self):

        im = cv2.imread(filename)
        newIm = self.selectTemplate(im)

        edge = self.cannyEdgeArg(newIm)

        contourList, contourCount = self.contouring(edge)

        perim = 0
        for contour in contourList:
            perim += cv2.arcLength(contour, True)

        total = perim / contourCount * 100
        type = ""
        if total > 850:
            type = "Spiral"
            print total, "= Spiral"
        else:
            type = "Elliptical"
            print total, "= Elliptical"

        filenameShort = os.path.splitext(os.path.split(filename)[1])[0]

        returnString = filenameShort, ": Type: ", type, ": Average Perimeter Total: ", total

        return returnString






    # -------------------------------------------------------------------------------------------------------------------
    # Plots the histogram by taking x & y coordinates called from init
    # -------------------------------------------------------------------------------------------------------------------

    def plotHist(self, x, y, fn):
        pylab.figure()
        pylab.xlim(0, 255)
        pylab.grid()
        pylab.title("Histogram of " + fn)
        pylab.xlabel("grey level")
        pylab.ylabel("number of occurrences")
        pylab.bar(x, y, align="center")
        pylab.show()

    # -------------------------------------------------------------------------------------------------------------------
    # Will be used to display help window
    # -------------------------------------------------------------------------------------------------------------------
    def help(self):
        top = Toplevel()
        top.title("Help Menu")
        top.geometry('500x300')

        msg = Message(top, text="(Here's a list of some quick references to help you navigate your way around the application)"
                                "\n\nTo import images to the application, select 'File' then select 'Open Files'"
                                "\n\nTo save the classification results, select 'File' then select 'Save Results'"
                                "\n\nIf you wish to clear the list, Click the clear list button next to the list or select 'Edit' and then select 'Clear list'"
                                "\n\nHistogram Generation: To generate a histogram of the selected image, click 'Tools' and then select 'Generate Histogram'"
                                "\n\nIf you wish to see the details behind the classification process, select 'Tools' and then select 'Classification Details'")
        msg.pack()

        exButton = Button(top, text="Close Menu", command=top.destroy)
        exButton.pack()

    # -------------------------------------------------------------------------------------------------------------------
    # Will be used for exporting data to file
    # -------------------------------------------------------------------------------------------------------------------
    def save(self):
        timestr = time.strftime("%d %m %Y - %H %M")

        file = open(str(timestr) + ".txt","w")

        file.write("=================================================================================================================")
        file.write("\nClassification Results for: " + timestr)
        file.write("\n=================================================================================================================")

        for img in fileList:
            im = cv2.imread(img)
            newIm = self.selectTemplate(im)

            edge = self.cannyEdgeArg(newIm)

            contourList, contourCount = self.contouring(edge)

            perim = 0
            for contour in contourList:
                perim += cv2.arcLength(contour, True)

            total = perim / contourCount * 100
            total = round(total, 2)
            type = ""
            if total > 850:
                type = "Spiral"
                print total, "= Spiral"
            else:
                type = "Elliptical"
                print total, "= Elliptical"

            # print newIm.std()

            filenameShort = os.path.splitext(os.path.split(filename)[1])[0]

            returnString = filenameShort, ": |Type|: ", type, ": |Average Perimeter Total|: ", total

            file.write("\n"+str(returnString))


    # -------------------------------------------------------------------------------------------------------------------
    # Opens a file explorer and allows user to select files they wish to import to the application
    # This is then saved into a list and added to the list box
    # -------------------------------------------------------------------------------------------------------------------
    def open(self):
        file = tkFileDialog.askopenfilenames(parent=self,title='Select files', defaultextension=".jpg", filetypes=(("Jpeg", "*.jpg"),("Png", "*.png"),("All files", "*.*")))

        fileName = []

        for i in self.tk.splitlist(file):
            fileName.append(i)
            fileList.append(i)

        if file != None:
            self.displayIm(fileName[0])
            self.currentFile(fileName[0])
            for item in self.tk.splitlist(file):
                listbox.insert(END, item)

    # -------------------------------------------------------------------------------------------------------------------
    # Takes user selection to be passed into display im
    # -------------------------------------------------------------------------------------------------------------------
    def changeIm(self,im):

        newIm = im.widget
        sel = newIm.curselection()
        value = newIm.get(sel[0])
        label3.config(text="")

        self.displayIm(value)



        #self.init(value)

    # -------------------------------------------------------------------------------------------------------------------
    # Changes the image to the selected image in the display box
    # -------------------------------------------------------------------------------------------------------------------
    def displayIm(self, file):

        self.currentFile(file)

        im = Image.open(file)
        im = im.resize(size, Image.ANTIALIAS)
        im2 = ImageTk.PhotoImage(im)


        label2 = Label(self, image=im2)
        label2.image = im2
        label2.place(relx=.5, rely=.3, anchor="center")
        global label3
        label3 = Label(self, text=self.detect(), background="#333", fg="white")
        label3.place(relx=.5, rely=.571, anchor="center")



    # -------------------------------------------------------------------------------------------------------------------
    # Clears list of imported files
    # -------------------------------------------------------------------------------------------------------------------
    def clearList(self):
        listbox.delete(0,END)
        fileList[:] = []

        label3.config(text="")


    # -------------------------------------------------------------------------------------------------------------------
    # Initialising GUI elements
    # -------------------------------------------------------------------------------------------------------------------
    def createWidgets(self):
        self.master.title("Galaxy Type Recognition")
        self.configure(background="#333")
        label1 = Label(self, text="Galactic Recognition Software", background="#333", fg="white")
        label1.place(relx=.5, rely=.03, anchor="center")


        scroll = Scrollbar(self, orient=VERTICAL)
        scroll.grid(row=0, column=1, sticky="NS")
        listbox.config(yscrollcommand=scroll.set)
        listbox.pack()
        scroll.pack(fill=Y)

        listbox.place(relx=.4, rely=.71, anchor="center")
        scroll['command'] = listbox.yview
        scroll.place(relx=.698, rely=.71, anchor="center")
        listbox.bind("<Double-Button-1>", self.changeIm)
        listbox.bind("<Return>", self.changeIm)
        menubar = Menu(root)

        fileMenu = Menu(menubar, tearoff=0)
        fileMenu.add_command(label="Open Files", command=self.open)
        fileMenu.add_command(label="Save Results", command=self.save)
        fileMenu.add_command(label="Help", command=self.help)
        fileMenu.add_command(label="Print File List (DEBUG)", command=self.printFileList)

        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Clear List", command=self.clearList)
        editmenu.add_command(label="Check Spiral Accuracy", command=self.checkSpiral)
        editmenu.add_command(label="Check Elliptical Accuracy", command=self.checkElliptical)
        editmenu.add_command(label="Check Spiral Accuracy (K-Means)", command=self.checkSpiralK)
        editmenu.add_command(label="Check Elliptical Accuracy (K-Means)", command=self.checkEllipticalK)
        editmenu.add_command(label="Close application", command=sys.exit)

        toolsmenu = Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Generate Histogram(Current Image)", command=self.init)
        toolsmenu.add_command(label="Generate Colour Histogram(Current Image)", command=self.colourHist)
        toolsmenu.add_command(label="Generate Default Classification Details(Current Image)", command=self.detect)
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Perform Contour Tracing On Current Image (Colour)", command=self.contourColour)
        toolsmenu.add_command(label="Perform Template Matching On Current Image (Colour)", command=self.templateMatch)
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Perform Edge Detection On Current Image", command=self.cannyEdge)
        toolsmenu.add_command(label="Fill Edge Detection On Current Image", command=self.fillCanny)
        toolsmenu.add_command(label="Perform Edge Detection Inside Template Match On Current Image", command=self.cannyInsideTemplate)
        toolsmenu.add_command(label="Count Number Of Contours Inside Template Area On Current Image", command=self.contourCountTemplateArea)
        toolsmenu.add_command(label="Estimate Average Contour Count", command=self.contourEstimate)
        toolsmenu.add_command(label="Classify Galaxy Based On Average Contour Count", command=self.classifyOverAverage)

        toolsmenu.add_separator()
        toolsmenu.add_command(label="Peform Circle Detection on current image", command=self.houghCircle)
        toolsmenu.add_command(label="Peform HOG current image", command=self.hog)
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Perform K-Means Quantization on image", command=self.kmeans)
        toolsmenu.add_command(label="Contour K-Means image", command=self.contourK)
        toolsmenu.add_command(label="Calculate average colour of K-means", command=self.kColourMean)



        menubar.add_cascade(label="File", menu=fileMenu)
        menubar.add_cascade(label="Edit", menu=editmenu)
        menubar.add_cascade(label="Tools", menu=toolsmenu)

        root.config(menu=menubar)

        clear = Button(self, text="Clear List", command=self.clearList, background="#666", fg="white")
        clear.pack()
        clear.place(relx=.75, rely=.65, anchor="center")


    # -------------------------------------------------------------------------------------------------------------------
    # Initialising frame
    # -------------------------------------------------------------------------------------------------------------------
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack(fill=BOTH, expand=1)
        global runonce
        runonce = 0
        global listbox
        listbox = Listbox(self, width=100, background="#666", fg="white")
        global fileList
        fileList = []
        global contourEstimateCount
        contourEstimateCount = 0
        self.createWidgets()



# -------------------------------------------------------------------------------------------------------------------
# Assigning name to frame
# -------------------------------------------------------------------------------------------------------------------
root = Tk()
root.resizable(False,False)
app = App(master=root)
root.minsize(width=1200, height=800)
Style().configure(root, background="#666")
classification = Tool.classification()
redundant = Tool.redundantClassification()
hist = Tool.histograms()
app.mainloop()
root.destroy()