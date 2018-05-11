import cv2
import sys
import numpy as np

def prepare_training_data(data_folder_path):
 
#------STEP-1--------
#get the directories (one directory for each subject) in data folder
dirs = os.listdir(data_folder_path)

#list to hold all subject faces
faces = []
#list to hold labels for all subjects
labels = []

#let's go through each directory and read images within it
for dir_name in dirs:
 
#our subject directories start with letter 's' so
#ignore any non-relevant directories if any
if not dir_name.startswith("s"):
continue;

label = int(dir_name.replace("s", ""))
 
#build path of directory containing images for current subject subject
#sample subject_dir_path = "training-data/s1"
subject_dir_path = data_folder_path + "/" + dir_name
 
#get the images names that are inside the given subject directory
subject_images_names = os.listdir(subject_dir_path)

# Get user supplied values
cap = cv2.VideoCapture(0)
subjects = ["", "Edogawa Conan", "Lisa"]
#cascPath = "haarcascade_frontalface_default.xml"
cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Faces found", frame)



