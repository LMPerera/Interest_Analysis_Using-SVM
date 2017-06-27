import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from scipy.ndimage import zoom
from sklearn import datasets


#Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))


# Detect a face in the input image using 'haarcascade_frontalface_default.xml'
def detectFaces(frame):
    cascPath = "/home/madushika/Documents/mywork/Intrest_Analysis_Using SVM/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(75, 75),
            flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, detected_faces


# Stretch and crop with set values to match the approximate format of the faces from the dataset
def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]

    #transform the extracted image
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                               64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

# Predicts if an extracted face is interesting or not
def predict_face_is_interesting(extracted_face):
    return True if svc_1.predict(extracted_face.reshape(1, -1)) else False


if __name__ == "__main__":

    # Initializing Classifier
    svc_1 = SVC(kernel='linear')

    # load the dataset
    faces_dataset = datasets.fetch_olivetti_faces()

    results = {}
    index = 0

    # Loading the classification result
    results = json.load(open("/home/madushika/Documents/mywork/Intrest_Analysis_Using SVM/results/results.xml"))

    # Building the dataset
    indices = [int(i) for i in results]

    # Image Data
    data = faces_dataset.data[indices, :]

    # Target Vector
    target = [results[i] for i in results]
    target = np.array(target).astype(np.int32)

    # Train the classifier using 5 fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

    # Matrix and Results
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detectFaces(frame)

        face_index = 0

        cv2.putText(frame, "Press Esc to QUIT", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # extract features
                extracted_face = extract_face_features(gray, face, (0.3, 0.05))

                # predict interest
                prediction_result = predict_face_is_interesting(extracted_face)

                # draw extracted face in the top right corner
                frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

                # annotate main image with a label
                if prediction_result is True:
                    cv2.putText(frame, "Interesting",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)
                else:
                    cv2.putText(frame, "Not Interesting",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 5)

                # increment counter
                face_index += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

