import cv2


def rotate_image_90(image):
    image = cv2.transpose(image)
    image = cv2.flip(image, flipCode=1)

    return image


def get_nr_of_faces(image):
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = haar_cascade_face.detectMultiScale(image,
                                               scaleFactor=1.1,
                                               minNeighbors=6,
                                               minSize=(140, 140),
                                               flags=cv2.CASCADE_SCALE_IMAGE
                                               )

    print('Faces found: ', len(faces))

    return len(faces)


def find_faces(image, mark_faces=False):
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = haar_cascade_face.detectMultiScale(image,
                                               scaleFactor=1.1,
                                               minNeighbors=6,
                                               minSize=(140, 140),
                                               flags=cv2.CASCADE_SCALE_IMAGE
                                               )

    print('Faces found: ', len(faces))
    #Note: modifying the image with delineation of detected faces
    if mark_faces is True:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 15)

    return image
