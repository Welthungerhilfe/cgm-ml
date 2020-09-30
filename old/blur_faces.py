import os
import shutil
import cv2
import face_recognition


def blur_faces(filepath):
	'''
	Function to help with face blur
	@params: path of image
	@returns: processed image with face blurred
	'''
    image=cv2.imread(filepath)[:,:,::-1]
    face_locations = []
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract the region of the image that contains the face
        face_image = image[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        image[top:bottom, left:right] = face_image

    return image[:,:,::-1]


def process_files():
	'''
	Main runner code.
	- Copy data while maintaing the original directory structure
	- Remove personal info.
	'''
	inputpath = "/whhdata/qrcode"
	for dirpath, dirnames, filenames in os.walk(inputpath):
	    structure = "/localssd/anondata/qrcode" + dirpath[len(inputpath):]

	    if not os.path.isdir(structure):
	        os.mkdir(structure)
	    else:
	        # Folder does already exits!
	        continue

	    for file in filenames:
	        if "measurements" in dirpath:
	            if file.endswith(".jpg") or file.endswith(".png"):
	                processed_image = blur_faces(dirpath + "/" + file)
	                cv2.imwrite(structure + "/" + file, processed_image)
	                continue
	        shutil.copy2(dirpath + "/" + file, structure + "/" + file)

if __name__ == '__main__':
	process_files()
