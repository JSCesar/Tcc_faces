import cv2
import imutils

def detect_face(image_path):
    cascPath = '/home/jscesar/face_detect/haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=300, height=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.09
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = image[y:y + h, x:x + w]
        break

    if len(image) > 0:
        return image

def rotate_images(image_path, destination):
    print(image_path)
    path = destination
    image = cv2.imread(image_path)
    im90 = imutils.rotate(image, 90)
    im180 = imutils.rotate(image, 180)
    im270 = imutils.rotate(image, 270)

    destination = destination.replace('.jpg', '_rotate90.jpg')
    cv2.imwrite(destination, im90)
    destination = path
    destination =  destination.replace('.jpg', '_rotate180.jpg')
    cv2.imwrite(destination, im180)
    destination= path
    destination = destination.replace('.jpg', '_rotate270.jpg')
    cv2.imwrite(destination, im270)
    destination = path
    cv2.imwrite(destination, image)