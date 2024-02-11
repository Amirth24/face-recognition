import numpy as np
import cv2
from keras_facenet import FaceNet

face_locator = cv2.CascadeClassifier(
        'model/haarcascade_frontalface_default.xml')
# Load Model
embedder = FaceNet()
cap = cv2.VideoCapture(0)

enrolled_faces = []


def process_embedd(embedd):
    global enrolled_faces

    if len(enrolled_faces) == 0:
        enrolled_faces = [embedd]
        print("Face 0 Enrolled")
        return 0
    max_sim = -1.0
    max_idx = -1
    for i, face_embedd in enumerate(enrolled_faces):
        sim = np.dot(embedd, face_embedd.T)
        if sim > max_sim:
            max_sim = sim
            max_idx = i
    if max_idx != -1 and max_sim > 0.6:
        print("Found Face ", max_idx, "with", max_sim.squeeze())
    if max_sim <= 0.2:
        print(f"face {len(enrolled_faces)+1} enrolled")
        enrolled_faces.append(embedd)
    return max_idx


def resize(img, shape=(160, 160)):
    return cv2.resize(
        img,
        shape,
        interpolation=cv2.INTER_AREA if img.shape < shape else cv2.INTER_CUBIC)


COLORS = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (120, 230, 120),
    (255, 0, 255),
]
classifier_config = {
    "scaleFactor": 1.2,
    "minNeighbors": 10,
    "minSize": (15, 15),
    "flags": cv2.CASCADE_SCALE_IMAGE
}

while True:
    ok, img = cap.read()
    if not ok:
        print('Not OK')
        cap.release()
        cv2.destroyAllWindows()
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_locator.detectMultiScale(gray, **classifier_config)

    for (x, y, w, h) in faces:
        w_rm = int(0.3 * w / 2)
        face_img = img[y: y + h, x + w_rm: x + w - w_rm]
        face_img = resize(face_img)
        embedd = embedder.embeddings([face_img])
        face_idx = process_embedd(embedd)
        cv2.rectangle(img, (x, y), (x+w, y+h), COLORS[face_idx % len(COLORS)])

    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
