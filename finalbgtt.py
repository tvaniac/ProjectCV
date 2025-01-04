import os
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist 

# Functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance 

# Path ke dataset
dataset_path = './data_cropped/'  # Ubah path ini sesuai dengan lokasi folder dataset Anda
categories = ['Active', 'Sleep', 'Yawn']  # Nama folder yang menjadi kategori

# Load detector and predictor
detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') 

# Ground truth dan prediksi
ground_truth = []
predictions = []

# Thresholds
EYE_AR_THRESH = 0.3
YAWN_THRESH = 20

# Fungsi untuk membaca dataset
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)  # Label berdasarkan folder
    return images, labels

# Baca dataset dan buat ground truth
all_images = []
all_labels = []

for idx, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    images, labels = load_images_from_folder(folder_path, idx)  # 0=active, 1=sleep, 2=yawn
    all_images.extend(images)
    all_labels.extend(labels)

ground_truth = all_labels

# Fungsi evaluasi untuk Haar Cascade
def evaluate_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, _, _ = final_ear(shape)
        distance = lip_distance(shape)

        if ear < EYE_AR_THRESH:
            return 1  # Sleep
        elif distance > YAWN_THRESH:
            return 2  # Yawn
        else:
            return 0  # Active

    return 0  # Default ke Active jika wajah tidak terdeteksi

# Evaluasi dataset
for img in all_images:
    pred = evaluate_image(img)
    predictions.append(pred)

# Hitung metrik evaluasi
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='macro')
recall = recall_score(ground_truth, predictions, average='macro')
f1 = f1_score(ground_truth, predictions, average='macro')
confusion = confusion_matrix(ground_truth, predictions)

# Tampilkan hasil evaluasi
print("### Evaluation Metrics ###")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(confusion)
