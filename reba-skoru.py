import cv2
import mediapipe as mp
import math
from enum import Enum

# Mediapipe Pose sınıfını yükle
mp_pose = mp.solutions.pose

# Video akışını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera


class landmark_index(Enum):
    NOSE = 0
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_HIP = 3
    RIGHT_HIP = 4
    LEFT_ELBOW = 5
    RIGHT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_WRIST = 8
    LEFT_KNEE = 9
    RIGHT_KNEE = 10
    LEFT_ANKLE = 11
    RIGHT_ANKLE = 12


class landmark_enum(Enum):
    NOSE = mp_pose.PoseLandmark.NOSE
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
    LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE
    RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE


def getNeckAngle(neck_angle):
    reba_score = 0
    if neck_angle:
        if neck_angle < 10:
            reba_score += 0
        elif neck_angle < 20:
            reba_score += 1
        elif neck_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score


# PoseNet modelini yükle
def getTorsoAngle(torso_angle):
    reba_score = 0
    if torso_angle:
        if torso_angle < 10:
            reba_score += 0
        elif torso_angle < 20:
            reba_score += 1
        elif torso_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score


def getLeftLegAngle(left_leg_angle):
    reba_score = 0
    if left_leg_angle:
        if left_leg_angle < 10:
            reba_score += 0
        elif left_leg_angle < 20:
            reba_score += 1
        elif left_leg_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score


def getRightLegAngle(right_leg_angle):
    reba_score = 0
    if right_leg_angle:
        if right_leg_angle < 10:
            reba_score += 0
        elif right_leg_angle < 20:
            reba_score += 1
        elif right_leg_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score


def getLeftUpperArmAngle(left_upper_arm_angle):
    reba_score = 0
    if left_upper_arm_angle:
        if left_upper_arm_angle < 10:
            reba_score += 0
        elif left_upper_arm_angle < 20:
            reba_score += 1
        elif left_upper_arm_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

def getRightUpperArmAngle(right_upper_arm_angle):
    reba_score = 0
    if right_upper_arm_angle:
        if right_upper_arm_angle < 10:
            reba_score += 0
        elif right_upper_arm_angle < 20:
            reba_score += 1
        elif right_upper_arm_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

def getLeftLowerArmAngle(left_lower_arm_angle):
    reba_score = 0
    if left_lower_arm_angle:
        if left_lower_arm_angle < 10:
            reba_score += 0
        elif left_lower_arm_angle < 20:
            reba_score += 1
        elif left_lower_arm_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

def getRightLowerArmAngle(right_lower_arm_angle):
    reba_score = 0
    if right_lower_arm_angle:
        if right_lower_arm_angle < 10:
            reba_score += 0
        elif right_lower_arm_angle < 20:
            reba_score += 1
        elif right_lower_arm_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

def getLeftWristAngle(left_wrist_angle):
    reba_score = 0
    if left_wrist_angle:
        if left_wrist_angle < 10:
            reba_score += 0
        elif left_wrist_angle < 20:
            reba_score += 1
        elif left_wrist_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

def getRightWristAngle(right_wrist_angle):
    reba_score = 0
    if right_wrist_angle:
        if right_wrist_angle < 10:
            reba_score += 0
        elif right_wrist_angle < 20:
            reba_score += 1
        elif right_wrist_angle < 30:
            reba_score += 2
        else:
            reba_score += 3
    return reba_score

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera bağlantısı başarısız.")
            break

        # Görüntüyü BGR'den RGB'ye dönüştür
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe Pose analizini uygula
        results = pose.process(image_rgb)
        if results.pose_landmarks is not None:

            # PoseLandmark noktalarını al
            landmarks = []
            for l in landmark_enum:
                if results.pose_landmarks is not None:
                    landmark = results.pose_landmarks.landmark[l.value]
                    landmarks.append(landmark)
                else:
                    landmarks.append(None)

            # Duruş açılarını hesapla
            neck_angle = None
            torso_angle = None
            left_leg_angle = None
            right_leg_angle = None
            left_upper_arm_angle = None
            right_upper_arm_angle = None
            left_lower_arm_angle = None
            right_lower_arm_angle = None
            left_wrist_angle = None
            right_wrist_angle = None

            if all(landmarks):
                # Boyun açısını hesapla
                neck_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_SHOULDER.value].y - landmarks[landmark_index.LEFT_SHOULDER.value].y,
                                                    landmarks[landmark_index.RIGHT_SHOULDER.value].x - landmarks[landmark_index.LEFT_SHOULDER.value].x))

                # Gövde açısını hesapla
                torso_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_HIP.value].y - landmarks[landmark_index.LEFT_HIP.value].y,
                                                    landmarks[landmark_index.RIGHT_HIP.value].x - landmarks[landmark_index.LEFT_HIP.value].x))

                # Sol bacak açısını hesapla
                left_leg_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_KNEE.value].y - landmarks[landmark_index.LEFT_HIP.value].y,
                                                        landmarks[landmark_index.RIGHT_KNEE.value].x - landmarks[landmark_index.LEFT_HIP.value].x))

                # Sağ bacak açısını hesapla
                right_leg_angle = math.degrees(math.atan2(landmarks[landmark_index.LEFT_KNEE.value].y - landmarks[landmark_index.RIGHT_HIP.value].y,
                                                        landmarks[landmark_index.LEFT_KNEE.value].x - landmarks[landmark_index.RIGHT_HIP.value].x))

                # Sol üst kol açısını hesapla
                left_upper_arm_angle = math.degrees(math.atan2(landmarks[landmark_index.LEFT_ELBOW.value].y - landmarks[landmark_index.LEFT_SHOULDER.value].y,
                                                                landmarks[landmark_index.LEFT_ELBOW.value].x - landmarks[landmark_index.LEFT_SHOULDER.value].x) -
                                                    math.atan2(landmarks[landmark_index.LEFT_WRIST.value].y - landmarks[landmark_index.LEFT_ELBOW.value].y,
                                                                landmarks[landmark_index.LEFT_WRIST.value].x - landmarks[landmark_index.LEFT_ELBOW.value].x))

                # Sağ üst kol açısını hesapla
                right_upper_arm_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_ELBOW.value].y - landmarks[landmark_index.RIGHT_SHOULDER.value].y,
                                                                landmarks[landmark_index.RIGHT_ELBOW.value].x - landmarks[landmark_index.RIGHT_SHOULDER.value].x) -
                                                    math.atan2(landmarks[landmark_index.RIGHT_WRIST.value].y - landmarks[landmark_index.RIGHT_ELBOW.value].y,
                                                                landmarks[landmark_index.RIGHT_WRIST.value].x - landmarks[landmark_index.RIGHT_ELBOW.value].x))

                # Sol alt kol açısını hesapla
                left_lower_arm_angle = math.degrees(math.atan2(landmarks[landmark_index.LEFT_WRIST.value].y - landmarks[landmark_index.LEFT_ELBOW.value].y,
                                                                landmarks[landmark_index.LEFT_WRIST.value].x - landmarks[landmark_index.LEFT_ELBOW.value].x) -
                                                    math.atan2(landmarks[landmark_index.LEFT_ELBOW.value].y - landmarks[landmark_index.LEFT_SHOULDER.value].y,
                                                                landmarks[landmark_index.LEFT_ELBOW.value].x - landmarks[landmark_index.LEFT_SHOULDER.value].x))

                # Sağ alt kol açısını hesapla
                right_lower_arm_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_WRIST.value].y - landmarks[landmark_index.RIGHT_ELBOW.value].y,
                                                                landmarks[landmark_index.RIGHT_WRIST.value].x - landmarks[landmark_index.RIGHT_ELBOW.value].x) -
                                                    math.atan2(landmarks[landmark_index.RIGHT_ELBOW.value].y - landmarks[landmark_index.RIGHT_SHOULDER.value].y,
                                                                landmarks[landmark_index.RIGHT_ELBOW.value].x - landmarks[landmark_index.RIGHT_SHOULDER.value].x))

                # Sol bilek açısını hesapla
                left_wrist_angle = math.degrees(math.atan2(landmarks[landmark_index.LEFT_WRIST.value].y - landmarks[landmark_index.LEFT_ELBOW.value].y,
                                                        landmarks[landmark_index.LEFT_WRIST.value].x - landmarks[landmark_index.LEFT_ELBOW.value].x) -
                                                math.atan2(landmarks[landmark_index.LEFT_WRIST.value].y - landmarks[landmark_index.LEFT_ELBOW.value].y,
                                                            landmarks[landmark_index.LEFT_WRIST.value].x - landmarks[landmark_index.LEFT_ELBOW.value].x))

                # Sağ bilek açısını hesapla
                right_wrist_angle = math.degrees(math.atan2(landmarks[landmark_index.RIGHT_WRIST.value].y - landmarks[landmark_index.RIGHT_ELBOW.value].y,
                                                            landmarks[landmark_index.RIGHT_WRIST.value].x - landmarks[landmark_index.RIGHT_ELBOW.value].x) -
                                                math.atan2(landmarks[landmark_index.RIGHT_WRIST.value].y - landmarks[landmark_index.RIGHT_ELBOW.value].y,
                                                            landmarks[landmark_index.RIGHT_WRIST.value].x - landmarks[landmark_index.RIGHT_ELBOW.value].x))
            reba_score = 0

            # REBA puanını hesapla
            reba_score += getNeckAngle(neck_angle)

            reba_score += getTorsoAngle(torso_angle)

            reba_score +=  getLeftLegAngle(left_leg_angle)

            reba_score +=  getRightLegAngle(right_leg_angle)

            reba_score +=  getLeftUpperArmAngle(left_upper_arm_angle)

            reba_score +=  getRightUpperArmAngle(right_upper_arm_angle)

            reba_score += getLeftLowerArmAngle(left_lower_arm_angle)

            reba_score +=  getRightLowerArmAngle(right_lower_arm_angle)

            reba_score += getLeftWristAngle(left_wrist_angle)
            # Sağ Bilek Açısı
            reba_score += getRightWristAngle(right_wrist_angle)

            # Ekrana yazdır
            cv2.putText(image, f"Boyun Açısı: {neck_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Gövde Açısı: {torso_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sol Bacak Açısı: {left_leg_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sağ Bacak Açısı: {right_leg_angle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sol Üst Kol Açısı: {left_upper_arm_angle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sağ Üst Kol Açısı: {right_upper_arm_angle:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sol Alt Kol Açısı: {left_lower_arm_angle:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sağ Alt Kol Açısı: {right_lower_arm_angle:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sol Bilek Açısı: {left_wrist_angle:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Sağ Bilek Açısı: {right_wrist_angle:.2f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.putText(image, f"REBA Puanı: {reba_score}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # Görüntüyü göster
            cv2.imshow("PEBA Skoru", image)

        # Çıkış için 'q' tuşuna basıldığında döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()
