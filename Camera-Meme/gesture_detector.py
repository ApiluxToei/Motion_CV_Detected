import os
import urllib.request
from collections import deque, Counter
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


class GestureDetector:
    """
    คลาสสำหรับตรวจจับและวิเคราะห์ท่าทางของมือ (Gesture Detection)
    พร้อมระบบ Temporal Smoothing & Debounce ป้องกันการกระพริบของหน้าจอ
    """

    def __init__(self):
        self._ensure_model_exists()
        self.recognizer = self._init_recognizer()

        # ประวัติท่าทางย้อนหลังสำหรับทำ Majority Voting
        self.history = deque(maxlen=config.SMOOTHING_WINDOW_SIZE)

        # ตัวนับการคงสถานะท่าทาง (Hold Counter) เพื่อกันภาพกระพริบ
        self.active_gesture = None
        self.hold_frames_left = 0

    def _ensure_model_exists(self):
        """ตรวจสอบและดาวน์โหลดโมเดล MediaPipe Task หากยังไม่มีในโฟลเดอร์"""
        if not os.path.exists(config.MODEL_PATH):
            print(f"[GestureDetector] กำลังดาวน์โหลดโมเดล {config.MODEL_PATH}...")
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
            print("[GestureDetector] ดาวน์โหลดโมเดลสำเร็จ!")

    def _init_recognizer(self):
        """กำหนดค่าและสร้างอินสแตนซ์ของ MediaPipe GestureRecognizer"""
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        return vision.GestureRecognizer.create_from_options(options)

    @staticmethod
    def _is_finger_extended(tip, pip, mcp):
        """
        ตรวจสอบการกางของนิ้ว โดยเทียบระยะปลาย-โคน กับ ข้อกลาง-โคน
        ใช้สูตรระยะทางแบบ 3 มิติ (X, Y, Z) เพื่อความทนทานต่อมุมกล้อง
        """
        d_tip_mcp = (tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2 + (tip.z - mcp.z) ** 2
        d_pip_mcp = (pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2 + (pip.z - mcp.z) ** 2
        return d_tip_mcp > d_pip_mcp * 1.3

    def _classify_pointing(self, landmarks):
        """
        จำแนกท่าที่ชี้นิ้ว 1 นิ้ว (ชี้ที่หัว หรือ ชี้เข้าตัว):
        - นิ้วชี้กาง นิ้วกลาง-นาง-ก้อยพับ
        - ถ้าอยู่ระดับศีรษะ (y < 0.45 หรือ wrist.y < 0.50) -> 'head'
        - ถ้าอยู่ระดับกลาง-ล่าง และมีทิศทางชี้เข้าหาลำตัว -> 'self'
        """
        wrist = landmarks[0]
        mcp = landmarks[5]
        tip = landmarks[8]

        index_open = self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5])
        middle_open = self._is_finger_extended(landmarks[12], landmarks[10], landmarks[9])
        ring_open = self._is_finger_extended(landmarks[16], landmarks[14], landmarks[13])
        pinky_open = self._is_finger_extended(landmarks[20], landmarks[18], landmarks[17])

        if not (index_open and not middle_open and not ring_open and not pinky_open):
            return None

        # ท่าที่ 3: ชี้นิ้วที่หัว / ขมับ (ระดับสูง y < 0.45)
        if tip.y < 0.45 or wrist.y < 0.50:
            return "head"

        # ท่าที่ 2: ชี้นิ้วเข้าหาตัวเอง (ชี้เข้าในแกนลึก Z หรือแกนกึ่งกลาง X)
        dx = tip.x - mcp.x
        dz = tip.z - mcp.z
        pointing_towards_body_z = dz > -0.05
        pointing_inward_x = (wrist.x > 0.40 and dx < 0.05) or (wrist.x < 0.60 and dx > -0.05)

        if pointing_towards_body_z or pointing_inward_x:
            return "self"

        return None

    def _get_raw_gesture(self, result):
        """สกัดท่าทางดิบ (Raw Gesture) จากเฟรมปัจจุบัน"""
        if not result.gestures or not result.hand_landmarks:
            return None

        for i, gestures_list in enumerate(result.gestures):
            landmarks = result.hand_landmarks[i]

            # ตรวจสอบท่าชู 2 นิ้ว (Victory)
            for gesture in gestures_list:
                if (
                    gesture.category_name == "Victory"
                    and gesture.score >= config.VICTORY_SCORE_THRESHOLD
                ):
                    return "peace"

            # ตรวจสอบท่าชี้นิ้ว (Head หรือ Self)
            pt_gesture = self._classify_pointing(landmarks)
            if pt_gesture:
                return pt_gesture

        return None

    def _smooth_gesture(self, raw_gesture):
        """
        ระบบเพิ่มความเสถียร (Temporal Smoothing & Debounce):
        1. ใช้ Rolling Window Majority Voting ป้องกัน Noise ชั่วขณะ
        2. ใช้ Hold Counter ยึดรูปไว้ไม่ให้ภาพกระพริบดับหายไปเมื่อแสงหรือกล้องส่าย
        """
        self.history.append(raw_gesture)

        # หาค่าท่าทางที่พบมากที่สุดในหน้าต่างประวัติล่าสุด
        valid_votes = [g for g in self.history if g is not None]

        if valid_votes:
            # ใช้เสียงส่วนใหญ่ที่มากกว่าครึ่งหนึ่งของประวัติ
            counts = Counter(valid_votes)
            most_common, count = counts.most_common(1)[0]
            if count >= 2:  # ยืนยันเมื่อพบอย่างน้อย 2 ใน N เฟรม
                self.active_gesture = most_common
                self.hold_frames_left = config.GESTURE_HOLD_FRAMES
                return self.active_gesture

        # ถ้าในเฟรมปัจจุบันไม่พบท่าทาง แต่ยังมีโควตา Hold Frames เหลืออยู่
        if self.hold_frames_left > 0:
            self.hold_frames_left -= 1
            return self.active_gesture

        # เมื่อหมดเวลา Hold ให้เคลียร์สถานะกลับเป็นปกติ
        self.active_gesture = None
        return None

    def process_frame(self, frame):
        """
        ประมวลผลเฟรมภาพแบบ Real-time:
        คืนค่าเป็น Tuple: (ท่าทางที่ผ่านการกันสั่นแล้ว, ผลลัพธ์ Landmarks ของมือ)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.recognizer.recognize(mp_image)

        raw_gesture = self._get_raw_gesture(result)
        smoothed_gesture = self._smooth_gesture(raw_gesture)

        hand_landmarks = result.hand_landmarks if result.hand_landmarks else []
        return smoothed_gesture, hand_landmarks
