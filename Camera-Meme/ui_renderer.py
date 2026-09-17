import os
import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

import config


class UIRenderer:
    """
    คลาสสำหรับจัดการการแสดงผลกราฟิกและหน้าจอทั้งหมด:
    - โหลดและแคชรูปภาพมีม
    - วาด Hand Skeleton บนจอซ้าย (Webcam)
    - สร้างจอขวาแบบ Clean Minimal ไร้ตัวหนังสือ
    - รวมเป็น Split Screen (ซ้าย-ขวา)
    """

    COLOR_MAP = {
        "peace": (0, 255, 128),  # เขียวนีออน
        "self": (50, 200, 255),  # ฟ้านีออน
        "head": (0, 215, 255),   # เหลืองทอง
        None: (0, 200, 255),     # ฟ้าอมเขียว (ค่าปกติ)
    }

    STATUS_TEXT_MAP = {
        "peace": "CAMERA | [PEACE OUT] 2 FINGERS",
        "self": "CAMERA | [WHO, ME?!] POINT AT SELF",
        "head": "CAMERA | [THINK ABOUT IT] POINT TO HEAD",
        None: "CAMERA | READY",
    }

    def __init__(self):
        self.memes = self._load_memes()

    def _load_memes(self):
        """โหลดรูปภาพมีมทั้งหมดขึ้นหน่วยความจำล่วงหน้า"""
        memes = {}
        for gesture_name, path in config.MEME_PATHS.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                memes[gesture_name] = img
            else:
                print(f"[UIRenderer] คำเตือน: ไม่พบไฟล์รูป {path}")
                memes[gesture_name] = None
        return memes

    def _draw_hand_skeleton(self, frame, hand_landmarks, active_gesture):
        """วาดจุดและเส้นโครงกระดูกมือบนหน้าจอกล้องฝั่งซ้าย"""
        if not hand_landmarks:
            return

        h, w, _ = frame.shape
        line_color = self.COLOR_MAP.get(active_gesture, self.COLOR_MAP[None])

        for landmarks in hand_landmarks:
            # วาดเส้นเชื่อมต่อนิ้วมือ
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                pt1 = (int(landmarks[conn.start].x * w), int(landmarks[conn.start].y * h))
                pt2 = (int(landmarks[conn.end].x * w), int(landmarks[conn.end].y * h))
                cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

            # วาดจุดข้อต่อนิ้ว
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)

    def _draw_camera_status(self, frame, active_gesture):
        """แสดงแถบสถานะการตรวจจับเล็กๆ ที่มุมซ้ายบนของกล้อง"""
        status_text = self.STATUS_TEXT_MAP.get(active_gesture, self.STATUS_TEXT_MAP[None])
        status_color = self.COLOR_MAP.get(active_gesture, (200, 200, 200))

        # พื้นหลังป้ายสถานะแบบกึ่งโปร่งใส
        cv2.putText(
            frame,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

    def _render_right_panel(self, w, h, active_gesture):
        """
        สร้างจอฝั่งขวา:
        - เมื่อตรวจพบท่าทาง: ขยายรูปมีมตรงกลางแบบเต็มจอ สวยงาม คมชัด
        - ไม่มีตัวหนังสือหรือกรอบข้อความใดๆ รบกวนภาพ (Clean & Minimal)
        - เมื่อไม่มีท่าทาง: เป็นหน้าจอสีดำสนิท
        """
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        img = self.memes.get(active_gesture)

        if img is not None:
            img_h, img_w = img.shape[:2]

            # คำนวณสเกลให้พอดีกับจอขวาแบบเต็มพื้นที่โดยคงสัดส่วนเดิม (Aspect Ratio)
            scale = min(w / img_w, h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            resized = cv2.resize(img, (new_w, new_h))

            # วางให้อยู่กึ่งกลางหน้าจอ
            pos_x = (w - new_w) // 2
            pos_y = (h - new_h) // 2

            # รองรับทั้งภาพ PNG มี Alpha และภาพ JPG ปกติ
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    panel[pos_y : pos_y + new_h, pos_x : pos_x + new_w, c] = (
                        (1.0 - alpha) * panel[pos_y : pos_y + new_h, pos_x : pos_x + new_w, c]
                        + alpha * resized[:, :, c]
                    )
            else:
                panel[pos_y : pos_y + new_h, pos_x : pos_x + new_w] = resized[:, :, :3]

        return panel

    def render(self, frame, active_gesture, hand_landmarks):
        """
        รวมการแสดงผลทั้งหมด:
        1. วาดโครงกระดูกและสถานะบนเฟรมกล้องซ้าย
        2. สร้างจอขวาสำหรับแสดงรูปมีม
        3. ต่อเป็นหน้าจอ Split Screen
        """
        # วาดบนจอซ้าย
        self._draw_hand_skeleton(frame, hand_landmarks, active_gesture)
        self._draw_camera_status(frame, active_gesture)

        # สร้างจอขวา
        h, w, _ = frame.shape
        right_panel = self._render_right_panel(w, h, active_gesture)

        # รวมเป็น Split Screen แนวนอน
        return np.hstack([frame, right_panel])
