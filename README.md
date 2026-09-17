# Camera-Meme 🎥

โปรเจกต์ทดลองที่สร้างขึ้นเพื่อศึกษา **Computer Vision และ Hand Gesture Recognition** โดยใช้ Webcam ตรวจจับมือและท่าทางแบบ Real-time

แนวคิดของโปรเจกต์คือ เมื่อกล้องตรวจจับ Gesture ที่กำหนดไว้ได้ ระบบจะแสดงรูป Meme ที่เกี่ยวข้องขึ้นมาทางด้านขวาของหน้าจอ ทำให้การทดลองกับ Computer Vision มีความ Interactive มากขึ้น

โปรเจกต์นี้ทำขึ้นเพื่อการเรียนรู้ โดยมีการใช้ AI ช่วยในบางส่วนของการพัฒนา เช่น การหาแนวทางแก้ปัญหา อธิบายโค้ด และช่วยเขียนโค้ดบางส่วน

## ✨ ตอนนี้ทำอะไรได้บ้าง

ปัจจุบันระบบสามารถตรวจจับ Gesture หลัก ๆ ได้ 3 แบบ

| Gesture             | ผลลัพธ์                    |
| ------------------- | -------------------------- |
| ✌️ Victory / Peace  | แสดง Meme "Peace Out"      |
| 👉 Pointing to Head | แสดง Meme "Think About It" |
| 👉 Pointing to Self | แสดง Meme "Who, me?!"      |

หน้าจอจะแบ่งออกเป็น 2 ส่วน

* **ด้านซ้าย** — ภาพจาก Webcam พร้อม Hand Skeleton และสถานะ Gesture
* **ด้านขวา** — Meme ที่ตรงกับ Gesture ที่ตรวจจับได้

## 🧠 ใช้อะไรในการทำ

* **Python**
* **OpenCV** — ใช้จัดการ Webcam และประมวลผลภาพ
* **MediaPipe** — ใช้ตรวจจับมือและ Gesture
* **MediaPipe Gesture Recognizer** — ใช้โมเดลสำหรับจำแนกท่าทาง
* AI — ใช้เป็นตัวช่วยระหว่างการพัฒนาและศึกษาโค้ด

## 📁 โครงสร้างโปรเจกต์

```text
Camera-Meme/
│
├── config.py
├── gesture_detector.py
├── ui_renderer.py
├── main.py
│
├── gesture_recognizer.task
│
├── meme_peace_out.jpg
├── meme_who_me.jpg
├── meme_think.jpg
│
└── .gitignore
```

### แต่ละไฟล์ทำอะไร?

**`main.py`**

เป็นตัวควบคุมหลักของโปรแกรม

* เปิด Webcam ด้วย OpenCV
* พลิกภาพเป็นแบบ Mirror
* ส่งภาพไปให้ Gesture Detector
* รับผลการตรวจจับและส่งให้ UI Renderer
* รับ Input จาก Keyboard
* กด `q` เพื่อปิดโปรแกรม

**`gesture_detector.py`**

เป็นส่วนที่จัดการเกี่ยวกับการตรวจจับ Gesture

* โหลด MediaPipe Gesture Recognizer
* ตรวจจับมือและท่าทาง
* รองรับ Gesture ที่ใช้ในโปรเจกต์
* มีระบบ Smoothing เพื่อลดการเปลี่ยน Gesture ที่เร็วเกินไป
* ใช้ Rolling Window Majority Voting และ Hold Counter เพื่อช่วยลดอาการกระพริบของผลลัพธ์

**`ui_renderer.py`**

จัดการส่วนที่แสดงผลบนหน้าจอ

* โหลดและ Cache รูป Meme ไว้ใน RAM
* วาด Hand Landmark และ Skeleton
* แสดงสถานะ Gesture บนภาพกล้อง
* จัดตำแหน่งและปรับขนาด Meme
* รวมภาพกล้องและ Meme เป็น Split Screen

**`config.py`**

เก็บค่าต่าง ๆ ที่ใช้ร่วมกันในโปรเจกต์ เช่น

* Camera Index
* Path ของโมเดล
* Path ของ Meme
* ค่า Sensitivity
* ค่า Smoothing
* จำนวน Frame ที่ต้อง Hold Gesture

การแยก Config ออกมาแบบนี้ทำให้สามารถปรับค่าต่าง ๆ ได้โดยไม่ต้องเข้าไปแก้หลายไฟล์

## 🚀 วิธีติดตั้ง

Clone โปรเจกต์ก่อน

```bash
git clone <repository-url>
cd Camera-Meme
```

จากนั้นติดตั้ง Library ที่จำเป็น

```bash
pip install opencv-python mediapipe
```

## ▶️ วิธีรัน

ใช้คำสั่ง

```bash
python main.py
```

หลังจากโปรแกรมเปิดขึ้นมา ให้ลองทำ Gesture หน้ากล้อง

กด

```text
q
```

เพื่อออกจากโปรแกรม

## 🔧 การทำงานโดยรวม

การทำงานของโปรแกรมประมาณนี้:

```text
Webcam
   ↓
OpenCV รับภาพ
   ↓
MediaPipe ตรวจจับมือ
   ↓
Gesture Recognition
   ↓
Smoothing / Debounce
   ↓
ตรวจสอบ Gesture
   ↓
เลือก Meme
   ↓
UI Renderer
   ↓
Split Screen
```

ส่วนที่น่าสนใจของโปรเจกต์ไม่ได้มีแค่การตรวจจับ Gesture แต่ยังลองแก้ปัญหาที่เกิดขึ้นกับการทำงานแบบ Real-time ด้วย เช่น ผล Gesture เปลี่ยนไปมาเร็วเกินไปหรือภาพ Meme กระพริบ

จึงมีการเพิ่ม **Smoothing และ Debounce** เข้ามาช่วยให้ผลลัพธ์นิ่งขึ้น

## 📌 สิ่งที่อยากลองทำต่อ

โปรเจกต์นี้ยังเป็นโปรเจกต์ทดลองและน่าจะสามารถต่อยอดได้อีกหลายอย่าง เช่น

* เพิ่ม Gesture ใหม่
* เพิ่ม Meme ได้มากขึ้น
* ให้ผู้ใช้กำหนด Gesture → Meme เองได้
* เพิ่ม Sound Effect เมื่อเจอ Gesture
* เพิ่ม Animation ตอน Meme ปรากฏ
* ทดลองใช้ Gesture ที่สร้างขึ้นเอง
* ปรับปรุงความเร็วและความแม่นยำของการตรวจจับ
* ทดลองใช้ Computer Vision เทคนิคอื่น ๆ นอกเหนือจาก Gesture Recognition

## 🎯 จุดประสงค์ของโปรเจกต์

โปรเจกต์นี้ไม่ได้ตั้งใจทำให้เป็นระบบใหญ่ แต่ทำขึ้นเพื่อ **ลองเรียนรู้ Computer Vision จากการลงมือทำจริง**

เริ่มจากการเปิดกล้อง → ตรวจจับมือ → จำแนก Gesture → เอาผลที่ได้ไปทำอะไรบางอย่างต่อ

ระหว่างทำก็ได้ลองเจอปัญหาจริง เช่น การตรวจจับที่ไม่นิ่ง การเปลี่ยน Gesture เร็วเกินไป และการจัดการภาพแบบ Real-time ซึ่งเป็นส่วนที่ทำให้ได้เรียนรู้มากกว่าการเขียนโค้ดตามตัวอย่างอย่างเดียว

> **This is a learning project focused on exploring Computer Vision, real-time gesture recognition, and building something fun with it.**
