import cv2
import mediapipe as mp

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Mở webcam
cap = cv2.VideoCapture(0)

# Định nghĩa màu sắc dựa trên số ngón tay giơ lên
colors = [
    (0, 0, 255),   # 0 ngón - Đỏ
    (0, 165, 255), # 1 ngón - Cam
    (0, 255, 255), # 2 ngón - Vàng
    (0, 255, 0),   # 3 ngón - Lục
    (255, 0, 0),   # 4 ngón - Lam
    (128, 0, 128)  # 5 ngón - Tím
]

def count_fingers(hand_landmarks):
    """Hàm đếm số ngón tay giơ lên"""
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Điểm đầu ngón tay
    
    # Ngón cái
    if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Các ngón khác
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return sum(fingers)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Lật khung hình
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    num_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            num_fingers = count_fingers(hand_landmarks)
    
    # Hiển thị số ngón tay giơ lên
    cv2.putText(frame, f'Fingers: {num_fingers}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị bóng đèn với màu tương ứng
    color = colors[min(num_fingers, 5)]  # Lấy màu theo số ngón tay
    cv2.circle(frame, (frame.shape[1] - 50, 50), 30, color, -1)
    
    # Hiển thị khung hình
    cv2.imshow("Hand Detection", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
