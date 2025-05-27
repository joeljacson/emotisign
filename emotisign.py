import cv2
import mediapipe as mp
from deepface import DeepFace
import pyttsx3
import time

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()
else:
    print("✅ Webcam opened successfully")

# Prepare Output Window
cv2.namedWindow("EmotiSign - Gesture + Emotion + Voice", cv2.WINDOW_NORMAL)

last_emotion = ""
last_speak_time = 0

# Emotion-based Messages
emotion_responses = {
    "happy": "You look happy 😊. Enjoy your day!",
    "sad": "You seem a bit sad 😔. Hope things get better soon.",
    "angry": "You look angry 😡. Try taking a deep breath.",
    "surprise": "You look surprised 😮. Something unexpected happened?",
    "fear": "You seem worried 😨. Stay calm, everything's okay.",
    "neutral": "You look calm and neutral 😐. Keep going steady!",
    "disgust": "You look displeased 😣. Maybe something bothered you?"
}

def identify_simple_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]

    # Gesture Conditions
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "Thumbs Up 👍"
    elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
        return "Peace ✌️"
    else:
        return "Hand Detected 🖐️"

print("🎬 Starting EmotiSign... Press 'q' to quit.")

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gesture_text = ""
    # Detect Hand Gestures
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = identify_simple_gesture(hand_landmarks)
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print("🖐️", gesture_text)

    # Analyze Emotion Every 2 Seconds
    current_time = time.time()
    if current_time - last_speak_time > 2:
        try:
            print("🔍 Analyzing emotion...")
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            print(f"😊 Emotion detected: {emotion}")

            if emotion != last_emotion:
                emotion_msg = emotion_responses.get(emotion, f"You look {emotion}")
                combined_msg = f"{gesture_text}. {emotion_msg}"
                print(f"🔊 Speaking: {combined_msg}")
                engine.say(combined_msg)
                engine.runAndWait()
                last_emotion = emotion
                last_speak_time = current_time

            # Display Emotion on Frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, emotion_responses.get(emotion, ""), (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            print("❌ Emotion detection error:", e)
            cv2.putText(frame, "Emotion analysis failed", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show Final Output
    cv2.imshow("EmotiSign - Gesture + Emotion + Voice", frame)

    # Exit Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting EmotiSign...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
