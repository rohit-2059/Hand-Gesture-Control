import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=2
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Configuration parameters
SCROLL_AMOUNT = 40
SCROLL_REPETITIONS = 3
volume_control_delay = 0.5
last_volume_event_time = 0
VOLUME_PRESS_COUNT = 5

# Enhanced cursor control parameters
CURSOR_SMOOTHING = 0.7
CURSOR_SPEED = 1.8
PINCH_THRESHOLD = 0.04
CLICK_COOLDOWN = 0.1
last_click_time = 0

# New gesture detection parameters
THUMB_ANGLE_THRESHOLD = 60  # Degrees for thumb up/down detection
FINGER_DISTANCE_THRESHOLD = 0.1  # Distance between fingers for volume gesture

# Variables for cursor control
screen_width, screen_height = pyautogui.size()
cursor_x, cursor_y = pyautogui.position() 
prev_index_positions = []
POSITION_HISTORY = 5

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y])
    
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def is_volume_gesture(landmarks):
    """
    Enhanced volume gesture detection:
    - Thumb must be clearly up/down
    - Other fingers must be closed
    - Hand must be in vertical orientation
    """
    # Check thumb position relative to palm
    thumb_angle = calculate_angle(
        landmarks[4],  # Thumb tip
        landmarks[2],  # Thumb base
        landmarks[0]   # Wrist
    )
    
    # Check if other fingers are closed
    fingers_closed = all(
        landmarks[tip].y > landmarks[pip].y
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]  # finger tips and pips
    )
    
    # Check if thumb is clearly extended
    thumb_extended = abs(landmarks[4].x - landmarks[2].x) > FINGER_DISTANCE_THRESHOLD
    
    # Check if hand is in vertical orientation
    vertical_orientation = abs(landmarks[9].x - landmarks[0].x) < 0.1  # Middle finger base to wrist
    
    return thumb_extended and fingers_closed and vertical_orientation

def moving_average(positions):
    """Calculate moving average of positions for smoother cursor movement."""
    if not positions:
        return None
    return np.mean(positions, axis=0)

def is_palm_open(landmarks):
    """Check if the hand is open (fingers spread apart)."""
    thumb_dist = abs(landmarks[4].x - landmarks[0].x)
    pinky_dist = abs(landmarks[20].x - landmarks[0].x)
    return thumb_dist > 0.1 and pinky_dist > 0.1

def is_thumb_up(landmarks):
    """Enhanced thumb up detection."""
    if not is_volume_gesture(landmarks):
        return False
    return landmarks[4].y < landmarks[3].y

def is_thumb_down(landmarks):
    """Enhanced thumb down detection."""
    if not is_volume_gesture(landmarks):
        return False
    return landmarks[4].y > landmarks[3].y

def get_pinch_distance(landmarks):
    """Get normalized distance between thumb and index finger."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return ((thumb_tip.x - index_tip.x) ** 2 + 
            (thumb_tip.y - index_tip.y) ** 2 + 
            (thumb_tip.z - index_tip.z) ** 2) ** 0.5

def is_pinch_gesture(landmarks):
    """Enhanced pinch detection with z-axis consideration."""
    distance = get_pinch_distance(landmarks)
    return distance < PINCH_THRESHOLD

def handle_click():
    """Handle click with cooldown."""
    global last_click_time
    current_time = time.time()
    
    if current_time - last_click_time >= CLICK_COOLDOWN:
        pyautogui.click()
        last_click_time = current_time
        return True
    return False
# [Previous imports and initial setup remain the same until cursor control function]

def control_cursor(hand_landmarks):
    """Enhanced cursor control with improved smoothing and boundary protection."""
    global cursor_x, cursor_y, prev_index_positions
    
    # Only control cursor if not in volume gesture
    if not is_volume_gesture(hand_landmarks.landmark):
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[8]
        
        # Convert to screen coordinates with speed multiplier
        target_x = int(screen_width * (1 - index_tip.x) * CURSOR_SPEED)
        target_y = int(screen_height * index_tip.y * CURSOR_SPEED)
        
        # Add safety margins to prevent triggering failsafe
        MARGIN = 5  # pixels from screen edge
        target_x = max(MARGIN, min(target_x, screen_width - MARGIN))
        target_y = max(MARGIN, min(target_y, screen_height - MARGIN))
        
        # Add current position to history
        prev_index_positions.append([target_x, target_y])
        if len(prev_index_positions) > POSITION_HISTORY:
            prev_index_positions.pop(0)
        
        # Calculate smoothed position
        smoothed_position = moving_average(prev_index_positions)
        if smoothed_position is not None:
            # Apply additional smoothing between current and target position
            cursor_x = int(cursor_x + (smoothed_position[0] - cursor_x) * CURSOR_SMOOTHING)
            cursor_y = int(cursor_y + (smoothed_position[1] - cursor_y) * CURSOR_SMOOTHING)
            
            # Apply safety margins again after smoothing
            cursor_x = max(MARGIN, min(cursor_x, screen_width - MARGIN))
            cursor_y = max(MARGIN, min(cursor_y, screen_height - MARGIN))
            
            try:
                pyautogui.moveTo(cursor_x, cursor_y)
            except pyautogui.FailSafeException:
                # If failsafe triggers, move cursor to a safe position
                cursor_x = screen_width // 2
                cursor_y = screen_height // 2
                pyautogui.moveTo(cursor_x, cursor_y)

# [Rest of the code remains the same]

def handle_media_controls(hand_landmarks):
    """Handle media playback controls."""
    if is_palm_open(hand_landmarks.landmark):
        pyautogui.press('space')
        return "Play/Pause"
    elif is_pinch_gesture(hand_landmarks.landmark):
        if handle_click():
            return "Click"
    return ""

def adjust_volume(hand_landmarks, hand_label):
    """Enhanced volume control with strict gesture detection."""
    global last_volume_event_time
    current_time = time.time()
    
    if current_time - last_volume_event_time < volume_control_delay:
        return ""

    if hand_label == "Right" and is_volume_gesture(hand_landmarks.landmark):
        if is_thumb_up(hand_landmarks.landmark):
            for _ in range(VOLUME_PRESS_COUNT):
                pyautogui.press('volumeup')    
            last_volume_event_time = current_time
            return "Volume Up"
        elif is_thumb_down(hand_landmarks.landmark):
            for _ in range(VOLUME_PRESS_COUNT):
                pyautogui.press('volumedown')  
            last_volume_event_time = current_time
            return "Volume Down"
    return ""

def handle_scroll(hand_landmarks, hand_label):
    """Handle scrolling."""
    if hand_label == "Left":
        if is_thumb_up(hand_landmarks.landmark):
            for _ in range(SCROLL_REPETITIONS):
                pyautogui.scroll(SCROLL_AMOUNT)
            return "Scroll Up"
        elif is_thumb_down(hand_landmarks.landmark):
            for _ in range(SCROLL_REPETITIONS):
                pyautogui.scroll(-SCROLL_AMOUNT)
            return "Scroll Down"
    return ""

# Safety feature to prevent PyAutoGUI from failing
pyautogui.FAILSAFE = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    action_message = ""

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_label = hand_info.classification[0].label

            # Handle cursor control with right hand
            if hand_label == "Right":
                control_cursor(hand_landmarks)
                media_action = handle_media_controls(hand_landmarks)
                if media_action:
                    action_message = media_action
                volume_action = adjust_volume(hand_landmarks, hand_label)
                if volume_action:
                    action_message = volume_action
   
            # Handle scrolling with left hand
            if hand_label == "Left":
                scroll_action = handle_scroll(hand_landmarks, hand_label)
                if scroll_action:
                    action_message = scroll_action

    if action_message:
        cv2.putText(frame, action_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()