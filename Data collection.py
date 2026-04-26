import cv2
import numpy as np
import mediapipe as mp
import os
import time

DATA_PATH    = 'DATASET'   # record directly into DATASET
NO_SEQUENCES = 30
ACTIONS = [
    'I LOVE YOU',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
SEQUENCE_LENGTH = 30
COUNTDOWN_SEC   = 2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


def mediapipe_detection(frame, holistic_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic_model.process(rgb)
    rgb.flags.writeable = True
    return results


def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

def draw_landmarks(frame, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw_style
            .get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw_style.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())


def create_folders():
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            path = os.path.join(DATA_PATH, action, str(seq))
            os.makedirs(path, exist_ok=True)
    print(f"[INFO] Folder structure ready under '{DATA_PATH}/'")


def get_existing_sequences(action):
    complete = 0
    for seq in range(NO_SEQUENCES):
        seq_path = os.path.join(DATA_PATH, action, str(seq))
        frames = [f for f in os.listdir(seq_path)
                  if f.endswith('.npy')] if os.path.exists(seq_path) else []
        if len(frames) >= SEQUENCE_LENGTH:
            complete += 1
    return complete


def draw_landmarks(frame, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw_style
            .get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw_style.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())


def create_folders():

    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            path = os.path.join(DATA_PATH, action, str(seq))
            os.makedirs(path, exist_ok=True)
    print(f"[INFO] Folder structure ready under '{DATA_PATH}/'")


def get_existing_sequences(action):
    complete = 0
    for seq in range(NO_SEQUENCES):
        seq_path = os.path.join(DATA_PATH, action, str(seq))
        frames = [f for f in os.listdir(seq_path)
                  if f.endswith('.npy')] if os.path.exists(seq_path) else []
        if len(frames) >= SEQUENCE_LENGTH:
            complete += 1
    return complete


def draw_ui(frame, action, sequence, frame_num, state, total_actions, action_idx):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"Sign: {action}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)

    cv2.putText(frame, f"Action {action_idx + 1}/{total_actions}",
                (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.putText(frame, f"Seq {sequence + 1}/{NO_SEQUENCES}",
                (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    if state == 'WAITING':
        msg = "Press SPACE to record"
        color = (0, 215, 255)
    elif state == 'COUNTDOWN':
        msg = f"Get ready..."
        color = (0, 165, 255)
    elif state == 'RECORDING':
        msg = f"RECORDING  Frame {frame_num + 1}/{SEQUENCE_LENGTH}"
        color = (0, 60, 220)
        # Red recording dot
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 220), -1)
    elif state == 'DONE':
        msg = "All sequences collected!"
        color = (0, 200, 100)
    else:
        msg = ""
        color = (255, 255, 255)

    cv2.putText(frame, msg, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    if state == 'RECORDING':
        bar_w = w - 30
        filled = int(bar_w * (frame_num / SEQUENCE_LENGTH))
        cv2.rectangle(frame, (15, h - 20), (15 + bar_w, h - 8), (80, 80, 80), -1)
        cv2.rectangle(frame, (15, h - 20), (15 + filled, h - 8), (0, 60, 220), -1)

    cv2.putText(frame, "Q: Quit", (15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


def main():
    create_folders()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[INFO] Data Collection Started")
    print(f"       Actions : {ACTIONS}")
    print(f"       Sequences: {NO_SEQUENCES} x {SEQUENCE_LENGTH} frames each")
    print(f"       Press SPACE to start recording each sequence\n")

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        for action_idx, action in enumerate(ACTIONS):
            existing = get_existing_sequences(action)
            print(f"\n[ACTION] '{action}' — {existing}/{NO_SEQUENCES} sequences already complete")

            for sequence in range(NO_SEQUENCES):
                seq_path = os.path.join(DATA_PATH, action, str(sequence))
                existing_frames = [f for f in os.listdir(seq_path)
                                   if f.endswith('.npy')] if os.path.exists(seq_path) else []

                if len(existing_frames) >= SEQUENCE_LENGTH:
                    print(f"  Seq {sequence:02d} — already complete, skipping")
                    continue

                print(f"  Seq {sequence:02d} — waiting for SPACE...")
                state = 'WAITING'


                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)
                    draw_ui(frame, action, sequence, 0, state,
                            len(ACTIONS), action_idx)
                    cv2.imshow('Data Collection', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Quit by user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    if key == ord(' '):
                        break

                # Countdown
                state = 'COUNTDOWN'
                countdown_start = time.time()
                while time.time() - countdown_start < COUNTDOWN_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)
                    draw_ui(frame, action, sequence, 0, state,
                            len(ACTIONS), action_idx)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1)

                # Record frames
                state = 'RECORDING'
                print(f"  Seq {sequence:02d} — recording...")
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)
                    draw_ui(frame, action, sequence, frame_num, state,
                            len(ACTIONS), action_idx)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1)

                    # Save keypoints
                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_PATH, action,
                                             str(sequence), str(frame_num))
                    np.save(save_path, keypoints)

                print(f"  Seq {sequence:02d} — saved ✓")

            print(f"[DONE] '{action}' complete!")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] All data collected successfully!")
    print(f"[INFO] Dataset saved to '{DATA_PATH}/'")


if __name__ == '__main__':
    main()