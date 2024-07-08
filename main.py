import cv2
import time
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# All the points which are painted
paint_lines = set()

# Indexes of all the finger tips
WRIST_IDX = mp.solutions.hands.HandLandmark.WRIST
THUMB_IDX = mp.solutions.hands.HandLandmark.THUMB_TIP
INDEX_IDX = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_IDX = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
RING_IDX = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
PINKY_IDX = mp.solutions.hands.HandLandmark.PINKY_TIP

CONF_MODE = False
ERASE_MODE = False
CONF_MODE_THRESHOLD = 20

BRUSH_THICKNESS = 8
BRUSH_RADIUS = 4
BRUSH_COLOR = (0, 235, 9)


def calculate_distance(pos1, pos2):
    """Helper function to calculate the distance between two points"""
    if isinstance(pos1, tuple) and isinstance(pos2, tuple):
        return ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5
    else:
        return ((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2) ** 0.5


def check_conf_mode(landmarks):
    """
    Measures the distance between finger tips and wrist to figure out the current guesture shown by the user and
    switches into the mode accordingly.
    """

    global CONF_MODE, BRUSH_THICKNESS, BRUSH_RADIUS, BRUSH_COLOR, ERASE_MODE

    # The coordinates returned by mediapipe is normalised by the image height and width
    middle_wrist_dist = (
        calculate_distance(landmarks[MIDDLE_IDX], landmarks[WRIST_IDX]) * 100
    )
    pinky_wrist_dist = (
        calculate_distance(landmarks[WRIST_IDX], landmarks[PINKY_IDX]) * 100
    )

    # Erase mode - Peace symbol
    if middle_wrist_dist <= 20 and pinky_wrist_dist >= 19:
        ERASE_MODE = True
        return
    elif ERASE_MODE:
        ERASE_MODE = False

    if middle_wrist_dist >= CONF_MODE_THRESHOLD:
        CONF_MODE = True

        # Bringing thumb and pinky close activates brush size change mode.
        thumb_pinky_dist = (
            calculate_distance(landmarks[THUMB_IDX], landmarks[PINKY_IDX]) * 100
        )
        ring_thumb_dist = (
            calculate_distance(landmarks[RING_IDX], landmarks[THUMB_IDX]) * 100
        )

        index_tip_x = int(landmarks[INDEX_IDX].x * img_w)
        index_tip_y = int(landmarks[INDEX_IDX].y * img_h)
        index_tip_z = int(landmarks[INDEX_IDX].z * img_w)

        # Brush color adjust mode - Victory symbol
        if thumb_pinky_dist <= 15 and ring_thumb_dist <= 8:
            red = int((255 / img_w) * index_tip_x)
            blue = int((255 / img_h) * index_tip_y)
            green = abs(int((255 / img_w) * index_tip_z))

            BRUSH_COLOR = (red, green, blue)

        # Brush size adjust mode - Three symbol
        elif thumb_pinky_dist <= 15:
            # Brush thickness decreases from left to right
            BRUSH_THICKNESS = 20 - int(index_tip_x * 0.06)
            # Brush size increases from down to up
            BRUSH_RADIUS = 20 - int(index_tip_y * 0.04)

    elif CONF_MODE:
        CONF_MODE = False


def process_image(result: HandLandmarkerResult):
    global CONF_MODE, ERASE_MODE

    if result.handedness:
        # The image is flipped before processing it, therefore right hand is detected as the left hand
        # Comment this line and the program chooses the first hand it detects as the brush.
        if result.handedness[0][0].display_name == "Left":

            index_finger = result.hand_landmarks[0][INDEX_IDX]

            cur_pos_x, cur_pos_y = (
                int(index_finger.x * img_w),
                int(index_finger.y * img_h),
            )

            if not (CONF_MODE or ERASE_MODE):
                paint_lines.add(
                    (
                        (cur_pos_x, cur_pos_y),
                        (BRUSH_THICKNESS, BRUSH_RADIUS, BRUSH_COLOR),
                    )
                )

            elif ERASE_MODE:
                del_points = []
                # Can't delete an element from the set while iterating through it
                # So store the points in a temp list and batch delete them
                for pos, b_data in paint_lines:
                    dist = calculate_distance(pos, (cur_pos_x, cur_pos_y))

                    if dist <= BRUSH_RADIUS * 2:
                        del_points.append((pos, b_data))

                [paint_lines.discard((p, d)) for p, d in del_points]
            return cur_pos_x, cur_pos_y


i = 0
img_h = img_w = 0
lastFrameTime = 0
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    num_hands=1,
)
with HandLandmarker.create_from_options(options) as landmarker:
    videoCap = cv2.VideoCapture(2)
    videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # videoCap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    while True:
        readStatus, img = videoCap.read()

        if readStatus:
            thisFrameTime = time.time()
            fps = 1 / (thisFrameTime - lastFrameTime)
            lastFrameTime = thisFrameTime

            # Skip processing frames and catch up if we are lagging behind.
            if fps < 30:
                continue

            img_h, img_w, _ = img.shape

            img = cv2.flip(img, 1)
            p_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

            landmarker_result = landmarker.detect_for_video(p_img, i)
            finger_cords = process_image(landmarker_result)

            if finger_cords:
                check_conf_mode(landmarker_result.hand_landmarks[0])

            if paint_lines:
                # This suprisingly runs fast even when there are LOTS of points.
                for coords, (b_thickness, b_radius, b_color) in paint_lines:
                    img = cv2.circle(img, coords, b_radius, b_color, b_thickness)

            if CONF_MODE:
                img = cv2.circle(
                    img, finger_cords, BRUSH_RADIUS, BRUSH_COLOR, BRUSH_THICKNESS
                )

            if ERASE_MODE:
                img = cv2.circle(
                    img, finger_cords, BRUSH_RADIUS * 2, (255, 255, 255), 5
                )

            cv2.putText(
                img,
                f"FPS:{int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if CONF_MODE:
                mode = "CONF"
            elif ERASE_MODE:
                mode = "ERASE"
            else:
                mode = "DRAW"

            cv2.putText(
                img,
                f"MODE: {mode}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            img = cv2.resize(img, (1280, 780))
            cv2.imshow("Digital Canvas", img)
            cv2.waitKey(2)
            i += 1
