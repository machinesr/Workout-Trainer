import cv2
import numpy as np
import time
import PoseModule as pm


class Exercise:
    """
    Handles the logic for a specific exercise, tracking state, counting
    repetitions, validating form, and checking timing.
    """

    def __init__(
        self,
        landmarks,
        angle_range,
        progress_type,
        form_checks=None,
        timing=None,
        visibility_check=None,
    ):
        self.landmarks = landmarks
        self.angle_range = angle_range
        self.type = progress_type
        self.form_checks = form_checks if form_checks else []
        self.visibility_check = visibility_check if visibility_check else {}

        if timing is None:
            timing = {"concentric": 3, "hold": 1, "eccentric": 3, "tolerance": 0.5}
        self.concentric_time = timing.get("concentric", 3)
        self.hold_time = timing.get("hold", 1)
        self.eccentric_time = timing.get("eccentric", 3)
        self.time_tolerance = timing.get("tolerance", 0.5)
        self.total_rep_time = self.concentric_time + self.hold_time + self.eccentric_time

        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "down"
        self.stage_start_time = time.time()
        self.feedback_set_time = time.time()
        self.form_feedback = "START"
        self.speed_feedback = "START"
        self.rep_timing_is_good = True

    def _calculate_angle(self, detector, lmList, p1, p2, p3, img):
        """Internal helper to find and draw an angle."""
        return detector.findAngle(lmList[p1], lmList[p2], lmList[p3], img=img, draw=True)

    # <<< NEW METHOD TO HANDLE PERCENTAGE LOGIC >>>
    def _calculate_percentage(self, angle, angle_range, progress_type):
        """
        Calculates the exercise percentage, inverting the logic for 'lift' types.
        """
        # For lifts, a larger angle means higher percentage (e.g., glute bridge)
        if progress_type == "inverse":
            # Maps from [min_angle, max_angle] to [0, 100]
            return np.interp(angle, (angle_range[0], angle_range[1]), (0, 100))
        # For curls/presses, a smaller angle means higher percentage (e.g., bicep curl)
        else:
            # Maps from [min_angle, max_angle] to [100, 0]
            return np.interp(angle, (angle_range[0], angle_range[1]), (100, 0))

    def update(self, img, detector, lmList):
        """
        Processes a new frame to update exercise state, returning UI values.
        """
        pace_progress = 0.0
        inter_stage_warnings = ["TOO FAST", "TOO SLOW", "HOLD AT TOP"]

        if isinstance(self.landmarks[0], list):
            angle1 = self._calculate_angle(detector, lmList, *self.landmarks[0], img)
            angle2 = self._calculate_angle(detector, lmList, *self.landmarks[1], img)
            angle = (angle1 + angle2) / 2
        else:
            angle = self._calculate_angle(detector, lmList, *self.landmarks, img)

        # <<< MODIFIED: Use the new helper function for all percentage and bar calculations >>>
        per = self._calculate_percentage(angle, self.angle_range, self.type)
        # The bar calculation should always go from 100% to 0% of its height
        bar = np.interp(per, (0, 100), (650, 100))

        # <<< MODIFIED: Upgraded form checking logic >>>
        current_form_is_good = True
        self.form_feedback = "GOOD"
        for check in self.form_checks:
            # Default to 'angle' check if type is not specified
            check_type = check.get("check_type", "angle")

            if check_type == "visibility":
                if not check_body_visibility(detector.lmList, check["landmarks"]):
                    self.form_feedback = check["feedback"]
                    current_form_is_good = False
                    break

            # <<< NEW: Logic to handle 'positional' checks >>>
            elif check_type == "positional":
                lm1_id, lm2_id = check["landmarks"]
                # Ensure landmarks are available before checking
                if lm1_id < len(lmList) and lm2_id < len(lmList):
                    if check["condition"](lmList[lm1_id], lmList[lm2_id]):
                        self.form_feedback = check["feedback"]
                        current_form_is_good = False
                        break

            elif check_type == "angle":
                check_angle = detector.findAngle(
                    lmList[check["landmarks"][0]],
                    lmList[check["landmarks"][1]],
                    lmList[check["landmarks"][2]],
                    img=img,
                    draw=False,
                )
                if check["condition"](check_angle, check["threshold"]):
                    self.form_feedback = check["feedback"]
                    current_form_is_good = False
                    break
        # <<< END of MODIFIED block >>>

        current_time = time.time()
        elapsed_stage_time = current_time - self.stage_start_time

        is_moving = self.stage in ["going_up", "hold", "going_down"]
        if (not current_form_is_good and is_moving) or (per <= 5 and self.stage in ["going_up", "hold"]):
            self.stage = "down"
            self.speed_feedback = "REP RESET"
            self.feedback_set_time = current_time

        if self.stage == "down":
            final_rep_messages = [
                "GOOD REP!",
                "BAD TIMING",
                "REP RESET",
            ] + inter_stage_warnings
            if self.speed_feedback in final_rep_messages and (current_time - self.feedback_set_time < 1.0):
                pass
            else:
                self.speed_feedback = "LIFT UP"

            if per >= 10:
                self.stage = "going_up"
                self.stage_start_time = current_time
                self.rep_timing_is_good = True

        elif self.stage == "going_up":
            is_warning_active = self.speed_feedback in inter_stage_warnings
            if is_warning_active and (current_time - self.feedback_set_time < 0.5):
                pass
            else:
                self.speed_feedback = "GO"
            progress_time = min(elapsed_stage_time, self.concentric_time)
            pace_progress = progress_time / self.total_rep_time
            if per >= 90:
                if abs(elapsed_stage_time - self.concentric_time) > self.time_tolerance:
                    self.speed_feedback = "TOO FAST" if elapsed_stage_time < self.concentric_time else "TOO SLOW"
                    self.feedback_set_time = current_time
                    self.rep_timing_is_good = False
                self.stage = "hold"
                self.stage_start_time = current_time
            elif elapsed_stage_time > self.concentric_time + self.time_tolerance:
                self.speed_feedback = "TOO SLOW"
                self.rep_timing_is_good = False

        elif self.stage == "hold":
            is_warning_active = self.speed_feedback in inter_stage_warnings
            if is_warning_active and (current_time - self.feedback_set_time < 0.5):
                pass
            else:
                self.speed_feedback = "HOLD"
            concentric_progress = self.concentric_time / self.total_rep_time
            hold_progress = elapsed_stage_time / self.total_rep_time
            pace_progress = concentric_progress + hold_progress
            if per < 90:
                self.speed_feedback = "HOLD AT TOP"
                self.rep_timing_is_good = False
                self.feedback_set_time = current_time
                self.stage = "going_down"
                self.stage_start_time = current_time
            elif elapsed_stage_time >= self.hold_time:
                self.stage = "going_down"
                self.stage_start_time = current_time

        elif self.stage == "going_down":
            is_warning_active = self.speed_feedback in inter_stage_warnings
            if is_warning_active and (current_time - self.feedback_set_time < 0.5):
                pass
            else:
                self.speed_feedback = "BACK SLOWLY"
            concentric_hold_progress = (self.concentric_time + self.hold_time) / self.total_rep_time
            eccentric_progress = elapsed_stage_time / self.total_rep_time
            pace_progress = concentric_hold_progress + eccentric_progress
            if per <= 5:
                if abs(elapsed_stage_time - self.eccentric_time) > self.time_tolerance:
                    self.speed_feedback = "TOO FAST" if elapsed_stage_time < self.eccentric_time else "TOO SLOW"
                    self.rep_timing_is_good = False
                if self.rep_timing_is_good and current_form_is_good:
                    self.good_reps += 1
                    self.speed_feedback = "GOOD REP!"
                else:
                    self.bad_reps += 1
                    if self.speed_feedback not in inter_stage_warnings:
                        self.speed_feedback = "BAD TIMING"
                self.feedback_set_time = current_time
                self.stage = "down"

        pace_progress = min(pace_progress, 1.0)
        return (
            bar,
            per,
            self.good_reps,
            self.bad_reps,
            self.form_feedback,
            self.speed_feedback,
            pace_progress,
        )


# ===============================================================
#                        CONFIGURATION
# ===============================================================

EXERCISE_CONFIG = {
    "bicep_curl": {
        "landmarks": [12, 14, 16],
        "angle_range": [40, 160],
        "progress_type": "normal",
        "timing": {"concentric": 2, "hold": 1, "eccentric": 3, "tolerance": 0.7},
        "form_checks": [
            {
                "check_type": "angle",
                "landmarks": [24, 12, 14],
                "threshold": 30,
                "condition": lambda a, t: a > t,
                "feedback": "PIN YOUR ELBOW",
            },
            {
                "check_type": "visibility",
                "landmarks": [12, 14, 16, 24],
                "feedback": "KEEP RIGHT UPPER BODY VISIBLE",
            },
        ],
        "visibility_check": {
            "landmarks": [12, 14, 16, 24],
            "feedback": "SHOW RIGHT UPPER BODY",
        },
    },
    "squat": {
        "landmarks": [24, 26, 28],
        "angle_range": [90, 175],
        "progress_type": "normal",
        "timing": {"concentric": 2, "hold": 1, "eccentric": 1, "tolerance": 1},
        "form_checks": [
            {
                "check_type": "angle",  # <<< MODIFIED: Explicitly defined
                "landmarks": [12, 24, 26],
                "threshold": 80,
                "condition": lambda a, t: a < t,
                "feedback": "KEEP CHEST UP",
            },
            {
                "check_type": "angle",
                "landmarks": [24, 26, 28],
                "threshold": 80,
                "condition": lambda a, t: a < t,
                "feedback": "SQUAT TOO DEEP",
            },
            {
                "check_type": "visibility",
                "landmarks": [12, 24, 26, 28],
                "feedback": "KEEP WHOLE RIGHT BODY VISIBLE",
            },
        ],
        "visibility_check": {
            "landmarks": [12, 24, 26, 28, 32],
            "feedback": "SHOW RIGHT FULL BODY",
        },
    },
    "wall_push_up": {
        "landmarks": [12, 14, 16],
        "angle_range": [90, 160],
        "progress_type": "normal",
        "timing": {"concentric": 4, "hold": 1, "eccentric": 2, "tolerance": 0.7},
        "form_checks": [
            {
                "check_type": "angle",
                "landmarks": [12, 24, 28],  # R_Shoulder, R_Hip, R_Ankle
                "threshold": 160,
                "condition": lambda a, t: a < t,
                "feedback": "KEEP BACK STRAIGHT",
            },
            {
                "check_type": "positional",
                "landmarks": [12, 24],  # R_Shoulder, R_Hip
                "condition": lambda shoulder, hip: (abs(shoulder[1] - hip[1]) / abs(shoulder[2] - hip[2])) < 0.15,
                "feedback": "LEAN FORWARD MORE",
            },
            {
                "check_type": "positional",
                "landmarks": [14, 12],  # R_Elbow, R_Shoulder
                "condition": lambda elbow, shoulder: elbow[2] < shoulder[2] - 30,  # -30 for pixel tolerance
                "feedback": "TUCK YOUR ELBOWS",
            },
            {
                "check_type": "positional",
                "landmarks": [16, 12],  # R_Wrist, R_Shoulder
                "condition": lambda wrist, shoulder: wrist[2] < shoulder[2],
                "feedback": "LOWER YOUR HANDS",
            },
            {
                "check_type": "visibility",
                "landmarks": [12, 14, 16, 24],
                "feedback": "KEEP RIGHT UPPER BODY VISIBLE",
            },
        ],
        "visibility_check": {
            "landmarks": [12, 14, 16, 24],
            "feedback": "SHOW RIGHT UPPER BODY",
        },
    },
    "glute_bridge": {
        "landmarks": [12, 24, 26],  # R_Shoulder, R_Hip, R_Knee
        "angle_range": [130, 170],
        "progress_type": "inverse",
        "timing": {"concentric": 2, "hold": 2, "eccentric": 3, "tolerance": 0.7},
        "form_checks": [
            {
                "check_type": "angle",
                "landmarks": [12, 24, 26],  # R_Shoulder, R_Hip, R_Knee
                # CHANGED: Threshold is now a realistic value to detect over-extension.
                "threshold": 180,
                "condition": lambda a, t: a > t,
                "feedback": "AVOID ARCHING BACK",
            },
            {
                "check_type": "angle",
                "landmarks": [24, 26, 28],  # R_Hip, R_Knee, R_Ankle
                "threshold": 110,
                "condition": lambda a, t: a > t,
                "feedback": "KEEP FEET CLOSER",
            },
            {
                "check_type": "positional",
                "landmarks": [12, 28],  # R_Shoulder, R_Ankle
                "condition": lambda shoulder, ankle: abs(shoulder[2] - ankle[2]) > 40,
                "feedback": "KEEP SHOULDERS & FEET ON GROUND",
            },
        ],
        "visibility_check": {
            "landmarks": [12, 24, 26, 28],
            "feedback": "SHOW FULL RIGHT SIDE OF BODY",
        },
    },
    "seated_leg_raise": {
        "landmarks": [24, 26, 28],  # R_Hip, R_Knee, R_Ankle
        "angle_range": [90, 155],
        "progress_type": "inverse",
        "timing": {"concentric": 3, "hold": 2, "eccentric": 4, "tolerance": 0.7},
        "form_checks": [
            {
                "check_type": "angle",
                "landmarks": [12, 24, 26],  # R_Shoulder, R_Hip, R_Knee
                "threshold": 110,
                "condition": lambda a, t: a > t,
                "feedback": "SIT UP STRAIGHT",
            },
            {
                "check_type": "positional",
                "landmarks": [26, 24],  # R_Knee, R_Hip
                "condition": lambda knee, hip: knee[2] < hip[2],  # Checks if knee's y-coord is above hip's y-coord
                "feedback": "KEEP THIGH ON CHAIR",
            },
        ],
        "visibility_check": {
            "landmarks": [12, 24, 26, 28],
            "feedback": "SHOW FULL RIGHT SIDE OF BODY",
        },
    },
}

UI_CONFIG = {
    "colors": {
        "bg": (0, 0, 0),
        "good": (0, 255, 0),
        "bad": (0, 0, 255),
        "warning": (0, 165, 255),
        "neutral": (255, 255, 255),
        "text_dark_bg": (0, 0, 0),
    },
    "fonts": {"main": cv2.FONT_HERSHEY_PLAIN},
    "feedback_box": (280, 550, 770, 170),
    "rep_counter_box": (0, 450, 250, 270),
    "movement_bar": (1100, 100, 75, 550),
    "pace_bar": (300, 660, 730, 30),
}

# ===============================================================
#                        HELPER & UI FUNCTIONS
# ===============================================================


def check_body_visibility(full_lmList, required_ids, threshold=0.7):
    if not full_lmList:
        return False
    for landmark_id in required_ids:
        if (
            landmark_id >= len(full_lmList)
            or len(full_lmList[landmark_id]) < 4
            or full_lmList[landmark_id][3] < threshold
        ):
            return False
    return True


def draw_visibility_prompt(img, text):
    color, font = UI_CONFIG["colors"]["warning"], UI_CONFIG["fonts"]["main"]
    cv2.rectangle(img, (0, 250), (1280, 500), UI_CONFIG["colors"]["bg"], cv2.FILLED)
    cv2.putText(img, "POSITION CHECK", (300, 340), font, 5, color, 6)
    cv2.putText(img, text, (220, 430), font, 4, color, 5)


def draw_feedback_box(img, form_feedback, speed_feedback):
    x, y, w, h = UI_CONFIG["feedback_box"]
    colors = UI_CONFIG["colors"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["bg"], cv2.FILLED)
    form_color = colors["good"] if form_feedback == "GOOD" else colors["bad"]
    warning_msgs = ["TOO FAST", "TOO SLOW", "REP RESET", "HOLD AT TOP", "BAD TIMING"]
    speed_color = (
        colors["warning"]
        if speed_feedback in warning_msgs
        else colors["good"]
        if speed_feedback == "GOOD REP!"
        else colors["neutral"]
    )
    cv2.putText(
        img,
        f"FORM: {form_feedback}",
        (x + 20, y + 40),
        UI_CONFIG["fonts"]["main"],
        3,
        form_color,
        3,
    )
    cv2.putText(
        img,
        f"SPEED: {speed_feedback}",
        (x + 20, y + 85),
        UI_CONFIG["fonts"]["main"],
        3,
        speed_color,
        3,
    )


def draw_pace_bar(img, progress, handler):
    x, y, w, h = UI_CONFIG["pace_bar"]
    colors = UI_CONFIG["colors"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["neutral"], 3)
    filled_w = int(w * progress)
    cv2.rectangle(img, (x, y), (x + filled_w, y + h), colors["good"], cv2.FILLED)
    total_time = handler.total_rep_time
    if total_time > 0:
        concentric_end_x = x + int(w * (handler.concentric_time / total_time))
        hold_end_x = concentric_end_x + int(w * (handler.hold_time / total_time))
        cv2.line(
            img,
            (concentric_end_x, y),
            (concentric_end_x, y + h),
            colors["text_dark_bg"],
            4,
        )
        cv2.line(img, (hold_end_x, y), (hold_end_x, y + h), colors["text_dark_bg"], 4)


def draw_movement_bar(img, percentage, bar_value, form_is_good):
    x, y, w, h_max = UI_CONFIG["movement_bar"]
    color = UI_CONFIG["colors"]["good"] if form_is_good else UI_CONFIG["colors"]["bad"]
    cv2.rectangle(img, (x, y), (x + w, y + h_max), color, 3)
    cv2.rectangle(img, (x, int(bar_value)), (x + w, y + h_max), color, cv2.FILLED)
    cv2.putText(img, f"{int(percentage)}%", (x, y - 25), UI_CONFIG["fonts"]["main"], 4, color, 4)


def draw_rep_counter(img, good_reps, bad_reps):
    x, y, w, h = UI_CONFIG["rep_counter_box"]
    colors = UI_CONFIG["colors"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["bg"], cv2.FILLED)
    cv2.putText(img, "GOOD", (x + 25, y + 50), UI_CONFIG["fonts"]["main"], 3, colors["good"], 3)
    cv2.putText(
        img,
        str(int(good_reps)),
        (x + 45, y + 150),
        UI_CONFIG["fonts"]["main"],
        8,
        colors["good"],
        15,
    )
    cv2.putText(img, "BAD", (x + 35, y + 200), UI_CONFIG["fonts"]["main"], 3, colors["bad"], 3)
    cv2.putText(
        img,
        str(int(bad_reps)),
        (x + 45, y + 260),
        UI_CONFIG["fonts"]["main"],
        5,
        colors["bad"],
        5,
    )


def draw_header_info(img, exercise_name):
    color, font = UI_CONFIG["colors"]["text_dark_bg"], UI_CONFIG["fonts"]["main"]
    cv2.putText(img, f"Exercise: {exercise_name.upper()}", (50, 50), font, 2, color, 3)
    cv2.putText(img, "1-5: Change | 'r': reset | 'q': quit", (50, 100), font, 2, color, 3)


def draw_countdown(img, remaining_time):
    color, font = UI_CONFIG["colors"]["bad"], UI_CONFIG["fonts"]["main"]
    cv2.putText(img, "GET READY!", (350, 300), font, 7, color, 8)
    cv2.putText(img, str(remaining_time), (550, 450), font, 10, color, 10)


# ===============================================================
#                        MAIN APPLICATION
# ===============================================================


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = pm.poseDetector()
    exercise_keys = {
        ord("1"): "bicep_curl",
        ord("2"): "squat",
        ord("3"): "wall_push_up",
        ord("4"): "glute_bridge",
        ord("5"): "seated_leg_raise",
    }
    current_exercise_name = "bicep_curl"
    exercise_handler = Exercise(**EXERCISE_CONFIG[current_exercise_name])

    program_state = "WAITING_FOR_BODY"
    countdown_duration = 5
    countdown_start_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if program_state == "WAITING_FOR_BODY":
            draw_header_info(img, current_exercise_name)
            visibility_config = EXERCISE_CONFIG[current_exercise_name].get("visibility_check")
            if visibility_config:
                is_visible = check_body_visibility(detector.lmList, visibility_config["landmarks"])
                if is_visible:
                    program_state = "COUNTDOWN"
                    countdown_start_time = time.time()
                else:
                    draw_visibility_prompt(img, visibility_config["feedback"])
            else:
                program_state = "COUNTDOWN"
                countdown_start_time = time.time()

        elif program_state == "COUNTDOWN":
            draw_header_info(img, current_exercise_name)
            time_since_start = time.time() - countdown_start_time
            if time_since_start >= countdown_duration:
                program_state = "TRACKING"
            else:
                remaining_time = int(countdown_duration - time_since_start) + 1
                draw_countdown(img, remaining_time)

        elif program_state == "TRACKING":
            bar, per, pace_progress = 100, 0, 0.0
            form_feedback, speed_feedback = "NO PERSON DETECTED", ""
            good_count, bad_count = (
                exercise_handler.good_reps,
                exercise_handler.bad_reps,
            )

            if len(lmList) != 0:
                (
                    bar,
                    per,
                    good_count,
                    bad_count,
                    form_feedback,
                    speed_feedback,
                    pace_progress,
                ) = exercise_handler.update(img, detector, lmList)

            draw_feedback_box(img, form_feedback, speed_feedback)
            draw_pace_bar(img, pace_progress, exercise_handler)
            draw_movement_bar(img, per, bar, form_feedback == "GOOD")
            draw_rep_counter(img, good_count, bad_count)
            draw_header_info(img, current_exercise_name)

        cv2.imshow("AI Trainer", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("r") or key in exercise_keys:
            if key in exercise_keys:
                current_exercise_name = exercise_keys[key]
            exercise_handler = Exercise(**EXERCISE_CONFIG[current_exercise_name])
            program_state = "WAITING_FOR_BODY"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
