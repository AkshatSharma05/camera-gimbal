import cv2
import numpy as np


class VisionTargetController:
    def __init__(self, width, height):
        self.image_center = np.array([width / 2, height / 2], dtype=float)
        self.kp = 0.02
        self.ki = 0.001
        self.kd = 0.0001
        self.deadband = 4
        self.integral_limit = 100
        self.integral = np.zeros(2)
        self.previous_error = None

    def find_target(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(
            hsv, np.array([0, 100, 80]), np.array([12, 255, 255])
        )
        upper_red = cv2.inRange(
            hsv, np.array([168, 100, 80]), np.array([180, 255, 255])
        )
        mask = cv2.bitwise_or(lower_red, upper_red)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 25:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        return np.array([
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ])

    def update(self, bgr_image, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        target_center = self.find_target(bgr_image)
        if target_center is None:
            self.integral.fill(0)
            self.previous_error = None
            return None

        error = target_center - self.image_center
        if np.linalg.norm(error) <= self.deadband:
            self.integral.fill(0)
            self.previous_error = error
            return None

        derivative = (
            np.zeros(2)
            if self.previous_error is None
            else (error - self.previous_error) / dt
        )
        self.integral = np.clip(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        self.previous_error = error

        yaw_rate = -(
            self.kp * error[1]
            + self.ki * self.integral[1]
            + self.kd * derivative[1]
        )
        pitch_rate = (
            self.kp * error[0]
            + self.ki * self.integral[0]
            + self.kd * derivative[0]
        )
        return {"yaw": yaw_rate, "pitch": pitch_rate}
