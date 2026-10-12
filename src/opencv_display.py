import sys

import cv2
import numpy as np

WINDOW_NAME = "Gimbal camera"


def main():
    width, height = map(int, sys.argv[1:3])
    frame_size = width * height * 3
    frame_bytes = bytearray(frame_size)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = memoryview(frame_bytes)
            offset = 0
            while offset < frame_size:
                count = sys.stdin.buffer.readinto(frame[offset:])
                if count == 0:
                    if offset == 0:
                        return
                    raise EOFError("Received an incomplete camera frame")
                offset += count

            image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                height, width, 3
            )
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
