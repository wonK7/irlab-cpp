import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.api import image_pb2

def save_debug_image(image, title="Image", filename="output.png", cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def save_histogram(image, title, filename, otsu_thresh):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(title)
    plt.plot(hist)
    plt.xlabel('intensity')
    plt.ylabel('# of pixels')
    plt.axvline(x=otsu_thresh, color='red', linestyle='--', label=f"Otsu: {int(otsu_thresh)}")
    plt.savefig(filename)

def track(image_client: ImageClient):
    debug_capture = False
    use_otsu = True
    equalize = True

    while True:
        image_responses = image_client.get_image_from_sources(["frontleft_fisheye_image", "frontright_fisheye_image"])
    
    

        dtype = np.uint8

        gray_frame = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        gray_frame = cv2.imdecode(gray_frame, -1)

        if image_responses[0].source.name[0:5] == "front":
            gray_frame = cv2.rotate(gray_frame, cv2.ROTATE_90_CLOCKWISE)

        elif image_responses[0].source.name[0:5] == "right":
            gray_frame = cv2.rotate(gray_frame, cv2.ROTATE_180)

        # frame = cv2.resize(frame, (640, 480))

        gray_frame_overlay = gray_frame.copy()

        # Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Histogram equalization
        if(equalize):
            chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            eq_frame = chale.apply(blurred_frame)
        else:
            eq_frame = blurred_frame

        # if(debug_capture):
        #     otsu_t_blur, temp_bin_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Otsu thresholding / Adaptive thresholding
        if(use_otsu):
            otsu_t, bin_frame = cv2.threshold(eq_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            bin_frame = cv2.adaptiveThreshold(eq_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)
        closed_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, kernel)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(closed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw and label detected rectangles
        rectangle_count = 0
        if hierarchy is not None:
            for idx, cnt in enumerate(contours):
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                
                    area = cv2.contourArea(approx)
                    if area < 500: 
                        continue

                    if hierarchy[0][idx][3] == -1:
                        continue

                    pts = approx.reshape(4, 2)

                    def sort_pts(pts):
                        pts = sorted(pts, key=lambda p: (p[1], p[0]))  # sort by y then x
                        top = sorted(pts[:2], key=lambda p: p[0])  # top-left, top-right
                        bottom = sorted(pts[2:], key=lambda p: p[0])  # bottom-left, bottom-right
                        return np.array([top[0], top[1], bottom[1], bottom[0]])

                    pts = sort_pts(pts)
                
                    # Vector function
                    def distance(p1, p2):
                        return np.linalg.norm(p1 - p2)

                    def angle_between(v1, v2):
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

                    # Compute side vectors
                    v1 = pts[1] - pts[0]  # top
                    v2 = pts[2] - pts[1]  # right
                    v3 = pts[3] - pts[2]  # bottom
                    v4 = pts[0] - pts[3]  # left

                    # Check opposite sides roughly equal
                    len1 = distance(pts[0], pts[1])
                    len3 = distance(pts[2], pts[3])
                    len2 = distance(pts[1], pts[2])
                    len4 = distance(pts[3], pts[0])

                    if abs(len1 - len3) / max(len1, len3) < 0.2 and abs(len2 - len4) / max(len2, len4) < 0.2:
                        # Check angle between adjacent sides
                        angle1 = angle_between(v1, v2)
                        angle2 = angle_between(v2, v3)
                        if np.degrees(angle1) > 60 and np.degrees(angle1) < 120:
                            # Passed all checks: draw and label
                            cv2.drawContours(gray_frame_overlay, [approx], -1, (255, 0, 0), 2)
                            M = cv2.moments(approx)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(gray_frame_overlay, f"{rectangle_count+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            rectangle_count += 1

        # Show result
        cv2.imshow("Tictacspot", gray_frame_overlay)
        cv2.imshow("Tictacspot2", closed_frame)

        if debug_capture:
            save_debug_image(gray_frame, "grayscale", "input.png")
            save_debug_image(blurred_frame, "blurred", "blurred.png")
            save_debug_image(eq_frame, "equalized", "equalized.png" )
            #save_histogram(blurred_frame, "before equalization", "blurred_hist.png", otsu_t_blur)
            save_histogram(eq_frame, "histogram", "hist.png", otsu_t)
            save_debug_image(bin_frame, "binary img", "binary_img.png")
            save_debug_image(closed_frame, "closed_img", "closed_img.png")

            contour_preview = cv2.cvtColor(gray_frame.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 1)
            save_debug_image(contour_preview, "contours", "contours.png")
            save_debug_image(gray_frame_overlay, "final", "output.png")

            debug_capture = False

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("exiting")
    # Release camera and close windows
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('TicTacSPOT')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    assert not robot.is_estopped()
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    
    track(image_client)

    # with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True):
    #     robot.logger.info('Powering on robot... This may take several seconds.')
    #     robot.power_on(timeout_sec=20)
    #     assert robot.is_powered_on(), 'Robot power on failed.'
    #     robot.logger.info('Robot powered on.')

    #     track(image_client)

    #     robot.power_off(cut_immediately=False, timeout_sec=20)
    #     assert not robot.is_powered_on(), 'Robot power off failed.'

if __name__ == '__main__':
    main()