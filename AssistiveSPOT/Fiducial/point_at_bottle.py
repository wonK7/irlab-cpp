import argparse
import sys
import time
import cv2
import numpy as np
from scipy import ndimage
from dotenv import load_dotenv
import logging, traceback
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import image_pb2, arm_command_pb2, geometry_pb2, estop_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.time_sync import TimedOutError
from google.protobuf import wrappers_pb2
from bosdyn.api import (basic_command_pb2, geometry_pb2, image_pb2, manipulation_api_pb2,
                        network_compute_bridge_pb2, robot_state_pb2)
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, block_until_arm_arrives)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

# Keycodes
ESC = 27
Q = ord('q')

# Rotation correction for upright camera view
ROTATION_ANGLE = {'hand_color_image': 0}

def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception('Robot is estopped. Use an E-Stop client to release.')
    
def get_objects_and_image(network_compute_client, server, model, confidence, image_sources, label):
    for source in image_sources:
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(image_source=source)
        input_data = network_compute_bridge_pb2.NetworkComputeInputData(
            image_source_and_service=image_source_and_service,
            model_name=model,
            min_confidence=confidence,
            rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL)
        server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(service_name=server)
        process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
            input_data=input_data, server_config=server_data)

        try:
            resp = network_compute_client.network_compute_bridge_command(process_img_req)
            if not resp.image_response:
                continue

            frame = get_bounding_box_image(resp)
            cv2.imshow("Spot Camera Feed", frame)

            detected_objs = []
            if len(resp.object_in_image) > 0:
                print(f"[DEBUG] Detected {len(resp.object_in_image)} objects on source '{source}'")
                for obj in resp.object_in_image:
                    conf_msg = wrappers_pb2.FloatValue()
                    obj.additional_properties.Unpack(conf_msg)
                    conf = conf_msg.value

                    if conf < confidence:
                        continue  # Skip low confidence

                    xs = [v.x for v in obj.image_properties.coordinates.vertexes]
                    ys = [v.y for v in obj.image_properties.coordinates.vertexes]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    print(f"[DEBUG] Object '{obj.name}' with confidence {conf:.3f} at center ({center_x:.1f}, {center_y:.1f})")

                    try:
                        vision_tform_obj = frame_helpers.get_a_tform_b(
                            obj.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                            obj.image_properties.frame_name_image_coordinates)
                    except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                        vision_tform_obj = None

                    if vision_tform_obj is not None:
                        detected_objs.append((obj, vision_tform_obj, conf, center_x, center_y))

            return detected_objs, resp.image_response

        except Exception as e:
            print(f"Error on source {source}: {e}")

    return [], None

def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.frombuffer(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) < 3 or img.shape[2] == 1 else img

    for obj in response.object_in_image:
        conf_msg = wrappers_pb2.FloatValue()
        obj.additional_properties.Unpack(conf_msg)
        confidence = conf_msg.value

        polygon = []
        min_x = float('inf')
        min_y = float('inf')
        for v in obj.image_properties.coordinates.vertexes:
            polygon.append([v.x, v.y])
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)

        polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = f"{obj.name} {confidence:.2f}"
        cv2.putText(img, caption, (int(min_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

def image_to_opencv(image, auto_rotate=True):
    dtype = np.uint8
    img_data = np.frombuffer(image.shot.image.data, dtype=dtype)

    if image.shot.image.format == image_pb2.Image.FORMAT_JPEG:
        try:
            np_data = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARNING] OpenCV failed to decode JPEG image (len={len(img_data)})")
                return None
        except Exception as e:
            print(f"[ERROR] cv2.imdecode crashed: {e} (len={len(img_data)})")
            return None

    elif image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        height = image.shot.image.rows
        width = image.shot.image.cols
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            img = img_data.reshape((height, width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            img = img_data.reshape((height, width, 3))
        else:
            raise ValueError("Unsupported raw pixel format.")
    else:
        raise ValueError("Unsupported image format.")

    if auto_rotate:
        angle = ROTATION_ANGLE.get(image.source.name, 0)
        if angle != 0:
            img = ndimage.rotate(img, angle)
    return img

def find_service_by_port(robot, target_port):
    directory_client = robot.ensure_client('directory')
    services = directory_client.list()
    
    for service in services:
        for endpoint in service.service_entries:
            if endpoint.host_port.port == target_port:
                return service.name
    return None

def stream_camera(image_client, stop_flag):
    request = build_image_request('hand_color_image', quality_percent=50)
    # cv2.namedWindow('hand_color_image', cv2.WINDOW_NORMAL)

    while not stop_flag():
        try:
            image_proto = image_client.get_image([request])[0]
            image = image_to_opencv(image_proto, auto_rotate=True)

            if image is None:
                print("[WARNING] Skipping frame due to failed image decode.")
                continue

        except TimedOutError:
            print("[INFO] Image request timed out — continuing.")
            continue
        except Exception as err:
            print(f'[ERROR] Camera stream error (outer): {err}')
            traceback.print_exc()
            time.sleep(0.1)
            continue

def move_arm_and_stream(config):
    sdk = bosdyn.client.create_standard_sdk('SpotArmCamera')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    assert robot.has_arm()
    verify_estop(robot)

    stop_camera = False

    def should_stop():
        return stop_camera

    lease_client.take()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        blocking_stand(command_client, timeout_sec=10)

        unstow = RobotCommandBuilder.arm_ready_command()
        command_client.robot_command(unstow)
        time.sleep(3)

        orientation = geometry_pb2.Quaternion(w=1, x=0, y=0, z=0)
        robot_state = robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        pose_list = [
            {"x": 0.7, "y":  0.25, "z": 0.2},
            {"x": 0.7, "y": -0.25, "z": 0.2},
            {"x": 0.7, "y":  0.25, "z": 0.2},
            {"x": 0.7, "y":  0.25, "z": 0.3},
            {"x": 0.7, "y": -0.25, "z": 0.3},
            {"x": 0.7, "y":  0.25, "z": 0.3},
            {"x": 0.7, "y":  0.25, "z": 0.4},
            {"x": 0.7, "y": -0.25, "z": 0.4},
            {"x": 0.7, "y":  0.25, "z": 0.4},
        ]

        print("Press ESC or Q to quit.")
        # Start streaming in a non-blocking way using a thread
        import threading
        stream_thread = threading.Thread(target=stream_camera, args=(image_client, should_stop))
        stream_thread.start()

        arm_pointed = False

        for i, pos in enumerate(pose_list):
            move_fast = [0, 2, 3, 5, 6, 8]
            duration = 0.8 if i in move_fast else 4.0
            hand_pos = geometry_pb2.Vec3(x=pos["x"], y=pos["y"], z=pos["z"])
            hand_pose = geometry_pb2.SE3Pose(position=hand_pos, rotation=orientation)
            odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(hand_pose)

            arm_command = RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
                odom_T_hand.rot.w, odom_T_hand.rot.x,
                odom_T_hand.rot.y, odom_T_hand.rot.z,
                ODOM_FRAME_NAME, duration)

            cmd_id = command_client.robot_command(arm_command)
            print(f'Moving to pose {i+1} over {duration}s...')

            while True:
                feedback = command_client.robot_command_feedback(cmd_id)
                status = feedback.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status

                detected_objs, image_data = get_objects_and_image(
                    network_compute_client, config.ml_service, config.model,
                    confidence=0.5, image_sources=["hand_color_image"], label="med")
                
                if detected_objs:
                    print("detected true")
                else:
                    print("detected false")

                if detected_objs:
                    for obj, vision_tform_obj, conf, cx, cy in detected_objs:
                        print(f"🔍 Interrupting: Object '{obj.name}' detected with confidence {conf:.3f} at center ({cx:.1f}, {cy:.1f})")
                    command_client.robot_command(RobotCommandBuilder.stop_command())

                    # Use first detected object for aiming
                    obj, vision_tform_obj, _, _, _ = detected_objs[0]

                    robot_state = robot_state_client.get_robot_state()
                    odom_T_vision = get_a_tform_b(
                        robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, frame_helpers.VISION_FRAME_NAME)
                    odom_T_obj = odom_T_vision * vision_tform_obj

                    aim_command = RobotCommandBuilder.arm_pose_command(
                        odom_T_obj.x, odom_T_obj.y, odom_T_obj.z + 0.05,
                        orientation.w, orientation.x, orientation.y, orientation.z,
                        ODOM_FRAME_NAME, 1.0)
                    command_client.robot_command(aim_command)
                    block_until_arm_arrives(command_client, command_client.robot_command(aim_command), 2.0)
                    print("✅ Arm aimed at detected object.")
                    arm_pointed = True
                    break

                # Normal end condition
                if status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                    print(f"Pose {i+1} complete.")
                    break
                time.sleep(0.1)
            
            if arm_pointed:
                break

        stop_camera = True
        stream_thread.join()
        print("All poses complete. Stowing arm.")

        stow = RobotCommandBuilder.arm_stow_command()
        command_client.robot_command(stow)
        time.sleep(3)

        robot.power_off(cut_immediately=False, timeout_sec=20)

def main(argv):
    load_dotenv()
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-s', '--ml-service', default='yolov8-server', help='ML server service name.')
    parser.add_argument('-m', '--model', default='med.pt', help='Model name registered on the ML server.')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    args = parser.parse_args(argv)

    sdk = bosdyn.client.create_standard_sdk("main")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    print("Reached point 1")

    if not hasattr(cv2, 'imshow'):
        print("[FATAL] OpenCV built without GUI support.")
        sys.exit(1)

    move_arm_and_stream(args)

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
