import argparse
import sys
import time

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2, network_compute_bridge_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from dotenv import load_dotenv

kImageSources = ['hand_color_image']

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

def main(argv):
    load_dotenv()
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-s', '--ml-service', required=True, help='ML server service name.')
    parser.add_argument('-m', '--model', required=True, help='Model name registered on the ML server.')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    args = parser.parse_args(argv)

    sdk = bosdyn.client.create_standard_sdk("MLViewer")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    network_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client.take()
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        cv2.namedWindow("Spot Camera Feed")

        try:
            while True:
                for source in kImageSources:
                    img_source = network_compute_bridge_pb2.ImageSourceAndService(image_source=source)

                    input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                        image_source_and_service=img_source,
                        model_name=args.model,
                        min_confidence=args.confidence,
                        rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL
                    )

                    server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
                        service_name=args.ml_service
                    )

                    req = network_compute_bridge_pb2.NetworkComputeRequest(
                        input_data=input_data, server_config=server_data
                    )

                    try:
                        resp = network_client.network_compute_bridge_command(req)
                        if not resp.image_response:
                            continue

                        frame = get_bounding_box_image(resp)
                        cv2.imshow("Spot Camera Feed", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
                    except Exception as e:
                        print(f"Error on source {source}: {e}")

        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
