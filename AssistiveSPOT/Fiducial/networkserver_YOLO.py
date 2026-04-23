# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import io
import os
import sys
import time
import logging

import cv2
from PIL import Image
import numpy as np

from bosdyn.api import network_compute_bridge_service_pb2_grpc
from bosdyn.api import network_compute_bridge_pb2
from bosdyn.api import image_pb2
from bosdyn.api import header_pb2
import bosdyn.client
import bosdyn.client.util
import grpc
from concurrent import futures

from ultralytics import YOLO

import queue
import threading
from google.protobuf import wrappers_pb2
from dotenv import load_dotenv

kServiceAuthority = "fetch-tutorial-worker.spot.robot"

class YOLOv8ObjectDetectionModel:
    def __init__(self, model_path, label_path=None):
        print(f"Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        if label_path:
            with open(label_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = None
        self.name = os.path.basename(model_path)

    def predict(self, image):
        # image is BGR numpy array, convert to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        return results


def process_thread(args, request_queue, response_queue):
    # Load the model(s)
    models = {}
    for model in args.model:
        this_model = YOLOv8ObjectDetectionModel(model[0], model[1] if len(model) > 1 else None)
        models[this_model.name] = this_model

    print('')
    print('Service ' + args.name + ' running on port: ' + str(args.port))

    print('Loaded models:')
    for model_name in models:
        print('    ' + model_name)

    while True:
        request = request_queue.get()

        if isinstance(request, network_compute_bridge_pb2.ListAvailableModelsRequest):
            out_proto = network_compute_bridge_pb2.ListAvailableModelsResponse()
            for model_name in models:
                out_proto.models.data.append(network_compute_bridge_pb2.ModelData(model_name=model_name))
            response_queue.put(out_proto)
            continue
        else:
            out_proto = network_compute_bridge_pb2.NetworkComputeResponse()

        # Find the model
        if request.input_data.model_name not in models:
            err_str = 'Cannot find model "' + request.input_data.model_name + '" in loaded models.'
            print(err_str)

            # Set the error in the header.
            out_proto.header.error.code = header_pb2.CommonError.CODE_INVALID_REQUEST
            out_proto.header.error.message = err_str
            response_queue.put(out_proto)
            continue

        model = models[request.input_data.model_name]

        # Unpack the incoming image.
        if request.input_data.image.format == image_pb2.Image.FORMAT_RAW:
            pil_image = Image.open(io.BytesIO(request.input_data.image.data))
            if request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                # If the input image is grayscale, convert it to RGB.
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2RGB)

            elif request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                # Already an RGB image.
                image = np.array(pil_image)

            else:
                print('Error: image input in unsupported pixel format: ', request.input_data.image.pixel_format)
                response_queue.put(out_proto)
                continue

        elif request.input_data.image.format == image_pb2.Image.FORMAT_JPEG:
            dtype = np.uint8
            jpg = np.frombuffer(request.input_data.image.data, dtype=dtype)
            image = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

            if len(image.shape) < 3:
                # If the input image is grayscale, convert it to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        else:
            print('Error: unsupported image format:', request.input_data.image.format)
            response_queue.put(out_proto)
            continue

        image_height, image_width = image.shape[:2]

        results = model.predict(image)

        # YOLOv8 results:
        # results[0].boxes.xyxy: tensor Nx4 (x1,y1,x2,y2)
        # results[0].boxes.conf: tensor Nx1 confidence scores
        # results[0].boxes.cls: tensor Nx1 class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
        scores = results[0].boxes.conf.cpu().numpy()  # shape (N,)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        num_objects = 0

        for i in range(len(boxes)):
            if scores[i] < request.input_data.min_confidence:
                continue

            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            cls_id = classes[i]

            if model.labels and cls_id < len(model.labels):
                label = model.labels[cls_id]
            else:
                label = f'class_{cls_id}'

            num_objects += 1

            point1 = (x1, y1)
            point2 = (x2, y1)
            point3 = (x2, y2)
            point4 = (x1, y2)

            out_obj = out_proto.object_in_image.add()
            out_obj.name = f"obj{num_objects}_label_{label}"

            for pt in [point1, point2, point3, point4]:
                vertex = out_obj.image_properties.coordinates.vertexes.add()
                vertex.x = float(pt[0])
                vertex.y = float(pt[1])

            confidence = wrappers_pb2.FloatValue(value=float(conf))
            out_obj.additional_properties.Pack(confidence)

            if not args.no_debug:
                polygon = np.array([point1, point2, point3, point4], np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
                caption = f"{label}: {conf:.3f}"
                left_x = min(pt[0] for pt in [point1, point2, point3, point4])
                top_y = min(pt[1] for pt in [point1, point2, point3, point4])
                cv2.putText(image, caption, (int(left_x), int(top_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f'Found {num_objects} object(s)')

        if not args.no_debug:
            debug_image_filename = 'network_compute_server_output.jpg'
            cv2.imwrite(debug_image_filename, image)
            print('Wrote debug image output to: "' + debug_image_filename + '"')

        out_proto.status = network_compute_bridge_pb2.NetworkComputeStatus.NETWORK_COMPUTE_STATUS_SUCCESS
        response_queue.put(out_proto)


class NetworkComputeBridgeWorkerServicer(
        network_compute_bridge_service_pb2_grpc.NetworkComputeBridgeWorkerServicer):

    def __init__(self, thread_input_queue, thread_output_queue):
        super(NetworkComputeBridgeWorkerServicer, self).__init__()

        self.thread_input_queue = thread_input_queue
        self.thread_output_queue = thread_output_queue

    def NetworkCompute(self, request, context):
        print('Got NetworkCompute request')
        self.thread_input_queue.put(request)
        out_proto = self.thread_output_queue.get()
        return out_proto

    def ListAvailableModels(self, request, context):
        print('Got ListAvailableModels request')
        self.thread_input_queue.put(request)
        out_proto = self.thread_output_queue.get()
        return out_proto


def register_with_robot(options):
    """ Registers this worker with the robot's Directory."""
    ip = bosdyn.client.common.get_self_ip(options.hostname)
    print('Detected IP address as: ' + ip)

    sdk = bosdyn.client.create_standard_sdk("yolov8_network_compute_server")

    robot = sdk.create_robot(options.hostname)

    # Authenticate robot before being able to use it
    bosdyn.client.util.authenticate(robot)

    directory_client = robot.ensure_client(
        bosdyn.client.directory.DirectoryClient.default_service_name)
    directory_registration_client = robot.ensure_client(
        bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)

    # Check to see if a service is already registered with our name
    services = directory_client.list()
    for s in services:
        if s.name == options.name:
            print(f"WARNING: existing service with name, \"{options.name}\", removing it.")
            directory_registration_client.unregister(options.name)
            break

    # Register service
    print(f'Attempting to register {ip}:{options.port} onto {options.hostname} directory...')
    directory_registration_client.register(options.name, "bosdyn.api.NetworkComputeBridgeWorker", kServiceAuthority, ip, int(options.port))


def main(argv):
    load_dotenv()
    default_port = '50051'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='[MODEL_PATH] [LABELS_FILE.txt]: Path to YOLOv8 .pt model and optional labels txt file', action='append', nargs='+', required=True)
    parser.add_argument('-p', '--port', help='Server\'s port number, default: ' + default_port,
                        default=default_port)
    parser.add_argument('-d', '--no-debug', help='Disable writing debug images.', action='store_true')
    parser.add_argument('-n', '--name', help='Service name', default='yolov8-server')
    bosdyn.client.util.add_base_arguments(parser)

    options = parser.parse_args(argv)

    # Validate model paths
    for model in options.model:
        if not os.path.isfile(model[0]):
            print(f'Error: model file ({model[0]}) not found or is not a file.')
            sys.exit(1)
        if len(model) > 1 and not os.path.isfile(model[1]):
            print(f'Error: labels file ({model[1]}) not found or is not a file.')
            sys.exit(1)

    # Perform registration.
    register_with_robot(options)

    # Thread-safe queues for communication between the GRPC endpoint and the ML thread.
    request_queue = queue.Queue()
    response_queue = queue.Queue()

    # Start server thread
    thread = threading.Thread(target=process_thread, args=([options, request_queue, response_queue]))
    thread.start()

    # Set up GRPC endpoint
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
        NetworkComputeBridgeWorkerServicer(request_queue, response_queue), server)
    server.add_insecure_port('[::]:' + options.port)
    server.start()

    print('Running...')
    thread.join()

    return True


if __name__ == '__main__':
    logging.basicConfig()
    if not main(sys.argv[1:]):
        sys.exit(1)
