import argparse
import logging
import sys
import time
import os

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.time_sync import TimedOutError

_LOGGER = logging.getLogger(__name__)

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = '.png'
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = '.jpg'

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE.get(image.source.name, 0))

    return img, extension


def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help='Time [ms] to wait before the next capture',
                        type=int, default=100)
    parser.add_argument('-r', '--resize-ratio', help='Fraction to resize the image', type=float,
                        default=1)
    parser.add_argument('--disable-full-screen',
                        help='A single image source gets displayed full screen by default. This flag disables that.',
                        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args()

    # Set default image sources if none provided
    if not options.image_sources:
        options.image_sources = ['hand_color_image']

    # Add depth image source for internal use
    all_sources = list(options.image_sources)
    if 'hand_depth' not in all_sources:
        all_sources.append('hand_depth')

    # Ensure directories exist
    os.makedirs("frames", exist_ok=True)
    os.makedirs("frames_depth", exist_ok=True)

    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)

    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent,
                            resize_ratio=options.resize_ratio) for source in all_sources
    ]

    # Display only user-specified sources (e.g., hand_color_image)
    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    keystroke = None
    timeout_count_before_reset = 0
    t1 = time.time()
    image_count = 0
    frame_index = 2223
    depth_index = 2223

    while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
        try:
            images_future = image_client.get_image_async(requests, timeout=0.5)
            while not images_future.done():
                keystroke = cv2.waitKey(25)
                if keystroke in (VALUE_FOR_ESC_KEYSTROKE, VALUE_FOR_Q_KEYSTROKE):
                    sys.exit(1)
            images = images_future.result()
        except TimedOutError:
            if timeout_count_before_reset == 5:
                _LOGGER.info('Resetting image client after 5+ timeout errors.')
                image_client = reset_image_client(robot)
                timeout_count_before_reset = 0
            else:
                timeout_count_before_reset += 1
            continue
        except Exception as err:
            _LOGGER.warning(err)
            continue

        for image_proto in images:
            image, _ = image_to_opencv(image_proto, options.auto_rotate)
            source_name = image_proto.source.name

            if source_name == 'hand_depth':
                # Save raw depth image
                raw_filename = f"frames_depth/depth_{depth_index:06d}.png"
                cv2.imwrite(raw_filename, image)

                # Save normalized 8-bit visualization
                depth_vis = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = np.uint8(depth_vis)
                vis_filename = f"frames_depth/depth_{depth_index:06d}_vis.png"
                cv2.imwrite(vis_filename, depth_vis)

                depth_index += 1

            else:
                # Show and save color image
                cv2.imshow(source_name, image)
                filename = f"frames/frame_{frame_index:06d}.png"
                cv2.imwrite(filename, image)
                frame_index += 1

        keystroke = cv2.waitKey(options.capture_delay)
        image_count += 1
        print(f'Mean image retrieval rate: {image_count/(time.time() - t1):.2f} Hz')


if __name__ == '__main__':
    if not main():
        sys.exit(1)
