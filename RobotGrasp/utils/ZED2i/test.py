import pyzed.sl as sl
import cv2

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.sdk_verbose = 0
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : "+repr(err)+". Exit program.")
    exit(-1)

# Get camera information (ZED serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: {0}".format(zed_serial))

# Define the Object Detection module parameters
detection_parameters = sl.ObjectDetectionParameters()
detection_parameters.enable_tracking = True
detection_parameters.enable_segmentation = True

# Object tracking requires camera tracking to be enabled
if detection_parameters.enable_tracking:
    zed.enable_positional_tracking()

print("Object Detection: Loading Module...")
err = zed.enable_object_detection(detection_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print("Error {}, exit program".format(err))
    zed.close()
    exit()
