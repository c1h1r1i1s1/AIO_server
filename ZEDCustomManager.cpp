// This file manages the ZED camera, including its initialisation and frame grabbing
#include "ZEDCustomManager.hpp"
#include <iostream>
#include <cstring>
#include <assert.h>

using namespace sl;

sl::Camera* ZEDCustomManager::m_ZedCamera = new sl::Camera();;
sl::InitParameters ZEDCustomManager::m_InitParams = sl::InitParameters();
sl::RuntimeParameters ZEDCustomManager::m_RuntimeParams = sl::RuntimeParameters();
sl::Mat* ZEDCustomManager::m_LeftMat = nullptr;

int ZEDCustomManager::m_ImageWidth = 0;
int ZEDCustomManager::m_ImageHeight = 0;

ZEDCustomManager::ZEDCustomManager() {}

sl::Camera* ZEDCustomManager::getCamera() {
    return m_ZedCamera;
}

cv::Mat ZEDCustomManager::getCurrentMat() {
    return slMat2cvMat(*m_LeftMat);
}

bool ZEDCustomManager::InitializeZEDCamera(bool isZedStatic) {
    // Create the camera instance if not already created.
    if (!m_ZedCamera) {
        std::cout << "Should make new right??" << std::endl;
        m_ZedCamera = new sl::Camera();
    }

    // Set initialization parameters.
    m_InitParams.camera_resolution = RESOLUTION::HD720;
    m_InitParams.depth_mode = DEPTH_MODE::NEURAL;
    m_InitParams.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    m_InitParams.sdk_verbose = 1;

    // Open the camera.
    ERROR_CODE state = m_ZedCamera->open(m_InitParams);
    if (state != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open the ZED camera: " << (int)state << std::endl;
        return 1;
    }
    std::cout << "Camera opened successfully" << std::endl;

    ObjectDetectionParameters detection_parameters;
    detection_parameters.detection_model = OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = true;

    if (detection_parameters.enable_tracking) {
        PositionalTrackingParameters positional_tracking_parameters;
        m_ZedCamera->enablePositionalTracking(positional_tracking_parameters);
    }

    sl::ERROR_CODE zed_error = m_ZedCamera->enableObjectDetection(detection_parameters);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << "enableObjectDetection: " << zed_error << "\nExit program.";
        m_ZedCamera->close();
        exit(-1);
    }

    m_RuntimeParams = RuntimeParameters();
    if (!isZedStatic) {
        m_RuntimeParams.measure3D_reference_frame = sl::REFERENCE_FRAME::WORLD; // Tracks zed movement
    }

    // Retrieve image dimensions.
    sl::Resolution resolution = m_ZedCamera->getCameraInformation().camera_configuration.resolution;
    m_ImageWidth = resolution.width;
    m_ImageHeight = resolution.height;

    // Create the left image Mat if it hasn't been created.
    if (!m_LeftMat) {
        m_LeftMat = new sl::Mat(m_ImageWidth, m_ImageHeight, MAT_TYPE::U8_C4, MEM::CPU);
    }

    return 0;
}

bool ZEDCustomManager::CaptureFrame() {
    if (!m_ZedCamera) {
        std::cerr << "Camera not initialized." << std::endl;
        return 1;
    }
    sl::ERROR_CODE state = m_ZedCamera->grab(m_RuntimeParams);
    if (state != ERROR_CODE::SUCCESS) {
        LogStatement("Failed to grab frame");
        std::cerr << "Failed to grab frame: " << (int)state << std::endl;
        return 1;
    }
    state = m_ZedCamera->retrieveImage(*m_LeftMat, VIEW::LEFT);
    if (state != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to retrieve image from camera L" << std::endl;
        return 1;
    }
    return 0;
}

void ZEDCustomManager::CloseZEDCamera() {
    if (m_ZedCamera) {
        m_ZedCamera->disablePositionalTracking();
        m_ZedCamera->disableObjectDetection();
        m_ZedCamera->close();
        delete m_ZedCamera;
        m_ZedCamera = nullptr;
    }
    if (m_LeftMat) {
        delete m_LeftMat;
        m_LeftMat = nullptr;
    }
}

void ZEDCustomManager::GetCurrentPosition(sl::Pose m_camPose) {
    m_ZedCamera->getPosition(m_camPose, sl::REFERENCE_FRAME::WORLD);
}

ZEDCustomManager::~ZEDCustomManager() {
    ZEDCustomManager::CloseZEDCamera();
}