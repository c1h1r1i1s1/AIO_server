#include "aruco_detection.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>


void ArucoDetector::PrepareCameraParameters(ZEDCustomManager* zedCustomManager) {
    sl::Camera* zedCamera = zedCustomManager->getCamera();

    sl::CameraParameters leftCamParams = zedCamera->getCameraInformation().camera_configuration.calibration_parameters.left_cam;

    // Create camera matrix
    m_cameraMatrix[0] = leftCamParams.fx; m_cameraMatrix[1] = 0;             m_cameraMatrix[2] = leftCamParams.cx;
    m_cameraMatrix[3] = 0;             m_cameraMatrix[4] = leftCamParams.fy; m_cameraMatrix[5] = leftCamParams.cy;
    m_cameraMatrix[6] = 0;             m_cameraMatrix[7] = 0;             m_cameraMatrix[8] = 1;

    for (int i = 0; i < 12; ++i) {
        m_distCoeffs[i] = static_cast<float>(leftCamParams.disto[i]);
    }

    m_cameraMatrixMat = new cv::Mat(3, 3, CV_32F, m_cameraMatrix);
    m_distCoeffsMat = new cv::Mat(1, 5, CV_32F, m_distCoeffs);
}

bool ArucoDetector::arucoDetection(cv::Mat inputImage, float* rotationVector, float* translationVector) {
    try {
        // Convert the image to grayscale
        cv::Mat gray;
        cv::cvtColor(inputImage, gray, cv::COLOR_BGRA2GRAY);

        // Define ArUco dictionary and detector parameters
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        cv::Ptr<cv::aruco::Dictionary> dictionaryPtr = cv::makePtr<cv::aruco::Dictionary>(dictionary);  // Create pointer from dictionary
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::makePtr<cv::aruco::DetectorParameters>();


        // Detect markers
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<int> markerIds;
        cv::aruco::detectMarkers(gray, dictionaryPtr, markerCorners, markerIds, parameters);

        if (markerIds.empty()) {
            return 1; // No markers detected
        }

        // Estimate pose of the first detected marker
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, *m_cameraMatrixMat, *m_distCoeffsMat, rvecs, tvecs);

        if (!rvecs.empty() && !tvecs.empty()) {
            // Copy the rotation and translation vectors to the output pointers
            rotationVector[0] = static_cast<float>(rvecs[0][0]);
            rotationVector[1] = static_cast<float>(rvecs[0][1]);
            rotationVector[2] = static_cast<float>(rvecs[0][2]);

            translationVector[0] = static_cast<float>(tvecs[0][0]);
            translationVector[1] = static_cast<float>(tvecs[0][1]);
            translationVector[2] = static_cast<float>(tvecs[0][2]);

            return 0;
        }

        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in DetectArucoMarker: " << e.what() << std::endl;
        return false;
    }
}