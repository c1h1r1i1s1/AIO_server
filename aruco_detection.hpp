#ifndef ARUCO_DETECTION_HPP
#define ARUCO_DETECTION_HPP

#include "ZEDCustomManager.hpp"
#include <opencv2/opencv.hpp>

class ArucoDetector {
public:
	void PrepareCameraParameters(ZEDCustomManager* zedCustomManager);
	bool arucoDetection(cv::Mat inputImage, float* rotationVector, float* translationVector);
private:
	static float m_cameraMatrix[9];
	static float m_distCoeffs[12];
	static cv::Mat* m_cameraMatrixMat;
	static cv::Mat* m_distCoeffsMat;
};

#endif