#ifndef ZEDCUSTOMMANAGERPLUGIN_HPP
#define ZEDCUSTOMMANAGERPLUGIN_HPP

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

class ZEDCustomManager {
	public:
		explicit ZEDCustomManager();
		sl::Camera* getCamera();
		cv::Mat getCurrentMat();

		bool InitializeZEDCamera(bool isZedStatic);
		bool CaptureFrame();
		void GetCurrentPosition(sl::Pose m_camPose);
		
		~ZEDCustomManager();

	private:
		// member variables to hold our camera instance, image Mat, and parameters.
		static sl::Camera* m_ZedCamera;
		static sl::InitParameters m_InitParams;
		static sl::RuntimeParameters m_RuntimeParams;
		static sl::Mat* m_LeftMat;

		static int m_ImageWidth;
		static int m_ImageHeight;
		static const int m_Channels = 4; // Using MAT_8U_C4: 4 channels (BGRA)

		void CloseZEDCamera();
};

#endif // ZEDCUSTOMMANAGERPLUGIN_HPP
