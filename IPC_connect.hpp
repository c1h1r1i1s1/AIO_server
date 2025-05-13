#ifndef IPC_CONNECT_HPP
#define IPC_CONNECT_HPP

#include <iostream>
#include "framework.h"
#include "opencv2/opencv.hpp"
#include "utils.h"

class IPC_connect {
    public:
        IPC_connect();

        ~IPC_connect();

        bool inpaintBlob(const cv::Mat& inputBlob_L, const cv::Mat& inputBlob_R, cv::Mat& outputBlob_L, cv::Mat& outputBlob_R);

    private:
        HANDLE hMapFile = NULL;
        LPVOID pBuf = NULL;
        HANDLE hInputEvent = NULL;
        HANDLE hOutputEvent = NULL;
};

#endif