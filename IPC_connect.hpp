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

        bool inpaintBlob(const cv::Mat& inputBlob, cv::Mat& outputBlob);

    private:
        HANDLE hMapFile = NULL;
        LPVOID pBuf = NULL;
        HANDLE hInputEvent = NULL;
        HANDLE hOutputEvent = NULL;
};

#endif