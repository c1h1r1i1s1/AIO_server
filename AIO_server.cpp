#include <iostream>
#include "AIO_server.hpp"
#include "aruco_detection.hpp"
#include <stdlib.h>

float rotationVector[3];
float translationVector[3];
std::vector<int> removal_ids;

int main()
{
    //// Initialise Everything
    ZEDCustomManager* zedCustomManager = new ZEDCustomManager();
    if (zedCustomManager->InitializeZEDCamera()) {
        return 1;
    }
    
    SegPaintManager* segPaintManager = new SegPaintManager();
    if (segPaintManager->InitialiseEngines()) {
        return 1;
    }

    SocketManager* socketManager = new SocketManager();
    socketManager->start();

    ArucoDetector* arucoDetector = new ArucoDetector();
    arucoDetector->PrepareCameraParameters(zedCustomManager);

    
    //// First get headset position with aruco detection
    while (true) {
        zedCustomManager->CaptureFrame();
        cv::Mat frame = zedCustomManager->getCurrentMat();
        
        if (arucoDetector->arucoDetection(frame, rotationVector, translationVector)) {
            continue;
        } else {
            break;
        }
    }

    // Use vectors for headset object location
    // Possibly pass zed camera location to headset so user can confirm positioning


    //// Begin detection
    while (true) {
        zedCustomManager->CaptureFrame();

        int objCount = segPaintManager->ProcessFrame(zedCustomManager);
        if (objCount == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        std::vector<DetectedObject> frameObjects = segPaintManager->GetObjects();

        //Pass to headset in json format somehow
        //socketManager->broadcastBoundingBoxes(const std::string & boundingBoxData)

        // Get removal ids
        //selection_changes_buffer = socketManager->getSelectionChanges();
        // Iterate through and remove/add items back in

        cv::Mat paintedFrame = segPaintManager->InpaintFrame();
        cv::imshow("Output", paintedFrame);
    }

}