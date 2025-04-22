#include <iostream>
#include "AIO_server.hpp"

float rotationVector[3];
float translationVector[3];
std::vector<int> removal_ids;

int main()
{
    //// Initialise Everything
    std::cout << "Initialising..." << std::endl;
    ZEDCustomManager* zedCustomManager = new ZEDCustomManager();
    if (zedCustomManager->InitializeZEDCamera()) {
        return 1;
    }
    std::cout << "\tZedCamera Initialised" << std::endl;
    
    SegPaintManager* segPaintManager = new SegPaintManager();
    if (segPaintManager->InitialiseEngines()) {
        return 1;
    }
    std::cout << "\tSegPaint Initialised" << std::endl;

    ChangeQueue* g_changeQueue = new ChangeQueue();

    SocketManager* socketManager = new SocketManager(g_changeQueue);
    socketManager->start();
    std::cout << "\tHeadset Socket Initialised" << std::endl;

    //// Begin detection
    std::cout << "Main loop beginning" << std::endl;
    while (true) {
        zedCustomManager->CaptureFrame();

        int objCount = segPaintManager->ProcessFrame(zedCustomManager);
        if (objCount == 0) {
            std::cout << "No objects found" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        std::vector<DetectedObject> frameObjects = segPaintManager->GetObjects();

        // Pass to headset in json format
        socketManager->broadcastBoundingBoxes(frameObjects);

        // Remove selected objects
        ObjectChange change;
        while (g_changeQueue->popChange(change)) {
            if (change.selected) {
                segPaintManager->ErasePrivateObject(change.id);
            }
            else {
                segPaintManager->ShowPrivateObject(change.id);
            }
        }

        cv::Mat paintedFrame = segPaintManager->InpaintFrame();
        cv::Mat display;
        cv::cvtColor(paintedFrame, display, cv::COLOR_BGRA2RGB);
        cv::imshow("Output", display);
        if (cv::waitKey(1) == 27)  // exit on ESC
            break;
    }

}