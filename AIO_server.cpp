#include <iostream>
#include "AIO_server.hpp"

float rotationVector[3];
float translationVector[3];
std::vector<int> removal_ids;
sl::Pose camPose;

std::atomic<bool> g_shouldExit{ false };

void onSignal(int signum) {
    g_shouldExit = true;
}

int main()
{
    // Set up close handling
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

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
    while (!g_shouldExit) {
        zedCustomManager->CaptureFrame();

        int objCount = segPaintManager->ProcessFrame(zedCustomManager);
        if (objCount == 0) {
            std::cout << "No objects found" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        std::vector<DetectedObject> frameObjects = segPaintManager->GetObjects();

        // Get world pose of camera for correct object positions
        zedCustomManager->GetCurrentPosition(camPose);
        sl::Rotation rot = camPose.getRotationMatrix();
        sl::Translation tran = camPose.getTranslation();

        // Pass to headset in json format
        socketManager->broadcastBoundingBoxes(frameObjects, rot, tran);

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

        cv::Mat paintedFrame_L, paintedFrame_R;
        std::vector<cv::Mat> paintedFrames = segPaintManager->InpaintFrame(zedCustomManager);
        if (paintedFrames.size() < 2) {
            std::cout << "not enough painted frames produced" << std::endl;
            return 1;
        }
        paintedFrame_L = paintedFrames[0];
        paintedFrame_R = paintedFrames[1];
        cv::Mat display_L;
        cv::Mat display_R;
        cv::cvtColor(paintedFrame_L, display_L, cv::COLOR_BGRA2RGB);
        cv::cvtColor(paintedFrame_R, display_R, cv::COLOR_BGRA2RGB);
        cv::imshow("Output_L", display_L);
        cv::imshow("Output_R", display_R);
        if (cv::waitKey(1) == 27)  // exit on ESC
            break;
    }

    std::cout << "Program exiting..." << std::endl;
    
    delete segPaintManager;
    delete socketManager;
    delete zedCustomManager;

    return 0;
}