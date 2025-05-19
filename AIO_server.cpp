#include <iostream>
#include "AIO_server.hpp"

float rotationVector[3];
float translationVector[3];
std::vector<int> removal_ids;
sl::Pose camPose;
sl::Mat pcloud;

std::atomic<bool> g_shouldExit{ false };

int key{ 0 };

void onSignal(int signum) {
    g_shouldExit = true;
}

int main(int argc, char** argv)
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

    // 3D Viewer
    GLViewer viewer;
    sl::CameraConfiguration const camera_config{ zedCustomManager->getCamera()->getCameraInformation().camera_configuration};
    sl::Resolution const pc_resolution{ std::min(camera_config.resolution.width, static_cast<size_t>(1280UL)),
                                       std::min(camera_config.resolution.height, static_cast<size_t>(720UL)) };
    sl::CameraConfiguration const camera_info{ zedCustomManager->getCamera()->getCameraInformation(pc_resolution).camera_configuration };
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, true);

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
        std::vector<DetectedMask> frameMasks; // = segPaintManager->GetMasks();

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

        cv::Mat paintedFrame = segPaintManager->InpaintFrame(zedCustomManager);
        cv::Mat display;
        cv::cvtColor(paintedFrame, display, cv::COLOR_BGRA2RGBA);

        /// Point cloud
        zedCustomManager->getCamera()->retrieveMeasure(pcloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU, pc_resolution);

        int W = pcloud.getWidth(), H = pcloud.getHeight();
        auto ptr = pcloud.getPtr<sl::float4>(sl::MEM::CPU);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;

                cv::Vec4b px = display.at<cv::Vec4b>(y, x);
                // pack into 0xBBGGRRAA on little-endian
                uint32_t packed =
                    (uint32_t(px[2]) << 0)   // B
                    | (uint32_t(px[1]) << 8)   // G
                    | (uint32_t(px[0]) << 16)   // R
                    | (uint32_t(px[3]) << 24);  // A
                ptr[idx].w = *reinterpret_cast<float*>(&packed);
            }
        }

        sl::Mat pcloudGPU;
        pcloud.copyTo(pcloudGPU, sl::COPY_TYPE::CPU_GPU);

        viewer.updateData(pcloudGPU, frameMasks, camPose.pose_data);
        
        //cv::imshow("Output", display);

        int const cv_key{ cv::waitKey(10) };
        int const gl_key{ viewer.getKey() };
        key = (gl_key == -1) ? cv_key : gl_key;
        if (key == 'p' || key == 32) {
            viewer.setPlaying(!viewer.isPlaying());
        }
        while ((key == -1) && !viewer.isPlaying() && viewer.isAvailable()) {
            int const cv_key{ cv::waitKey(10) };
            int const gl_key{ viewer.getKey() };
            key = (gl_key == -1) ? cv_key : gl_key;
            if (key == 'p' || key == 32) {
                viewer.setPlaying(!viewer.isPlaying());
            }
        }
        if (!viewer.isAvailable())
            break;
    }

    std::cout << "Program exiting..." << std::endl;

    cv::destroyAllWindows();
    viewer.exit();
    
    delete segPaintManager;
    delete socketManager;
    delete zedCustomManager;

    return 0;
}