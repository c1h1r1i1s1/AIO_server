// This is the main program that contains the main function loop.
// It pulls in the various functions and manages everything
#include <iostream>
#include "AIO_server.hpp"


// Set to false if expecting ZED camera movement
// Slightly impacts object positioning accuracy but tracks well
bool isZedStatic = true;

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
    if (zedCustomManager->InitializeZEDCamera(isZedStatic)) {
        return 1;
    }
    camPose.pose_data.setIdentity();
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
    std::vector<DetectedMask> frameMasks;

    cv::HersheyFonts font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
    auto prev_frame_time = std::chrono::high_resolution_clock::now();;
    auto new_frame_time = std::chrono::high_resolution_clock::now();;
    double fps = 0;

    //// Begin detection
    std::cout << "Main loop beginning" << std::endl;
    while (!g_shouldExit) {
        zedCustomManager->CaptureFrame();
        int objCount = segPaintManager->ProcessFrame(zedCustomManager);

        std::vector<DetectedObject> frameObjects = segPaintManager->GetObjects();

        // Get any selections from headset
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

        // Composite inpainted frame onto point cloud. NOTE this is slow due to CPU_GPU copies
        zedCustomManager->getCamera()->retrieveMeasure(pcloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU, pc_resolution);
        int W = pcloud.getWidth(), H = pcloud.getHeight();
        auto ptr = pcloud.getPtr<sl::float4>(sl::MEM::CPU);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;
                cv::Vec4b px = display.at<cv::Vec4b>(y, x);
                uint32_t packed = (uint32_t(px[2]) << 0) | (uint32_t(px[1]) << 8) | (uint32_t(px[0]) << 16) | (uint32_t(px[3]) << 24);
                ptr[idx].w = *reinterpret_cast<float*>(&packed);
            }
        }
        sl::Mat pcloudGPU;
        pcloud.copyTo(pcloudGPU, sl::COPY_TYPE::CPU_GPU);

        zedCustomManager->GetCurrentPosition(camPose);
        viewer.updateData(pcloudGPU, frameMasks, camPose.pose_data);

        socketManager->broadcastBoundingBoxes(frameObjects);
        
        // Calculate FPS
        /*new_frame_time = std::chrono::high_resolution_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::microseconds>(new_frame_time - prev_frame_time).count() / 1000000.0;
        prev_frame_time = new_frame_time;
        fps = 1.0 / frame_time;
        int fps_int = std::round(fps);
        std::string fps_text = std::to_string(fps_int);
        cv::putText(display, fps_text, cv::Point(7, 70), font, 3, (100, 255, 0), 3, cv::LineTypes::LINE_AA);*/

        cv::imshow("Output", display);

        // Keyboard controls
        int const cv_key{ cv::waitKey(1) };
        int const gl_key{ viewer.getKey() };
        key = (gl_key == -1) ? cv_key : gl_key;
        if (key == 'p' || key == 32) {
            viewer.setPlaying(!viewer.isPlaying());
        }
        while ((key == -1) && !viewer.isPlaying() && viewer.isAvailable()) {
            int const cv_key{ cv::waitKey(1) };
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