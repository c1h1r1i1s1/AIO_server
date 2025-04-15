#ifndef SEGMENTATION_MODEL_HPP
#define SEGMENTATION_MODEL_HPP

#include "yolov11-seg.hpp"
#include <vector>
#include <string>

using namespace seg;

class SegmentationModel {
public:
    explicit SegmentationModel(const std::string& engineFilePath);

    ~SegmentationModel();

    int segmentFrame(const cv::Mat& input360,
        std::vector<DetectedObject>& detectedObjects,
        std::vector<DetectedMask>& detectedMasks,
        sl::Camera* zedCamera);

    // Prevent copying.
    SegmentationModel(const SegmentationModel&) = delete;
    SegmentationModel& operator=(const SegmentationModel&) = delete;

private:
    YOLOv11_seg* m_detector; // Owned detector instance

    // Configuration parameters for postprocessing.
    const int m_topk = 100;
    const int m_segChannels = 32;
    const float m_scoreThresh = 0.5F;
    const float m_iouThresh = 0.65F;
};

#endif // SEGMENTATION_MODEL_HPP
