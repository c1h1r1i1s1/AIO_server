#ifndef TRTX_YOLOV8SEG_UTILS_H_
#define TRTX_YOLOV8SEG_UTILS_H_

#include "common.hpp"
#include <string>
#include <vector>
#include <math.h>
#include "framework.h"
#include <cuda_fp16.h>

using namespace seg;

inline void LogStatement(const std::string& statement) {
    std::ostringstream oss;
    oss << "Incoming Log statement: " << statement;
    OutputDebugStringA(oss.str().c_str());
}

inline cv::Mat slMat2cvMat(sl::Mat const& input) {
    int cv_type = -1;
    switch (input.getDataType()) {
    case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
        break;
    case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
        break;
    case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
        break;
    case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
        break;
    case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
        break;
    case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
        break;
    case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
        break;
    case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
        break;
    default: break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

// Be careful about the memory owner, might want to use it like : 
//     cvMat2slMat(cv_mat).copyTo(sl_mat, sl::COPY_TYPE::CPU_CPU);
inline sl::Mat cvMat2slMat(cv::Mat const& input) {
    sl::MAT_TYPE sl_type;
    switch (input.type()) {
    case CV_32FC1: sl_type = sl::MAT_TYPE::F32_C1;
        break;
    case CV_32FC2: sl_type = sl::MAT_TYPE::F32_C2;
        break;
    case CV_32FC3: sl_type = sl::MAT_TYPE::F32_C3;
        break;
    case CV_32FC4: sl_type = sl::MAT_TYPE::F32_C4;
        break;
    case CV_8UC1: sl_type = sl::MAT_TYPE::U8_C1;
        break;
    case CV_8UC2: sl_type = sl::MAT_TYPE::U8_C2;
        break;
    case CV_8UC3: sl_type = sl::MAT_TYPE::U8_C3;
        break;
    case CV_8UC4: sl_type = sl::MAT_TYPE::U8_C4;
        break;
    default: break;
    }
    return sl::Mat(input.cols, input.rows, sl_type, input.data, input.step, sl::MEM::CPU);
}

// Inline helper to convert FP16 (stored as uint16_t) to float.
inline float fp16ToFloat(uint16_t h) {
    __half h_val = *reinterpret_cast<__half*>(&h);
    return __half2float(h_val);
}

inline cv::Mat compositeOutput(const cv::Mat& predictedImage360, cv::Mat& mask360, cv::Mat& inputCVMat720)
{

    // Upscale from 360p to 720p
    // 720p resolution is 1280x720.
    cv::Mat predictedImage720;
    cv::resize(predictedImage360, predictedImage720, cv::Size(1280, 720), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(predictedImage720, predictedImage720, cv::COLOR_RGB2BGR);

    // --- Process and Upscale the Mask ---
    if (mask360.channels() == 3)
        cv::cvtColor(mask360, mask360, cv::COLOR_BGR2GRAY);

    // Convert to CV_8U, scaling from [0,1] to [0,255].
    mask360.convertTo(mask360, CV_8U, 255.0);

    // Upscale the mask to 720p using nearest neighbor to preserve binary values.
    cv::Mat mask720;
    cv::resize(mask360, mask720, cv::Size(1280, 720), 0, 0, cv::INTER_NEAREST);

    cv::Mat invMask = cv::Mat::ones(mask720.size(), mask720.type()) * 255 - mask720;

    // --- Composite the Inpainted Region onto the Original 720p Image ---
    cv::Mat compositedImage = inputCVMat720.clone();
    // The non-zero values in mask720 indicate where to copy the predicted image.
    predictedImage720.copyTo(compositedImage, invMask);

    return compositedImage;
}

inline cv::Mat combineMasks(const std::vector<int> removal_ids, const cv::Mat input360, const std::vector<DetectedMask> detectedMasks) {

    // Create an unordered_set for quick lookup of removal IDs.
    std::unordered_set<int> removalSet(removal_ids.begin(), removal_ids.end());

    cv::Mat compositeMask(input360.size(), CV_32F, cv::Scalar(1.0));

    // Iterate over each detected mask.
    for (const auto& dm : detectedMasks) {
        // If the object's ID is in the removal list.
        if (removalSet.find(dm.id) != removalSet.end()) {
            cv::Rect2f bbox = dm.bbox;
            cv::Mat mask = dm.mask;

            cv::Mat binaryMask;

            // Set to 0/1 values
            cv::threshold(mask, binaryMask, 127, 1.0, cv::THRESH_BINARY);

            cv::Mat binaryMask32;
            binaryMask.convertTo(binaryMask32, CV_32F);

            // Convert single-channel mask to 3 channels to match the frame
            /*cv::Mat mask3Ch;
            cv::cvtColor(binaryMask32, mask3Ch, cv::COLOR_GRAY2RGB);*/

            // Get specific boxed section of composite mask and remove mask
            cv::Mat roi = compositeMask(bbox);
            cv::Mat result = roi.mul(1.0 - binaryMask32);
            result.copyTo(compositeMask(bbox));
        }
    }

    cv::Mat binaryMask;
    compositeMask.convertTo(binaryMask, CV_8U, -255.0, 255.0);

    int kernelSize = 4; // A 3x3 kernel; increase size for a larger buffer.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

    // Perform dilation on the full-frame mask.
    cv::Mat dilatedMask;
    cv::dilate(binaryMask, dilatedMask, kernel, cv::Point(-1, -1), 1);

    cv::Mat finalMask;
    dilatedMask.convertTo(finalMask, CV_32F, -1/255.0, 1.0);

    cv::Mat final3ChMask;
    cv::cvtColor(finalMask, final3ChMask, cv::COLOR_GRAY2RGB);

    return final3ChMask;
}

inline seg::float3 ComputeCenter(const seg::float3 corners[8]) {
    seg::float3 min = corners[0];
    seg::float3 max = corners[0];

    for (int i = 1; i < 8; ++i) {
        if (corners[i].x < min.x) min.x = corners[i].x;
        if (corners[i].y < min.y) min.y = corners[i].y;
        if (corners[i].z < min.z) min.z = corners[i].z;

        if (corners[i].x > max.x) max.x = corners[i].x;
        if (corners[i].y > max.y) max.y = corners[i].y;
        if (corners[i].z > max.z) max.z = corners[i].z;
    }

    seg::float3 center;
    center.x = (min.x + max.x) / 2000.0f;
    center.y = (min.y + max.y) / 2000.0f;
    center.z = (min.z + max.z) / 2000.0f;

    return center;
}

inline seg::float3 ComputeSize(const seg::float3 corners[8]) {
    seg::float3 min = corners[0];
    seg::float3 max = corners[0];

    for (int i = 1; i < 8; ++i) {
        if (corners[i].x < min.x) min.x = corners[i].x;
        if (corners[i].y < min.y) min.y = corners[i].y;
        if (corners[i].z < min.z) min.z = corners[i].z;

        if (corners[i].x > max.x) max.x = corners[i].x;
        if (corners[i].y > max.y) max.y = corners[i].y;
        if (corners[i].z > max.z) max.z = corners[i].z;
    }

    seg::float3 size;
    size.x = (max.x - min.x) / 1000.0f; // width
    size.y = (max.y - min.y) / 1000.0f; // height
    size.z = (max.z - min.z) / 1000.0f; // depth

    return size;
}

// This function assumes:
// - RGB360 is a 360x640 RGB image from the ZED camera (after conversion) CV_32FC3
// - boxMask is a CV_32FC3 binary (grayscale) mask from YOLO segmentation where the mask regions are non-zero
inline cv::Mat prepareMaskedImage(const cv::Mat& RGB360, const cv::Mat& combinedMask) {

    // Multiply the input image by the inverse mask
    // This will zero out the regions where the mask is 1.
    cv::Mat maskedImage = RGB360.mul(combinedMask);

    return maskedImage;
}

// Rearranging, scaling, and converting to an 8-bit image.
inline bool extractOutputs(cv::Mat& rawPaintData, cv::Mat& predictedImage360) {

    const int batch = 1;
    const int channels = 3;
    const int height = 360;
    const int width = 640;

    // Verify the raw data has the correct number of elements and type.
    if (rawPaintData.total() != static_cast<size_t>(batch * channels * height * width) ||
        rawPaintData.type() != CV_32F)
    {
        std::cerr << "Unexpected raw data size or type" << std::endl;
        return 1;
    }

    // Create an intermediate image to hold the FP32 data in HWC order.
    cv::Mat imageFP32(height, width, CV_32FC3);

    // rawPaintData is in NCHW format. For batch 1, we compute the index as:
    // index = c * (height * width) + h * width + w.
    float* rawData = reinterpret_cast<float*>(rawPaintData.data);
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = c * (height * width) + h * width + w;
                imageFP32.at<cv::Vec3f>(h, w)[c] = rawData[idx];
            }
        }
    }

    // Convert the normalized FP32 values from range [-1, 1] to [0, 255].
    // Adjust the scaling if your data is normalized differently.
    imageFP32 = (imageFP32 + 1.0f) * 127.5f;

    // Convert the image to an 8-bit unsigned integer image.
    imageFP32.convertTo(predictedImage360, CV_8UC3);

    return 0;
}

#endif  // TRTX_YOLOV8SEG_UTILS_H_

#pragma once
