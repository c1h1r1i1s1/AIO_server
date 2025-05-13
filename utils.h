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

inline cv::Mat compositeOutput(const cv::Mat& predictedImage360, cv::Mat& mask720, cv::Mat& inputCVMat720)
{

    // Upscale from 360p to 720p
    // 720p resolution is 1280x720.
    cv::Mat predictedImage720;
    cv::resize(predictedImage360, predictedImage720, cv::Size(1280, 720), 0, 0, cv::INTER_LINEAR);
    //cv::cvtColor(predictedImage720, predictedImage720, cv::COLOR_RGB2BGR);

    // --- Process and Upscale the Mask ---
    if (mask720.channels() == 3)
        cv::cvtColor(mask720, mask720, cv::COLOR_BGR2GRAY);

    // Convert to CV_8U, scaling from [0,1] to [0,255].
    mask720.convertTo(mask720, CV_8U, 255.0);

    cv::Mat invMask = cv::Mat::ones(mask720.size(), mask720.type()) * 255 - mask720;

    // --- Composite the Inpainted Region onto the Original 720p Image ---
    cv::Mat compositedImage = inputCVMat720.clone();
    // The non-zero values in mask720 indicate where to copy the predicted image.
    predictedImage720.copyTo(compositedImage, invMask);

    return compositedImage;
}

inline cv::Mat combineMasks(const std::vector<int> removal_ids, const cv::Mat input720, const std::vector<DetectedMask> detectedMasks) {

    // Create an unordered_set for quick lookup of removal IDs.
    std::unordered_set<int> removalSet(removal_ids.begin(), removal_ids.end());

    cv::Mat compositeMask(input720.size(), CV_32F, cv::Scalar(1.0));

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

            // Get specific boxed section of composite mask and remove mask
            cv::Mat roi = compositeMask(bbox);
            cv::Mat result = roi.mul(1.0 - binaryMask32);
            result.copyTo(compositeMask(bbox));
        }
    }

    cv::Mat binaryMask;
    compositeMask.convertTo(binaryMask, CV_8U, -255.0, 255.0);

    int kernelSize = 5; // A 3x3 kernel; increase size for a larger buffer.
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

inline void ConvertCameraPose(
    const seg::float3 corners[8],
    seg::float3 cornersCam[8],
    sl::Rotation rot,
    const sl::Translation tran) {
    
    for (int i = 0; i < 8; ++i) {
        float wx = corners[i].x;
        float wy = corners[i].y;
        float wz = corners[i].z;

        // 1) Translate so camera is at the origin:
        float dx = wx - tran[0];
        float dy = wy - tran[1];
        float dz = wz - tran[2];

        // 2) Apply the inverse rotation:
        cornersCam[i].x = rot(0, 0) * dx + rot(1, 0) * dy + rot(2, 0) * dz;
        cornersCam[i].y = rot(0, 1) * dx + rot(1, 1) * dy + rot(2, 1) * dz;
        cornersCam[i].z = rot(0, 2) * dx + rot(1, 2) * dy + rot(2, 2) * dz;
    }
}

inline seg::float3 ComputeCenter(seg::float3 corners[8]) {
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

inline seg::float3 ComputeSize(seg::float3 corners[8]) {
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
// - RGB360 is a 720p RGB image from the ZED camera (after conversion) CV_32FC3
// - boxMask is a CV_32FC3 binary (grayscale) mask from YOLO segmentation where the mask regions are non-zero
inline cv::Mat prepareMaskedImage(const cv::Mat& RGB720, const cv::Mat& combinedMask) {

    // Multiply the input image by the inverse mask
    // This will zero out the regions where the mask is 1.
    cv::Mat maskedImage = RGB720.mul(combinedMask);

    // Rescale to 360 for inpainting
    cv::Mat RGB360;
    RGB360.create(360, 640, RGB720.type());
    cv::resize(maskedImage, RGB360, cv::Size(640, 360));

    return RGB360;
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

inline cv::Mat warpAndCombineMasksToRight(
    sl::Camera* zed,
    const cv::Mat& input720,
    const std::vector<int>& removal_ids,
    const std::vector<DetectedMask>& detectedMasks)
{
    // get disparity and clean it up
    sl::Mat dispZed;
    zed->retrieveMeasure(dispZed, sl::MEASURE::DISPARITY, sl::MEM::CPU);
    cv::Mat dispCV(
        dispZed.getHeight(), dispZed.getWidth(),
        CV_32F, dispZed.getPtr<sl::float1>(sl::MEM::CPU),
        dispZed.getStepBytes(sl::MEM::CPU)
    );
    cv::medianBlur(dispCV, dispCV, 3);
    cv::patchNaNs(dispCV, 0.0f);

    // Prepare fast lookup of which IDs to remove
    std::unordered_set<int> removalSet(removal_ids.begin(), removal_ids.end());

    // This will hold 0 or 255 per pixel for the right mask removal regions
    cv::Mat outBin = cv::Mat::zeros(input720.size(), CV_8U);

    // Process each detected mask on its own
    for (auto& dm : detectedMasks) {
        if (removalSet.count(dm.id) == 0)
            continue;

        cv::Rect bbox_roi = dm.bbox;
        if (bbox_roi.area() == 0) continue;

        // threshold the float mask into a 0/255 CV_8U mask
        cv::Mat mask8U;
        dm.mask.convertTo(mask8U, CV_8U, 255.0);  // now 0 or 255

        // compute mean disparity inside that region
        cv::Mat dispROI = dispCV(bbox_roi);
        
        cv::Scalar meanDisp = cv::mean(dispROI, mask8U);
        float meanD = static_cast<float>(meanDisp[0]);

        if (std::abs(meanD) < 1e-3f)
            continue;
        int dx = std::lround(meanD);

        // build a full-frame binary image with just this mask
        cv::Mat frameMask = cv::Mat::zeros(input720.size(), CV_8U);
        mask8U.copyTo(frameMask(bbox_roi));

        // warp it horizontally by dx
        cv::Mat T = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, 0);
        cv::Mat shifted;
        cv::warpAffine(
            frameMask, shifted, T, frameMask.size(),
            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0)
        );

        // accumulate into our final removal map
        outBin |= shifted;
    }

    // dilate a little to buffer edges
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 5)
    );
    cv::dilate(outBin, outBin, kernel, cv::Point(-1, -1), 1);

    cv::Mat outFloat;
    outBin.convertTo(outFloat, CV_32F, -1.0f / 255.0f, 1.0f);

    // make 3-channel float RGB
    cv::Mat out3Ch;
    cv::cvtColor(outFloat, out3Ch, cv::COLOR_GRAY2RGB);

    return out3Ch;
}

#endif  // TRTX_YOLOV8SEG_UTILS_H_

#pragma once
