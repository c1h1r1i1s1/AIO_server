// This manages the YOLO model and provides a set of functions for accessing the model I/O
#include "seg_engine.hpp"
#include "ZEDCustomManager.hpp"
#include <cstring>
#include <sstream>
#include <filesystem>

SegmentationModel::SegmentationModel(const std::string& engineFilePath)
{
    m_detector = new YOLOv11_seg(engineFilePath);
    m_detector->make_pipe();
}

SegmentationModel::~SegmentationModel() {
    if (m_detector) {
        delete m_detector;
        m_detector = nullptr;
    }
}

// Runs segmentation on the input image and populates the detected objects and masks.
int SegmentationModel::segmentFrame(const cv::Mat& input720,
    std::vector<DetectedObject>& detectedObjects,
    std::vector<DetectedMask>& detectedMasks,
    sl::Camera* zedCamera)
{
    // Clear any previous results.
    detectedObjects.clear();
    detectedMasks.clear();

    std::vector<DetectedMaskPre> detectedMasksPre;

    // Run segmentation with the detector.
    m_detector->copy_from_Mat(input720);
    m_detector->infer();

    // Postprocess the outputs to get segmentation objects.
    std::vector<seg::Object> objs;
    m_detector->postprocess(objs, m_scoreThresh, m_iouThresh, m_topk, m_segChannels);

    // Prepare data for the ZED SDK.
    std::vector<sl::CustomMaskObjectData> objects_in;
    objects_in.reserve(objs.size());
    detectedMasksPre.reserve(objs.size());

    for (seg::Object& obj : objs) {
        objects_in.emplace_back();
        sl::CustomMaskObjectData& tmp{ objects_in.back() };
        tmp.unique_object_id = sl::generate_unique_id();
        tmp.probability = obj.prob;
        tmp.label = obj.label;

        // Currently set to ignore people, tables, phones and controllers
        if (tmp.label == 0 || tmp.label == 60 || tmp.label == 65 || tmp.label == 67) {
            continue;
        }

        tmp.bounding_box_2d = convertCvRect2SdkBbox(obj.rect);
        // Ground object if its a person
        tmp.is_grounded = (obj.label == 0);
        // others are tracked in full 3D space

        cvMat2slMat(obj.boxMask).copyTo(tmp.box_mask, sl::COPY_TYPE::CPU_CPU);

        detectedMasksPre.emplace_back();
        DetectedMaskPre& tmpMask{ detectedMasksPre.back() };
        tmpMask.bbox = obj.rect;
        tmpMask.unique_object_id = tmp.unique_object_id;
        tmpMask.mask = obj.boxMask;
    }

    sl::Objects objects;
    sl::CustomObjectDetectionRuntimeParameters cod_rt_param;
    zedCamera->ingestCustomMaskObjects(objects_in);
    zedCamera->retrieveObjects(objects, cod_rt_param);

    for (size_t i = 0; i < objects.object_list.size(); i++) {
        DetectedObject tmpObject;
        DetectedMask tmpMask;

        tmpObject.id = objects.object_list.at(i).id;
        tmpObject.probability = objects.object_list.at(i).confidence;
        tmpObject.label = objects.object_list.at(i).raw_label;

        // If bounding boxes are broken or empty, skip
        if (objects.object_list.at(i).bounding_box.size() >= 8) {
            memcpy(tmpObject.bounding_box_3d,
                objects.object_list.at(i).bounding_box.data(),
                8 * sizeof(sl::float3));
        }
        else {
            LogStatement("Skipping id=" + std::to_string(objects.object_list[i].id) +
                " (got " + std::to_string(objects.object_list.at(i).bounding_box.size()) + " corners)\n");
            continue;
        }

        detectedObjects.push_back(tmpObject);

        for (DetectedMaskPre preMask : detectedMasksPre) {
            if (objects.object_list.at(i).unique_object_id == preMask.unique_object_id) {
                tmpMask.id = objects.object_list.at(i).id;
                tmpMask.position = objects.object_list.at(i).position;
                tmpMask.tracking_state = objects.object_list.at(i).tracking_state;
                tmpMask.raw_label = objects.object_list.at(i).raw_label;
                tmpMask.zed_bb = objects.object_list.at(i).bounding_box;
                tmpMask.mask = preMask.mask;
                tmpMask.bbox = preMask.bbox;
                detectedMasks.push_back(tmpMask);
            }
        }
    }

    return static_cast<int>(detectedObjects.size());
}
