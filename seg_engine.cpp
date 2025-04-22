#include "seg_engine.hpp"
#include "ZEDCustomManager.hpp"
#include <cstring>
#include <sstream>
#include <filesystem>

// Constructor: create the detector instance and perform initial setup.
SegmentationModel::SegmentationModel(const std::string& engineFilePath)
{
    // Create the detector and load the engine.
    m_detector = new YOLOv11_seg(engineFilePath);
    // The make_pipe call allocates buffers; 'false' means no warmup.
    m_detector->make_pipe();
}

// Destructor: release the detector.
SegmentationModel::~SegmentationModel() {
    if (m_detector) {
        delete m_detector;
        m_detector = nullptr;
    }
}

// Runs segmentation on the input image and populates the detected objects and masks.
int SegmentationModel::segmentFrame(const cv::Mat& input360,
    std::vector<DetectedObject>& detectedObjects,
    std::vector<DetectedMask>& detectedMasks,
    sl::Camera* zedCamera)
{
    // Clear any previous results.
    detectedObjects.clear();
    detectedMasks.clear();

    std::vector<DetectedMaskPre> detectedMasksPre;

    // Run segmentation with the detector.
    m_detector->copy_from_Mat(input360);
    m_detector->infer();

    // Postprocess the outputs to get segmentation objects.
    std::vector<seg::Object> objs;
    m_detector->postprocess(objs, m_scoreThresh, m_iouThresh, m_topk, m_segChannels);

    // Prepare data for the ZED SDK.
    std::vector<sl::CustomMaskObjectData> objects_in;
    objects_in.reserve(objs.size());

    for (seg::Object& obj : objs) {
        DetectedMaskPre tmpMaskPre;
        sl::CustomMaskObjectData tmp;

        tmpMaskPre.bbox = obj.rect;
        tmp.unique_object_id = sl::generate_unique_id();
        tmpMaskPre.unique_object_id = tmp.unique_object_id;
        tmp.probability = obj.prob;
        tmp.label = obj.label;
        tmp.bounding_box_2d = convertCvRect2SdkBbox(obj.rect);

        tmpMaskPre.mask = obj.boxMask;

        tmp.is_grounded = (obj.label == 0);
        cvMat2slMat(obj.boxMask).copyTo(tmp.box_mask, sl::COPY_TYPE::CPU_CPU);

        objects_in.push_back(tmp);
        detectedMasksPre.push_back(tmpMaskPre);

        // Possible danger here with conversion of bboxes modifying tmMask data
    }

    sl::Objects objects;
    sl::CustomObjectDetectionRuntimeParameters cod_rt_param;
    zedCamera->ingestCustomMaskObjects(objects_in);
    zedCamera->retrieveObjects(objects, cod_rt_param);

    bool person_found = false;

    for (size_t i = 0; i < objects.object_list.size(); i++) {
        DetectedObject tmpObject;
        DetectedMask tmpMask;

        tmpObject.id = objects.object_list.at(i).id;
        tmpObject.probability = objects.object_list.at(i).confidence;
        tmpObject.label = objects.object_list.at(i).raw_label;

        if (tmpObject.label == 0) {
            person_found = true;
        }

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
                tmpMask.mask = preMask.mask;
                tmpMask.bbox = preMask.bbox;
                detectedMasks.push_back(tmpMask);
            }
        }
    }

    return static_cast<int>(detectedObjects.size());
}
