#include "seg_paint_manager.hpp"
#include "ZEDCustomManager.hpp"

std::vector<DetectedObject> SegPaintManager::m_detectedObjects;
std::vector<DetectedMask> SegPaintManager::m_detectedMasks;
std::vector<int> SegPaintManager::m_removal_ids;

bool SegPaintManager::InitialiseEngines() {
        
    try {

        m_IPC_connector = new IPC_connect();
        m_segmentationModel = new SegmentationModel("yolov11m-seg.engine");

    } catch (const std::exception& e) {
        LogStatement("Loading engines failed");
        LogStatement(e.what());
        std::cerr << "Loading engines failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;

}

int SegPaintManager::ProcessFrame(ZEDCustomManager* zedCustomManager) {

    m_inputCVMat720 = zedCustomManager->getCurrentMat();
        
    // Scale down to 360p
    cv::cvtColor(m_inputCVMat720, m_inputCVMat720, cv::COLOR_BGRA2RGB);
    m_input360.create(360, 640, m_inputCVMat720.type());
    cv::resize(m_inputCVMat720, m_input360, cv::Size(640, 360));

    // Segment
    m_detectedObjects.clear();
    m_detectedMasks.clear();
    int objectCount = m_segmentationModel->segmentFrame(m_input360, m_detectedObjects, m_detectedMasks, zedCustomManager->getCamera());

    return objectCount;

}

std::vector<DetectedObject> SegPaintManager::GetObjects() const { // Could use maxObjects

    return m_detectedObjects;

}

void SegPaintManager::getSegFrame() {
    // Combine selected masks
    cv::Mat combinedMask = combineMasks(m_removal_ids, m_input360, m_detectedMasks);
}

bool SegPaintManager::ErasePrivateObject(int id) {

    // Check if already there
    for (int i = 0; i < m_removal_ids.size(); i++) {
        if (m_removal_ids.at(i) == id) {
            return 0;
        }
    }

    m_removal_ids.push_back(id);

    return 0;

}

bool SegPaintManager::ShowPrivateObject(int id) {

    auto it = std::find(m_removal_ids.begin(), m_removal_ids.end(), id);

    if (it != m_removal_ids.end()) {
        m_removal_ids.erase(it);
        return 0;
    }

    return 1;

}

bool SegPaintManager::ClearRemovals() {

    m_removal_ids.clear();

    return 0;

}

cv::Mat SegPaintManager::InpaintFrame() {

    if (m_removal_ids.size() == 0) {
        return m_inputCVMat720;
    }

    // 1. Convert input image from BGR to RGB for inpainting model
    cv::Mat imageRGB;
    cv::cvtColor(m_input360, imageRGB, cv::COLOR_BGR2RGB);

    // 2. Convert the image to float and normalize pixel values to [-1, 1]
    cv::Mat imageFloat;
    imageRGB.convertTo(imageFloat, CV_32FC3, 1.0 / 127.5, -1.0);  // (img/127.5) - 1

    // Combine selected masks
    cv::Mat combinedMask = combineMasks(m_removal_ids, m_input360, m_detectedMasks);

    imageFloat = prepareMaskedImage(imageFloat, combinedMask);

    // 5. Convert the masked image to a blob with shape (1, 3, 360, 640)
    // blobFromImage converts from HWC (360x640x3) to CHW and adds a batch dimension.
    cv::Mat imageBlobFP32 = cv::dnn::blobFromImage(imageFloat);

    cv::Mat rawPaintData;
    if (m_IPC_connector->inpaintBlob(imageBlobFP32, rawPaintData)) {
        return imageRGB;
    }
        
    // Process output

    cv::Mat predictedImage360;
    int res = extractOutputs(rawPaintData, predictedImage360);
    if (res) {
        LogStatement("Issue with extracting outputs into HCW format");
        return imageRGB;
    }

    cv::Mat composite = compositeOutput(predictedImage360, combinedMask, m_inputCVMat720);

    return composite;

}

void SegPaintManager::UnloadEngines() {

    delete m_segmentationModel;
    delete m_IPC_connector;

}