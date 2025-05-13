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

    m_inputCVMat720_L = zedCustomManager->getCurrentLeftMat();
    m_inputCVMat720_R = zedCustomManager->getCurrentRightMat();
        
    // Scale down to 360p
    cv::cvtColor(m_inputCVMat720_L, m_inputCVMat720_L, cv::COLOR_BGRA2RGB);
    m_input360_L.create(360, 640, m_inputCVMat720_L.type());
    cv::resize(m_inputCVMat720_L, m_input360_L, cv::Size(640, 360));
    
    cv::cvtColor(m_inputCVMat720_R, m_inputCVMat720_R, cv::COLOR_BGRA2RGB);
    m_input360_R.create(360, 640, m_inputCVMat720_R.type());
    cv::resize(m_inputCVMat720_R, m_input360_R, cv::Size(640, 360));

    // Segment
    m_detectedObjects.clear();
    m_detectedMasks.clear();
    int objectCount = m_segmentationModel->segmentFrame(m_inputCVMat720_L, m_detectedObjects, m_detectedMasks, zedCustomManager->getCamera());

    return objectCount;

}

std::vector<DetectedObject> SegPaintManager::GetObjects() const { // Could use maxObjects

    return m_detectedObjects;

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

std::vector<cv::Mat> SegPaintManager::InpaintFrame(ZEDCustomManager* zedCustomManager) {

    if (m_removal_ids.size() == 0) {
        return { m_inputCVMat720_L, m_inputCVMat720_R };
    }

    // 2. Convert the image to float and normalize pixel values to [-1, 1]
    cv::Mat imageFloat_L;
    cv::Mat imageFloat_R;
    m_inputCVMat720_L.convertTo(imageFloat_L, CV_32FC3, 1.0 / 127.5, -1.0);  // (img/127.5) - 1
    m_inputCVMat720_R.convertTo(imageFloat_R, CV_32FC3, 1.0 / 127.5, -1.0);

    // Combine selected masks (720p)
    cv::Mat combinedMask_L = combineMasks(m_removal_ids, m_inputCVMat720_L, m_detectedMasks);
    cv::Mat combinedMask_R = warpAndCombineMasksToRight(zedCustomManager->getCamera(),
        m_inputCVMat720_R,
        m_removal_ids,
        m_detectedMasks);
    //cv::Mat combinedMask_R = warpMaskToRight(zedCustomManager->getCamera(), combinedMask_L);

    imageFloat_L = prepareMaskedImage(imageFloat_L, combinedMask_L); // Combine mask and input
    imageFloat_R = prepareMaskedImage(imageFloat_R, combinedMask_R);

    // 5. Convert the masked image to a blob with shape (1, 3, 360, 640)
    // blobFromImage converts from HWC (360x640x3) to CHW and adds a batch dimension.
    cv::Mat imageBlobFP32_L = cv::dnn::blobFromImage(imageFloat_L);
    cv::Mat imageBlobFP32_R = cv::dnn::blobFromImage(imageFloat_R);

    cv::Mat rawPaintData_L;
    cv::Mat rawPaintData_R;
    if (m_IPC_connector->inpaintBlob(imageBlobFP32_L, imageBlobFP32_R, rawPaintData_L, rawPaintData_R)) {
        return { m_inputCVMat720_L, m_inputCVMat720_R };
    }
        
    // Process output
    cv::Mat predictedImage360_L;
    cv::Mat predictedImage360_R;
    int res = extractOutputs(rawPaintData_L, predictedImage360_L);
    if (res) {
        LogStatement("Issue with extracting outputs into HCW format");
        return { m_inputCVMat720_L, m_inputCVMat720_R };
    }
    res = extractOutputs(rawPaintData_R, predictedImage360_R);
    if (res) {
        LogStatement("Issue with extracting outputs into HCW format");
        return { m_inputCVMat720_L, m_inputCVMat720_R };
    }

    cv::Mat composite_L = compositeOutput(predictedImage360_L, combinedMask_L, m_inputCVMat720_L);
    cv::Mat composite_R = compositeOutput(predictedImage360_R, combinedMask_R, m_inputCVMat720_R);

    return { composite_L, composite_R };

}

void SegPaintManager::UnloadEngines() {

    delete m_segmentationModel;
    delete m_IPC_connector;

}

SegPaintManager::~SegPaintManager() {
    UnloadEngines();
}