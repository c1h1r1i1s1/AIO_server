#ifndef SEG_PAINT_HPP
#define SEG_PAINT_HPP

#include "seg_engine.hpp"
#include "ZEDCustomManager.hpp"
#include "IPC_connect.hpp"
#include <vector>
#include <cuda_runtime.h>
#include "common.hpp"

class SegPaintManager {
public:
	bool InitialiseEngines();
	int ProcessFrame(ZEDCustomManager* zedCustomManager);
	std::vector<DetectedObject> GetObjects() const;
	bool ErasePrivateObject(int id);
	bool ShowPrivateObject(int id);
	bool ClearRemovals();
	std::vector<cv::Mat> InpaintFrame(ZEDCustomManager* zedCustomManager);
	void UnloadEngines();
	~SegPaintManager();

private:
	SegmentationModel* m_segmentationModel;
	IPC_connect* m_IPC_connector;
	static std::vector<DetectedObject> m_detectedObjects;
	static std::vector<DetectedMask> m_detectedMasks;
	static std::vector<int> m_removal_ids;
	cv::Mat m_input360_L;
	cv::Mat m_inputCVMat720_L;
	cv::Mat m_input360_R;
	cv::Mat m_inputCVMat720_R;
};

#endif