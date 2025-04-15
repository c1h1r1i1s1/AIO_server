#ifndef SEGMENT_NORMAL_YOLOV11_SEG_HPP
#define SEGMENT_NORMAL_YOLOV11_SEG_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "common.hpp"

#define trt_name_engine_get_binding_name getIOTensorName
#define trt_name_engine_get_nb_binding getNbIOTensors

using namespace seg;

class YOLOv11_seg {
public:
    // Constructor/Destructor declarations (Fixed input resolution version: 360x640).
    explicit YOLOv11_seg(const std::string& engine_file_path);
    ~YOLOv11_seg();

    void make_pipe();

    void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size);

    void copy_from_Mat(const cv::Mat& image);

    void infer();

    void postprocess(std::vector<seg::Object>& objs,
        float score_thres = 0.25f,
        float iou_thres = 0.65f,
        int topk = 100,
        int seg_channels = 32);

private:
    int num_bindings = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    cudaStream_t stream = nullptr;

    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

public:
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;
};

#endif // SEGMENT_NORMAL_YOLOV11_SEG_HPP
