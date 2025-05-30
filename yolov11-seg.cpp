// This interfaces directly with the YOLO segmentation and object detection model
// The code has been modified from ZED's yolo integration example
#include "yolov11-seg.hpp"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cuda_fp16.h>

#include <cassert>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>


// ======================================================================
// YOLOv11_seg Constructor (Fixed Input Resolution)
// ======================================================================
YOLOv11_seg::YOLOv11_seg(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->trt_name_engine_get_nb_binding();

    // Loop over bindings and assume fixed input resolution (360x640)
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = this->engine->getTensorDataType(this->engine->getIOTensorName(i));
        std::string name = this->engine->trt_name_engine_get_binding_name(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        if (engine->getTensorIOMode(engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);

            this->context->setInputShape(name.c_str(), dims);
        }
        else {
            dims = this->engine->getTensorShape(this->engine->getIOTensorName(i));
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

// ======================================================================
// YOLOv11_seg Destructor
// ======================================================================
YOLOv11_seg::~YOLOv11_seg()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;

    if (this->stream) {
        cudaStreamDestroy(this->stream);
        this->stream = nullptr;
    }
    for (auto& ptr : this->device_ptrs) {
        if (ptr) {
            CHECK(cudaFree(ptr));
        }
    }
    for (auto& ptr : this->host_ptrs) {
        if (ptr) {
            CHECK(cudaFreeHost(ptr));
        }
    }
}

// ======================================================================
// make_pipe: Allocate buffers and perform warmup (if desired)
// ======================================================================
void YOLOv11_seg::make_pipe()
{
    for (auto& binding : this->input_bindings) {
        void* d_ptr = nullptr;
        CHECK(cudaMallocAsync(&d_ptr, binding.size * binding.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }
    for (auto& binding : this->output_bindings) {
        void* d_ptr = nullptr;
        void* h_ptr = nullptr;
        size_t size = binding.size * binding.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }
}

// ======================================================================
// letterbox: Resize image while maintaining aspect ratio and add borders
// ======================================================================
void YOLOv11_seg::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size)
{
    const float inp_h = static_cast<float>(size.height);
    const float inp_w = static_cast<float>(size.width);
    float height = static_cast<float>(image.rows);
    float width = static_cast<float>(image.cols);

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if (static_cast<int>(width) != padw || static_cast<int>(height) != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    // Save parameters for later postprocessing.
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

// ======================================================================
// copy_from_Mat: Overload 1 - Using fixed input size (640x360) to 640x384
// ======================================================================
void YOLOv11_seg::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat blob;
    cv::Size fixedSize(640, 384); // Fixed resolution: width=640, height=384
    this->letterbox(image, blob, fixedSize);
    // No dynamic input shape setting is needed here.
    CHECK(cudaMemcpyAsync(this->device_ptrs[0],
        blob.ptr<float>(),
        blob.total() * blob.elemSize(),
        cudaMemcpyHostToDevice,
        this->stream));
}

// ======================================================================
// infer: Run network inference and copy outputs to host pointers
// ======================================================================
void YOLOv11_seg::infer()
{
    for (unsigned int i = 0; i < this->num_inputs; ++i) {
        this->context->setTensorAddress(this->input_bindings[i].name.c_str(), this->device_ptrs[i]);
    }
    for (unsigned int i = 0; i < this->num_outputs; ++i) {
        this->context->setTensorAddress(this->output_bindings[i].name.c_str(),
            this->device_ptrs[i + this->num_inputs]);
    }
    this->context->enqueueV3(this->stream);

    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i],
            this->device_ptrs[i + this->num_inputs],
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// ======================================================================
// postprocess: Process network outputs to extract detection & segmentation info
// ======================================================================
void YOLOv11_seg::postprocess(std::vector<seg::Object>& objs,
    float score_thres,
    float iou_thres,
    int topk,
    int seg_channels)
{
    objs.clear();
    // Using fixed model resolution (384×640)
    int input_h = 384;
    int input_w = 640;

    int seg_h = input_h / 4;
    int seg_w = input_w / 4;
    int num_channels, num_anchors, num_classes;
    bool found = false;
    int bid = 0, bcnt = -1;
    for (auto& o : this->output_bindings) {
        ++bcnt;
        if (o.dims.nbDims == 3) {
            num_channels = o.dims.d[1];
            num_anchors = o.dims.d[2];
            found = true;
            bid = bcnt;
        }
    }
    assert(found);
    num_classes = num_channels - seg_channels - 4;

    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
        static_cast<float*>(this->host_ptrs[bid]));
    output = output.t();
    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F,
        static_cast<float*>(this->host_ptrs[1 - bid]));

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bbox_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto mask_ptr = row_ptr + 4 + num_classes;
        auto max_elem = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_elem;
        if (score > score_thres) {
            float x = *bbox_ptr++ - dw;
            float y = *bbox_ptr++ - dh;
            float w = *bbox_ptr++;
            float h = *bbox_ptr;
            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, static_cast<float>(width));
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, static_cast<float>(height));
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, static_cast<float>(width));
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, static_cast<float>(height));
            int label = static_cast<int>(max_elem - scores_ptr);
            cv::Rect_<float> bbox(x0, y0, x1 - x0, y1 - y0);
            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, mask_ptr);
            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            mask_confs.push_back(mask_conf);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    cv::Mat masks;
    int cnt = 0;
    for (auto& idx : indices) {
        if (cnt >= topk) break;
        seg::Object obj;
        obj.label = labels[idx];
        obj.rect = bboxes[idx];
        obj.prob = scores[idx];
        masks.push_back(mask_confs[idx]);
        objs.push_back(obj);
        ++cnt;
    }

    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(static_cast<int>(indices.size()), { seg_h, seg_w });
        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = static_cast<int>(dw / static_cast<float>(input_w) * seg_w);
        int scale_dh = static_cast<int>(dh / static_cast<float>(input_h) * seg_h);
        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);
        for (int i = 0; i < static_cast<int>(indices.size()); i++) {
            cv::Mat exp_neg, sigmoid, mask;
            cv::exp(-maskChannels[i], exp_neg);
            sigmoid = 1.0 / (1.0 + exp_neg);
            sigmoid = sigmoid(roi);
            cv::resize(sigmoid, mask, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}