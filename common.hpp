#ifndef SEGMENT_NORMAL_COMMON_HPP
#define SEGMENT_NORMAL_COMMON_HPP

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sl/Camera.hpp>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>         // For _access on Windows
#define F_OK 0          // Check for file existence
#else
#include <unistd.h>     // For access on POSIX systems
#endif

#ifndef S_ISREG
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

// Macro for CUDA error checking
#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// ----------------------------------------------------------------------
// Logger Declaration
// ----------------------------------------------------------------------
class Logger : public nvinfer1::ILogger {
public:
    // Inline constructor
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {}

    // Inline override of log, matching the base class exactly.
    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

private:
    nvinfer1::ILogger::Severity reportableSeverity;
};

// ----------------------------------------------------------------------
// Inline Utility Functions
// ----------------------------------------------------------------------
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:  return 4;
    case nvinfer1::DataType::kHALF:   return 2;
    case nvinfer1::DataType::kINT32:  return 4;
    case nvinfer1::DataType::kINT8:   return 1;
    case nvinfer1::DataType::kBOOL:   return 1;
    default:                         return 4;
    }
}

inline float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

inline std::vector<sl::uint2> convertCvRect2SdkBbox(cv::Rect_<float> const& bbox_in) {
    std::vector<sl::uint2> bbox_out;
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x),
        static_cast<unsigned int>(bbox_in.y)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x + bbox_in.width),
        static_cast<unsigned int>(bbox_in.y)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x + bbox_in.width),
        static_cast<unsigned int>(bbox_in.y + bbox_in.height)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x),
        static_cast<unsigned int>(bbox_in.y + bbox_in.height)));
    return bbox_out;
}

inline cv::Rect sdkBboxToCvRect(const std::vector<sl::uint2>& sdk_bbox)
{
    if (sdk_bbox.size() != 4) {
        return cv::Rect();
    }

    unsigned int min_x = sdk_bbox[0].x;
    unsigned int min_y = sdk_bbox[0].y;
    unsigned int max_x = sdk_bbox[0].x;
    unsigned int max_y = sdk_bbox[0].y;

    for (const auto& pt : sdk_bbox) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    return cv::Rect(static_cast<int>(min_x),
        static_cast<int>(min_y),
        static_cast<int>(max_x - min_x),
        static_cast<int>(max_y - min_y));
}

// ----------------------------------------------------------------------
// Namespace "seg" Declarations
// ----------------------------------------------------------------------
namespace seg {
    struct Binding {
        size_t         size = 1;
        size_t         dsize = 1;
        nvinfer1::Dims dims;
        std::string    name;
    };

    struct Object {
        cv::Rect_<float> rect;
        int              label = 0;
        float            prob = 0.0f;
        cv::Mat          boxMask;
    };

    struct PreParam {
        float ratio = 1.0f;
        float dw = 0.0f;
        float dh = 0.0f;
        float height = 0;
        float width = 0;
    };

    struct float3 {
        float x;
        float y;
        float z;
    };

    struct DetectedObject {
        int id;
        float3 bounding_box_3d[8];
        int label;
        float probability;
    };

    struct DetectedMaskPre {
        cv::Mat mask;       // Expected to be CV_32F (binary, with 0.0 and 1.0 values)
        cv::Rect bbox;
        sl::String unique_object_id;
    };

    struct DetectedMask {
        cv::Mat mask;       // Expected to be CV_32F (binary, with 0.0 and 1.0 values)
        int id;
        cv::Rect bbox;
    };

    struct SocketData {
        
    };
}

#endif // SEGMENT_NORMAL_COMMON_HPP
