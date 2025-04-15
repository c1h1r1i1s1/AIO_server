#include "IPC_connect.hpp"

const size_t FP32_BLOB_SIZE = 1ull * 3 * 360 * 640 * sizeof(uint32_t);

// Define shared memory and event names (without "Global\\" if not needed).
const wchar_t* SHARED_MEMORY_NAME = L"InpaintingSharedMemory";
const wchar_t* EVENT_INPUT_READY = L"InputReadyEvent";
const wchar_t* EVENT_OUTPUT_READY = L"OutputReadyEvent";

// Constructor: Open the shared memory mapping and the named events.
IPC_connect::IPC_connect() {
    // Open existing shared memory.
    hMapFile = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, SHARED_MEMORY_NAME);
    if (!hMapFile) {
        std::wcerr << L"OpenFileMappingW failed with error: " << GetLastError() << std::endl;
        // Handle error as needed.
    }
    // Map shared memory into our address space.
    pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FP32_BLOB_SIZE);
    if (!pBuf) {
        std::wcerr << L"MapViewOfFile failed with error: " << GetLastError() << std::endl;
        // Handle error as needed.
    }
    // Open the named events.
    hInputEvent = OpenEventW(EVENT_MODIFY_STATE, FALSE, EVENT_INPUT_READY);
    if (!hInputEvent) {
        std::wcerr << L"OpenEventW for input event failed with error: " << GetLastError() << std::endl;
        // Handle error as needed.
    }
    hOutputEvent = OpenEventW(SYNCHRONIZE, FALSE, EVENT_OUTPUT_READY);
    if (!hOutputEvent) {
        std::wcerr << L"OpenEventW for output event failed with error: " << GetLastError() << std::endl;
        // Handle error as needed.
    }
}

// Destructor: Clean up all handles.
IPC_connect::~IPC_connect() {
    if (hInputEvent) CloseHandle(hInputEvent);
    if (hOutputEvent) CloseHandle(hOutputEvent);
    if (pBuf) UnmapViewOfFile(pBuf);
    if (hMapFile) CloseHandle(hMapFile);
}

// inpaintBlob: Sends an input FP32 blob and receives the processed FP32 blob.
// Both inputBlob and outputBlob are expected to have a total size of FP32_BLOB_SIZE.
bool IPC_connect::inpaintBlob(const cv::Mat& inputBlob, cv::Mat& outputBlob) {
    // Verify input blob size.
    if (inputBlob.total() * inputBlob.elemSize() != FP32_BLOB_SIZE) {
        LogStatement("Input blob size does not match expected FP32_BLOB_SIZE.");
        std::cerr << "Input blob size does not match expected FP32_BLOB_SIZE." << std::endl;
        return 1;
    }
    // Copy input FP32 data into shared memory.
    memcpy(pBuf, inputBlob.data, FP32_BLOB_SIZE);

    // Signal the server that the input is ready.
    if (!SetEvent(hInputEvent)) {
        LogStatement("SetEvent for input event failed");
        std::wcerr << L"SetEvent for input event failed: " << GetLastError() << std::endl;
        return 1;
    }

    // Wait for the server to process and signal that output is ready (5 sec timeout).
    DWORD waitResult = WaitForSingleObject(hOutputEvent, 5000);
    if (waitResult != WAIT_OBJECT_0) {
        LogStatement("WaitForSingleObject on output event failed");
        std::wcerr << L"WaitForSingleObject on output event failed: " << GetLastError() << std::endl;
        return 1;
    }

    // Copy the output FP32 data from shared memory into outputBlob.
    // Create a cv::Mat header for the output (same size and type).
    outputBlob = cv::Mat(1, 3 * 360 * 640, CV_32F);
    memcpy(outputBlob.data, pBuf, FP32_BLOB_SIZE);

    return 0;
}