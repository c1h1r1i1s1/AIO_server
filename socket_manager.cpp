#include "socket_manager.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <thread>
#include <chrono>
#include <mutex>

using json = nlohmann::json;
using namespace std::chrono_literals;
using namespace seg;

std::set<uWS::WebSocket<false, true, SocketData>*> SocketManager::m_clients;
std::mutex SocketManager::m_clientsMutex;
std::mutex SocketManager::posConfirmMutex;
ChangeQueue* SocketManager::gc_changeQueue;
bool SocketManager::posConfirm = false;

SocketManager::SocketManager(ChangeQueue* g_changeQueue) : m_running(false) {
    gc_changeQueue = g_changeQueue;
}


SocketManager::~SocketManager() {
    stop();
}

bool SocketManager::getPosConfirm() {
    std::lock_guard<std::mutex> lock(posConfirmMutex);
    return posConfirm;
}

// Start the server and broadcaster thread.
void SocketManager::start() {
    m_running = true;

    // Make sure the lifetime of object exceeds lifetime callback
    m_listenerThread = std::thread([this]() {
        uWS::App().ws<SocketData>("/*", {
            .open = [this](uWS::WebSocket<false, true, SocketData>* ws) {
                std::cout << "Client connected" << std::endl;
                std::lock_guard<std::mutex> lock(m_clientsMutex);
                m_clients.insert(ws);
            },
            .message = [this](uWS::WebSocket<false, true, SocketData>* ws, std::string_view message, uWS::OpCode opCode) {
                try {
                    auto data = json::parse(message);
                    if (data.contains("comType")) {
                        if (data["comType"] == "selection") {
                            if (data.contains("id") && data.contains("selected")) {
                                std::cout << "Selected " << data["id"] << ": " << data["selected"] << std::endl;
                                int id = data["id"].get<int>();
                                bool selected = data["selected"].get<bool>();

                                ObjectChange change{ id, selected };
                                gc_changeQueue->pushChange(change);
                            }
                        }
                        else if (data["comType"] == "posConfirm") {
                            std::lock_guard<std::mutex> lock(posConfirmMutex);
                            posConfirm = true;
                            std::cout << "Confirmed camera position" << std::endl;
                        }
                    }
                }
                catch (std::exception& e) {
                    std::cerr << "JSON parse error: " << e.what() << std::endl;
                }
            },
            .close = [this](uWS::WebSocket<false, true, SocketData>* ws, int /*code*/, std::string_view /*msg*/) {
                std::cout << "Client disconnected" << std::endl;
                std::lock_guard<std::mutex> lock(m_clientsMutex);
                m_clients.erase(ws);
            }
        })
        .listen(12345, [this](us_listen_socket_t* token) {
            m_listenSocket = token;
            if (m_listenSocket) {
                std::cout << "Listening on port 12345" << std::endl;
            }
        })
        .run();
    });
}

void SocketManager::stop() {
    if (!m_running)
        return;

    m_running = false;

    {
        std::lock_guard<std::mutex> lock(m_clientsMutex);
        for (auto ws : m_clients) {
            ws->close();
        }
        m_clients.clear();
    }

    if (m_listenSocket) {
        us_listen_socket_close(0, m_listenSocket);
        m_listenSocket = nullptr;
    }

    // If needed, join the listener thread in the destructor or here.
    if (m_listenerThread.joinable()) {
        m_listenerThread.join();
    }

    uWS::Loop::get()->free();
}

void SocketManager::broadcastBoundingBoxes(std::vector<DetectedObject> boundingBoxData) {
    
    json boxData;
    boxData["boxes"] = json::array();
    for (DetectedObject boxInstance: boundingBoxData) {
        json box;
        box["id"] = boxInstance.id;
        box["label"] = boxInstance.label;

        if (box["label"] == 59) {
            continue;
        }

        seg::float3 center = ComputeCenter(boxInstance.bounding_box_3d);
        seg::float3 size = ComputeSize(boxInstance.bounding_box_3d);

        box["x"] = center.x;
        box["y"] = center.y;
        box["z"] = center.z;

        // Dimensions of the box.
        box["w"] = size.x;
        box["h"] = size.y;
        box["d"] = size.z;
        
        boxData["boxes"].push_back(box);
    };
    
    const std::string stringBoxData = boxData.dump();
    
    std::set<uWS::WebSocket<false, true, SocketData>*> clientsCopy;
    {
        std::lock_guard<std::mutex> lock(m_clientsMutex);
        clientsCopy = m_clients;
    }
    for (auto client : clientsCopy) {
        if (client) {
            client->send(stringBoxData, uWS::OpCode::TEXT);
        }
    }
}