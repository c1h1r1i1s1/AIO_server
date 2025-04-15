#include "socket_manager.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <thread>
#include <chrono>
#include <mutex>

using json = nlohmann::json;
using namespace std::chrono_literals;

//// Dummy function for bounding box json generation.
//std::string getBoundingBoxJson() {
//    json j;
//    j["headsetPose"] = { {"x", 1.1}, {"y", 1.7}, {"z", 0.4},
//                          {"qx", 0.0}, {"qy", 0.7}, {"qz", 0.0}, {"qw", 0.7} };
//    j["boxes"] = {
//        { {"id", "box1"}, {"category", "chair"}, {"x", 1}, {"y", 1}, {"z", 2}, {"w", 0.5}, {"h", 1}, {"d", 0.5} },
//        { {"id", "box2"}, {"category", "lamp"},  {"x", -1}, {"y", 1.2}, {"z", 2.5}, {"w", 0.3}, {"h", 0.8}, {"d", 0.3} }
//    };
//    return j.dump();
//}

SocketManager::SocketManager() {}

SocketManager::~SocketManager() {
    stop();
}

// Start the server and broadcaster thread.
void SocketManager::start() {

    // Make sure the lifetime of object exceeds lifetime callback
    uWS::App().ws<SocketData>("/*", {
        .open = [this](uWS::WebSocket<false, true, SocketData>* ws) {
            std::cout << "Client connected" << std::endl;
            std::lock_guard<std::mutex> lock(m_clientsMutex);
            m_clients.insert(ws);
        },
        .message = [](uWS::WebSocket<false, true, SocketData>* ws, std::string_view message, uWS::OpCode opCode) {
            try {
                auto data = json::parse(message);
                if (data.contains("id") && data.contains("selected")) {
                    std::cout << "Selected " << data["id"] << ": " << data["selected"] << std::endl;
                    if (data["selected"] == "1") {
                        m_removal_ids.emplace_back(data["id"]); // Really should fix this up before it works
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
        .listen(12345, [](auto* listenSocket) {
        if (listenSocket) {
            std::cout << "Listening on port 12345" << std::endl;
        }
            })
        .run();
}

void SocketManager::broadcastBoundingBoxes(const std::string& boundingBoxData) {
    std::set<uWS::WebSocket<false, true, SocketData>*> clientsCopy;
    {
        std::lock_guard<std::mutex> lock(m_clientsMutex);
        clientsCopy = m_clients;
    }
    for (auto client : clientsCopy) {
        if (client) {
            client->send(boundingBoxData, uWS::OpCode::TEXT);
        }
    }
}

std::vector<int> SocketManager::getSelectionChanges() {
    return m_removal_ids;
}