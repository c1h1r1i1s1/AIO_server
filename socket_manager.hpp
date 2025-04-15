#ifndef SOCKET_MANAGER_HPP
#define SOCKET_MANAGER_HPP

#include <uwebsockets/App.h>
#include "common.hpp"
#include <iostream>

using namespace seg;

class SocketManager {
    public:
        SocketManager();
        ~SocketManager();

        void broadcastBoundingBoxes(const std::string& boundingBoxData);
        std::vector<int> getRemovalIds();
   
        void start();
        void stop();
private:

    static std::set<uWS::WebSocket<false, true, SocketData>*> m_clients;
    static std::mutex m_clientsMutex;

    static std::vector<int, int> m_selection_changes;
};

#endif