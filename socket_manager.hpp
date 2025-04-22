#ifndef SOCKET_MANAGER_HPP
#define SOCKET_MANAGER_HPP

#include <uwebsockets/App.h>
#include "common.hpp"
#include "utils.h"
#include <iostream>

#include "ChangeQueue.hpp"

struct SocketData {};

class SocketManager {
    public:
        SocketManager(ChangeQueue* g_changeQueue);
        ~SocketManager();
        
        bool getPosConfirm();
        void broadcastBoundingBoxes(std::vector<DetectedObject> boundingBoxData);
   
        void start();
        void stop();
private:

    static std::set<uWS::WebSocket<false, true, SocketData>*> m_clients;
    static std::mutex m_clientsMutex;
    static ChangeQueue* gc_changeQueue;
    static bool posConfirm;
    static std::mutex posConfirmMutex;

    std::thread m_listenerThread;
    bool m_running;
};

#endif