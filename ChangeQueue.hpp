#ifndef CHANGE_QUEUE_HPP
#define CHANGE_QUEUE_HPP

#include <queue>
#include <mutex>

struct ObjectChange {
    int id;
    bool selected; // true: remove object; false: show object
};

class ChangeQueue {
public:
    void pushChange(const ObjectChange& change) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(change);
    }

    bool popChange(ObjectChange& outChange) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return false;
        outChange = m_queue.front();
        m_queue.pop();
        return true;
    }

private:
    std::queue<ObjectChange> m_queue;
    std::mutex m_mutex;
};

#endif