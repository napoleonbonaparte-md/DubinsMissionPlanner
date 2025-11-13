#include "DubinsIterator.h"
extern "C" {
    // Constructor wrapper
    void* create_dubins_iterator(float start[3], float goal[3], float turning_radius, float step_size) {
        return new DubinsIterator(start, goal, turning_radius, step_size);
    }
    // Destructor wrapper
    void destroy_dubins_iterator(void* obj) {
        delete static_cast<DubinsIterator*>(obj);
    }
    // getNextPoint wrapper
    PathPoint dubins_iterator_get_next_point(void* obj) {
        return static_cast<DubinsIterator*>(obj)->getNextPoint();
    }
    // hasNext wrapper
    bool dubins_iterator_has_next(void* obj) {
        return static_cast<DubinsIterator*>(obj)->hasNext();
    }
    // reset wrapper
    void dubins_iterator_reset(void* obj) {
        static_cast<DubinsIterator*>(obj)->reset();
    }
}
