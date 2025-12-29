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
    // Returns pointer to array of PathPoint and sets *num_points to the count
    PathPoint* dubins_iterator_get_all_points(void* obj, int* num_points) {
        static std::vector<PathPoint> points; // static to keep memory alive for Python
        points = static_cast<DubinsIterator*>(obj)->getAllPoints();
        if (num_points) *num_points = points.size();
        return points.empty() ? nullptr : points.data();
    }
    // getSegmentPoints wrapper
    PathPoint* dubins_iterator_get_segment_points(void* obj, int* num_points) {
        static std::vector<PathPoint> points;
        points = static_cast<DubinsIterator*>(obj)->getSegmentPoints();
        if (num_points) *num_points = points.size();
        return points.empty() ? nullptr : points.data();
    }
}
