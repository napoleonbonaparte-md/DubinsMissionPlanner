#include "Dubins.h"
class DubinsIterator{
public:

    DubinsIterator(float start[3], float goal[3], 
                   float turning_radius, float step_size);
    // Default constructor
    DubinsIterator();
    // Copy constructor
    DubinsIterator(const DubinsIterator& other);
    // Assignment operator
    DubinsIterator& operator=(const DubinsIterator& other);
    std::vector<PathPoint> getAllPoints();
    std::vector<PathPoint> getSegmentPoints();
    PathPoint getNextPoint();
    bool hasNext() const;
    void reset();
    DubinsPath path;
    float current_t;
    float step;
    float path_length;
    bool is_valid;
    float start_[3];
    float goal_[3];
    float turning_radius_;
};