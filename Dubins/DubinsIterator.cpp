#include "DubinsIterator.h"

DubinsIterator::DubinsIterator(float start[3], float goal[3], 
                float turning_radius, float step_size) 
    : current_t(0.0), step(step_size), path_length(0.0), turning_radius_(turning_radius) {
    if(!start || !goal){
        is_valid = false;
        return;
    }
    for (int i = 0; i < 3; ++i) {
        start_[i] = start[i];
        goal_[i] = goal[i];
    }
    if (dubins_shortest_path(&path, start_, goal_, turning_radius_) == EDUBOK) {
        path_length = dubins_path_length(&path);
        is_valid = true;
    } else {
        is_valid = false;
    }
}

// Default constructor
DubinsIterator::DubinsIterator()
    : current_t(0.0), step(1.0), path_length(0.0), is_valid(false), turning_radius_(1.0f) {
    for (int i = 0; i < 3; ++i) {
        start_[i] = 0.0f;
        goal_[i] = 0.0f;
    }
}

// Copy constructor
DubinsIterator::DubinsIterator(const DubinsIterator& other)
    : current_t(other.current_t), step(other.step), path_length(other.path_length),
      is_valid(other.is_valid), turning_radius_(other.turning_radius_) {
    for (int i = 0; i < 3; ++i) {
        start_[i] = other.start_[i];
        goal_[i] = other.goal_[i];
    }
    // Re-initialize path
    if (dubins_shortest_path(&path, start_, goal_, turning_radius_) == EDUBOK) {
        path_length = dubins_path_length(&path);
        is_valid = true;
    } else {
        is_valid = false;
    }
}

// Assignment operator
DubinsIterator& DubinsIterator::operator=(const DubinsIterator& other) {
    if (this != &other) {
        current_t = other.current_t;
        step = other.step;
        path_length = other.path_length;
        is_valid = other.is_valid;
        turning_radius_ = other.turning_radius_;
        for (int i = 0; i < 3; ++i) {
            start_[i] = other.start_[i];
            goal_[i] = other.goal_[i];
        }
        // Re-initialize path
        if (dubins_shortest_path(&path, start_, goal_, turning_radius_) == EDUBOK) {
            path_length = dubins_path_length(&path);
            is_valid = true;
        } else {
            is_valid = false;
        }
    }
    return *this;
}

PathPoint DubinsIterator::getNextPoint() {
    PathPoint point = {0.0, 0.0, 0.0, current_t, false};
    if (!is_valid || current_t > path_length) {
        return point;
    }

    float sample_t = current_t;
    // Clamp to path_length if next step would overshoot
    if (current_t + step > path_length && current_t < path_length) {
        sample_t = path_length;
    }

    float q[3];
    if (dubins_path_sample(&path, sample_t, q) == EDUBOK) {
        point.x = q[0];
        point.y = q[1];
        point.theta = q[2];
        point.t = sample_t;
        point.valid = true;
    }

    current_t = sample_t + step;
    return point;
}

bool DubinsIterator::hasNext() const {
    return is_valid && current_t <= path_length;
}

void DubinsIterator::reset() {
    current_t = 0.0;
}

std::vector<PathPoint> DubinsIterator::getAllPoints() {
    std::vector<PathPoint> points;
    reset();
    while (hasNext()) {
        points.push_back(getNextPoint());
    }
    return points;
}

std::vector<PathPoint> DubinsIterator::getSegmentPoints() {
    std::vector<PathPoint> points;
    // Use the new collect_dubins_segment_dots logic
    std::vector<PathPoint> seg_points = collect_dubins_segment_dots(start_, goal_, turning_radius_, step);
    for (const auto& pt : seg_points) {
        points.push_back(pt);
    }
    return points;
}
