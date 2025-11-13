#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include "Dubins.h"
#include <float.h>

#define EPSILON (10e-10)

/* The segment types for each of the Path types */
const SegmentType DIRDATA[][3] = {
    { L_SEG, S_SEG, L_SEG },
    { L_SEG, S_SEG, R_SEG },
    { R_SEG, S_SEG, L_SEG },
    { R_SEG, S_SEG, R_SEG },
    { R_SEG, L_SEG, R_SEG },
    { L_SEG, R_SEG, L_SEG }
};
std::vector<PathPoint> collect_dubins_segment_dots(float q0[3], float q1[3], float rho, float step_size) {
    std::vector<PathPoint> points;
    DubinsPath path;
    if (dubins_shortest_path(&path, q0, q1, rho) != EDUBOK) {
        return points;
    }
    const SegmentType* types = DIRDATA[path.type];
    float seg_lengths[3] = { path.param[0] * rho, path.param[1] * rho, path.param[2] * rho };
    float t_global = 0.0;
    for (int i = 0; i < 3; ++i) {
        SegmentType type = types[i];
        float seg_length = seg_lengths[i];
        float t_start = t_global;
        float t_end = t_global + seg_length;
        if (type == S_SEG) {
            // Only first and last point
            float q_first[3], q_last[3];
            if (dubins_path_sample(&path, t_start, q_first) == EDUBOK)
                points.push_back({q_first[0], q_first[1], q_first[2], t_start, true});
            if (dubins_path_sample(&path, t_end, q_last) == EDUBOK)
                points.push_back({q_last[0], q_last[1], q_last[2], t_end, true});
        } else {
            // All points in turn
            float t = t_start;
            while (t < t_end) {
                float q[3];
                if (dubins_path_sample(&path, t, q) == EDUBOK)
                    points.push_back({q[0], q[1], q[2], t, true});
                t += step_size;
            }
            // Ensure last point is included
            float q_last[3];
            if (dubins_path_sample(&path, t_end, q_last) == EDUBOK)
                points.push_back({q_last[0], q_last[1], q_last[2], t_end, true});
        }
        t_global += seg_length;
    }
    return points;
}

std::vector<PathPoint> collect_dubins_path(float q0[3], float q1[3], 
                                         float rho, float step_size) {
    std::vector<PathPoint> path_points;
    DubinsPath path;
    
    if (dubins_shortest_path(&path, q0, q1, rho) != EDUBOK) {
        return path_points; // Return empty vector if path creation fails
    }

    // Get total path length
    float length = dubins_path_length(&path);
    
    // Sample points along the path
    float t = 0.0;
    while (t <= length) {
        float q[3];
        if (dubins_shortest_path(&path, q0, q1, rho) == EDUBOK) {
            if (dubins_path_sample(&path, t, q) == EDUBOK) {
                path_points.push_back({q[0], q[1], q[2], t});
            }
        }
        t += step_size;
    }

    // Ensure we get the exact end point
    float q_end[3];
    if (dubins_path_endpoint(&path, q_end) == EDUBOK) {
        path_points.push_back({q_end[0], q_end[1], q_end[2], length});
    }

    return path_points;
}

int dubins_word(DubinsIntermediateResults* in, DubinsPathType pathType, float out[3]);
int dubins_intermediate_results(DubinsIntermediateResults* in, float q0[3], float q1[3], float rho);

/**
 * Floating point modulus suitable for rings
 *
 * fmod doesn't behave correctly for angular quantities, this function does
 */
float fmodr( float x, float y)
{
    return x - y*floor(x/y);
}

float mod2pi( float theta )
{
    return fmodr( theta, 2 * M_PI );
}

int dubins_shortest_path(DubinsPath* path, float q0[3], float q1[3], float rho)
{
    /* Core function of dubins path. */
    int i, errcode;
    DubinsIntermediateResults in;
    float params[3];
    float cost;
    float best_cost = INFINITY;
    int best_word = -1;
    errcode = dubins_intermediate_results(&in, q0, q1, rho);
    if(errcode != EDUBOK) {
        return errcode;
    }


    path->qi[0] = q0[0];
    path->qi[1] = q0[1];
    path->qi[2] = q0[2];
    path->rho = rho;
 
    for( i = 0; i < 6; i++ ) {
        DubinsPathType pathType = (DubinsPathType)i;
        errcode = dubins_word(&in, pathType, params);
        if(errcode == EDUBOK) {
            cost = params[0] + params[1] + params[2];
            if(cost < best_cost) {
                best_word = i;
                best_cost = cost;
                path->param[0] = params[0];
                path->param[1] = params[1];
                path->param[2] = params[2];
                path->type = pathType;
            }
        }
    }
    if(best_word == -1) {
        return EDUBNOPATH;
    }
    return EDUBOK;
}

int dubins_path(DubinsPath* path, float q0[3], float q1[3], float rho, DubinsPathType pathType)
{
    /* 
    generate a path with a specific word.
    */
    int errcode;
    DubinsIntermediateResults in;
    errcode = dubins_intermediate_results(&in, q0, q1, rho);
    if(errcode == EDUBOK) {
        float params[3];
        errcode = dubins_word(&in, pathType, params);
        if(errcode == EDUBOK) {
            path->param[0] = params[0];
            path->param[1] = params[1];
            path->param[2] = params[2];
            path->qi[0] = q0[0];
            path->qi[1] = q0[1];
            path->qi[2] = q0[2];
            path->rho = rho;
            path->type = pathType;
        }
    }
    return errcode;
}

float dubins_path_length( DubinsPath* path )
{
    float length = 0.;
    length += path->param[0];
    length += path->param[1];
    length += path->param[2];
    length = length * path->rho;
    return length;
}

float dubins_segment_length( DubinsPath* path, int i )
{
    if( (i < 0) || (i > 2) )
    {
        return INFINITY;
    }
    return path->param[i] * path->rho;
}

float dubins_segment_length_normalized( DubinsPath* path, int i )
{
    if( (i < 0) || (i > 2) )
    {
        return INFINITY;
    }
    return path->param[i];
} 

DubinsPathType dubins_path_type( DubinsPath* path ) 
{
    return path->type;
}

void dubins_segment( float t, float qi[3], float qt[3], SegmentType type)
{
    float st = sin(qi[2]);
    float ct = cos(qi[2]);
    if( type == L_SEG ) {
        qt[0] = +sin(qi[2]+t) - st;
        qt[1] = -cos(qi[2]+t) + ct;
        qt[2] = t;
    }
    else if( type == R_SEG ) {
        qt[0] = -sin(qi[2]-t) + st;
        qt[1] = +cos(qi[2]-t) - ct;
        qt[2] = -t;
    }
    else if( type == S_SEG ) {
        qt[0] = ct * t;
        qt[1] = st * t;
        qt[2] = 0.0;
    }
    qt[0] += qi[0];
    qt[1] += qi[1];
    qt[2] += qi[2];
}

int dubins_path_sample( DubinsPath* path, float t, float q[3] )
{
    /* tprime is the normalised variant of the parameter t */
    float tprime = t / path->rho;
    float qi[3]; /* The translated initial configuration */
    float q1[3]; /* end-of segment 1 */
    float q2[3]; /* end-of segment 2 */
    const SegmentType* types = DIRDATA[path->type];
    float p1, p2;

    if( t < 0 || t > dubins_path_length(path) ) {
        return EDUBPARAM;
    }

    /* initial configuration */
    qi[0] = 0.0;
    qi[1] = 0.0;
    qi[2] = path->qi[2];

    /* generate the target configuration */
    p1 = path->param[0];
    p2 = path->param[1];
    dubins_segment( p1,      qi,    q1, types[0] );
    dubins_segment( p2,      q1,    q2, types[1] );
    if( tprime < p1 ) {
        dubins_segment( tprime, qi, q, types[0] );
    }
    else if( tprime < (p1+p2) ) {
        dubins_segment( tprime-p1, q1, q,  types[1] );
    }
    else {
        dubins_segment( tprime-p1-p2, q2, q,  types[2] );
    }

    /* scale the target configuration, translate back to the original starting point */
    q[0] = q[0] * path->rho + path->qi[0];
    q[1] = q[1] * path->rho + path->qi[1];
    q[2] = mod2pi(q[2]);

    return EDUBOK;
}


int dubins_path_endpoint( DubinsPath* path, float q[3] )
{
    return dubins_path_sample( path, dubins_path_length(path) - EPSILON, q );
}

int dubins_extract_subpath( DubinsPath* path, float t, DubinsPath* newpath )
{
    /* calculate the true parameter */
    float tprime = t / path->rho;

    if((t < 0) || (t > dubins_path_length(path)))
    {
        return EDUBPARAM; 
    }

    /* copy most of the data */
    newpath->qi[0] = path->qi[0];
    newpath->qi[1] = path->qi[1];
    newpath->qi[2] = path->qi[2];
    newpath->rho   = path->rho;
    newpath->type  = path->type;

    /* fix the parameters */
    newpath->param[0] = fmin( path->param[0], tprime );
    newpath->param[1] = fmin( path->param[1], tprime - newpath->param[0]);
    newpath->param[2] = fmin( path->param[2], tprime - newpath->param[0] - newpath->param[1]);
    return 0;
}

int dubins_intermediate_results(DubinsIntermediateResults* in, float q0[3], float q1[3], float rho)
{
    float dx, dy, D, d, theta, alpha, beta;
    if( rho <= 0.0 ) {
        return EDUBBADRHO;
    }

    dx = q1[0] - q0[0];
    dy = q1[1] - q0[1];
    D = sqrt( dx * dx + dy * dy );
    d = D / rho;
    theta = 0;

    /* test required to prevent domain errors if dx=0 and dy=0 */
    if(d > 0) {
        theta = mod2pi(atan2( dy, dx ));
    }
    alpha = mod2pi(q0[2] - theta);
    beta  = mod2pi(q1[2] - theta);

    in->alpha = alpha;
    in->beta  = beta;
    in->d     = d;
    in->sa    = sin(alpha);
    in->sb    = sin(beta);
    in->ca    = cos(alpha);
    in->cb    = cos(beta);
    in->c_ab  = cos(alpha - beta);
    in->d_sq  = d * d;

    return EDUBOK;
}

int dubins_LSL(DubinsIntermediateResults* in, float out[3]) 
{
    float tmp0, tmp1, p_sq;
    
    tmp0 = in->d + in->sa - in->sb;
    p_sq = 2 + in->d_sq - (2*in->c_ab) + (2 * in->d * (in->sa - in->sb));

    if(p_sq >= 0) {
        tmp1 = atan2( (in->cb - in->ca), tmp0 );
        out[0] = mod2pi(tmp1 - in->alpha);
        out[1] = sqrt(p_sq);
        out[2] = mod2pi(in->beta - tmp1);
        return EDUBOK;
    }
    return EDUBNOPATH;
}


int dubins_RSR(DubinsIntermediateResults* in, float out[3]) 
{
    float tmp0 = in->d - in->sa + in->sb;
    float p_sq = 2 + in->d_sq - (2 * in->c_ab) + (2 * in->d * (in->sb - in->sa));
    if( p_sq >= 0 ) {
        float tmp1 = atan2( (in->ca - in->cb), tmp0 );
        out[0] = mod2pi(in->alpha - tmp1);
        out[1] = sqrt(p_sq);
        out[2] = mod2pi(tmp1 -in->beta);
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_LSR(DubinsIntermediateResults* in, float out[3]) 
{
    float p_sq = -2 + (in->d_sq) + (2 * in->c_ab) + (2 * in->d * (in->sa + in->sb));
    if( p_sq >= 0 ) {
        float p    = sqrt(p_sq);
        float tmp0 = atan2( (-in->ca - in->cb), (in->d + in->sa + in->sb) ) - atan2(-2.0, p);
        out[0] = mod2pi(tmp0 - in->alpha);
        out[1] = p;
        out[2] = mod2pi(tmp0 - mod2pi(in->beta));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_RSL(DubinsIntermediateResults* in, float out[3]) 
{
    float p_sq = -2 + in->d_sq + (2 * in->c_ab) - (2 * in->d * (in->sa + in->sb));
    if( p_sq >= 0 ) {
        float p    = sqrt(p_sq);
        float tmp0 = atan2( (in->ca + in->cb), (in->d - in->sa - in->sb) ) - atan2(2.0, p);
        out[0] = mod2pi(in->alpha - tmp0);
        out[1] = p;
        out[2] = mod2pi(in->beta - tmp0);
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_RLR(DubinsIntermediateResults* in, float out[3]) 
{
    float tmp0 = (6. - in->d_sq + 2*in->c_ab + 2*in->d*(in->sa - in->sb)) / 8.;
    float phi  = atan2( in->ca - in->cb, in->d - in->sa + in->sb );
    if( fabs(tmp0) <= 1) {
        float p = mod2pi((2*M_PI) - acos(tmp0) );
        float t = mod2pi(in->alpha - phi + mod2pi(p/2.));
        out[0] = t;
        out[1] = p;
        out[2] = mod2pi(in->alpha - in->beta - t + mod2pi(p));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_LRL(DubinsIntermediateResults* in, float out[3])
{
    float tmp0 = (6. - in->d_sq + 2*in->c_ab + 2*in->d*(in->sb - in->sa)) / 8.;
    float phi = atan2( in->ca - in->cb, in->d + in->sa - in->sb );
    if( fabs(tmp0) <= 1) {
        float p = mod2pi( 2*M_PI - acos( tmp0) );
        float t = mod2pi(-in->alpha - phi + p/2.);
        out[0] = t;
        out[1] = p;
        out[2] = mod2pi(mod2pi(in->beta) - in->alpha -t + mod2pi(p));
        return EDUBOK;
    }
    return EDUBNOPATH;
}

int dubins_word(DubinsIntermediateResults* in, DubinsPathType pathType, float out[3]) 
{
    int result;
    switch(pathType)
    {
    case LSL:
        result = dubins_LSL(in, out);
        break;
    case RSL:
        result = dubins_RSL(in, out);
        break;
    case LSR:
        result = dubins_LSR(in, out);
        break;
    case RSR:
        result = dubins_RSR(in, out);
        break;
    case LRL:
        result = dubins_LRL(in, out);
        break;
    case RLR:
        result = dubins_RLR(in, out);
        break;
    default:
        result = EDUBNOPATH;
    }
    return result;
}