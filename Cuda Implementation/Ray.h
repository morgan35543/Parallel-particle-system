#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { rayOri = a; rayDir = b; }
        __device__ vec3 origin() const       { return rayOri; }
        __device__ vec3 direction() const    { return rayDir; }
        __device__ vec3 point_at_parameter(float t) const { return rayOri + t*rayDir; }

        vec3 rayOri;
        vec3 rayDir;
};

#endif