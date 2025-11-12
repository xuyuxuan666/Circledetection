#pragma once
#include <limits>

#ifndef WellRow
#define WellRow 2
#endif
#ifndef WellCol
#define WellCol 8
#endif
#ifndef PointRow
#define PointRow 6
#endif
#ifndef PointCol
#define PointCol 3
// #define PointCol 2
#endif

#ifndef STD_IMG_W
#define STD_IMG_W 2048
#endif
#ifndef STD_IMG_H
#define STD_IMG_H 1536
#endif

#ifndef HAVE_TYPEDEF_USHORT
#define HAVE_TYPEDEF_USHORT
typedef unsigned short ushort;
#endif

struct _POINTPOSITIONINFO {
    float x = std::numeric_limits<float>::quiet_NaN();
    float y = std::numeric_limits<float>::quiet_NaN();
    bool  measured = false;
    bool  valid    = false;
};

void PerformShapeDetection(
    ushort usImage[],
    _POINTPOSITIONINFO (&PostionArray)[WellRow][WellCol][PointRow][PointCol]);

void PerformShapeDetectionDyn(
    const ushort* usImage, int width, int height,
    _POINTPOSITIONINFO (&PostionArray)[WellRow][WellCol][PointRow][PointCol]);
