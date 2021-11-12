#ifndef __DETECT_UTILS_H__
#define __DETECT_UTILS_H__

#include <NvInfer.h>

#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/timespec.h>
#include <jetson-utils/logging.h>

#include "cudaUtility.h"

/**
 * Object Detection result.
 */
typedef struct Detection {
    // Object Info
    uint32_t Instance;  /**< Index of this unique object instance */
    uint32_t ClassID;   /**< Class index of the detected object. */
    float Confidence;   /**< Confidence value of the detected object. */

    // Bounding Box Coordinates
    float Left;     /**< Left bounding box coordinate (in pixels) */
    float Right;        /**< Right bounding box coordinate (in pixels) */
    float Top;      /**< Top bounding box cooridnate (in pixels) */
    float Bottom;       /**< Bottom bounding box coordinate (in pixels) */

    /**< Calculate the width of the object */
    inline float Width() const
    {
        return Right - Left;
    }

    /**< Calculate the height of the object */
    inline float Height() const
    {
        return Bottom - Top;
    }

    /**< Calculate the area of the object */
    inline float Area() const
    {
        return Width() * Height();
    }

    /**< Calculate the width of the bounding box */
    static inline float Width( float x1, float x2 )
    {
        return x2 - x1;
    }

    /**< Calculate the height of the bounding box */
    static inline float Height( float y1, float y2 )
    {
        return y2 - y1;
    }

    /**< Calculate the area of the bounding box */
    static inline float Area( float x1, float y1, float x2, float y2 )
    {
        return Width(x1, x2) * Height(y1, y2);
    }

    /**< Return the center of the object */
    inline void Center( float *x, float *y ) const
    {
        if (x) *x = Left + Width() * 0.5f;
        if (y) *y = Top + Height() * 0.5f;
    }

    /**< Return true if the coordinate is inside the bounding box */
    inline bool Contains( float x, float y ) const
    {
        return x >= Left && x <= Right && y >= Top && y <= Bottom;
    }

    /**< Return true if the bounding boxes intersect and exceeds area % threshold */
    inline bool Intersects( const Detection &det, float areaThreshold = 0.0f ) const
    {
        return (IntersectionArea(det) / fmaxf(Area(), det.Area()) > areaThreshold);
    }

    /**< Return true if the bounding boxes intersect and exceeds area % threshold */
    inline bool Intersects( float x1, float y1, float x2, float y2, float areaThreshold = 0.0f ) const
    {
        return (IntersectionArea(x1, y1, x2, y2) / fmaxf(Area(), Area(x1, y1, x2, y2)) > areaThreshold);
    }

    /**< Return the area of the bounding box intersection */
    inline float IntersectionArea( const Detection &det ) const
    {
        return IntersectionArea(det.Left, det.Top, det.Right, det.Bottom);
    }

    /**< Return the area of the bounding box intersection */
    inline float IntersectionArea( float x1, float y1, float x2, float y2 ) const
    {
        if (!Overlaps(x1, y1, x2, y2)) return 0.0f;
        return (fminf(Right, x2) - fmaxf(Left, x1)) * (fminf(Bottom, y2) - fmaxf(Top, y1));
    }

    /**< Return true if the bounding boxes overlap */
    inline bool Overlaps( const Detection &det ) const
    {
        return !(det.Left > Right || det.Right < Left || det.Top > Bottom || det.Bottom < Top);
    }

    /**< Return true if the bounding boxes overlap */
    inline bool Overlaps( float x1, float y1, float x2, float y2 ) const
    {
        return !(x1 > Right || x2 < Left || y1 > Bottom || y2 < Top);
    }

    /**< Expand the bounding box if they overlap (return true if so) */
    inline bool Expand( float x1, float y1, float x2, float y2 )
    {
        if (!Overlaps(x1, y1, x2, y2)) return false;
        Left = fminf(x1, Left);
        Top = fminf(y1, Top);
        Right = fmaxf(x2, Right);
        Bottom = fmaxf(y2, Bottom);
        return true;
    }

    /**< Expand the bounding box if they overlap (return true if so) */
    inline bool Expand( const Detection &det )
    {
        if (!Overlaps(det)) return false;
        Left = fminf(det.Left, Left);
        Top = fminf(det.Top, Top);
        Right = fmaxf(det.Right, Right);
        Bottom = fmaxf(det.Bottom, Bottom);
        return true;
    }

    /**< Reset all member variables to zero */
    inline void Reset()
    {
        Instance = 0;
        ClassID = 0;
        Confidence = 0;
        Left = 0;
        Right = 0;
        Top = 0;
        Bottom = 0;
    }

    /**< Default constructor */
    inline Detection()
    {
        Reset();
    }
} Detection;


#endif
