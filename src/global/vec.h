// vec - the few operations on points of the plane and of space that geometry code needs
//
// Syntax:   Point3 n = cross(a, b);   double s = dot(a, b);   double len = norm(a);
//           double z = cross(p, q);   (the same for Point, with sub and segment_distance)
// See also: cora/cont_set/cont_set.h (Point, Point3)

#pragma once

#include "cora/cont_set/cont_set.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

inline double cross(const Point &a, const Point &b) { return a[0] * b[1] - a[1] * b[0]; }

inline Point sub(const Point &a, const Point &b) { return {a[0] - b[0], a[1] - b[1]}; }

inline double norm(const Point &a) { return std::hypot(a[0], a[1]); }

inline double dot(const Point &a, const Point &b) { return a[0] * b[0] + a[1] * b[1]; }

/// The distance of p to the segment ab.
inline double segment_distance(const Point &p, const Point &a, const Point &b) {
    const Point d = sub(b, a);
    const double len2 = dot(d, d);
    const double t = len2 > 0 ? std::clamp(dot(sub(p, a), d) / len2, 0.0, 1.0) : 0.0;
    return norm(sub(p, {a[0] + t * d[0], a[1] + t * d[1]}));
}

/// The signed area of a polygon: positive if it runs counter-clockwise.
inline double area(const Polygon &p) {
    double a = 0;
    for (std::size_t i = 0; i < p.size(); ++i) a += cross(p[i], p[(i + 1) % p.size()]);
    return a / 2;
}

inline Point3 cross(const Point3 &a, const Point3 &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double dot(const Point3 &a, const Point3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const Point3 &a) { return std::sqrt(dot(a, a)); }

/// a + s * b.
inline Point3 add(const Point3 &a, const Point3 &b, double s = 1) {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
}

inline Point3 sub(const Point3 &a, const Point3 &b) { return add(a, b, -1); }

inline Point3 scale(const Point3 &a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
