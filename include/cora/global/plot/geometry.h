// geometry - union of polygons, thinning of polygons, faces of a convex hull
//
// Syntax:   loops = union_polygons(polygons);   p = thin_polygon(p, {0.01, 0.01});
//           faces = hull_faces(points);
// See also: cora/global/plot/figure.h, cora/global/plot/plot.h

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/tensor/tolerance.h"


#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The union of the polygons as closed loops: outer boundaries counter-clockwise, holes clockwise,
/// so that filling them with the nonzero rule gives the union. Polygons (simple, convex or not) may
/// overlap, touch or share edges; those of fewer than three points are left out. Corners within
/// `resolution` times the extent of the polygons of the line through their neighbors are dropped
/// first (that much less exact, and much faster for polygons of many corners). If the boundary
/// cannot be traced (nearly degenerate input), the polygons are returned as they are,
/// counter-clockwise.
std::vector<Polygon> union_polygons(const std::vector<Polygon> &polygons,
                                    double resolution = numerics::kPlotResolution);

/// The polygon without the vertices that lie within resolution (per axis) of the line through their
/// neighbors, which also removes repeated points (Douglas-Peucker). A closed polygon keeps at
/// least three vertices if it had them.
Polygon thin_polygon(const Polygon &polygon, Point resolution, bool closed = true);

/// The faces of the convex hull of the points, each a polygon seen counter-clockwise from
/// outside, coplanar triangles merged. A flat point set gives one face, a segment a face of two
/// points, and a single point a face of one.
std::vector<std::vector<Point3>> hull_faces(const std::vector<Point3> &points);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
