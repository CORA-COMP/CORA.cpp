// geometry - union of polygons, thinning of polygons, faces of a convex hull
//
// Syntax:   loops = union_polygons(polygons);   p = thin_polygon(p, {0.01, 0.01});
//           faces = hull_faces(points);
// See also: cora/global/plot/figure.h, cora/global/plot/plot.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/tensor/tolerance.h"


#include <array>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The union of the polygons as closed loops: outer boundaries counter-clockwise, holes
/// clockwise; corners within `resolution` of the extent are dropped first.
std::vector<Outline> union_polygons(const std::vector<Outline> &polygons,
                                    double resolution = numerics::kPlotResolution);

/// The polygon without the vertices that lie within resolution (per axis) of the line through their
/// neighbors, which also removes repeated points (Douglas-Peucker). A closed polygon keeps at
/// least three vertices if it had them.
Outline thin_polygon(const Outline &polygon, Point resolution, bool closed = true);

/// The faces of the convex hull of the points, each a polygon seen counter-clockwise from
/// outside, coplanar triangles merged. A flat point set gives one face, a segment a face of two
/// points, and a single point a face of one.
std::vector<std::vector<Point3>> hull_faces(const std::vector<Point3> &points);

/// The corner for a legend of size w x h in the unit square of the axes (y up) that covers the
/// least of the boxes {x0, x1, y0, y1} of the drawn sets: 0 upper right, 1 upper left, 2 lower
/// left, 3 lower right, the first on a tie (as matplotlib's and pgfplots' default order).
int legend_corner(const std::vector<std::array<double, 4>> &boxes, double w, double h);

/// The boxes {x0, x1, y0, y1} of the polygons of a figure's layers in the unit square of the
/// limits, for legend_corner.
std::vector<std::array<double, 4>> unit_boxes(const std::vector<std::vector<Outline>> &layers,
                                              double x0, double x1, double y0, double y1);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
