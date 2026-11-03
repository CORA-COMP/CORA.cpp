// test_fullspace_vertices_3d - fullspace vertices_3d: the eight corners of the unbounded box
//
// CORA: contSet/@fullspace/vertices_.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

#include <cmath>
#include <set>

using namespace cora;
using test::check;

namespace {

void the_corners_are_at_infinity(const std::string &b) {
    const std::vector<std::vector<Point3>> V = Fullspace(3).vertices_3d();
    check(V.size() == 1 && V[0].size() == 8, b + ": one list of eight corners");
    std::set<std::array<double, 3>> signs;
    for (const Point3 &p : V[0]) {
        check(std::isinf(p[0]) && std::isinf(p[1]) && std::isinf(p[2]), b + ": all infinite");
        signs.insert(p);
    }
    check(signs.size() == 8, b + ": every sign combination occurs once");
}

/// A set of two dimensions has no corners in space; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { Fullspace(2).vertices_3d(); }, Error::not_supported),
          "vertices_3d: two dimensions");
    ChecksScope off(false);
    check(Fullspace(3).vertices_3d()[0].size() == 8, "vertices_3d: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_corners_are_at_infinity);
    checks_and_their_switch();
    return test::finish("fullspace vertices_3d");
}
