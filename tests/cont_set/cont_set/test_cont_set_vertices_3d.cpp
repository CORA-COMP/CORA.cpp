// test_cont_set_vertices_3d - ContSet::vertices_3d: three-dimensional sets only
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A set of two dimensions is refused; a box has eight vertices.
void only_three_dimensions() {
    const Interval I2 = Interval(test::vec({0, -1}), test::vec({1, 1}));
    const ContSet &S2 = I2;
    check(test::raises([&] { S2.vertices_3d(); }, Error::not_supported), "two dimensions");
    const Interval I = Interval(test::vec({0, -1, 0}), test::vec({1, 1, 2}));
    const ContSet &S = I;
    check(S.vertices_3d()[0].size() == 8, "a box has eight vertices");
    ChecksScope off(false);
    check(S.vertices_3d()[0].size() == 8, "the checks off");
}

/// Through the abstract class, a zonotope gives its corners too, and the projection of a
/// four-dimensional set onto three dimensions is what the plot uses.
void through_a_projection() {
    const Zonotope Z(test::column({0, 0, 0, 0}),
                     Tensor({{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}));
    const Set P = Z.project({0, 1, 2});
    check(P->vertices_3d()[0].size() == 8, "a projected cube has eight corners");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    only_three_dimensions();
    through_a_projection();
    return test::finish("cont_set vertices_3d");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
