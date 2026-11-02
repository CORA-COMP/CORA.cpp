// test_cont_set_vertices - ContSet::vertices: two-dimensional sets only

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// A set of three dimensions is refused; a box has four vertices.
void only_two_dimensions() {
    const Interval I3 = Interval(test::column({0, -1, 0}), test::column({1, 1, 2}));
    const ContSet &S3 = I3;
    check(test::raises([&] { S3.vertices(); }, Error::not_supported), "three dimensions");
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    const ContSet &S = I;
    check(S.vertices()[0].size() == 4, "a box has four vertices");
    ChecksScope off(false);
    check(S.vertices()[0].size() == 4, "the checks off");
}

} // namespace

int main() {
    only_two_dimensions();
    return test::finish("cont_set vertices");
}
