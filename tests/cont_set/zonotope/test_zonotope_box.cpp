// test_zonotope_box - zonotope box: the box around the zonotope, as a zonotope
//
// CORA: contSet/@zonotope/box.m
// PROPERTY: contSet/@zonotope/box.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The center stays, the generators become diag(sum |G|).
void box_of_a_known_zonotope(const std::string &b) {
    const Zonotope Z(column({1.0, -2.0}), Tensor({{1.0, -3.0, 0.5}, {-2.0, 0.5, 0.0}}));
    const Zonotope B = Z.box();
    check(close(B.center(), Z.center(), 0), b + ": the center stays");
    check(close(B.generators(), std::vector<double>{4.5, 0, 0, 2.5}, 0), b + ": diag(sum |G|)");
    const Zonotope batch = Zonotope::stack({Z, Z + column({1.0, 1.0})}).box();
    check(batch.generators().shape() == std::vector<int64_t>({2, 2, 2}), b + ": batch shape");
}

/// Stated property: the box has the interval hull of the zonotope, contains it, and is a box.
void box_has_the_interval_hull(const std::string &b) {
    cora::Rng rng(12);
    const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 7);
    const Zonotope B = Z.box();
    check(close(B.interval().inf(), Z.interval().inf(), 0) &&
              close(B.interval().sup(), Z.interval().sup(), 0),
          b + ": the same interval hull");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 4);
        check(test::support(B, d) >= test::support(Z, d) - 1e-12, b + ": the box contains Z");
    }
    check(B.box().generators().shape() == B.generators().shape(), b + ": a box is its own box");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        box_of_a_known_zonotope(b);
        box_has_the_interval_hull(b);
    });
    ChecksScope off(false);
    const Zonotope Z(column({0.0}), Tensor({{1.0, -2.0}}));
    check(close(Z.box().generators(), std::vector<double>{3}, 0),
          "box: the same with the checks off");
    return test::finish("zonotope box");
}
