// test_cont_set_linear_map - ContSet::linear_map: M S in the class of S, the matrix is checked
//
// CORA: contSet/@contSet/mtimes.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// A shear maps a box to the hull of its image and a zonotope exactly; a row matrix lowers the
/// dimension.
void the_image_of_any_set(const std::string &b) {
    const Tensor M = Tensor::from_data({1.0, 1.0, 0.0, 1.0}, {2, 2});
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0}));
    const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    const Set RI = static_cast<const ContSet &>(I).linear_map(M);
    // center (0, 1) -> (1, 1); radius (1, 1) -> (2, 1)
    check(RI.is<Interval>() && test::close(RI.as<Interval>().inf(), {-1.0, 0.0}) &&
              test::close(RI.as<Interval>().sup(), {3.0, 2.0}),
          b + ": the hull of the image of a box");
    const Set RZ = static_cast<const ContSet &>(Z).linear_map(M);
    check(RZ.is<Zonotope>() && test::close(RZ.as<Zonotope>().center(), {1.0, 0.0}) &&
              test::close(RZ.as<Zonotope>().generators(), {0.5, 0.5, 0.0, 0.5}),
          b + ": the image of a zonotope");
    const Tensor row = Tensor::from_data({1.0, -1.0}, {1, 2});
    check(static_cast<const ContSet &>(Z).linear_map(row).get().dim() == 1,
          b + ": a row matrix gives one dimension");
}

/// A matrix of the wrong width is refused; a class without the map raises noops; the checks off
/// skip the check.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    const ContSet &S = I;
    const Tensor wide = Tensor::from_data({1.0, 0.0, 0.0}, {1, 3});
    check(test::raises([&] { S.linear_map(wide); }, Error::dimension_mismatch), "three columns");
    check(test::raises([] { test::UnregisteredSet().linear_map(Tensor::eye(2)); }, Error::noops),
          "noops");
    ChecksScope off(false);
    check(S.linear_map(Tensor::eye(2)).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_image_of_any_set);
    errors_and_their_switch();
    return test::finish("cont_set linear_map");
}
