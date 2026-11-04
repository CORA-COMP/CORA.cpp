// test_cont_set_is_full_dim - ContSet::is_full_dim: whether a set has a nonempty interior
//
// CORA: contSet/@contSet/isFullDim.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// Boxes and zonotopes through the base class; tol relaxes the width test.
void full_dimension_of_any_set(const std::string &b) {
    const Interval box(column({-1.0, 0.0}), column({1.0, 2.0})),
        flat(column({-1.0, 1.0}), column({1.0, 1.0})), thin(column({0.0}), column({1e-4}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2})),
        segment(column({0.0, 0.0}), Tensor::from_data({1.0, 1.0}, {2, 1}));
    const ContSet &A = box, &B = flat, &D = Z, &E = segment, &T = thin;
    check(A.is_full_dim() && !B.is_full_dim(), b + ": boxes");
    check(D.is_full_dim() && !E.is_full_dim(), b + ": zonotopes");
    // a width of 1e-4 is zero for a loose tolerance only
    check(T.is_full_dim() && !T.is_full_dim(1e-3), b + ": tol relaxes the width test");
}

/// A negative tol is refused, a class without the query raises noops, and the checks off skip tol.
void errors_and_their_switch() {
    const Interval I(column({0.0}), column({1.0}));
    const ContSet &S = I;
    check(test::raises([&] { S.is_full_dim(-1.0); }, Error::wrong_value), "tol -1");
    check(test::raises([] { test::UnregisteredSet().is_full_dim(); }, Error::noops), "noops");
    ChecksScope off(false);
    check(S.is_full_dim(-1.0), "the checks off skip tol");
}

} // namespace

int main() {
    test::for_each_backend(full_dimension_of_any_set);
    errors_and_their_switch();
    return test::finish("cont_set is_full_dim");
}
