// test_ellipsoid_is_full_dim - Ellipsoid::is_full_dim: the rank of the shape matrix is n
//
// CORA: contSet/@ellipsoid/isFullDim.m
// PROPERTY: contSet/@ellipsoid/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a ball, a segment, a point, the empty set; the tolerance decides near rank loss.
void is_full_dim_known(const std::string &b) {
    check(Ellipsoid(Tensor({{4.0, 0.0}, {0.0, 1.0}})).is_full_dim(), b + ": full rank");
    check(!Ellipsoid(Tensor({{1.0, 0.0}, {0.0, 0.0}})).is_full_dim(), b + ": a segment");
    check(!Ellipsoid(Tensor::zeros({2, 2})).is_full_dim(), b + ": a point");
    check(!Ellipsoid::empty(2).is_full_dim(), b + ": the empty ellipsoid");
    // the second singular value is 1e-8 of the largest: full rank for the default tolerance, lost
    // for CORA's 1e-6 of the object and for an explicit tolerance
    const Tensor Q({{1.0, 0.0}, {0.0, 1e-8}});
    check(Ellipsoid(Q).is_full_dim(), b + ": a thin ellipsoid at the default tolerance");
    check(!Ellipsoid(Q, 1e-6).is_full_dim(), b + ": a thin ellipsoid at the tolerance 1e-6");
    check(!Ellipsoid(Q).is_full_dim(1e-6), b + ": the argument replaces the tolerance");
    // a batch is full-dimensional if every member is
    const Tensor both = Tensor::stack({Tensor({{1.0, 0.0}, {0.0, 1.0}}), Q * 100.0}, 0);
    check(Ellipsoid(both).is_full_dim(), b + ": a batch of full rank");
    const Tensor mixed =
        Tensor::stack({Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor::zeros({2, 2})}, 0);
    check(!Ellipsoid(mixed).is_full_dim(), b + ": a batch with a member of lower rank");
    const Set S = Ellipsoid(Tensor({{1.0}}));
    check(S->is_full_dim(), b + ": through the interface of every set");
}

/// Stated property: Q = A A' with A of n x r has rank r, so the set is full-dimensional iff r = n.
void is_full_dim_property(const std::string &b) {
    cora::Rng rng(5);
    for (int n = 1; n <= 5; ++n)
        for (int r = 1; r <= n; ++r)
            check(test::random_ellipsoid(rng, n, r).is_full_dim() == (r == n),
                  b + ": the rank of A A' decides");
}

/// A negative tolerance is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.is_full_dim(-1.0); }, Error::wrong_value), "is_full_dim: tol >= 0");
    const bool on = E.is_full_dim();
    ChecksScope off(false);
    check(E.is_full_dim() == on, "is_full_dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        is_full_dim_known(b);
        is_full_dim_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
