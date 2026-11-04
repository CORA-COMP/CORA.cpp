// test_capsule_is_full_dim - Capsule is_full_dim: a nonempty interior
//
// CORA: contSet/@capsule/isFullDim.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// A ball and a thick capsule have an interior; a segment and a point in R^2 have none; in R^1 a
/// segment of positive length is an interval (CORA's r > 0 says no).
void is_full_dim_known(const std::string &b) {
    check(Capsule(column({0.0, 0.0}), column({1.0, 0.0}), 0.1).is_full_dim(), b + ": r > 0");
    check(Capsule(column({0.0, 0.0}), column({0.0, 0.0}), 0.1).is_full_dim(), b + ": a ball");
    check(!Capsule(column({0.0, 0.0}), column({1.0, 0.0}), 0.0).is_full_dim(),
          b + ": a segment in R^2");
    check(!Capsule(column({0.0, 0.0})).is_full_dim(), b + ": a point in R^2");
    check(Capsule(column({0.0}), column({1.0}), 0.0).is_full_dim(), b + ": a segment in R^1");
    check(!Capsule(column({0.0}), column({0.0}), 0.0).is_full_dim(), b + ": a point in R^1");
    check(Capsule(column({0.0}), column({0.0}), 0.5).is_full_dim(), b + ": a ball in R^1");
    check(!Capsule::empty(2).is_full_dim() && !Capsule::empty(1).is_full_dim(),
          b + ": the empty capsule has no interior");
    check(Set(Capsule(column({0.0, 0.0}), column({1.0, 0.0}), 0.1))->is_full_dim(),
          b + ": through a Set");
}

/// A batch is full-dimensional if every member is.
void is_full_dim_batch(const std::string &b) {
    const Tensor c = Tensor::zeros({2, 2, 1}), g = Tensor::ones({2, 2, 1});
    check(Capsule(c, g, Tensor::from_data({0.1, 0.2}, {2, 1, 1})).is_full_dim(),
          b + ": every member");
    check(!Capsule(c, g, Tensor::from_data({0.1, 0.0}, {2, 1, 1})).is_full_dim(),
          b + ": one segment");
}

/// A negative tolerance is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Capsule C(column({0.0, 0.0}), column({1.0, 0.0}), 0.1);
    check(test::raises([&] { C.is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol is nonnegative");
    ChecksScope off(false);
    check(C.is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        is_full_dim_known(b);
        is_full_dim_batch(b);
    });
    checks_and_their_switch();
    return test::finish("capsule is_full_dim");
}
