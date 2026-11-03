// test_cont_set_project_high_dim - ContSet::project_high_dim: a set embedded in more dimensions
//
// CORA: contSet/@contSet/projectHighDim.m
// CORA: contSet/@contSet/projectHighDim_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The coordinates move to proj and the others are zero: by the box's own rule and by the
/// selection matrix of the generic one.
void the_embedding(const std::string &b) {
    const Interval I(column({1.0, 3.0}), column({2.0, 4.0}));
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.25}, {2, 2}));
    const Set RI = static_cast<const ContSet &>(I).project_high_dim(4, {2, 0});
    check(test::close(RI.as<Interval>().inf(), {3.0, 0.0, 1.0, 0.0}) &&
              test::close(RI.as<Interval>().sup(), {4.0, 0.0, 2.0, 0.0}),
          b + ": a box");
    const Set RZ = static_cast<const ContSet &>(Z).project_high_dim(4, {2, 0});
    check(RZ.is<Zonotope>() && test::close(RZ.as<Zonotope>().center(), {2.0, 0.0, 1.0, 0.0}) &&
              test::close(RZ.as<Zonotope>().generators(),
                          {0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0}),
          b + ": a zonotope");
    check(static_cast<const ContSet &>(Z).project_high_dim(2, {0, 1}).get().dim() == 2,
          b + ": the same dimension");
}

/// A smaller space, a wrong number of coordinates and a coordinate outside are refused; a class
/// without the map raises noops; the checks off skip the checks.
void errors_and_their_switch() {
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.25}, {2, 2}));
    const ContSet &S = Z;
    check(test::raises([&] { S.project_high_dim(1, {0, 0}); }, Error::wrong_value), "N below n");
    check(test::raises([&] { S.project_high_dim(3, {0}); }, Error::wrong_value), "one coordinate");
    check(test::raises([&] { S.project_high_dim(3, {0, 3}); }, Error::wrong_value), "past N");
    check(test::raises([] { test::UnregisteredSet().project_high_dim(3, {0, 1}); }, Error::noops),
          "noops");
    ChecksScope off(false);
    check(S.project_high_dim(3, {0, 2}).get().dim() == 3, "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_embedding);
    errors_and_their_switch();
    return test::finish("cont_set project_high_dim");
}
