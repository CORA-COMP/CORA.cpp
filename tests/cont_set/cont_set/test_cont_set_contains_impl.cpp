// test_cont_set_contains_impl - the unchecked containment of two sets (CORA's contains_)
//
// CORA: contSet/@contSet/contains_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The unchecked form gives what the entry gives: a zonotope contains a small box, not a far one.
void contains_impl_agrees_with_the_entry(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})),
        J(column({0.5, 1.0}), column({3.0, 3.0})), K(column({-0.25, 0.75}), column({0.25, 1.25}));
    const Zonotope Z(column({0.0, 1.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    check(cora::detail::contains_impl(Z, K, std::nullopt) && !cora::detail::contains_impl(Z, J, std::nullopt),
          b + ": a zonotope contains a box");
    for (const ContSet *B : {static_cast<const ContSet *>(&J), static_cast<const ContSet *>(&K),
                             static_cast<const ContSet *>(&Z)})
        check(cora::detail::contains_impl(Z, *B, std::nullopt) == contains(Z, *B), b + ": as the entry");
}

/// A pair without an entry is noops; the entry checks the dimensions.
void a_pair_without_an_entry() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2})),
        Z1(column({0.0}), Tensor::from_data({1.0}, {1, 1}));
    check(test::raises([&] { cora::detail::contains_impl(test::UnregisteredSet(), I, std::nullopt); },
                       Error::noops),
          "contains_: no pair");
    check(test::raises([&] { contains(Z1, I); }, Error::dimension_mismatch),
          "the entry checks the dimensions");
    ChecksScope off(false);
    check(cora::detail::contains_impl(Z, I, std::nullopt), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(contains_impl_agrees_with_the_entry);
    a_pair_without_an_entry();
    return test::finish("cont_set contains_impl");
}
