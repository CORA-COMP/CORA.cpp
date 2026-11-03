// test_cont_set_is_intersecting_impl - the unchecked intersection test (CORA's isIntersecting_)
//
// CORA: contSet/@contSet/isIntersecting_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The unchecked form gives what the entry gives, for the class pairs of the table.
void is_intersecting_impl_agrees_with_the_entry(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})),
        J(column({0.5, 1.0}), column({3.0, 3.0})), K(column({-0.25, 0.75}), column({0.25, 1.25}));
    const Zonotope Z(column({0.0, 1.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    for (const ContSet *A : {static_cast<const ContSet *>(&I), static_cast<const ContSet *>(&Z)})
        for (const ContSet *B : {static_cast<const ContSet *>(&J), static_cast<const ContSet *>(&K),
                                 static_cast<const ContSet *>(&Z)})
            check(cora::detail::is_intersecting_impl(*A, *B, std::nullopt) == is_intersecting(*A, *B),
                  b + ": as the entry");
    check(cora::detail::is_intersecting_impl(I, J, std::nullopt) &&
              cora::detail::is_intersecting_impl(I, K, std::nullopt),
          b + ": the boxes meet");
    const Interval far(column({5.0, 5.0}), column({6.0, 6.0}));
    check(!cora::detail::is_intersecting_impl(I, far, std::nullopt), b + ": far boxes do not");
}

/// A pair without an entry is noops; the entry checks the dimensions.
void a_pair_without_an_entry() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0})), I1(column({0.0}), column({1.0}));
    check(test::raises([&] { cora::detail::is_intersecting_impl(test::UnregisteredSet(), I, std::nullopt); },
                       Error::noops),
          "isIntersecting_: no pair");
    check(test::raises([&] { is_intersecting(I, I1); }, Error::dimension_mismatch),
          "the entry checks the dimensions");
    ChecksScope off(false);
    check(cora::detail::is_intersecting_impl(I, I, std::nullopt), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(is_intersecting_impl_agrees_with_the_entry);
    a_pair_without_an_entry();
    return test::finish("cont_set is_intersecting_impl");
}
