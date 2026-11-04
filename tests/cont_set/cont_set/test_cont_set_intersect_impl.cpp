// test_cont_set_intersect_impl - the unchecked intersection of two sets (CORA's and_)
//
// CORA: contSet/@contSet/and_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The unchecked form gives what the entry gives, for every class pair of the table.
void intersect_impl_agrees_with_the_entry(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})),
        J(column({0.5, 1.0}), column({3.0, 3.0})), K(column({-0.25, 0.75}), column({0.25, 1.25}));
    const Zonotope Z(column({0.0, 1.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    const Set X = cora::detail::intersect_impl(I, J);
    check(X.is<Interval>() && test::close(X.as<Interval>().inf(), {0.5, 1.0}) &&
              test::close(X.as<Interval>().sup(), {1.0, 2.0}),
          b + ": the intersection of two boxes");
    for (const ContSet *B : {static_cast<const ContSet *>(&J), static_cast<const ContSet *>(&Z)}) {
        const Set Y = cora::detail::intersect_impl(Z, *B), Y2 = intersect(Z, *B);
        check(test::close(test::support(Y, {1.0, 0.0}), test::support(Y2, {1.0, 0.0}), 0.0),
              b + ": the same set as the entry");
    }
    check(test::close(cora::detail::intersect_impl(I, K).as<Interval>().inf(), K.inf(), 0.0),
          b + ": a box inside the other is the intersection");
}

/// A pair without an entry is noops; the entry checks the dimensions, the unchecked form does not.
void a_pair_without_an_entry() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0})), I1(column({0.0}), column({1.0}));
    check(test::raises([&] { cora::detail::intersect_impl(I, test::UnregisteredSet()); },
                       Error::noops),
          "and_: no pair");
    check(test::raises([&] { intersect(I, I1); }, Error::dimension_mismatch),
          "the entry checks the dimensions");
    ChecksScope off(false);
    check(cora::detail::intersect_impl(I, I).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(intersect_impl_agrees_with_the_entry);
    a_pair_without_an_entry();
    return test::finish("cont_set intersect_impl");
}
