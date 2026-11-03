// test_empty_set_project - EmptySet project: the empty set with one dimension per entry
//
// CORA: contSet/@emptySet/project.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_projection_keeps_only_the_dimension(const std::string &b) {
    const Set P = EmptySet(4).project({2, 0});
    check(P.is<EmptySet>() && P.as<EmptySet>().dim() == 2, b + ": two of four dimensions");
    check(EmptySet(3).project({1, 1, 1}).as<EmptySet>().dim() == 3, b + ": a repeated entry counts");
}

/// A dimension outside 0..n-1 is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(3);
    check(test::raises([&] { E.project({3}); }, Error::wrong_value), "project: dimension 3 of 3");
    check(test::raises([&] { E.project({-1}); }, Error::wrong_value), "project: dimension -1");
    ChecksScope off(false);
    check(E.project({0, 2}).as<EmptySet>().dim() == 2, "project: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_projection_keeps_only_the_dimension);
    checks_and_their_switch();
    return test::finish("empty_set project");
}
