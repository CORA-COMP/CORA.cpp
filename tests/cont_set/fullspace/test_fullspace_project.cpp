// test_fullspace_project - fullspace project: the whole space of the kept dimensions
//
// CORA: contSet/@fullspace/project.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_projection_keeps_only_the_dimension(const std::string &b) {
    const Set P = Fullspace(4).project({2, 0});
    check(P.is<Fullspace>() && P.as<Fullspace>().dim() == 2, b + ": two of four dimensions");
    check(Fullspace(3).project({1}).as<Fullspace>().dim() == 1, b + ": one dimension");
    check(Fullspace(3).project({1, 1}).as<Fullspace>().dim() == 2,
          b + ": a repeated entry gives the enclosure, as in CORA");
}

/// A dimension outside 0..n-1, none, and R^0 are refused; with the checks off the result is the
/// same.
void checks_and_their_switch() {
    const Fullspace F(3);
    check(test::raises([&] { F.project({3}); }, Error::wrong_value), "project: dimension 3 of 3");
    check(test::raises([&] { F.project({-1}); }, Error::wrong_value), "project: dimension -1");
    check(test::raises([&] { F.project({}); }, Error::wrong_value), "project: no dimension");
    check(test::raises([&] { Fullspace(0).project({0}); }, Error::wrong_value), "project: R^0");
    ChecksScope off(false);
    check(F.project({0, 2}).as<Fullspace>().dim() == 2, "project: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_projection_keeps_only_the_dimension);
    checks_and_their_switch();
    return test::finish("fullspace project");
}
