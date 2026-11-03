// test_lean_fullspace_predicates - the predicates and the center of Fullspace against the oracle
//
// ORACLE: contSet/@fullspace/center.m
// ORACLE: contSet/@fullspace/contains_.m
// ORACLE: contSet/@fullspace/isBounded.m
// ORACLE: contSet/@fullspace/isFullDim.m
// ORACLE: contSet/@fullspace/isIntersecting_.m
// ORACLE: contSet/@fullspace/isequal.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/fullspace.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("fullspace.center", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5})
            test::oracle::check(Fullspace(n).center(), lean::Fullspace(n).center());
    });
    test::oracle::run("fullspace.contains", [](test::oracle::Random &rnd) {
        const Interval I = rnd.box(3);
        test::oracle::check(Fullspace(3).contains(I), lean::Fullspace(3).contains(I));
    });
    test::oracle::run("fullspace.isBounded", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5})
            test::oracle::check(Fullspace(n).is_bounded(), lean::Fullspace(n).is_bounded());
    });
    test::oracle::run("fullspace.isFullDim", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5})
            test::oracle::check(Fullspace(n).is_full_dim(), lean::Fullspace(n).is_full_dim());
    });
    test::oracle::run("fullspace.isIntersecting", [](test::oracle::Random &rnd) {
        const Interval I = rnd.box(3);
        test::oracle::check(Fullspace(3).is_intersecting(I), lean::Fullspace(3).is_intersecting(I));
    });
    test::oracle::run("fullspace.isequal", [](test::oracle::Random &) {
        test::oracle::check(Fullspace(3).isequal(Fullspace(3)),
                            lean::Fullspace(3).isequal(lean::Fullspace(3)));
    });
    return test::finish("lean fullspace predicates");
}
