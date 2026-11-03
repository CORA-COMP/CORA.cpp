// test_lean_fullspace_sets - the set operations and constructors of Fullspace against the CORALean oracle
//
// ORACLE: contSet/@fullspace/Inf.m
// ORACLE: contSet/@fullspace/fullspace.m
// ORACLE: contSet/@fullspace/and_.m
// ORACLE: contSet/@fullspace/box.m
// ORACLE: contSet/@fullspace/convHull_.m
// ORACLE: contSet/@fullspace/enclosePoints.m
// ORACLE: contSet/@fullspace/interval.m
// ORACLE: contSet/@fullspace/lift_.m
// ORACLE: contSet/@fullspace/plus.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/fullspace.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("fullspace.Inf", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5}) test::oracle::check(Fullspace::inf(n), lean::Fullspace::inf(n));
    });
    test::oracle::run("fullspace.box", [](test::oracle::Random &) {
        for (const int n : {1, 5}) test::oracle::check(Fullspace(n).box(), lean::Fullspace(n).box());
    });
    test::oracle::run("fullspace.interval", [](test::oracle::Random &) {
        for (const int n : {1, 2, 5})
            test::oracle::check(Fullspace(n).interval(), lean::Fullspace(n).interval());
    });
    test::oracle::run("fullspace.and", [](test::oracle::Random &rnd) {
        const Fullspace F(3);
        const Interval I = rnd.box(3);
        test::oracle::check(F.intersect(Fullspace(3)).as<Fullspace>(),
                            lean::Fullspace(3).intersect(lean::Fullspace(3)));
        test::oracle::check(F.intersect(I).as<Interval>(), lean::Fullspace(3).intersect(I));
    });
    test::oracle::run("fullspace.plus", [](test::oracle::Random &) {
        test::oracle::check(Fullspace(4).plus(Fullspace(4)),
                            lean::Fullspace(4).plus(lean::Fullspace(4)));
    });
    test::oracle::run("fullspace.convHull", [](test::oracle::Random &) {
        test::oracle::check(Fullspace(4).convex_hull(Fullspace(4)),
                            lean::Fullspace(4).convex_hull(lean::Fullspace(4)));
    });
    test::oracle::run("fullspace.enclosePoints", [](test::oracle::Random &rnd) {
        const Tensor points = rnd.matrix(4, 7);
        test::oracle::check(Fullspace::enclose_points(points),
                            lean::Fullspace::enclose_points(points));
    });
    test::oracle::run("fullspace.lift", [](test::oracle::Random &) {
        const std::vector<int64_t> dims{3, 0, 4};
        test::oracle::check(Fullspace(3).lift(6, dims), lean::Fullspace(3).lift(6, dims));
    });
    return test::finish("lean fullspace sets");
}
