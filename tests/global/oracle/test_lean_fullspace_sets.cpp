// test_lean_fullspace_sets - the set operations and constructors of Fullspace against the CORALean
//
// oracle
//
// ORACLE: contSet/@fullspace/Inf.m
// ORACLE: contSet/@fullspace/fullspace.m
// ORACLE: contSet/@fullspace/and_.m
// ORACLE: contSet/@fullspace/box.m
// ORACLE: contSet/@fullspace/convHull_.m
// ORACLE: contSet/@fullspace/enclosePoints.m
// ORACLE: contSet/@fullspace/interval.m
// ORACLE: contSet/@fullspace/lift_.m
// ORACLE: contSet/@fullspace/mtimes.m
// ORACLE: contSet/@fullspace/not.m
// ORACLE: contSet/@fullspace/plus.m
// ORACLE: contSet/@fullspace/project.m
// ORACLE: contSet/@fullspace/projectHighDim_.m
// ORACLE: contSet/@fullspace/radius.m
// ORACLE: contSet/@fullspace/supportFunc_.m
// ORACLE: contSet/@fullspace/volume_.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/empty_set.h"
#include "cora/global/oracle/fullspace.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("fullspace.Inf", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5})
            test::oracle::check(Fullspace::inf(n), lean::Fullspace::inf(n));
    });
    test::oracle::run("fullspace.box", [](test::oracle::Random &) {
        for (const int n : {1, 5})
            test::oracle::check(Fullspace(n).box(), lean::Fullspace(n).box());
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
        test::oracle::check(Fullspace(4).plus(Fullspace(4)).as<Fullspace>(),
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
    test::oracle::run("fullspace.project", [](test::oracle::Random &) {
        const std::vector<int64_t> dims{2, 0};
        test::oracle::check(Fullspace(3).project(dims).as<Fullspace>(),
                            lean::Fullspace(3).project(dims));
    });
    // CORA and CORALean agree at N = n; for a larger N CORALean gives R^N and CORA refuses
    test::oracle::run("fullspace.projectHighDim", [](test::oracle::Random &) {
        const std::vector<int64_t> proj{1, 0, 2};
        test::oracle::check(Fullspace(3).project_high_dim(3, proj),
                            lean::Fullspace(3).project_high_dim(3, proj));
    });
    // the image of R^n under a matrix of full row rank is the whole space of its rows; CORALean
    // gives that for every matrix, CORA a box where M has a zero row (see the next op)
    test::oracle::run("fullspace.mtimes", [](test::oracle::Random &rnd) {
        for (const int64_t m : {1, 3, 7}) {
            const Tensor M = rnd.matrix(m, 3);
            test::oracle::check(Fullspace(3).linear_map(M).as<Fullspace>(),
                                lean::Fullspace(3).linear_map(M));
        }
        // a zero row: the box with that coordinate at 0 is inside the whole space of the oracle
        const Tensor M = Tensor::concat({rnd.matrix(2, 3), Tensor::zeros({1, 3})}, -2);
        test::oracle::check((*Fullspace(3).linear_map(M)).dim(),
                            lean::Fullspace(3).linear_map(M).dim());
    });
    // Inf in every direction but the zero one, where the support function is 0
    test::oracle::run("fullspace.supportFunc", [](test::oracle::Random &rnd) {
        for (const int n : {1, 3, 5}) {
            const Tensor dir = rnd.direction(n), zero = dir.zeros_like();
            test::oracle::check(Fullspace(n).support_function(dir),
                                lean::Fullspace(n).support_function(dir));
            test::oracle::check(Fullspace(n).support_function(zero),
                                lean::Fullspace(n).support_function(zero));
        }
    });
    // R^0 is left out: CORA answers NaN there, CORALean Inf
    test::oracle::run("fullspace.radius", [](test::oracle::Random &) {
        for (const int n : {1, 2, 5})
            test::oracle::check(Fullspace(n).radius(), lean::Fullspace(n).radius());
    });
    test::oracle::run("fullspace.volume", [](test::oracle::Random &) {
        for (const int n : {1, 2, 5})
            test::oracle::check(Fullspace(n).volume(), lean::Fullspace(n).volume());
    });
    test::oracle::run("fullspace.not", [](test::oracle::Random &) {
        for (const int n : {0, 1, 5})
            test::oracle::check(Fullspace(n).complement().dim(),
                                lean::Fullspace(n).complement().dim());
    });
    return test::finish("lean fullspace sets");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
