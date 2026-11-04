// test_lean_fullspace_predicates - the predicates and the center of Fullspace against the oracle
//
// ORACLE: contSet/@fullspace/center.m
// ORACLE: contSet/@fullspace/contains_.m
// ORACLE: contSet/@fullspace/isBounded.m
// ORACLE: contSet/@fullspace/isFullDim.m
// ORACLE: contSet/@fullspace/isIntersecting_.m
// ORACLE: contSet/@fullspace/isequal.m
// ORACLE: contSet/@fullspace/representsa_.m
// ORACLE: contSet/@fullspace/vertices_.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/fullspace.h"
#include "oracle_testing.h"

#include <set>

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
    // the oracle refuses R^0; it proves the whole space is the whole space and no other type
    test::oracle::run("fullspace.representsa", [](test::oracle::Random &) {
        for (const int n : {1, 2, 5})
            for (const char *type : {"emptySet", "point", "origin", "fullspace"})
                test::oracle::check(Fullspace(n).representsa(type, 0.0),
                                    lean::Fullspace(n).representsa(type));
    });
    // the plane has four corners (+-Inf, +-Inf); the oracle lists them as the columns of V, CORA.cpp
    // as a polygon, so they are compared as a set
    test::oracle::run("fullspace.vertices", [](test::oracle::Random &) {
        const std::vector<double> V = lean::Fullspace(2).vertices().gather().data();
        std::set<std::pair<double, double>> theirs, mine;
        for (std::size_t j = 0; j < V.size() / 2; ++j) theirs.insert({V[j], V[V.size() / 2 + j]});
        const std::vector<Polygon> corners = Fullspace(2).vertices();
        for (const Point &p : corners[0]) mine.insert({p[0], p[1]});
        test::oracle::check(mine == theirs && theirs.size() == 4, true);
    });
    return test::finish("lean fullspace predicates");
}
