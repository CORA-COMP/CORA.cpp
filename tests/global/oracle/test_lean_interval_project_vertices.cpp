// test_lean_interval_project_vertices - Interval::project and Interval::vertices against CORALean
//
// The corners are the same doubles; CORA.cpp lists them counter-clockwise from the lower left, the
// oracle in its own order, so the two lists are compared as sets of corners, each bit for bit.
//
// ORACLE: contSet/@interval/project.m
// ORACLE: contSet/@interval/vertices_.m
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <array>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Corner = std::array<double, 2>;

/// The corners of the oracle's (2, k) matrix, in a canonical order.
std::vector<Corner> aux_corners(const Tensor &V) {
    const std::vector<double> d = V.data();
    const std::size_t k = d.size() / 2;
    std::vector<Corner> corners;
    for (std::size_t i = 0; i < k; ++i) corners.push_back({d[i], d[k + i]});
    std::sort(corners.begin(), corners.end());
    return corners;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.project", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5}) {
            const Interval I = rnd.box(n);
            for (const std::vector<int64_t> &dims : {std::vector<int64_t>{0}, {n - 1, 0}, {1, 0}})
                test::oracle::check(I.project(dims).as<Interval>(),
                                    lean::Interval(I).project(dims));
        }
    });
    // the corners of a planar box
    test::oracle::run("interval.vertices", [](test::oracle::Random &rnd) {
        for (int i = 0; i < 4; ++i) {
            const Interval I = rnd.box(2);
            std::vector<Corner> mine;
            const std::vector<Polygon> polygons = I.vertices();
            for (const auto &p : polygons[0]) mine.push_back({p[0], p[1]});
            std::sort(mine.begin(), mine.end());
            const std::vector<Corner> theirs = aux_corners(lean::Interval(I).vertices().V.gather());
            test::check(mine == theirs, "interval.vertices: the same corners, bit for bit");
        }
    });
    return test::finish("lean interval project vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
