// test_lean_mat_zonotope_expm_enclosure - MatZonotope::expm and expm_independent against the
// CORALean oracle, as enclosure of the same exponentials
//
// The oracle has one op for both (`matZonotope.expm` and `expmInd`): the Taylor series to a maximum
// order and a remainder box from a norm bound, as for the interval matrix; its box and CORA.cpp's
// do not contain one another. So the check is that the exponential of the vertices of the set and
// of random matrices C + sum b_i G_i lies in the oracle's box and in CORA.cpp's, and the two boxes
// meet. Only the sound mode runs, as the lean backend has no matrix exponential. There is no ORACLE
// tag.

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    for (const char *op : {"matZonotope.expm", "matZonotope.expmInd"})
        test::oracle::run(op, [op](test::oracle::Random &rnd) {
            if (test::oracle::mode() == test::oracle::Mode::exact) return;
            const bool full = std::string(op) == "matZonotope.expm";
            const int n = 2, h = 2, order = 6;
            const MatZonotope B = rnd.mat_zonotope(n, n, h);
            const MatZonotope A(B.center() * 0.2, B.generators() * 0.2);
            const MatZonotope mine = full ? A.expm(order) : A.expm_independent(order);
            const lean::MatZonotope theirs = full ? lean::MatZonotope(A).expm(order)
                                                  : lean::MatZonotope(A).expm_independent(order);
            // the members: the vertices (b in {-1, 1}^h) and random coefficients
            std::vector<std::vector<double>> coefficients;
            for (int vertex = 0; vertex < (1 << h); ++vertex) {
                std::vector<double> b(h);
                for (int i = 0; i < h; ++i) b[i] = ((vertex >> i) & 1) ? 1.0 : -1.0;
                coefficients.push_back(b);
            }
            for (int k = 0; k < 6; ++k) {
                std::vector<double> b(h);
                for (double &x : b) x = rnd.uniform(-1, 1);
                coefficients.push_back(b);
            }
            for (const std::vector<double> &b : coefficients) {
                const Tensor weights = Tensor::from_data(b, {h, 1, 1});
                const Tensor E = (A.center() + A.generators().multiply(weights).sum(-3)).expm();
                test::oracle::check_enclosed(op, E, theirs.interval_matrix());
                test::check(mine.interval_matrix().contains(E, "exact", 1e-12),
                            std::string(op) + ": a member in CORA.cpp's box");
            }
            const IntervalMatrix a = mine.interval_matrix();
            const auto [lo, hi] = theirs.interval_matrix().gather();
            const bool meet =
                (lo.less_equal(a.interval().sup()).multiply(a.interval().inf().less_equal(hi)))
                    .min()
                    .item() != 0;
            test::check(meet, std::string(op) + ": the boxes meet");
        });
    return test::finish("lean mat_zonotope expm enclosure");
}
