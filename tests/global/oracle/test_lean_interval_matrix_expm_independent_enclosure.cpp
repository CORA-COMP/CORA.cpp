// test_lean_interval_matrix_expm_independent_enclosure - IntervalMatrix::expm_independent against
//
// the CORALean oracle, as enclosure of the same exponentials
//
// The statement of CORALean (`exp_mem_expIntervalEnclosure`) is that exp(M) lies in the enclosure
// for every matrix M of the set. The oracle's remainder is a norm bound (`expRemainderBound`) and
// CORA's is `exponentialRemainder`; neither box contains the other (the oracle's is smaller on some
// sides), so the check is the statement itself: the exponential of the corners and of random
// matrices of the set lies in the box of the oracle and in CORA.cpp's, and the two boxes meet. Only
// the sound mode runs, as the lean backend has no matrix exponential. No bit-exact test exists: no
// ORACLE tag.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.expmInd", [](test::oracle::Random &rnd) {
        if (test::oracle::mode() == test::oracle::Mode::exact) return;
        const int n = 2, order = 6;
        const Interval box = rnd.box_matrix(n, n);
        const IntervalMatrix A(Interval(box.inf() * 0.2, box.sup() * 0.2));
        const lean::Interval theirs =
            lean::IntervalMatrix{lean::Interval(A.interval())}.expm_independent(order);
        const IntervalMatrix mine = A.expm_independent(order).enclosure;
        // the members: every corner of the set and random matrices between the bounds
        std::vector<Tensor> members;
        for (int corner = 0; corner < (1 << (n * n)); ++corner) {
            std::vector<double> w(n * n);
            for (int i = 0; i < n * n; ++i) w[i] = (corner >> i) & 1;
            members.push_back(
                A.infimum() +
                (A.interval().sup() - A.infimum()).multiply(Tensor::from_data(w, {n, n})));
        }
        for (int k = 0; k < 8; ++k) {
            std::vector<double> w(n * n);
            for (double &x : w) x = rnd.uniform(0, 1);
            members.push_back(
                A.infimum() +
                (A.interval().sup() - A.infimum()).multiply(Tensor::from_data(w, {n, n})));
        }
        for (const Tensor &M : members) {
            const Tensor E = M.expm();
            test::oracle::check_enclosed("intervalMatrix.expmInd: a member", E, theirs);
            // members of the exponential
            test::check(mine.contains(E, "exact", 1e-12),
                        "expm_independent: a member in CORA.cpp's box");
        }
        // two enclosures of the same exponentials meet
        const auto [lo, hi] = theirs.gather();
        const bool meet =
            (lo.less_equal(mine.interval().sup()).multiply(mine.interval().inf().less_equal(hi)))
                .min()
                .item() != 0;
        test::check(meet, "expm_independent: the boxes meet");
    });
    return test::finish("lean interval_matrix expm_independent enclosure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
