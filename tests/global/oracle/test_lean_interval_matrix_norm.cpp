// test_lean_interval_matrix_norm - IntervalMatrix::norm against the CORALean oracle
//
// The norms one and inf are sums and maxima of the same entries: bit for bit on the lean backend.
// The Frobenius norm is a square root, which the oracle bounds from above by Newton steps: CORA.cpp's
// correctly rounded root is inside the bound and within kRelative of it, but not its bits (one case
// in four differs), so the row is a property, not ORACLE.
//
// PROPERTY: matrixSet/@intervalMatrix/norm.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "cora/global/rng.h"
#include "oracle_testing.h"

#include <limits>

using namespace cora;

namespace {

constexpr double kRelative = 4 * std::numeric_limits<double>::epsilon();

}

int main() {
    if (!test::oracle::available()) return 0;
    cora::Rng rng(4);
    test::oracle::run("intervalMatrix.norm", [&](test::oracle::Random &rnd) {
        for (const auto [m, n] : {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            test::oracle::check(A.norm(Tensor::Norm::one), L.norm("1"));
            test::oracle::check(A.norm(Tensor::Norm::inf), L.norm("inf"));
            const double root = A.norm(Tensor::Norm::fro).data()[0], bound = L.norm("fro");
            test::oracle::check_enclosed("intervalMatrix.norm fro", root, bound);
            test::check(root >= bound * (1 - kRelative), "intervalMatrix.norm fro: close to the bound");
            // the oracle's bound is above the norm of every matrix of the set
            for (int trial = 0; trial < 10; ++trial) {
                std::vector<double> u(m * n);
                rng.uniform(u.data(), u.size(), 0.0, 1.0);
                const Tensor w = Tensor::from_data(u, {m, n});
                const Tensor M = A.infimum() + (A.interval().sup() - A.infimum()).multiply(w);
                const Tensor member_norm = M.matrix_norm(Tensor::Norm::fro);
                test::oracle::check_enclosed("intervalMatrix.norm fro: a member", member_norm, bound);
            }
        }
    });
    return test::finish("lean interval_matrix norm");
}
