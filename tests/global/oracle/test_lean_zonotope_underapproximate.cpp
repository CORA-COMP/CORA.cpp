// test_lean_zonotope_underapproximate - Zonotope::underapproximate against the CORALean oracle
//
// The oracle gives, per direction, the vertices c +- sum_j sign(s . g_j) g_j, and nothing for a
// vertex it cannot certify (a sum that rounds): those are skipped. CORA.cpp's columns are the plus
// and minus vertex of each direction in turn. A vertex of the oracle is a sum without rounding, so
// it is CORA.cpp's in both modes. The center and the generators are multiples of 1/8, so that the
// sums are exact and the oracle certifies them.
//
// ORACLE: contSet/@zonotope/underapproximate.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// The numbers rounded to multiples of 1/8 (of magnitude below 2^10, so that sums of them are
/// exact).
Tensor aux_eighths(const Tensor &x) { return (x * 8.0).round() * 0.125; }

} // namespace

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.underapproximate", [](oracle::Random &rnd) {
        int compared = 0;
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 5}, {5, 12}}) {
            const Zonotope Z(aux_eighths(rnd.vector(n)), aux_eighths(rnd.matrix(n, m)));
            const int directions = 4;
            std::vector<Tensor> S;
            std::vector<lean::Tensor> theirs_S;
            for (int k = 0; k < directions; ++k) {
                S.push_back(rnd.direction(n));
                theirs_S.push_back(lean::Tensor::from(S.back()));
            }
            const Tensor V = Z.underapproximate(Tensor::concat(S, -1));
            const std::vector<lean::Zonotope::VertexPair> pairs =
                lean::Zonotope(Z).underapproximate(theirs_S);
            test::check(pairs.size() == S.size(), "underapproximate: a pair per direction");
            for (int k = 0; k < directions; ++k) {
                const std::array<std::optional<lean::Tensor>, 2> vertices{pairs[k].plus,
                                                                          pairs[k].minus};
                for (int side = 0; side < 2; ++side)
                    if (vertices[side] && ++compared)
                        oracle::check_exact(oracle::op_name(),
                                            V.slice(-1, 2 * k + side, 2 * k + side + 1),
                                            *vertices[side]);
            }
        }
        test::check(compared > 0, "underapproximate: the oracle certifies some vertices");
    });
    return test::finish("lean zonotope underapproximate");
}
