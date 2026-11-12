// test_lean_zonotope_quad_map - Zonotope::quad_map against CORALean's Zonotope.quadMapTri, bit for
// bit
//
// CORA's quadMap of one zonotope (the center kept, 1 + N + N(N+1)/2 generators per output) is the
// float definition Zonotope.quadMapTri (MR !921): the entries of M = [c G]' Q [c G] formed as
// ([c G]' Q) [c G] with every dot product summed from its last entry, the center M_00 plus the sum
// of the halved squares, the squares first and then the strict lower triangle of M + M' down its
// columns. The caller's nonzeroFilter and the box of a single nonzero output are exact operations
// on it: a random Q with at least two outputs has none to filter, one output is checked at its
// center and its box. The inputs are random zonotopes and the Hessians and zonotope of the
// linearization of a step of the traffic scenario (TRAF22) and of Robertson (ROBE25_1) dumped from
// CORA.cpp.
//
// CORA: contSet/@zonotope/quadMap.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Z.quad_map(Q) against quadMapTri: CORA's nonzeroFilter (no zero generator) on the oracle's
/// zonotope for two or more nonzero outputs, else the box of its generators (its center kept).
void aux_compare(const std::string &what, const Zonotope &Z, const std::vector<Tensor> &Q) {
    const std::vector<lean::Tensor> theirs_Q(Q.begin(), Q.end());
    const lean::Zonotope theirs = lean::Zonotope(Z).quad_map_tri(theirs_Q);
    int nonzero = 0;
    for (const Tensor &M : Q) nonzero += M.abs().sum().item() != 0 ? 1 : 0;
    const Zonotope mine = Z.quad_map(Q);
    if (nonzero >= 2) {
        oracle::check(what, mine, theirs.compact("zeros", 0.0));
        return;
    }
    const Zonotope boxed(theirs.center().gather(), theirs.generators().gather().abs_sum());
    test::check(oracle::agree(what + ".c", mine.center(), theirs.center()), what + ": the center");
    test::check(oracle::agree(what + ".box", mine.generators(), boxed.generators()),
                what + ": the box");
}

/// The recorded calls of the file: the zonotope and the matrices after each line "Z n N k".
void aux_recorded(const std::string &name) {
    const std::filesystem::path file =
        std::filesystem::path(__FILE__).parent_path() / "data" / "quad_map_steps.txt";
    std::ifstream in(file);
    test::check(bool(in), "cannot read " + file.string());
    // the sections are named by the comment lines "# TRAF22"
    std::string line, section;
    const auto read = [&](int64_t count) {
        std::vector<double> v(static_cast<size_t>(count));
        for (double &x : v) in >> x;
        return v;
    };
    int index = 0;
    while (std::getline(in, line)) {
        if (line.rfind("# ", 0) == 0) section = line.substr(2);
        if (line.rfind("Z ", 0) != 0 || section != name) continue;
        int64_t n = 0, N = 0, k = 0;
        char tag = 0;
        std::istringstream(line) >> tag >> n >> N >> k;
        const std::vector<double> c = read(n), G = read(n * N);
        const Zonotope Z(Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, N}));
        std::vector<Tensor> Q;
        Q.reserve(size_t(k));
        for (int64_t i = 0; i < k; ++i) Q.push_back(Tensor::from_data(read(n * n), {n, n}));
        in >> std::ws;
        aux_compare(name + " call " + std::to_string(index++), Z, Q);
    }
    test::check(index > 0, "no recorded call of " + name);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.quadMapTri", [](oracle::Random &rnd) {
        for (const auto &[n, m, k] : std::vector<std::tuple<int, int, int>>{{1, 1, 2},
                                                                            {2, 3, 2},
                                                                            {3, 5, 3},
                                                                            {4, 6, 2},
                                                                            {6, 4, 6},
                                                                            {5, 9, 4},
                                                                            {3, 0, 2},
                                                                            {3, 4, 1},
                                                                            {2, 2, 1}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            std::vector<Tensor> Q;
            Q.reserve(std::size_t(k));
            for (int i = 0; i < k; ++i) Q.push_back(rnd.matrix(n, n));
            aux_compare("n = " + std::to_string(n) + ", m = " + std::to_string(m) +
                            ", k = " + std::to_string(k),
                        Z, Q);
        }
        // a zero output among the others
        const Zonotope Z = rnd.zonotope(3, 4);
        aux_compare("a zero output", Z,
                    {rnd.matrix(3, 3), Tensor::zeros({3, 3}), rnd.matrix(3, 3)});
        // a step of the traffic scenario (TRAF22) and of Robertson (ROBE25_1)
        aux_recorded("TRAF22");
        aux_recorded("ROBE25_1");
    });
    return test::finish("lean zonotope quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
