// test_lean_con_zonotope_interval_cert - ConZonotope::interval_cert against the CORALean oracle
//
// The box of the weak-duality bounds at given multipliers is CORALean's
// `ConZonotope.intervalCertWith` bit for bit: at the multipliers of CORA.cpp's linear programs
// (sent as `y`, for a constrained zonotope and for a cut with inequality rows), and at the
// multipliers the oracle's own exact simplex returns. A set without points has no counterpart in
// the oracle and is not compared.
//
// CORA.cpp only: tests interval_cert, CORALean's float checker of dual solutions
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether two multipliers agree in every bit.
bool aux_same(const ConZonotope::CertWeights &a, const ConZonotope::CertWeights &b) {
    const auto same = [](const std::vector<double> &x, const std::vector<double> &y) {
        return x.size() == y.size() &&
               (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0);
    };
    return same(a.y, b.y) && same(a.zu, b.zu) && same(a.zl, b.zl);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("conZonotope.intervalCert", [](test::oracle::Random &rnd) {
        // a constrained zonotope at the multipliers of CORA.cpp's programs, which are sent
        const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
        const std::optional<ConZonotope::Certificate> w =
            ConZonotope::Cut{cz.c(), cz.generators(),       cz.A(),
                             cz.b(), Tensor::zeros({0, 5}), Tensor::zeros({0, 1})}
                .certificate();
        test::check(w.has_value(), "intervalCert: the programs have a solution");
        if (!w) return;
        const lean::ConZonotope::Cert theirs = lean::ConZonotope(cz).interval_cert(*w);
        test::oracle::check(cz.interval_cert(*w), theirs.I);
        test::oracle::check(cz.interval_cert(), theirs.I);
        // the oracle echoes the multipliers it was sent
        bool echoed = theirs.weights.up.size() == w->up.size();
        for (std::size_t i = 0; echoed && i < w->up.size(); ++i)
            echoed = aux_same(theirs.weights.up[i], w->up[i]) &&
                     aux_same(theirs.weights.lo[i], w->lo[i]);
        test::check(echoed, "intervalCert: the multipliers sent are the ones used");
        // arbitrary multipliers, including negative ones, are clamped alike
        ConZonotope::Certificate scrambled = *w;
        for (auto &end : scrambled.up) {
            end.zu[0] = -2.5;
            end.zl[1] = 7.25;
        }
        test::oracle::check(cz.interval_cert(scrambled),
                            lean::ConZonotope(cz).interval_cert(scrambled).I);
    });
    // a cut with inequality rows, at the multipliers of CORA.cpp's programs
    test::oracle::run("conZonotope.intervalCert", [](test::oracle::Random &rnd) {
        const ConZonotope cz = rnd.con_zonotope(2, 4, 1);
        const Polytope guard(Tensor::from_data({1.0, 0.5, -0.3, 1.0}, {2, 2}),
                             cz.c() + Tensor::from_data({0.2, 0.1}, {2, 1}));
        const ConZonotope::Cut cut = cz.cut(guard);
        const std::optional<ConZonotope::Certificate> w = cut.certificate();
        test::check(w.has_value(), "intervalCert: the cut has a solution");
        if (!w) return;
        test::oracle::check(cut.interval_cert(*w), lean::ConZonotope(cut).interval_cert(*w).I);
    });
    // the multipliers of the oracle's own simplex, read back and applied by CORA.cpp
    test::oracle::run("conZonotope.intervalCert", [](test::oracle::Random &rnd) {
        const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
        const lean::ConZonotope::Cert theirs = lean::ConZonotope(cz).interval_cert();
        test::oracle::check(cz.interval_cert(theirs.weights), theirs.I);
    });
    return test::finish("lean con zonotope interval cert");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
