// test_lean_zono_bundle_enclose - ZonoBundle::enclose against the CORALean oracle, bit for bit
//
// The oracle's `zonoBundle.encloseCora` pairs the members up as CORA does: member i of the result
// is the enclosure (Zonotope.linComb) of member i of each bundle. `zonoBundle.enclose` is the
// generic route over every pair of members and is not compared. As for zonotope.enclose, CORA lists
// the generators (G+G2)/2, (c-c2)/2, (G-G2)/2 and CORALean (G+G2)/2, (G-G2)/2, (c-c2)/2: the same
// center and the same generators bit for bit, in another order.
//
// ORACLE: contSet/@zonoBundle/enclose.m
// Authors:       Claude Sonnet 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/zono_bundle.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

#include <algorithm>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators of Z, one vector per generator, sorted.
std::vector<std::vector<double>> aux_sorted_columns(const Zonotope &Z) {
    const int64_t n = Z.dim(), m = Z.nr_generators();
    const std::vector<double> data = Z.generators().matrix_transpose().data();
    std::vector<std::vector<double>> columns(m, std::vector<double>(n));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) columns[j][i] = data[i * m + j];
    std::sort(columns.begin(), columns.end());
    return columns;
}

/// Member by member: the same center and the same generators, in another order.
void aux_check(const ZonoBundle &mine, const lean::ZonoBundle &theirs) {
    const ZonoBundle other = theirs.nominal();
    test::check(mine.parallel_sets() == other.parallel_sets(), "encloseCora: member count");
    for (std::size_t i = 0; i < mine.zonotopes().size(); ++i) {
        const Zonotope &a = mine.zonotopes()[i], &b = other.zonotopes()[i];
        test::check(a.nr_generators() == b.nr_generators(), "encloseCora: generator count");
        test::check(oracle::agree("encloseCora.c", a.center().expand_dims(-1),
                                  b.center().expand_dims(-1)),
                    "encloseCora: the same center");
        test::check(aux_sorted_columns(a) == aux_sorted_columns(b),
                    "encloseCora: the same generators, in another order");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonoBundle.encloseCora", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3})
            for (const int64_t k : {1, 2, 3}) {
                const ZonoBundle a = rnd.zono_bundle(n, k, 3), b = rnd.zono_bundle(n, k, 2);
                aux_check(a.enclose(b), lean::ZonoBundle(a).enclose_cora(lean::ZonoBundle(b)));
            }
    });
    return test::finish("lean zono_bundle enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
