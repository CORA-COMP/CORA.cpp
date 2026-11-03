// priv_containment_sadraddini_tedrake - zonotope in zonotope, sufficient test, by a linear program
//
// The inner zonotope (c_in, G_in) lies in the outer one if G_in = G_out Gamma and
// c_in - c_out = G_out beta with |[Gamma beta]|_inf (the largest absolute row sum) at most 1
// (Sadraddini and Tedrake, CDC 2019, Thm. 3). The program minimizes that row sum.
//
// Syntax:   double scaling = priv_containment_sadraddini_tedrake(inner, outer);
// Inputs:   inner, outer - zonotopes
// Outputs:  scaling - the least row sum; at most 1 proves that inner is in outer
// See also: priv_containment_vertex_enumeration
//
// CORA: contSet/@zonotope/private/priv_zonotopeContainment_SadraddiniTedrake.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/linprog.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

double priv_containment_sadraddini_tedrake(const HostZonotope &inner, const HostZonotope &outer) {
    // the inner generators and the offset of the centers, as one matrix (n x mi)
    const int64_t n = inner.n, mi = inner.m + 1, mc = outer.m;
    std::vector<double> Gin(std::size_t(n * mi));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j + 1 < mi; ++j) Gin[i * mi + j] = inner.G[i * inner.m + j];
        Gin[i * mi + mi - 1] = inner.c[i] - outer.c[i];
    }

    // variables: Gamma (mc x mi, column by column), Lambda (the same), tau
    const int64_t block = mc * mi;
    LinProg lp;
    lp.n = int(2 * block + 1);
    lp.f.assign(lp.n, 0.0);
    lp.f.back() = 1.0;
    // G_out Gamma = G_in, column by column
    for (int64_t j = 0; j < mi; ++j)
        for (int64_t i = 0; i < n; ++i) {
            std::vector<double> row(lp.n, 0.0);
            for (int64_t k = 0; k < mc; ++k) row[j * mc + k] = outer.G[i * mc + k];
            lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
            lp.beq.push_back(Gin[i * mi + j]);
        }
    // |Gamma| <= Lambda
    for (int64_t v = 0; v < block; ++v)
        for (const double sign : {1.0, -1.0}) {
            std::vector<double> row(lp.n, 0.0);
            row[v] = sign;
            row[block + v] = -1.0;
            lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
            lp.bineq.push_back(0.0);
        }
    // the row sums of Lambda are at most tau
    for (int64_t k = 0; k < mc; ++k) {
        std::vector<double> row(lp.n, 0.0);
        for (int64_t j = 0; j < mi; ++j) row[block + j * mc + k] = 1.0;
        row.back() = -1.0;
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
        lp.bineq.push_back(0.0);
    }
    const LinProgResult r = linprog(lp);
    return r.status == LinProgResult::Status::optimal ? r.fval
                                                      : std::numeric_limits<double>::infinity();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
