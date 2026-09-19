#include "lp.h"

#include <glpk.h>

#include <mutex>
#include <stdexcept>

namespace cora {
namespace {
// GLPK keeps process-wide state, and the containment queries come from several threads.
std::mutex glpk_lock;
} // namespace

bool contains_lp(const Eigen::Ref<const Eigen::MatrixXd> &g,
                 const Eigen::Ref<const Eigen::VectorXd> &r, double tol) {
    const int n = static_cast<int>(g.rows()), m = static_cast<int>(g.cols());
    std::lock_guard<std::mutex> guard(glpk_lock);

    glp_prob *lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);

    // Columns 1..m are the coefficients b, column m+1 is the bound t, the only cost.
    glp_add_cols(lp, m + 1);
    for (int j = 1; j <= m; ++j) {
        glp_set_col_bnds(lp, j, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, j, 0.0);
    }
    glp_set_col_bnds(lp, m + 1, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, m + 1, 1.0);

    // Rows 1..n are G b = r; then b_j - t <= 0 and -b_j - t <= 0 for each j.
    glp_add_rows(lp, n + 2 * m);
    std::vector<int> index(m + 2);
    std::vector<double> value(m + 2);
    for (int i = 1; i <= n; ++i) {
        glp_set_row_bnds(lp, i, GLP_FX, r(i - 1), r(i - 1));
        for (int j = 1; j <= m; ++j) {
            index[j] = j;
            value[j] = g(i - 1, j - 1);
        }
        glp_set_mat_row(lp, i, m, index.data(), value.data());
    }
    for (int j = 1; j <= m; ++j) {
        for (int side = 0; side < 2; ++side) {
            const int row = n + 2 * (j - 1) + side + 1;
            glp_set_row_bnds(lp, row, GLP_UP, 0.0, 0.0);
            const int cols[3] = {0, j, m + 1};
            const double coef[3] = {0.0, side == 0 ? 1.0 : -1.0, -1.0};
            glp_set_mat_row(lp, row, 2, cols, coef);
        }
    }

    glp_smcp options;
    glp_init_smcp(&options);
    options.msg_lev = GLP_MSG_OFF;
    options.presolve = GLP_ON;
    const int status = glp_simplex(lp, &options);

    bool inside = false;
    if (status == 0 && glp_get_status(lp) == GLP_OPT) {
        inside = glp_get_obj_val(lp) <= 1.0 + tol;
    } else if (glp_get_status(lp) == GLP_NOFEAS || status == GLP_ENOPFS) {
        // No b reaches the point at all, whatever its norm.
        inside = false;
    } else {
        glp_delete_prob(lp);
        throw std::runtime_error("zonotope containment LP failed");
    }
    glp_delete_prob(lp);
    return inside;
}

} // namespace cora
