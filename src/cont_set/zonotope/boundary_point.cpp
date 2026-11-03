// boundary_point - where a ray from a point of a zonotope leaves it, as CORA's boundaryPoint
//
// Syntax:   x = Z.boundary_point(dir);   x = Z.boundary_point(dir, start);
// Inputs:   dir - direction, a nonzero column (n, 1)
//           start - a point of Z the ray starts at (default: the center)
// Outputs:  x - start + t dir for the largest t that stays in Z: from the zonotope norm of dir for
//           the center, from a linear program for another start, directly for n = 1. Not a batch
// See also: contains, support_function
//
// CORA: contSet/@zonotope/boundaryPoint.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/linprog.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The largest t with start + t dir in the zonotope: max t s.t. c + G beta = start + t dir,
/// |beta|_inf <= 1, t >= 0 (a linear program in (t, beta)).
double aux_exit_step(const HostZonotope &Z, const std::vector<double> &dir,
                     const std::vector<double> &start) {
    LinProg lp;
    lp.n = int(1 + Z.m);
    lp.f.assign(lp.n, 0.0);
    lp.f[0] = -1.0;
    lp.Aeq.assign(std::size_t(Z.n * lp.n), 0.0);
    for (int64_t i = 0; i < Z.n; ++i) {
        lp.Aeq[i * lp.n] = dir[i];
        for (int64_t j = 0; j < Z.m; ++j) lp.Aeq[i * lp.n + 1 + j] = -Z.G[i * Z.m + j];
        lp.beq.push_back(Z.c[i] - start[i]);
    }
    lp.lb.assign(lp.n, -1.0);
    lp.ub.assign(lp.n, 1.0);
    lp.lb[0] = 0.0;
    lp.ub[0] = std::numeric_limits<double>::infinity();
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal)
        CORA_FAIL(Error::solver_issue, "boundary_point: the linear program has no solution");
    return -r.fval;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::boundary_point(const Tensor &dir) const { return boundary_point(dir, c_); }

Tensor Zonotope::boundary_point(const Tensor &dir, const Tensor &start) const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("boundary_point: a linear program per zonotope; it supports a single "
                           "zonotope, not a batch");
    });
    CORA_CHECK(is_column(dir, dim()) && dir.ndim() == 2, Error::dimension_mismatch, [&] {
        return "boundary_point: dir must be a column of the dimension " + std::to_string(dim()) +
               " of the zonotope, got the shape " + shape_text(dir.shape());
    });
    CORA_CHECK(is_column(start, dim()) && start.ndim() == 2, Error::dimension_mismatch, [&] {
        return "boundary_point: start must be a column of the dimension " + std::to_string(dim()) +
               " of the zonotope, got the shape " + shape_text(start.shape());
    });
    // a zero direction has no boundary point, a start outside no ray
    const std::vector<double> d = dir.data(), s = start.data();
    bool zero = true;
    for (const double x : d) zero = zero && within_tol(x, 0.0);
    CORA_CHECK(!zero, Error::wrong_value,
               [] { return std::string("boundary_point: dir has to be a nonzero vector"); });
    CORA_CHECK(contains(start, "exact"), Error::wrong_value, [] {
        return std::string("boundary_point: start has to be a point of the zonotope");
    });

    // in one dimension the zonotope is the segment c +- sum |G|
    if (dim() == 1) return d[0] < 0 ? c_ - G_.abs_sum() : c_ + G_.abs_sum();
    const HostZonotope Z = priv_host_zonotope(*this);
    bool at_center = true;
    for (int64_t i = 0; i < Z.n; ++i) at_center = at_center && within_tol(Z.c[i], s[i]);
    if (at_center) { // the zonotope norm of dir is the factor 1 / t
        const HostZonotope centered{Z.n, Z.m, std::vector<double>(Z.n, 0.0), Z.G};
        return c_ + dir / priv_zonotope_norm(centered, d);
    }
    // from another start the program gives the step
    return start + dir * aux_exit_step(Z, d, s);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
