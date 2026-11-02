// linear_sys - lean::LinearSys, the reachability of x' = A x computed by CORALean
//
// See also: lean/zonotope.h

#include "cora/global/oracle/linear_sys.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {


// ===========================================  MAIN  =========================================== //

LinearSys::LinearSys(const cora::Tensor &A) : A_(Tensor::from(A)) {
    detail::Entry entry;
    CORA_CHECK(A_.rows() == A_.cols(), Error::wrong_input_in_constructor,
               [] { return std::string("lean::LinearSys: the dynamics matrix must be square"); });
}

ReachSet LinearSys::reach(const Zonotope &X0, double t_final, const alg::Standard &alg) const {
    detail::Entry entry;
    if (entry.check()) alg.validate();
    const double time_step = alg.time_step;
    const int taylor_terms = alg.taylor_terms, zonotope_order = alg.zonotope_order;
    CORA_CHECK(X0.center().dtype() == A_.dtype(), Error::wrong_value, [] {
        return std::string("lean reach: the system and the initial set must share one dtype");
    });
    const double steps = t_final / time_step;
    CORA_CHECK(
        t_final > 0 && std::abs(steps - std::round(steps)) <= 1e-9 * steps, Error::wrong_value,
        [] { return std::string("lean reach: t_final must be a positive multiple of time_step"); });
    // the times are exact doubles, not values of the dtype
    const Reply r = ask("linearSysReach", {{"A", A_}, {"X0", X0}, {"timeStep", time_step},
                                           {"tFinal", t_final}, {"taylorTerms", taylor_terms},
                                           {"zonotopeOrder", zonotope_order}});
    return {r.get<std::vector<Zonotope>>("timePoint"),
            r.get<std::vector<Zonotope>>("timeInterval")};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
