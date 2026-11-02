// linear_sys - lean::LinearSys, the reachability of x' = A x computed by CORALean
//
// See also: lean/zonotope.h

#include "cora/global/oracle/linear_sys.h"
#include "cora/global/error.h"

#include "cora/global/oracle/oracle.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// A double as the exact text the oracle reads (times need not be representable in the dtype).
Json aux_number(double x) { return Json(Tensor::encode(x)); }

std::vector<Zonotope> aux_sets(const std::string &dtype, const Json &list) {
    std::vector<Zonotope> sets;
    for (const Json &j : list.items) sets.push_back(Zonotope::from_json(dtype, j));
    return sets;
}

} // namespace


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
    Json request = Json::object();
    request.set("op", "linearSysReach").set("dtype", A_.dtype());
    request.set("A", A_.to_json()).set("X0", X0.to_json());
    // the times are exact doubles, not values of the dtype
    request.set("timeStep", aux_number(time_step)).set("tFinal", aux_number(t_final));
    request.set("taylorTerms", static_cast<int64_t>(taylor_terms));
    request.set("zonotopeOrder", static_cast<int64_t>(zonotope_order));
    const Json response = call(request);
    return {aux_sets(A_.dtype(), response.at("timePoint")),
            aux_sets(A_.dtype(), response.at("timeInterval"))};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
