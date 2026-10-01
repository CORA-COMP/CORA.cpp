// linearSys - lean::LinearSys, the reachability of x' = A x computed by CORALean
//
// See also: lean/zonotope.h

#include "global/oracle/linearSys.h"

#include "global/oracle/oracle.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// A double as the exact text the oracle reads (times need not be representable in the dtype).
Json aux_number(double x) {
    return Json(Tensor::encode(x));
}

std::vector<Zonotope> aux_sets(const std::string &dtype, const Json &list) {
    std::vector<Zonotope> sets;
    for (const Json &j : list.items) sets.push_back(Zonotope::fromJson(dtype, j));
    return sets;
}

} // namespace


// ===========================================  MAIN  =========================================== //

LinearSys::LinearSys(const cora::Tensor &A) : A_(Tensor::from(A)) {
    if (A_.rows() != A_.cols())
        throw std::invalid_argument("lean: the dynamics matrix of a linear system must be square");
}

ReachSet LinearSys::reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms,
                          int zonotopeOrder) const {
    if (X0.c.dtype() != A_.dtype())
        throw std::invalid_argument("lean: the system and the initial set must share one dtype");
    const double steps = tFinal / timeStep;
    if (!(timeStep > 0) || std::abs(steps - std::round(steps)) > 1e-9 * steps)
        throw std::invalid_argument("lean: tFinal must be a positive multiple of timeStep");
    Json request = Json::object();
    request.set("op", "linearSysReach").set("dtype", A_.dtype());
    request.set("A", A_.toJson()).set("X0", X0.toJson());
    // the times are exact doubles, not values of the dtype
    request.set("timeStep", aux_number(timeStep)).set("tFinal", aux_number(tFinal));
    request.set("taylorTerms", static_cast<int64_t>(taylorTerms));
    request.set("zonotopeOrder", static_cast<int64_t>(zonotopeOrder));
    const Json response = call(request);
    return {aux_sets(A_.dtype(), response.at("timePoint")),
            aux_sets(A_.dtype(), response.at("timeInterval"))};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
