# Out of scope

CORA methods CORA.cpp does not port, one row each with its reason (`group/class/method`). The
coverage tables count them separately; a method that a source tags cannot be listed.

| Row | Reason |
|---|---|
| `contSet/contSet/initEmptySet` | deprecated in CORA: replaced by contSet.empty |
| `contSet/contSet/isConHyperplane` | deprecated in CORA: replaced by representsa(S,'conHyperplane') |
| `contSet/contSet/isHyperplane` | deprecated in CORA: replaced by representsa(S,'hyperplane') |
| `contSet/contSet/isInterval` | deprecated in CORA: replaced by representsa(S,'interval') |
| `contSet/contSet/isParallelotope` | deprecated in CORA: replaced by representsa(S,'parallelotope') |
| `contSet/contSet/isPolytope` | deprecated in CORA: replaced by representsa(S,'polytope') |
| `contSet/contSet/isZero` | deprecated in CORA: replaced by representsa(S,'origin') |
| `contSet/contSet/isZonotope` | deprecated in CORA: replaced by representsa(S,'zonotope') |
| `contSet/contSet/isempty` | deprecated in CORA: replaced by representsa(S,'emptySet') |
| `contSet/polygon/simplify` | deprecated in CORA: replaced by compact(polygon,'douglasPeucker',tol) |
| `contSet/conZonotope/deleteZeros` | deprecated in CORA: replaced by compact(cZ,'zeros') |
| `contSet/polyZonotope/deleteZeros` | deprecated in CORA: replaced by compact(pZ,'zeros') |
| `contSet/zonotope/deleteZeros` | deprecated in CORA: replaced by compact(Z,'zeros') |
| `contSet/zonotope/deleteAligned` | deprecated in CORA: replaced by compact(Z,'aligned') |
