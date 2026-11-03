# Differences from CORA

Where CORA.cpp deliberately differs from CORA's MATLAB code, because CORA's result is wrong or unsound.

| CORA file | What CORA does | What CORA.cpp does | Minimal input |
|---|---|---|---|
| `contSet/@interval/atan2.m` | Returns the hull of `atan2` at the four corners, which misses +-pi across the branch cut | Follows CORALean's guarded definition: the corner hull if `0 < inf x`, `0 < inf y` or `sup y < 0`, else [-pi, pi] rounded outward | `y = [-1, 1]`, `x = [-2, -1]`: CORA gives [-2.678, 2.678] |
| `contSet/@interval/boundaryPoint.m` | For a box unbounded on one side, `Inf*0 = NaN` enters the bound, and `x(inf_dirs) = Inf*sign_dir(d_inf)` uses the wrong mask, so it returns +-Inf | Returns the exit point of the ray; +-Inf only where the ray never leaves (coordinates with `dir = 0` keep `start`) | `I = [0, Inf]`, `dir = -1`, `start = 1`: CORA gives -Inf, the exit point is 0 |
| `contSet/@interval/acosh.m` | The out-of-domain error says the valid domain is '>= -1 && <= 1' | The domain is `>= 1`, as the code checks; the message says so | `acosh(interval(0.5, 2))` |
