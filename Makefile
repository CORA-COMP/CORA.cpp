# CORA.cpp — one binary, no build system beyond this file.
#
# Eigen is header-only and GLPK is the only library the CPU path needs; -march=native is
# safe because the tool is built on the worker it runs on. rng.cpp is the one file
# compiled with -ffast-math, so the compiler may vectorize the logarithm and the sine of
# Box–Muller through libmvec; everything that the results depend on keeps strict IEEE
# arithmetic.
#
# `make TORCH=/path/to/libtorch` builds the libtorch backend in as well, which is what
# reaches the GPU and carries autograd; without it torch_none.cpp stands in and gpu
# instances report unsupported. TORCH_ABI must be the _GLIBCXX_USE_CXX11_ABI that
# libtorch itself was built with — install_tool.sh reads it off the install.

CXX       ?= g++
EIGEN     ?= /usr/include/eigen3
BLAS      ?=
TORCH     ?=
TORCH_ABI ?= 1
WARN       = -Wall -Wextra
# The Python module is a shared object, so every object it links must be relocatable.
PIC        = -fPIC
PYTHON    ?= python3
CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
INCLUDES   = -Isrc -Icompetition -isystem $(EIGEN)
LDLIBS     = -lglpk
LDFLAGS    = -fopenmp

# The library in src/ mirrors MATLAB CORA's folders, one operation per file; competition/ is the
# harness. Sources are found, so a new operation needs no edit here. The libtorch files and the
# programs with their own main are added below.
LIB_SRC  = $(filter-out src/python/% src/tensor/torch.cpp,$(shell find src -name '*.cpp' | sort))
COMP_SRC = $(filter-out competition/main.cpp competition/torch_backend.cpp \
                        competition/torch_none.cpp competition/sets/torch_contains.cpp, \
                        $(shell find competition -name '*.cpp' | sort))
SRC      = $(LIB_SRC) $(COMP_SRC)

ifeq ($(TORCH),)
SRC += competition/torch_none.cpp
else
SRC      += competition/torch_backend.cpp competition/sets/torch_contains.cpp \
            src/tensor/torch.cpp
DEFS      = -DCORACPP_TORCH -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES += -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LDFLAGS  += -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib
LDLIBS   += -ltorch -ltorch_cpu -lc10
# --no-as-needed: nothing references libtorch_cuda directly, but without it linked the
# CUDA kernels are not registered and every gpu instance would report unsupported.
ifneq ($(wildcard $(TORCH)/lib/libtorch_cuda.so),)
LDLIBS += -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed
endif
endif

# Eigen's own GEMM reaches about a fifth of this machine's peak; an external BLAS is
# what the catalog's largest matMul needs to finish at all. Opt-in, since it changes
# which library does the arithmetic.
ifneq ($(BLAS),)
DEFS   += -DEIGEN_USE_BLAS
LDLIBS += -lopenblas
endif

OBJ = $(SRC:%.cpp=build/%.o)

all: build/coracpp

build/coracpp: $(OBJ) build/competition/main.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/src/global/rng.o: src/global/rng.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

build/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -c -o $@ $<

# Tests sit in tests/ in the folders of what they test; those ending in _torch need libtorch.
TEST_SRC = $(shell find tests -name 'test_*.cpp' | sort)
TESTS    = $(patsubst %.cpp,build/%,$(filter-out %_torch.cpp,$(TEST_SRC)))
ifneq ($(TORCH),)
TESTS += $(patsubst %.cpp,build/%,$(filter %_torch.cpp,$(TEST_SRC)))
endif

build/tests/%: tests/%.cpp $(OBJ) tests/testing.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -Itests $(LDFLAGS) -o $@ $(filter %.cpp %.o,$^) $(LDLIBS)

# Examples: one small file per topic in examples/cpp; `make example` builds and runs them all.
# The ones that name libtorch types need it.
EXAMPLE_SRC = $(sort $(wildcard examples/cpp/example_*.cpp))
EXAMPLES    = $(patsubst %.cpp,build/%,$(filter-out %_torch.cpp %_gpu.cpp %_batch.cpp %_gradient.cpp,$(EXAMPLE_SRC)))
ifneq ($(TORCH),)
EXAMPLES = $(patsubst %.cpp,build/%,$(EXAMPLE_SRC))
endif

build/examples/cpp/%: examples/cpp/%.cpp $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

example: $(EXAMPLES)
	@for e in $(EXAMPLES); do echo "== $$e"; ./$$e || exit 1; done

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# `make python TORCH=...` builds the `coracpp` package into build/coracpp (the compiled module
# and the Python files beside it); put build/ on PYTHONPATH. It needs libtorch, since torch
# tensors are its inputs.
PYEXT = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYINC = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")

PYFILES = $(wildcard src/python/coracpp/*.py)

python: build/coracpp/_coracpp$(PYEXT) $(PYFILES:src/python/coracpp/%=build/coracpp/%)

build/coracpp/%.py: src/python/coracpp/%.py
	@mkdir -p $(@D)
	cp $< $@

build/coracpp/_coracpp$(PYEXT): src/python/bindings.cpp $(OBJ)
	@test -n "$(TORCH)" || { echo "make python needs TORCH=/path/to/libtorch"; exit 1; }
	@mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -I$(PYINC) $(LDFLAGS) \
	    -o $@ $^ $(LDLIBS) -L$(TORCH)/lib -ltorch_python

clean:
	rm -rf build

.PHONY: all test clean python example
