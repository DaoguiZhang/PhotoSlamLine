# amax/amin vs max(dim)/min(dim) gradient semantics

Standalone LibTorch test. Build and run:

```bash
mkdir -p /tmp/amaxmin && cp tests/amax_amin_grad_semantics.cpp /tmp/amaxmin/t.cpp
cd /tmp/amaxmin
cat > CMakeLists.txt <<'CM'
cmake_minimum_required(VERSION 3.18)
project(amaxmin CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
add_executable(t t.cpp)
target_link_libraries(t "${TORCH_LIBRARIES}")
CM
cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)") . && make && ./t
```

Result on LibTorch 2.12.0 / CUDA, threshold 1.0 (term active):

| case | loss | gradient wrt `scaling_` |
|---|---|---|
| unique max / unique min | identical | **identical** |
| tied max (two axes equal the max) | identical | **DIFFERS, rel 0.025** |
| tied min (two axes equal the min) | identical | **DIFFERS, rel 0.50** |
| **actual initialisation pattern** (`scale_axis1 == scale_axis2` exactly, 0.08000002056) | identical | **DIFFERS, rel 0.50** |
| realistic random scales, no ties | identical | identical |

`max(dim).values` routes the gradient to the single argmax/argmin; `amax`/`amin` spread it
over every tied extremum. The values are identical, the gradients are not.

The repository's initialisation produces an **exact tie** between the two perpendicular
axes, so the substitution changed training semantics from the first iteration. It was
therefore reverted; the rest of the Stage-1 work is unaffected.
