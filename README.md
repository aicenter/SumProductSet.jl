[![Run tests](https://github.com/aicenter/SumProductSet.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aicenter/SumProductSet.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# SumProductSet.jl

is a package for probabilistic learning of tree-structured, heterogeneous graph data based on sum-product networks. The package extends the standard sum-product networks by introducing a novel computational node---the set node---and creating model-builders for the networks. The implementation follows the methodology of the [SumProductTransform.jl](https://github.com/pevnak/SumProductTransform.jl) package. The package [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) is utilized to process tree-structured graphs embodied by the JSON format.

To reproduce this project, do the following:

1. Download this repository.
2. Open the Julia (preferably Julia 1.9) console and type:
   ```julia
   using Pkg
   Pkg.activate("path/to/the/project")
   Pkg.instantiate()
   ```
   These commands will download and install all necessary packages defined in the project environment.
3. Use the package:
   ``` julia
   using SumProductSet
   
   ```
   
   Basic examples are in the [examples](https://github.com/aicenter/SumProductSet.jl/tree/dev/examples) folder. Examples requiring libraries that are
   not included in SumProductSet.jl package have a separate environment.
   
   ## Note
   Adopting `import` Mill.jl or Distribution.jl is preferred to `using`, since it does not create ambiguity between these packages and the SumProductSet.jl
   package.
