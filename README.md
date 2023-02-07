[![Run tests](https://github.com/aicenter/SumProductSet.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aicenter/SumProductSet.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# SumProductSet.jl

is package for probailistic learning of hierarchicaly structured heterogeneous data based on SumProduct networks. The package extends standard SumProduct networks by introducing new type of model node and implementing rules for building models over aforementioned data. It is based on [SumProductTransform.jl](https://github.com/pevnak/SumProductTransform.jl) package. The package [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) is used as a framework that unifies accepted data format.



To reproduce this project, do the following:

1. Download this code base repository.
2. Open a Julia (preferably Julia 1.8) console and type:
   ```julia
   using Pkg
   Pkg.activate("path/to/the/project")
   Pkg.instantiate()
   ```
   This will download and install all necessary packages defined in project environment for you.
3. Use the package:
   ``` julia
   using SumProductSet
   
   ```
   
   Basic examples are shown in [examples](https://github.com/aicenter/SumProductSet.jl/tree/dev/examples) folder. Examples that require externel libraries
   not included in SumProductSet.jl package have separate environment.
   
   ## Note
   Importing Mill.jl or Distribution.jl is prefered to `using` these packages since it does not create ambiguity between the packages and SumProductSet.jl
   package.
