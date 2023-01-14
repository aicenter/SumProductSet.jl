# SumProductSet.jl

is package for probailistic learning of hierarchicaly structured heterogeneous data based on SumProduct networks. The package extends standard SumProduct networks by introducing new type of model node and implementing rules for building models over aforementioned data. It is based on [SumProductTransform.jl](https://github.com/pevnak/SumProductTransform.jl) package. The package [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) is used as a framework that unifies accepted data format.



To reproduce this project, do the following:

1. Download this code base repository.
2. Open a Julia console and type:
   ```julia
   using Pkg
   Pkg.activate("path/to/the/project")
   Pkg.instantiate()
   ```
3. Use the package:
   ``` julia
   using SumProductSet
   
   ```


This will download and install all necessary packages defined in project environment for you.
