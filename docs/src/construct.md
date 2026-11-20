# Construct an instance

To construct an instance of the TOBS algorithm, use [`TOBSAlg`](@ref):
```julia
alg = TOBSAlg()
```
When optimizing a model using `TOBSAlg`, all the variables are assumed to be binary if their lower and upper bounds are 0 and 1 respectively even if the `isinteger` flag was not used. If there are variables with other bounds' values, the optimization will give an error. Equality constraints are rejected with an `ArgumentError`.
