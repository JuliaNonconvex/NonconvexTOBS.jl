<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexTOBS.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexTOBS.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexTOBS.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexTOBS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaNonconvex/NonconvexTOBS.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexTOBS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexTOBS.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexTOBS.jl

Topology optimization of binary structures

The TOBS binary optimization algorithm for material distribution problems.

NonconvexTOBS.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
alg = TOBSAlg()
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexTOBS.jl/stable](https://julianonconvex.github.io/NonconvexTOBS.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexTOBS.jl/dev](https://julianonconvex.github.io/NonconvexTOBS.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexTOBS")
```
