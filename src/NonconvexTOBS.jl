module NonconvexTOBS

export TOBSAlg, TOBSOptions

using Reexport, Parameters, SparseArrays, Zygote, JuMP, Cbc
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!, _dot
import NonconvexCore: optimize!

struct TOBSAlg <: AbstractOptimizer end

struct TOBSOptions
    nt::NamedTuple # one of the fields of the nt is milp_options
end

function TOBSOptions(
    ;β::Real = 0.1, # move limit parameter
    N::Int = 5, # number of past iterations for moving average calculation
    τ::Real = 0.0001, # convergence parameter (upper bound of error)
    ϵ::Real = 0.01, # constraint relaxation parameter
    timeLimit::Real = 1.0,
    optimizer = Cbc.Optimizer,
)
    return TOBSOptions((; β, N, τ, ϵ, timeLimit, optimizer))
end

@params mutable struct TOBSWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    options::TOBSOptions
    counter::Base.RefValue{Int}
end
function TOBSWorkspace(
    model::VecModel, x0::AbstractVector = Nonconvex.getinit(model);
    options = TOBSOptions(), kwargs...,
)
    return TOBSWorkspace(model, copy(x0), options, Ref(0))
end
@params struct TOBSResult <: AbstractResult
    minimizer
    minimum
    fcalls::Int
end

function optimize!(workspace::TOBSWorkspace)
    @unpack model, x0, options, counter = workspace
    counter[] = 0
    @unpack β, N, τ, ϵ, timeLimit, optimizer = options.nt
    milp_solver = optimizer_with_attributes(optimizer)
    numVars = length(NonconvexCore.getinit(model))
    count = 1 # iteration counter
    comps = zeros(N) # recent history of compliance values
    if any(NonconvexCore.getmin(model) .!= 0) || any(NonconvexCore.getmax(model) .!= 1)
        throw(ArgumentError("Lower bound must be 0 and upper bound must be 1."))
    end
    er = 1
    # x = NonconvexCore.getinit(model)
    x = ones(numVars)
    currentConstr = model.ineq_constraints(x)

    m = JuMP.Model(milp_solver)
    while τ < er || any(currentConstr .> 0)
        count > 1 && (m = JuMP.Model(milp_solver))
        set_optimizer_attribute(m, "logLevel", 0)
        set_optimizer_attribute(m, "seconds", timeLimit)

        sens = Zygote.gradient(NonconvexCore.getobjective(model), x)[1]
        jacConstr = Zygote.jacobian(model.ineq_constraints, x)[1]

        # Define optimization subproblem variables (change in each variable of original problem for this iteration)
        @variable(m, deltaX[1:numVars], Int)
        set_lower_bound.(deltaX, -x)
        set_upper_bound.(deltaX, 1 .- x)
        # Constrain volume change per iteration
        @constraint(m, sum(deltaX) <= β*numVars)
        # Constraint relaxation
        currentConstr = model.ineq_constraints(x)
        Δ = map(currentConstr) do c
            if 0 < (1 - ϵ) * c
                return -ϵ * c
            elseif 0 > (1 + ϵ) * c
                return ϵ * c
            else
                return -c
            end
        end
        @constraint(m, jacConstr * deltaX .<= Δ)
        # Define optimization objective
        @objective(m, Min, sens' * deltaX)
        # Optimize linearized problem
        JuMP.optimize!(m)
        # Perform step (update pseudo-densities)
        x += JuMP.value.(deltaX)
        # Store recent history of objectives
        comps[1:end-1] .= comps[2:end]
        comps[end] = getobjective(model)(x)
        if count > 2 * N - 1
            er = abs(sum([comps[i] - comps[i - 1] for i in 2:N])) / sum(comps)
        end
        
        @info "iter = $count, obj = $(round.(comps[end]; digits=3)), constraint violation norm = $(round(norm(currentConstr), digits=3)), er = $(round(er, digits=3))"
        count += 1
    end

    return TOBSResult(copy(x), NonconvexCore.getobjective(model)(x), counter[])
end

function Workspace(model::VecModel, optimizer::TOBSAlg, x0::AbstractVector; kwargs...,)
    return TOBSWorkspace(model, x0; kwargs...)
end

end