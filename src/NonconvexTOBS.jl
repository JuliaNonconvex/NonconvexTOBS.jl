module NonconvexTOBS

export TOBSAlg, TOBSOptions

using Reexport, Parameters, SparseArrays, Zygote, Cbc
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize!
import JuMP

struct TOBSAlg <: AbstractOptimizer end

struct TOBSOptions
    nt::NamedTuple # one of the fields of the nt is milp_options
end

function TOBSOptions(
    ;β::Real = 0.05, # move limit parameter
    N::Int = 20, # number of past iterations for moving average calculation
    τ::Real = 0.001, # convergence parameter (upper bound of error)
    ϵ::Real = 0.05, # constraint relaxation parameter
    timeLimit::Real = 1.0,
    optimizer = Cbc.Optimizer,
    maxiter::Int = 200
)
    return TOBSOptions((; β, N, τ, ϵ, timeLimit, optimizer, maxiter))
end

@params mutable struct TOBSWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    options::TOBSOptions
end
function TOBSWorkspace(
    model::VecModel, x0::AbstractVector = NonconvexCore.getinit(model);
    options = TOBSOptions(), kwargs...,
)
    return TOBSWorkspace(model, copy(x0), options)
end
@params struct TOBSResult <: AbstractResult
    minimizer
    minimum
    error
end

function optimize!(workspace::TOBSWorkspace)
    @unpack model, x0, options = workspace
    @unpack β, N, τ, ϵ, timeLimit, optimizer, maxiter = options.nt
    milp_solver = JuMP.optimizer_with_attributes(optimizer)
    numVars = length(NonconvexCore.getinit(model))
    count = 1 # iteration counter
    comps = zeros(N) # recent history of compliance values
    if any(NonconvexCore.getmin(model) .!= 0) || any(NonconvexCore.getmax(model) .!= 1)
        throw(ArgumentError("Lower bound must be 0 and upper bound must be 1."))
    end
    er = 1
    x = ones(numVars)
    currentConstr, jacConstr = NonconvexCore.value_jacobian(model.ineq_constraints, x)
    objval, objgrad = NonconvexCore.value_gradient(getobjective(model), x)
    best_sol = (x, objval, currentConstr, norm(currentConstr))

    m = JuMP.Model(milp_solver)
    skip_step = false
    while (τ < er || any(currentConstr .> 0)) && count < maxiter
        count > 1 && (m = JuMP.Model(milp_solver))
        JuMP.set_optimizer_attribute(m, "logLevel", 0)
        JuMP.set_optimizer_attribute(m, "seconds", timeLimit)
        if !skip_step || count == 1
            if count > 1
                currentConstr, jacConstr = NonconvexCore.value_jacobian(model.ineq_constraints, x)
            end
            violation = norm(currentConstr)
            if (violation <= best_sol[4] - 1e-8 || violation < 1e-8 && objval < best_sol[2])
                best_sol = (x, objval, currentConstr, violation)
            end
        end
        skip_step = false
        # Define optimization subproblem variables (change in each variable of original problem for this iteration)
        JuMP.@variable(m, deltaX[1:numVars], Int)
        JuMP.set_lower_bound.(deltaX, -x)
        JuMP.set_upper_bound.(deltaX, 1 .- x)
        JuMP.@variable(m, absdeltaX[1:numVars])
        JuMP.set_lower_bound.(absdeltaX, 0)
        JuMP.set_upper_bound.(absdeltaX, 1)
        JuMP.@constraint(m, deltaX .<= absdeltaX)
        JuMP.@constraint(m, .-deltaX .<= absdeltaX)
        # Constrain volume change per iteration
        JuMP.@constraint(m, sum(absdeltaX) <= β*numVars)
        # Constraint relaxation
        Δ = map(currentConstr) do c
            abs(c) < ϵ ? -c : -ϵ * c
        end
        JuMP.@constraint(m, jacConstr * deltaX .<= Δ)
        # Define optimization objective
        JuMP.@objective(m, Min, objgrad' * deltaX)
        # Optimize linearized problem
        JuMP.optimize!(m)
        # Check if infeasible
        if JuMP.termination_status(m) == JuMP.INFEASIBLE
            skip_step = true
        end
        # Sometimes value errors even if feasible??
        try
            if !skip_step
                x += JuMP.value.(deltaX)
            end
        catch
            skip_step = true
        end
        if skip_step
            @warn "Subproblem is infeasible. Temporarily relaxing the subproblem."
            β *= 1.1
            ϵ *= 1.1
        else
            β = options.nt.β
            ϵ = options.nt.ϵ
            # Store recent history of objectives
            comps[1:end-1] .= comps[2:end]
            objval, objgrad = NonconvexCore.value_gradient(getobjective(model), x)
            comps[end] = objval
            if count > N
                er = abs(sum([comps[i] - comps[i - 1] for i in 2:N])) / sum(comps)
            end
            @info "iter = $count, obj = $(round.(comps[end]; digits=3)), constr_vio_norm = $(round(norm(currentConstr), digits=3)), er = $(round(er, digits=3)) Δ = $(Δ[1])"
        end
        count += 1
    end
    return TOBSResult(best_sol[1], best_sol[2], er)
end

function Workspace(model::VecModel, optimizer::TOBSAlg, x0::AbstractVector; kwargs...,)
    return TOBSWorkspace(model, x0; kwargs...)
end

end