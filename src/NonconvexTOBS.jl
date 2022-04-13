module NonconvexTOBS

export TOBSalg, TOBSOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!, _dot
import NonconvexCore: optimize!
using TOBS

@with_kw struct TOBSOptions
    nt::NamedTuple # one of the fields of the nt is milp_options
    timeLimit::Real = 1.0
    optimizer::TOBSalg{S} # ?
    # optimSolver = Cbc.optimizer
    β::Real = 0.1 # move limit parameter
    N::Int = 5 # number of past iterations for moving average calculation
    τ::Real = 0.0001 # convergence parameter (upper bound of error)
    ϵ::Real = 0.01 # constraint relaxation parameter
end

function TOBSOptions(; kwargs...,)
    return TOBSOptions(kwargs)
end

@params mutable struct TOBSWorkspace <: Workspace
    model::VecModel
    problem # ?
    x0::AbstractVector
    options::TOBSOptions
    counter::Base.RefValue{Int}
end
function TOBSWorkspace(
    model::VecModel, x0::AbstractVector = Nonconvex.getinit(model);
    options = TOBSOptions(), kwargs...,
)
    problem, counter = get_TOBS_problem(
        model, copy(x0),
        options.nt.hessian_approximation == "limited-memory",
        options.nt.jac_c_constant == "yes" && options.nt.jac_d_constant == "yes",
    )
    return TOBSWorkspace(model, problem, copy(x0), options, counter)
end
@params struct TOBSResult <: AbstractResult
    minimizer
    minimum
    problem
    status
    fcalls::Int
end

function optimize!(workspace::TOBSWorkspace)
    @unpack model, problem, x0, options, counter = workspace
    problem.x .= x0
    counter[] = 0
    foreach(keys(options.nt)) do k
        v = options.nt[k]
        addOption(problem, string(k), v)
    end
    solvestat = TOBSSolve(problem, workspace)
    return TOBSResult(
        copy(problem.x), NonconvexCore.getobjective(model)(problem.x),
        problem, solvestat, counter[]
    )
end

struct TOBSalg{S} <: AbstractOptimizer
    milp::S
end

function TOBSSolve(workspace)
    # milp_solver = optimizer_with_attributes(Cbc.Optimizer)
    milp_solver = optimizer_with_attributes(workspace.options.optimizer)
    workspace.problem.x .= workspace.x0
    numVars = length(workspace.problem.x)
    count = 1 # iteration counter
    comps = zeros(workspace.options.N) # recent history of compliance values

    m = JuMP.Model(milp_solver)
    while workspace.options.τ < workspace.options.er || currentConstr > constrVal
        count > 1 && (m = JuMP.Model(milp_solver))
        set_optimizer_attribute(m, "logLevel", 0)
        set_optimizer_attribute(m, "seconds", workspace.options.timeLimit)

        sens = Zygote.gradient(NonconvexCore.getobjective(workspace.model), x)[1]
        # gradConstr = Zygote.gradient(constr, x)[1]
        jacConstr = Zygote.jacobian(m.ineqconstraints, x)[1]
        # in slack: NonconvexCore.getineqconstraints(model) ?

        # Define optimization subproblem variables (change in each variable of original problem for this iteration)
        @variable(m, deltaX[1:numVars], Int)
        set_lower_bound.(deltaX, -x)
        set_upper_bound.(deltaX, 1 .- x)
        # Constrain volume change per iteration
        @constraint(m, sum(x -> x^2, deltaX) <= workspace.options.β*numVars)
        # Constraint relaxation
        if constrVal < (1 - workspace.options.ϵ) * currentConstr
            Δ = -workspace.options.ϵ * currentConstr
        elseif constrVal > (1 + workspace.options.ϵ) * currentConstr
            Δ = workspace.options.ϵ * currentConstr
        else
            Δ = constrVal - currentConstr
        end
        @constraint(m, jacConstr' * deltaX <= Δ)
        # g(x) = model.ineq_constraints(x) -> g(x) <= 0
        # Define optimization objective
        @objective(m, Min, sens' * deltaX)
        # Optimize linearized problem
        JuMP.optimize!(m)
        # Perform step (update pseudo-densities)
        x += JuMP.value.(deltaX)
        # Store recent history of objectives
        comps[1:end-1] .= comps[2:end]
        comps[end] = workspace.model.objective.f(x)
        if count > 2 * N - 1
            workspace.options.er = abs(sum([comps[i] - comps[i - 1] for i in 2:N])) / sum(comps)
        end
        # currentConstr = volfrac(filter(x))
        
        @info "iter = $count, obj = $(round.(comps[end]; digits=3)), vol_frac = $(round(currentConstr, digits=3)), er = $(round(workspace.options.er, digits=3))"
        count += 1        
    end
end

function Workspace(model::VecModel, optimizer::TOBSalg, x0::AbstractVector; kwargs...,)
    return TOBSWorkspace(model, x0; kwargs...)
end

function get_TOBS_problem()
    
end

end