module NonconvexTOBS

export IpoptAlg, IpoptOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!, _dot
import NonconvexCore: optimize!
using Ipopt

@params struct TOBSOptions
    nt::NamedTuple # one of the fields of the nt is milp_options
end
function TOBSOptions(; kwargs...,)
    return TOBSOptions(kwargs)
end
# TOBSOptions(milp_options = (time_limit = ..,))

@params mutable struct TOBSWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    options::TOBSOptions
    counter::Base.RefValue{Int}
end
function TOBSWorkspace(
    model::VecModel, x0::AbstractVector = getinit(model);
    options = TOBSOptions(), kwargs...,
)
    problem, counter = get_ipopt_problem(
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
    @unpack model, problem, options, counter, x0 = workspace
    problem.x .= x0
    counter[] = 0
    foreach(keys(options.nt)) do k
        v = options.nt[k]
        addOption(problem, string(k), v)
    end
    solvestat = Ipopt.IpoptSolve(problem)
    return TOBSResult(
        copy(problem.x), getobjective(model)(problem.x),
        problem, solvestat, counter[]
    )
end

struct TOBSAlg{S} <: AbstractOptimizer
    milp::S
end

function Workspace(model::VecModel, optimizer::TOBSAlg, x0::AbstractVector; kwargs...,)
    return TOBSWorkspace(model, x0; kwargs...)
end

# getobjective(model)
# g(x) = model.ineq_constraints(x) -> g(x) <= 0
# getmin(model)
# getmax(model)
# getinit(model)

end
