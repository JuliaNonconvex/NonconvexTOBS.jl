module NonconvexTOBS

export TOBSAlg, TOBSOptions

using Reexport, Parameters, SparseArrays, HiGHS
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize!
import JuMP

struct TOBSAlg <: AbstractOptimizer end

struct TOBSOptions
    nt::NamedTuple # one of the fields of the nt is milp_options
end

function TOBSOptions(;
    movelimit::Real=0.1, # move limit parameter
    pastN::Int=20, # number of past iterations for moving average calculation
    convParam::Real=0.001, # convergence parameter (upper bound of error)
    constrRelax::Real=0.9, # constraint relaxation parameter
    timeLimit::Real=1.0,
    optimizer=HiGHS.Optimizer,
    maxiter::Int=200,
    timeStable::Bool=true,
    relaxDecay::Real=0.9, # factor to shrink inflated movelimit/constrRelax toward defaults on success
    maxInfeasible::Int=5, # consecutive infeasible subproblems before feasibility restoration
)
    return TOBSOptions((;
        movelimit,
        pastN,
        convParam,
        constrRelax,
        timeLimit,
        optimizer,
        maxiter,
        timeStable,
        relaxDecay,
        maxInfeasible,
    ))
end

mutable struct TOBSWorkspace{TM<:VecModel,TX<:AbstractVector,TO<:TOBSOptions} <: Workspace
    model::TM
    x0::TX
    options::TO
end
function TOBSWorkspace(
    model::VecModel,
    x0::AbstractVector=NonconvexCore.getinit(model);
    options=TOBSOptions(),
    kwargs...,
)
    return TOBSWorkspace(model, copy(x0), options)
end
struct TOBSResult{TM1,TM2,TE} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    error::TE
end

function optimize!(workspace::TOBSWorkspace)
    @unpack model, x0, options = workspace
    @unpack movelimit,
    pastN,
    convParam,
    constrRelax,
    timeLimit,
    optimizer,
    maxiter,
    timeStable,
    relaxDecay,
    maxInfeasible = options.nt
    milp_solver = JuMP.optimizer_with_attributes(optimizer)
    numVars = length(NonconvexCore.getinit(model))
    count = 1 # iteration counter
    objHist = zeros(pastN) # recent history of compliance values
    if any(NonconvexCore.getmin(model) .!= 0) || any(NonconvexCore.getmax(model) .!= 1)
        throw(ArgumentError("Lower bound must be 0 and upper bound must be 1."))
    end
    er = 1.0
    x = copy(x0)
    currentConstr, jacConstr = NonconvexCore.value_jacobian(model.ineq_constraints, x)
    objval, objgrad = NonconvexCore.value_gradient(getobjective(model), x)
    pastGrad = copy(objgrad)
    best_sol = (x, objval, currentConstr, norm(currentConstr))
    best_feasible = false

    m = JuMP.Model(milp_solver)
    skip_step = false
    infeasible_count = 0
    while (convParam < er || any(currentConstr .> 0)) && count < maxiter
        count > 1 && (m = JuMP.Model(milp_solver))
        JuMP.set_optimizer_attribute(m, "log_to_console", false)
        JuMP.set_optimizer_attribute(m, "time_limit", timeLimit)
        if !skip_step || count == 1
            if count > 1
                currentConstr, jacConstr =
                    NonconvexCore.value_jacobian(model.ineq_constraints, x)
            end
            violation = norm(currentConstr)
            if (violation <= best_sol[4] - 1e-8 || violation < 1e-8 && objval < best_sol[2])
                best_sol = (x, objval, currentConstr, violation)
                if violation < 1e-8
                    best_feasible = true
                end
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
        # Constrain amount of change per iteration
        JuMP.@constraint(m, sum(absdeltaX) <= movelimit * numVars)
        # Constraint relaxation: when a constraint is satisfied (c < constrRelax)
        # we relax the linearized constraint to -c so the subproblem is easier to
        # solve. When violated (c >= constrRelax), we demand a fraction
        # (1 - constrRelax) of the violation be removed this iteration.
        Δ = map(currentConstr) do c
            c < constrRelax ? -c : (constrRelax - 1) * c
        end
        JuMP.@constraint(m, jacConstr * deltaX .<= Δ)
        # Objective: normally minimize the linearized objective. After too many
        # infeasible subproblems, switch to a feasibility-restoration phase that
        # minimizes the L1 norm of the *positive* (violated) part of the
        # linearized constraint residual. This finds the flips that push the
        # violated constraints down the most, ignoring the original objective.
        restore = infeasible_count >= maxInfeasible
        if restore
            # Artificial slack for each constraint (>= positive violation).
            JuMP.@variable(m, violSlack[1:length(currentConstr)] >= 0)
            # linearized constraint value + slack >= 0  =>  slack >= -(c + jac*dx)
            # i.e. slack captures any remaining (positive) violation after the step.
            JuMP.@constraint(m, violSlack .>= -(currentConstr + jacConstr * deltaX))
            JuMP.@objective(m, Min, sum(violSlack))
        else
            JuMP.@objective(m, Min, objgrad' * deltaX)
        end
        # Optimize linearized problem
        JuMP.optimize!(m)
        # Check if infeasible
        if JuMP.termination_status(m) == JuMP.INFEASIBLE
            # Subproblem infeasible, but a primal solution may still exist
            # (e.g. from a time-limited solve). Evaluate the *real* constraint
            # at the candidate point and accept the step only if it improves
            # the actual (nonlinear) feasibility, not just the linearization.
            primal = JuMP.primal_status(m)
            if primal == JuMP.FEASIBLE_POINT || primal == JuMP.NEARLY_FEASIBLE_POINT
                x_trial = x + JuMP.value.(deltaX)
                trialConstr = model.ineq_constraints(x_trial)
                trial_vio = norm(trialConstr)
                if trial_vio < norm(currentConstr) - 1e-10
                    @info "Subproblem infeasible but step improves real feasibility (vio $(round(norm(currentConstr), digits=3)) → $(round(trial_vio, digits=3))). Accepting."
                else
                    skip_step = true
                end
            else
                skip_step = true
            end
        end
        # Apply the step if not skipped. Re-fetch deltaX.value to handle both
        # the normal-feasible case and the infeasible-but-accepted case.
        try
            if !skip_step
                x += JuMP.value.(deltaX)
            end
        catch
            skip_step = true
        end
        if skip_step
            infeasible_count += 1
            if infeasible_count == maxInfeasible
                @warn "Subproblem infeasible $infeasible_count times. Switching to feasibility restoration."
            elseif infeasible_count > maxInfeasible
                # Even restoration failed: keep bumping move limit to give the
                # restoration phase more freedom on the next attempt.
                @warn "Restoration subproblem still infeasible."
                movelimit *= 1.1
                constrRelax = min(constrRelax * 1.1, 0.999)
            else
                @warn "Subproblem is infeasible. Temporarily relaxing the subproblem."
                movelimit *= 1.1
                constrRelax = min(constrRelax * 1.1, 0.999)
            end
        else
            if restore
                @info "Restoration step succeeded; resuming normal iterations."
            end
            infeasible_count = 0
            # Decay inflated parameters back toward their defaults on success.
            movelimit = options.nt.movelimit + relaxDecay * (movelimit - options.nt.movelimit)
            constrRelax =
                options.nt.constrRelax + relaxDecay * (constrRelax - options.nt.constrRelax)
            # Store recent history of objectives
            objHist[1:(end-1)] .= objHist[2:end]
            objval, objgrad = NonconvexCore.value_gradient(getobjective(model), x)
            # Apply "time stabilization": average the current gradient with the
            # previous iteration's gradient to reduce oscillation.
            if timeStable
                newGrad = (objgrad + pastGrad) / 2
                pastGrad = copy(objgrad)  # save the ORIGINAL gradient, not the averaged one
                objgrad = newGrad
            end
            objHist[end] = objval
            if count > pastN
                # Use sum of ABSOLUTE differences so oscillations don't cancel out
                er = sum([abs(objHist[i] - objHist[i-1]) for i = 2:pastN]) / sum(objHist)
            end
            @info "iter = $count, obj = $(round.(objHist[end]; digits=3)), constr_vio_norm = $(round(norm(currentConstr), digits=3)), er = $(round(er, digits=3))"
        end
        count += 1
    end
    if !best_feasible
        @warn "No feasible solution was found during the optimization. Returning the best design found."
    end
    return TOBSResult(best_sol[1], best_sol[2], er)
end

function Workspace(model::VecModel, optimizer::TOBSAlg, x0::AbstractVector; kwargs...)
    return TOBSWorkspace(model, x0; kwargs...)
end

end
