"""
    Scheduler{T<:ParameterSchedulers.AbstractSchedule, O}
    Scheduler(schedule::ParameterSchedulers.AbstractSchedule, opt)

Wrap a `schedule` and optimizer together into a `Scheduler`.
The `schedule` is iterated each time the optimizer updates the gradients.
The `Scheduler` can be used anywhere a Flux optimizer is used.

`opt` can be either:
- an optimizer (the learning rate will be the scheduled parameter)
- a function that accepts the current parameter value and returns a new optimizer

# Examples
```jldoctest
julia> opt = Momentum();

julia> schedule = Schedule.Exp(λ = 0.01, γ = 0.5);

julia> sopt = Schedule.Scheduler(schedule, opt)

julia> 
"""
mutable struct Scheduler{T<:AbstractSchedule, O}
  schedule::T
  optim::O
end

Optimise.Optimisers.init(o::Scheduler, x::AbstractArray) = (t = 1, optim = init(o.optim, x))

function Optimise.Optimisers.apply(opt::Scheduler, x, dx, state)
  # set param
  o = opt.optim(opt.schedule[state.t])

  # do normal apply
  dx, s = Optimise.Optimisers.apply(o, x, dx, state.optim)

  return dx, (t = t + 1, optim = s)
end