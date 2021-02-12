using Zygote: Params

Optimisers.state(o, xs::Params) = [state(o, x) for x in xs]

function Optimisers.update(o, xs::Params, gs, states)
  new_states = []
  for (i, (x, s)) in enumerate(zip(xs, states))
    x̄, s̄ = update(o, x, gs[x], s)
    xs[x] .= x̄
    push!(new_state, s̄)
  end

  return xs, new_states
end

# """
#     ClipValue(thresh)

# Clip gradients when their absolute value exceeds `thresh`.
# """
# mutable struct ClipValue{T}
#     thresh::T
# end

# apply!(o::ClipValue, x, Δ) = clamp!(Δ, -o.thresh, o.thresh)

# """
#     ClipNorm(thresh)

# Clip gradients when their L2 norm exceeds `thresh`.
# """
# mutable struct ClipNorm{T}
#     thresh::T
# end

# function apply!(o::ClipNorm, x, Δ)
#     Δnrm = norm(Δ)
#     if Δnrm > o.thresh
#         rmul!(Δ, o.thresh / Δnrm)
#     end
#     return Δ
# end
