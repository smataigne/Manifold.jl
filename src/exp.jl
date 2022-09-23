@views function expeuclidean(S::StiefelVector, Δ::TangentVector)
    Δ₂ = gettangent(Δ, S)
    temp₁ = similar(Δ₂, size(Δ₂, 1), size(S, 1))
    temp₂ = similar(Δ₂, size(S, 1), size(Δ.A, 2))
    Exp = similar(Δ₂, size(S, 1), size(Δ.A, 2))
    mul!(temp₁, Δ₂, transpose(S.U))
    E₁ = exp(skewhermitian!(temp₁) .* 2)
    E₂ = exp(-Δ.A)
    mul!(temp₂, S.U, E₂)
    mul!(Exp, E₁, temp₂)
    return Orthonormal(Exp, false)
end

@views function expeuclidean(S::StiefelVector, Δ::AbstractMatrix)
    Δ₂ = Δ
    temp₁ = similar(Δ₂, size(Δ₂, 1), size(S, 1))
    temp₂ = similar(Δ₂, size(S, 1), size(S, 2))
    A = similar(Δ₂, size(S, 2), size(S, 2))
    Exp = similar(Δ₂, size(S, 1), size(S, 2))
    mul!(temp₁, Δ₂, transpose(S.U))
    #display(skewhermitian(temp₁.*2))
    E₁ = exp(skewhermitian!(temp₁.*2))
    mul!(A, transpose(S.U) , Δ, -1, 0)
    E₂ = exp(skewhermitian!(A))
    mul!(temp₂, S.U, E₂)
    mul!(Exp, E₁, temp₂)
    return Orthonormal(Exp, false)
end

@views function expcanonical(S::StiefelVector, Δ::TangentVector)
    Δ₂ = gettangent(Δ, S)
    m, n = size(S, 1), size(S, 2)
    temp₁ = similar(Δ₂, n, n)
    temp₂ = similar(Δ₂, m, n)
    temp₃ = similar(Δ₂, 2*n, 2*n)
    temp₄ = similar(Δ₂, m, 2*n)
    mul!(temp₁, transpose(S.U), Δ₂)
    mul!(temp₂, S.U, temp₁)
    Δ₂ .-= temp₂
    Q, R = gramschmidt!(Δ₂)
    temp₃[1:n, 1:n] .= Δ.A
    temp₃[n+1:end,1:n] .= R.*2
    E₁ = exp(skewhermitian!(temp₃))
    temp₄[:, 1:n] .= S.U
    temp₄[:, n+1:end] .= Q
    mul!(temp₂, temp₄, E₁[:, 1:n])
    return Orthonormal(temp₂, false)
end

@views function expcanonical(S::StiefelVector, Δ::AbstractMatrix)
    Δ₂ = Δ
    m, n = size(S, 1), size(S, 2)
    A = similar(Δ₂, n, n)
    temp₂ = similar(Δ₂, m, n)
    temp₃ = similar(Δ₂, 2*n, 2*n)
    temp₄ = similar(Δ₂, m, 2*n)
    mul!(A, transpose(S.U), Δ₂)
    mul!(temp₂, S.U, A)
    Δ₂ .-= temp₂
    Q, R = gramschmidt!(Δ₂)
    mul!(temp₃[1:n, 1:n], transpose(S.U), Δ)
    temp₃[n+1:end,1:n] .= R.*2
    E₁ = exp(skewhermitian!(temp₃))
    temp₄[:, 1:n] .= S.U
    temp₄[:, n+1:end] .= Q
    mul!(temp₂, temp₄, E₁[:, 1:n])
    return Orthonormal(temp₂, false)
end

function Base.exp(S::StiefelVector{T, METRIC}, Δ::TangentVector{T, METRIC}) where {T, METRIC}
    if iszero(METRIC)
        return expeuclidean(S, Δ)
    end
    return expcanonical(S, Δ)
end

function Base.exp(S::StiefelVector{T, METRIC}, Δ::AbstractMatrix{T}) where {T, METRIC}
    if iszero(METRIC)
        return expeuclidean(S, Δ)
    end
    return expcanonical(S, Δ)
end