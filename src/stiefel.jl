
mutable struct StiefelVector{T<:Number, METRIC, X<:Orthonormal{<:T}}
    U::X 
    Θ::Union{Nothing, Orthonormal{T}} #Θ is the orthonormal complement of U
    #METRIC: 0 =  Euclidean, 1 = Canonical
    function StiefelVector{T, METRIC, X}(U) where {T, METRIC, X}
        Base.require_one_based_indexing(U)
        new{T, METRIC, X}(U, nothing)
    end
end

"""
    StiefelVector(U, METRIC)

Type representing a point of the Stiefel Manifold.
Let S be a StiefelVector, S.U must be orthonormal, i.e `U^TU = I`.
S.Θ is the orthonormal complement of S.U and can be added/build to the structure
using `addcomplement` or `buildcomplement`.
"""
function StiefelVector(U::AbstractMatrix, METRIC = 0)
    isorthonormal(U) || throw(ArgumentError("Columns of U must be orthonormal"))
    return StiefelVector{eltype(U), METRIC, Orthonormal{<:eltype(U)}}(Orthonormal(U, false))
end

StiefelVector(U::Orthonormal, METRIC = 0) = StiefelVector{eltype(U), METRIC, typeof(U)}(U)
Base.size(S::StiefelVector) = size(S.U)
Base.size(S::StiefelVector, n) = size(S.U, n)
dims(S::StiefelVector) = size(S)

"""
    buildcomplement(S)

Construct the orthonormal complement of S.U using
Gram-Schmidt orthogonalization.
"""
@views function buildcomplement(S::StiefelVector{T}) where T
    m, n = dims(S)
    n₂ = m - n
    Θ = rand(T, m, n₂)
    for i = 1:n
        for j = 1:n₂
            α = dot(S.U[:,i], Θ[:, j] )
            Θ[:, j] .-= α.*S.U[:,i]
        end
    end
    for i = 1:n₂
        nm = norm(Θ[:, i])
        Θ[:, i] ./= nm
        for j = i+1:n₂
            α = dot(Θ[:,i], Θ[:, j] )
            Θ[:, j] .-= α.*Θ[:,i]
        end
    end
    S.Θ = Orthonormal(Θ, false)
    return S.Θ
end

@views function addcomplement(S::StiefelVector, Q::Orthonormal) 
    iszero(transpose(S.U) * Q) || throw(ArgumentError("Q must be the orthonormal complement of S.U"))
    S.Θ = Q
end

"""
    addcomplement(S, Q)

Add the orthonormal complement Q of S.U to S where S is a StiefelVector
"""
addcomplement(S::StiefelVector, Q::AbstractMatrix) = addcomplement(S, Orthonormal(Q))

function projection!(W::AbstractMatrix, S::StiefelVector)
    temp1 = similar(W, size(S, 2), size(W, 2))
    temp2 = similar(W, size(S, 1), size(W, 2))
    mul!(temp1, transpose(S.U), W)
    SkewLinearAlgebra.hermitian!(temp1)
    mul!(temp2, S.U, temp1)
    W .-= temp2
    return W
end

"""
    projection(W, S)
Returns the projection of any matrix W m×n on the
tangent space to the StiefelVector S ∈ St(m, n). 
"""
projection(W::AbstractMatrix, S::StiefelVector) = projection!(copy(W), S::StiefelVector)

mutable struct TangentVector{T<:Number, METRIC, Am<:SkewHermitian{T}, Bm<:Union{Nothing, AbstractMatrix{T}}}
    A::Am
    B::Bm
    function TangentVector{T, METRIC, Am, Bm}(A, B) where {T, METRIC, Am, Bm}
        Base.require_one_based_indexing(A)
        B === nothing || Base.require_one_based_indexing(B)
        new{T, METRIC, Am, Bm}(A, B)
    end
end

TangentVector(A::SkewHermitian{T}, B::Union{Nothing, AbstractMatrix{T}}, METRIC = 0) where T = TangentVector{T, METRIC, typeof(A), typeof(B)}(A, B)
TangentVector(A::StridedMatrix{T}, B::AbstractMatrix{T}, METRIC = 0) where {T} = TangentVector{T, METRIC, SkewHermitian{T},typeof(B)}(skewhermitian(A), B)

function gettangent(Δ::TangentVector, S::StiefelVector)
    if Δ.B === nothing 
        temp = similar(Δ.A, size(S, 1), size(Δ.A, 2))
        mul!(temp, S.U, Δ.A)
        return temp
    else
        S.Θ === nothing && throw(ArgumentError("Complementary space of S must be specified"))
        temp₁ = similar(Δ.A, size(S, 1), size(Δ.A, 2))
        temp₂ = similar(Δ.A, size(S, 1), size(Δ.A, 2))
        mul!(temp₁, S.U, Δ.A)
        mul!(temp₂, S.Θ, Δ.B)
        temp₁ .+= temp₂
        return temp₁
    end
end

@views function doteuclidean(Δ::TangentVector, Γ::TangentVector)
    Σ = 0
    n = size(Δ.A, 2)
    for i = 1:n
        Σ += dot(Δ.A[:, i], Γ.A[:, i])  
    end
    n = size(Δ.B, 2)
    for i = 1:n
        Σ += dot(Δ.B[:, i], Γ.B[:, i])
    end
    return Σ
end

@views function dotcanonical(Δ::TangentVector, Γ::TangentVector)
    Σ = 0
    n = size(Δ.A, 2)
    for i = 1:n
        Σ += dot(Δ.A[:, i], Γ.A[:, i])  
    end
    Σ ./= 2
    n = size(Δ.B, 2)
    for i = 1:n
        Σ += dot(Δ.B[:, i], Γ.B[:, i])
    end
    return Σ
end

function LA.dot(Δ::TangentVector{T, METRIC}, Γ::TangentVector{T, METRIC}) where {T, METRIC}
    if iszero(METRIC)
        return doteuclidean(Δ, Γ)
    end
    return dotcanonical(Δ, Γ)
end

@views function normeuclidean(Δ::TangentVector)
    return sqrt(dot(Δ, Δ))
end

@views function normcanonical(Δ::TangentVector)
    return sqrt(dot(Δ, Δ))
end

function LA.norm(Δ::TangentVector)
    if iszero(METRIC)
        return normeuclidean(Δ)
    end
    return normcanonical(Δ)
end 

@views function gramschmidt!(A::AbstractMatrix)
    n = size(A, 2)
    R = similar(A, n, n)
    iszero(A) && return Matrix(randorthonormal(eltype(A), size(A, 1), n)), R
    for i = 1:n
        R[i, i] = norm(A[:, i])
        A[:, i] ./= R[i,i]
        for j = i+1:n
            R[i, j] = dot(A[:, i], A[:, j])
            A[:, j] .-= R[i,j].*A[:, i]
        end
    end
    return A, R
end

gramschmidt(A::AbstractMatrix) = gramschmidt!(copy(A))




    

