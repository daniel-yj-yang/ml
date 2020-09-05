## Key Concepts

```Av = λv```

## Examples

<hr>

Clojure:
```Clojure
user=> (def A (matrix [[-6 3] [4 5]]))
#'user/A

user=> (decomp-eigenvalue A)
{:values (-7.0 6.0), :vectors [-0.9487 -0.2433
 0.3162 -0.9730]
}

;; Av = λv
user=> (mmult X (sel (:vectors (decomp-eigenvalue X)) :cols 1)) ;; Av
[-1.4595
-5.8381]

user=> (mult 6 (sel (:vectors (decomp-eigenvalue X)) :cols 1)) ;; λv, λ = 6
[-1.4595
-5.8381]
```

<hr>

R:
```R
> A <- matrix(c(-6, 3, 4, 5), 2, 2, byrow = T)

> A
     [,1] [,2]
[1,]   -6    3
[2,]    4    5

> eigen(A)
eigen() decomposition
$values
[1] -7  6

$vectors
           [,1]       [,2]
[1,] -0.9486833 -0.2425356
[2,]  0.3162278 -0.9701425
```
