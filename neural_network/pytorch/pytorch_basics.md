PyTorch Basics

``` python
>>> a = torch.tensor([[1,2],[3,4]])

>>> b = torch.tensor([[5,6],[7,8]])

>>> a
tensor([[1, 2],
        [3, 4]])
        
>>> b
tensor([[5, 6],
        [7, 8]])
        
>>> a * b  # Elementwise multiplication
tensor([[ 5, 12],
        [21, 32]])
        
>>> torch.mm(a,b) # Matrix multiplication
tensor([[19, 22],
        [43, 50]])
```
