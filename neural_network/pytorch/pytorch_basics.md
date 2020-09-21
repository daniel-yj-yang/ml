PyTorch Basics

#### Elementwise multiplication

``` python
>>> a = torch.tensor([[1,2],[3,4]])
>>> b = torch.tensor([[5,6],[7,8]])
>>> a
tensor([[1, 2],
        [3, 4]])
>>> b
tensor([[5, 6],
        [7, 8]])
>>> a * b
tensor([[ 5, 12],
        [21, 32]])
```

##### Matrix multiplication

```python
>>> torch.mm(a,b)
tensor([[19, 22],
        [43, 50]])
```
