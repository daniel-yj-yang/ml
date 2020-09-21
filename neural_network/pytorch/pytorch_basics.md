PyTorch Basics

``` python
# Vector operation

>>> x = torch.tensor([1,2,3,4,5])

>>> y = torch.tensor([6,7,8,9,10])

>>> torch.dot(x,y) # dot product
tensor(130)

>>> x[2:4]
tensor([3, 4])

################################################################

# Matrix Basics

>>> c = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])

>>> c
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]])

>>> c.size() # matrix size
torch.Size([3, 6])

>>> c.dim() # matrix dimensions
2

>>> df = pd.DataFrame({'a':[1, 2, 3], 'b':[4, 5, 6]})

>>> df
   a  b
0  1  4
1  2  5
2  3  6

>>> torch.tensor(df.values)  # convert pandas DataFrame to tensor
tensor([[1, 4],
        [2, 5],
        [3, 6]])

################################################################

# Matrix operation

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

################################################################

# Derivative

>>> x = torch.tensor(3., requires_grad = True) # x is now not just a float but also allowing derivative operation

>>> x
tensor(3., requires_grad=True)

>>> y = x**2 # y = x^2

>>> y
tensor(9., grad_fn=<PowBackward0>)

>>> y.backward() # calculate the derivate of y with respect to x, which is dy/dx = 2x

>>> x.grad # plug in x to see the value of dy/dx
tensor(6.)

################################################################

# Partial Derivative

>>> x = torch.tensor(4., requires_grad = True)

>>> y = torch.tensor(8., requires_grad = True)

>>> z = x**y + torch.sin(x)*torch.cos(y)

>>> z.backward() # calculate the partial derivatives here

>>> x.grad # plug in x to see the value of ∂z/∂x
tensor(131072.0938)
```
        
