PyTorch Basics

``` python
>>> x = torch.tensor([1,2,3,4,5])

>>> y = torch.tensor([6,7,8,9,10])

>>> torch.dot(x,y) # dot product
tensor(130)

>>> x[2:4]
tensor([3, 4])

################################################################

>>> c = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])

>>> c
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]])

>>> c.size() # matrix size
torch.Size([3, 6])

>>> c.dim() # matrix dimensions
2

################################################################

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