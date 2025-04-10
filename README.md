# torchcodes
This servers as notes to show my learnings and programs related to PyTorch. Currently learning from [learnpytorch](https://www.learnpytorch.io)

## Topics covered so far

<details>
  <summary>PyTorch fundamentals</summary>
  
  - `torch.tensor()`
  - `tensor.ndim`
  - `tensor.shape`
  - `tensor.dtype`
  - `scalar.item()`
  - `torch.rand(size)`
  - `torch.zeros(size)`
  - `torch.ones(size)`
  - `torch.rand_like(tensor)`
  - `torch.zeros_like(tensor)`
  - `torch.ones_like(tensor)`
  - `torch.arange(start, end, steps)`
  - `tensor.reshape(shape)`
  - `tensor.type(torch.type)` - **torch.types**:
    - `torch.float64`
    - `torch.float32`
    - `torch.float18`
    - `torch.int8`
  - Tensor operations
    - Addition `torch.add(tensor1, tensor2)`
    - Subtraction `torch.sub(tensor1, tensor2)`
    - Division `torch.div(tensor1, tensor2)`
    - Multiplication (element-wise) `torch.mal(tensor1, tensor2)`
    - Matrix multiplication `torch.matmul(tensor1, tensor2)` or `torch.mm(tensor1, tensor2)` or `tensor1 @ tensor2`
  - `tensor 1.T`
  - `torch.nn.Linear(in_features, out_features)`
  - `tensor.max()`
  - `tensor.min()`
  - `tensor.mean()`
  - `tensor.sum()`
  - `tensor.argmin()`
  - `tensor.argmax()`
  - `tensor.sort()`
    - `tensor.sort().indices`
    - `tensor.sort().values`
  - `tensor.view(shape)`
  - `torch.stack((tensor1, tensor2, tensor3,....))`
  - `tensor.permute(shape)`
  - `torch.from_numpy(ndarray)`
  - `torch.tensor.numpy(tensor)`
  - `torch.manual_seed(seed)`
  - `torch.cuda`
    - `torch.cuda.is_available()`
    - `torch.cuda.get_device_name()`
    - `tensor.is_cuda`
  -`tensor.device(device_name)`
</details>
