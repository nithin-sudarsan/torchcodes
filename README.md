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
<details>
  <summary>PyTorch workflow fundamentals</summary>
  
- Train-test split
- Visualising data
  - `matplotlib.pyplot`
  - `plt.plot(x,y,label)`
  - `plt.xlabel()`
  - `plt.ylabel()`
  - `plt.title()`
  - `plt.legend()`
- `nn.Module`
- `nn.Parameter`
- `def forward(self, tensor)`
- `torch.manual_seed()`
- `with torch.inference_mode()`
- `model.eval()`
- `model.state_dict()`
- `y_pred = model(x_test)`
- Loss functions
  - Mean Average Error (MAE): `torch.nn.L1Loss()`
  - Binary Cross Entropy (BCE) : `torch.nn.BCELoss()`
- Optimizers
  - Stochastic Gradient Descent (SGD): `torch.optim.SGD(params, lr)` [`lr` stands for learning rate]
  - Adam Optimizer: `torch.optim.Adam()`
- Steps in training loop
  - `y_pred = model(x_train)`
  - `loss = loss_fn(y_pred, y_train)`
  - `optimizer.zero_grad()`
  - `loss.backward()`
  - `optimizer.step()`
- Steps in testing loop
  - `test_pred = model(x_test)`
  - `testing_loss = loss_fn(test_pred, y_test.type(torch.float))`
  - `loss.detach().numpy()`
- `y_pred.detach()`
- Saving model
  - `MODEL_PATH = Path("directory_name")`
  - `MODEL_PATH.mkdir(parents = True, exist_ok = True)`
  - `MODEL_NAME = "model_name"`
  - `MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME`
  - `torch.save(obj = [model / state_dicts / tensors / objects], f = MODEL_SAVE_PATH`)
- Loading model
  - `loaded_model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))`
- `nn.Linear(in_features, out_features)`
- Training on GPU
  - `model.to(device)`
  - `x_train = x_train.to(device)`
  - similarly other parameters
  - `next(model.parameters()).device` To check the device in which model is loaded
  - `loss.cpu().detach().numpy()`
  - `y_pred.cpu().detach()`
</details>
