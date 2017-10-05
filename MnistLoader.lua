local MnistLoader, parent = torch.class('MnistLoader')

DATA_PATH = 'mnist.t7/'

function MnistLoader:__init(split, batch_size, num_labels)
  local num_labels = num_labels or -1
--  assert(split == 'train' or split == 'test', 'split must be train or test')
  assert(num_labels <= 0 or num_labels % 10 == 0, 'num labels must be divisible by 10')

  self.batch_size = batch_size
  if split=='train' or split=='test' then
    self.dataset = torch.load(('%s/%s_32x32.t7'):format(DATA_PATH, split), 'ascii')
  elseif split=='transfer' then
    dataset_train = torch.load(('%s/train_32x32.t7'):format(DATA_PATH), 'ascii')
    dataset_test = torch.load(('%s/test_32x32.t7'):format(DATA_PATH), 'ascii')
    self.dataset={}
    self.dataset.data=torch.ByteTensor(70000,1,32,32)
    self.dataset.labels=torch.ByteTensor(70000) 
   -- print(dataset_train.data:size())
   -- print(dataset_train.labels:size())
    self.dataset.data[{{1,60000},{},{},{}}]=dataset_train.data
    self.dataset.data[{{60001,70000},{},{},{}}]=dataset_test.data
    self.dataset.labels[{{1,60000}}]=dataset_train.labels
    self.dataset.labels[{{60001,70000}}]=dataset_test.labels
   end
  -- print('before')
  print(self.dataset.data:size())
  self.dataset.data = self.dataset.data[{{},{},{3,30},{3,30}}]:float()  --transform to 28x28
 -- print('after')
--  print(self.dataset.data:size())
  self.dataset.data:div(255)
  
  --print(self.dataset.labels)
----------smapling num_labels examples
  if num_labels > 0 then
    local selection = torch.LongTensor(num_labels)
   -- print(self.dataset.labels)
    for i = 1,10 do
      local idx = self.dataset.labels:eq(i):nonzero():squeeze():totable()
      --print(idx)
      selection[{{(i-1)*num_labels/10+1,i*num_labels/10}}] = 
        torch.Tensor(idx):index(1, torch.randperm(#idx)[{{1,num_labels/10}}]:long())
    end
    self.dataset.data = self.dataset.data:index(1, selection)
    self.dataset.labels = self.dataset.labels:index(1, selection)
   print('---------------label number---------------')
    print(self.dataset.labels:size())
  end

  local randperm = torch.randperm(self.dataset.data:size(1)):long()
  self.dataset.data = self.dataset.data:index(1, randperm):squeeze():float()
  self.dataset.labels = self.dataset.labels:index(1, randperm)
  self.num_batches = math.floor(self.dataset.data:size(1) / self.batch_size)

  self.batch_idx = 0
  self.num_class = 10
  self.num_labels = self.dataset.data:size(1)
  return self
end

function MnistLoader:reset()
  local randperm = torch.randperm(self.dataset.data:size(1)):long()
  self.dataset.data = self.dataset.data:index(1, randperm):squeeze()
  self.dataset.labels = self.dataset.labels:index(1, randperm)
  self.batch_idx = 0
end

function MnistLoader:next_batch(gpuid)
  if self.batch_idx == self.num_batches then
    self:reset()
  end
  self.batch_idx = self.batch_idx % self.num_batches + 1
  local idx1 = (self.batch_idx - 1) * self.batch_size + 1
  local idx2 = self.batch_idx * self.batch_size 
  local x = self.dataset.data:sub(idx1, idx2)
  local y = self.dataset.labels:sub(idx1, idx2)
  if gpuid >= 0 then
    x = x:cuda()
    y = y:cuda()
  end
  return x, y
end

return MnistLoader
