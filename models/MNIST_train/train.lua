local Trainer = torch.class('Trainer')

function Trainer:__init(model, criterion, opt)
  self.iter = 0
  self.params, self.grad_params = model:getParameters()
  self.opt = opt
  self.optim_state = {learningRate=opt.learning_rate, momentum=opt.momentum,weightDecay=opt.weightDecay}
  self.init_learning_rate = opt.learning_rate
  self.lr_decay_iter = opt.lr_decay_iter or -1
  self.max_iter = opt.max_iterations

  self.model = model
  self.criterion = criterion


  self.loader = MnistLoader('train', opt.batch_size, opt.num_labels, opt.norm_input)
  self.unlabeled_loader = MnistLoader('train', opt.batch_size_unlabel, -1, opt.norm_input)
  self.targets_unlabel = {}
  self.targets = {}

  self.loss_sum, self.dn_loss_sum = 0, 0
  self.loss_perIter, self.dn_loss_perIter=0, 0
  self.test_accu=0
  if opt.optim=='adam' then
      self.optimMethod=optim.adam
  else
      self.optimMethod=optim.sgd
  end
  self.confusion = optim.ConfusionMatrix(self.opt.num_class)
end

function Trainer:train()
  self.iter = self.iter + 1
  local i = self.iter
  local opt = self.opt
  local model, criterion = self.model, self.criterion

  local targets_unlabel = self.targets_unlabel
  local targets = self.targets

  local function feval()
    return self.criterion.output, self.grad_params
  end

  --local loss, dn_loss = nil, nil
  self.grad_params:zero()

  -------------------------------------------------------------
  -- Unlabled training
  local x_unlabeled = self.unlabeled_loader:next_batch(opt.gpuid)

  -- Compute clean activations (without noise)
  model:evaluate()
  for i = 1,#model.layer_sizes do
    model.bn_layers[i]:training()
  end
--print(model.bn_layers)
  local output = model:forward(x_unlabeled)

--print(output)
  -- dummy targets
  --targets[1] = torch.zeros(opt.batch_size):typeAs(x_unlabeled)
  targets_unlabel[1] = torch.ones(opt.batch_size_unlabel):typeAs(x_unlabeled)
  for i = 2,#output do
    targets_unlabel[i] = targets_unlabel[i] or output[i].new():resizeAs(output[i])
    targets_unlabel[i]:copy(output[i])
  end

--print(targets)
  for i = 0,#model.layer_sizes do
    targets_unlabel[#model.layer_sizes+3+i]:copy(output[2+i])
    if i >= 1 then
      if opt.flag_DBN==1 then 
       self.criterion.criterions[i+#model.layer_sizes+3]
        :setMeanStd(model.bn_layers[i].running_means,
                    model.bn_layers[i].running_projections)
      else
        self.criterion.criterions[i+#model.layer_sizes+3]
       :setMeanStd(model.bn_layers[i].save_mean,
                    model.bn_layers[i].save_std)
      end
     end
  end

  model:training()

  -- set criterion weights
  for i = 1,#criterion.weights do criterion.weights[i] = 0 end
  criterion.weights[#model.layer_sizes+3] = opt.unsup_weights[1] 
  criterion.weights[#model.layer_sizes+4] = opt.unsup_weights[2] 
  for i = 2,#model.layer_sizes do
    criterion.weights[#model.layer_sizes+3+i] = opt.unsup_weights[3]
  end

  
 --print(criterion.criterions[5].mean)
  local output_noisy = model:forward(x_unlabeled)
-- print(criterion.criterions[5].mean)
 -- print(output_noisy)
--  print(targets)
 -- print(criterion.criterions)
-- print(criterion.criterions[9].running_means)
self.dn_loss_perIter = criterion:forward(output_noisy, targets_unlabel)
  self.dn_loss_sum = self.dn_loss_sum + self.dn_loss_perIter
  local d = criterion:backward(output_noisy, targets_unlabel)
  model:backward(x_unlabeled, d)
--print(x_unlabeled)
  -------------------------------------------------------------
  -- Labeled training
  model:training()
  local x_labeled, y = self.loader:next_batch(opt.gpuid)
  local out = model:forward(x_labeled)

  for i = 1,#criterion.weights do criterion.weights[i] = 0 end
  criterion.weights[1] = 1
  
  targets[1]=torch.Tensor():resize(y:size()):copy(y)

  for i = 2,#out do
    targets[i] = targets[i] or out[i].new():resizeAs(out[i])
    targets[i]:copy(out[i])
  end
  self.loss_perIter = criterion:forward(out, targets)
  self.loss_sum = self.loss_sum + self.loss_perIter
  local d = criterion:backward(out, targets)
  model:backward(x_labeled, d)
  self.confusion:batchAdd(out[1], y)

  self.optimMethod(feval, self.params, self.optim_state)

  if i % self.opt.T_loss == 0 then
    local loss_mean = self.loss_sum / self.opt.T_loss
    self.loss_sum = 0

    dn_loss_mean = self.dn_loss_sum / self.opt.T_loss
    self.dn_loss_sum = 0

    print(('iteration %d: cls_loss = %f, denoise_loss = %f')
            :format(i, loss_mean, dn_loss_mean))
  end
  if i % opt.T_accu == 0 then
    self.confusion:updateValids()
    print('Train accuracy:', self.confusion.totalValid)
    local test_accuracy = test()
    self.test_accu=test_accuracy
  end
  if i * opt.batch_size % opt.num_labels == 0 then
    self.confusion:zero()
  end

  if self.lr_decay_iter > 0 and i > self.lr_decay_iter then
    local decay_rate = 1 - (i-self.lr_decay_iter) / (self.max_iter-self.lr_decay_iter)
    self.optim_state.learningRate =
      math.max(1e-8, self.init_learning_rate * decay_rate)
  end
end

return Trainer
