require 'optim'
require 'module/NormMSE'
require 'MnistLoader'

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:option('-architecture', 'ladder', 'GPU ID (only using cuda)')
cmd:option('-model', 'sgd', 'GPU ID (only using cuda)')
cmd:option('-optim', 'adam', 'GPU ID (only using cuda)')
cmd:option('-num_labels', 100, 'number of labeled data')
cmd:option('-batch_size', 100, 'batch size')
cmd:option('-batch_size_unlabel', 100, 'batch size of unlabeled data')
cmd:option('-learning_rate', 0.0002, 'learning rate')
cmd:option('-momentum', 0.9, '')
cmd:option('-weightDecay', 0.0, '')
cmd:option('-comb_func', 'gaussian', 'combinator function g')

cmd:option('-lr_decay_iter', 50000, 'learning rate decay iter')
cmd:option('-max_iterations', 75000, 'number of training iteration')
cmd:option('-m_perGroup', 50, 'number of training iteration')
cmd:option('-T_loss', 100, 'number of training iteration')
cmd:option('-T_accu', 1000, 'number of training iteration')
cmd:option('-gpuid', -1, 'GPU ID (only using cuda)')
cmd:option('-seed', 1, 'GPU ID (only using cuda)')
cmd:option('-unsup_weights', "{1000,10,0.1}", 'GPU ID (only using cuda)')
cmd:option('-flag_DBN', 0, '0:NoDBN, 1:All_DBN; 2:Encode_DBN; -1: for decode dont use weight normalization')
cmd:option('-mode_nonlinear', 0, 'nonlinear--0:sigmoid;1:tanh;2:relu;3ELU')
cmd:option('-T_update', 1000, 'update T for SVD or NNN')
cmd:option('-epcilo', 0.01, 'used for NNN')
cmd:option('-configure_layer', "{1000,500,250,250,250}", '')
cmd:option('-connection_mode', "Identity", 'nil; Linear; Linear_IdInit')
cmd:option('-noise_level', 0.3, 'nil; Linear; Linear_IdInit')
cmd:option('-modelSave',0,'the initial value for BN scale')

opt = cmd:parse(arg)
opt.unsup_weights = tonumber(opt.unsup_weights) or loadstring('return '..opt.unsup_weights)()
opt.configure_layer = tonumber(opt.configure_layer) or loadstring('return '..opt.configure_layer)()
print(opt)
local Trainer=nil
if opt.architecture=='ladder' then
  require 'models/model_ladder'
  Trainer = require 'models/MNIST_train/train'
end

threadNumber=1
torch.setnumthreads(threadNumber)
torch.manualSeed(opt.seed)

opt.num_class=10

model, criterion = createLadderAE(opt)

if opt.gpuid >= 0 then
  require 'cunn'
cutorch.manualSeed(opt.seed)
--  cutorch.setDevice(opt.gpuid+1)
  model:cuda()
  criterion:cuda()
end


local trainer = Trainer.new(model, criterion, opt)

local test_loader = MnistLoader('test', opt.batch_size, -1)
function test()
  test_loader:reset()
  model:evaluate()
  local cfm = optim.ConfusionMatrix(opt.num_class)
  for t = 1,test_loader.num_batches do
    local x, y = test_loader:next_batch(opt.gpuid)
    local pred
    if string.match(opt.architecture,'supervise_only') then
      pred= model:forward(x)
    else
      pred= model:forward(x)[1]
    end
    
    cfm:batchAdd(pred, y)
  end
  cfm:updateValids()
  -- print('Test confusion matrix:')
-- print(cfm)
  return cfm.totalValid
end
losses={}
dn_losses={}
test_accus={}
start_time=torch.tic()

for i = 1,opt.max_iterations do
--  print('-----------------iteration:'..i..'------------------------------')
  trainer:train()
  if i % opt.T_loss==0 then
      losses[#losses+1]=trainer.loss_perIter
      dn_losses[#dn_losses+1]=trainer.dn_loss_perIter
  end
  if i % opt.T_accu==0 then
      test_accus[#test_accus+1]=trainer.test_accu
      print('---------------------------------test accuracy:'..test_accus[#test_accus])
  end
end
results={}
results.opt=opt
results.losses=losses
results.dn_losses=dn_losses
results.test_accus=test_accus


baseString=opt.architecture..'_'..opt.connection_mode..'_'..opt.model
..'_DBN'..opt.flag_DBN..'_G'..opt.m_perGroup..'_'..opt.optim..'_lr'..opt.learning_rate
..'_WD'..opt.weightDecay..'_'..opt.comb_func..'_nl'..opt.mode_nonlinear
..'_b'..opt.batch_size..'_NumLabel'..opt.num_labels
..'_LrDecay'..opt.lr_decay_iter..'_Iter'..opt.max_iterations..'_noise'..opt.noise_level


 if opt.modelSave==1 then
    torch.save('SModel_Semi_MNIST_'..baseString..'.t7', model:clearState())
  end

torch.save('result_MNIST_'..baseString
..'_UW'..opt.unsup_weights[1]..'_'..opt.unsup_weights[2]..'_'..opt.unsup_weights[3]..'_seed'..opt.seed..'.dat', results)
