require 'nn'
require 'dpnn'
require 'nngraph'
require '../module/WhiteNoise_local'
require '../module/Linear_PN_EI'
local nninit = require 'nninit'


function createLadderAE(opt)
  local layer_sizes = opt.configure_layer
--  local layer_sizes = {layer_sizes[#layer_sizes]}
  local noise_level = opt.noise_level or 0.3
  local dropout = opt.dropout or 0.3
  layer_sizes[0] = 784
  local nonlinear
    if opt.mode_nonlinear==0 then  --sigmod
       nonlinear=nn.Sigmoid
     elseif opt.mode_nonlinear==1 then --tanh
      nonlinear=nn.Tanh
        elseif opt.mode_nonlinear==2 then --ReLU
        nonlinear=nn.ReLU
       elseif opt.mode_nonlinear==3 then --ReLU
       nonlinear=nn.ELU
     end



  -- Encoder
  local z, z_bn, z_noise = {}, {}, {}
  local input = nn.Identity()()
  z[0] = nn.Reshape(layer_sizes[0], true)(input)
  z_noise[0] = nn.WhiteNoise_local(0, noise_level)(z[0])
  prev_out = z_noise[0]

  local bn_layers = {}
  local SVD_layers={}
    local function getMul(sz, i) 
      if i == nil then return nn.CMul(sz) end
      return nn.CMul(sz):init('weight', nninit.constant, i)
    end
    local function getAdd(sz, i)
      if i == nil then return nn.Add(sz) end
      return nn.Add(sz):init('bias', nninit.constant, i)
    end
  
 
 for i = 1,#layer_sizes do
    local sz = layer_sizes[i]
   -- print(opt.model)
    if opt.model=='sgd' then
        print('------------------encode:sgd----------')
        z[i] = nn.Linear(layer_sizes[i-1], sz)(prev_out)
    elseif opt.model=='PN_EI' then
        print('------------------encode:PN_EI----------')
        SVD_layers[#SVD_layers+1]=nn.Linear_PN_EI(layer_sizes[i-1], sz) 
        z[i] = SVD_layers[#SVD_layers](prev_out)
    end
    
        bn_layers[i] = nn.BatchNormalization(sz, nil, nil, false)
    z_bn[i] = bn_layers[i](z[i])
    z_noise[i] = nn.WhiteNoise_local(0, noise_level)(z_bn[i])
    prev_out = nn.ReLU(true)(getMul(sz,1)(getAdd(sz,0)(z_noise[i])))
  end
  local y = nn.Linear(layer_sizes[#layer_sizes], opt.num_class)(prev_out)
  local y_bn = nn.BatchNormalization(opt.num_class)(y)
  local y_softmax = nn.SoftMax()(y_bn)

  -- Decoder
  local up_size = opt.num_class
  local up_layer = y_softmax
  local u, z_hat = {}, {}, {}, {}
  for i = #layer_sizes,0,-1 do
    local sz = layer_sizes[i]
    local decode_bn
        decode_bn=nn.BatchNormalization(sz, nil, nil, false)
    
    if opt.model=='sgd' then
        print('------------------decode:sgd----------')
        u[i] = decode_bn
            (nn.Linear(up_size, layer_sizes[i])(up_layer))
    elseif opt.model=='PN_EI' then
        print('------------------decode:PN_EI----------')
        SVD_layers[#SVD_layers+1]=nn.Linear_PN_EI(up_size, layer_sizes[i])
        u[i] = decode_bn
         (SVD_layers[#SVD_layers](up_layer))
    end


    local g

    if opt.comb_func == 'vanilla' then
      g = function(z_noise, u)
        local function AffineMul(sz, x, y)
          local xy = nn.CMulTable()({x, y})
          return getAdd(sz,0)(nn.CAddTable()({getMul(sz,1)(x), getMul(sz,0)(y), getMul(sz,0)(xy)}))
        end
        local a1 = AffineMul(sz, z_noise, u)
        local a2 = AffineMul(sz, z_noise, u)
        return nn.CAddTable()({a1, getMul(sz,1)(nonlinear()(a2))})
      end
    elseif opt.comb_func == 'vanilla-randinit' then
      g = function(z_noise, u)
        local function AffineMul(sz, x, y)
          local xy = nn.CMulTable()({x, y})
          return getAdd(sz)(nn.CAddTable()({getMul(sz)(x), getMul(sz)(y), getMul(sz)(xy)}))
        end
        local a1 = AffineMul(sz, z_noise, u)
        local a2 = AffineMul(sz, z_noise, u)
        return nn.CAddTable()({a1, getMul(sz)(nonlinear()(a2))})
      end
    elseif opt.comb_func == 'gaussian' then
      local function AddTwo(x, y) return nn.CAddTable()({x,y}) end
      local function SubTwo(x, y) return nn.CSubTable()({x,y}) end
      local function MulTwo(x, y) return nn.CMulTable()({x,y}) end
      local function Affine(sz, x) return getAdd(sz)(getMul(sz)(x)) end
      g = function (z_noise, u)
        local mu = AddTwo(Affine(sz, nonlinear()(Affine(sz, u))), getMul(sz)(u))
        local nu = AddTwo(Affine(sz, nonlinear()(Affine(sz, u))), getMul(sz)(u))
        return AddTwo(MulTwo(SubTwo(z_noise, mu), nu), mu)
      end
    elseif opt.comb_func == 'attention' then
      local function AddTwo(x, y) return nn.CAddTable()({x,y}) end
      local function MulTwo(x, y) return nn.CMulTable()({x,y}) end
      local function Affine(sz, x) return getAdd(sz)(getMul(sz)(x)) end
      g = function (z_noise, u)
        local mu = AddTwo(Affine(sz, nonlinear()(Affine(sz, u))), getMul(sz)(u))
         -- local xy = nn.CMulTable()({x, y})
        return nn.CMulTable()({z_noise, mu}) 
      end
    else
      error('unrecognized combinator function')
    end

    if opt.connection_mode== 'Identity' then
      print('---------------lateral connection--------')
        z_hat[i] = g(z_noise[i], u[i]) 
    elseif opt.connection_mode == 'Linear' then
      print('---------------Linear connection--------')
         local z_Map=nn.Linear(sz,sz)(z_noise[i])
          z_hat[i] = g(z_Map, u[i])
    elseif opt.connection_mode == 'SoftMax' then
      print('---------------SoftMax connection--------')
         local z_Map=nn.SoftMax()(z_noise[i])
          z_hat[i] = g(z_Map, u[i])
    else
      error('unrecognized connection mode')
    end

    up_size = sz
    up_layer = z_hat[i]
  end

  local outputs = {y_bn, z[0]}
  for i=1,#z_bn do table.insert(outputs, z_bn[i]) end
  for i=0,#z_hat do table.insert(outputs, z_hat[i]) end

  local model = nn.gModule({input}, outputs)

  model.layer_sizes = layer_sizes
  model.bn_layers = bn_layers
  model.SVD_layers = SVD_layers

  -- construct criterion
  local criterion = nn.ParallelCriterion()
  criterion:add(nn.CrossEntropyCriterion())

  -- dummy criterion for noisy output
  for i = 0,#layer_sizes do
    criterion:add(nn.MSECriterion(), 0) 
  end

  for i = 0,#layer_sizes do
    if opt.flag_DBN>=1 then
    criterion:add(nn.NormMSE_DBNCriterion())
    else
    criterion:add(nn.NormMSECriterion())
    end
  end
  return model, criterion
end


