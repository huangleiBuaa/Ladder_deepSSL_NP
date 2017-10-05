require 'nn'
--require 'dpnn'
--require 'nngraph'
--local nninit = require 'nninit'

local nmse, parent = torch.class('nn.NormMSECriterion', 'nn.Criterion')
function nmse:__init()
  parent.__init(self)
  self.mean = nil
  self.std = nil
  self.mse = nn.MSECriterion()
end
function nmse:setMeanStd(mean, std)
  self.mean = mean:view(1, -1):clone()
  self.std = std:view(1, -1):clone()
end
-- target is clean, input is noisy
function nmse:updateOutput(input, target)
  if self.mean and self.std then
    self.input_norm = (input - self.mean:expand(input:size())):cdiv(self.std:expand(input:size()))
    self.output = self.mse:updateOutput(self.input_norm, target)
  else
    self.output = self.mse:updateOutput(input, target)
  end
  return self.output
end
function nmse:updateGradInput(input, target)
  if self.mean and self.std then
    self.gradInput = self.mse:updateGradInput(self.input_norm, target)
    self.gradInput:cdiv(self.std:expand(input:size())) -- bug??
  else
    self.gradInput = self.mse:updateGradInput(input, target)
  end
  return self.gradInput
end
function nmse:type(...)
  parent.type(self, ...)
  self.mse:type(...)
end

