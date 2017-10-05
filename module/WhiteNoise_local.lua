local WhiteNoise_local, Parent = torch.class('nn.WhiteNoise_local', 'nn.Module')

function WhiteNoise_local:__init(mean, std)
   Parent.__init(self)
   -- std corresponds to 50% for MNIST training data std.
   self.mean = mean or 0
   self.std = std or 0.1
   self.noise = torch.Tensor()
end

function WhiteNoise_local:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train ~= false then
      self.noise:resizeAs(input)
      self.noise:normal(self.mean, self.std)
      self.output:add(self.noise)
   else
      if self.mean ~= 0 then
         self.output:add(self.mean)
      end
   end
   return self.output
end

function WhiteNoise_local:updateGradInput(input, gradOutput)
   if self.train ~= false then
      -- Simply return the gradients.
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   else
     -- error('backprop only defined while training')
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  end
   return self.gradInput
end

function WhiteNoise_local:__tostring__()
  return string.format('%s mean: %f, std: %f', 
                        torch.type(self), self.mean, self.std)
end
