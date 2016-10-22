-- model

local function createModel()
	local nPreviousOutputPlane

	local Convolution = nn.SpatialConvolution
	local Avg = nn.SpatialAveragePooling
	local ReLU = nn.ReLU
	local Max = nn.SpatialMaxPooling
	local SBatchNorm = nn.SpatialBatchNormalization

	-- 层间直连
	local function shortcut(nInputPlane,  nOutputPlane,  stride)
		if nInputPlane ~= nOutputPlane then
			return nn.Sequential()
				:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
				:add(SBatchNorm(nOutputPlane))
		else
			return nn.Identity()
		end
	end

	-- 残差模块
	local function residualBlock(nOutputPlane, stride)
		local nInputPlane = nPreviousOutputPlane
		nPreviousOutputPlane = nOutputPlane

		local s = nn.Sequential()
		s:add(Convolution(nInputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1))
		s:add(SBatchNorm(nOutputPlane))
		s:add(ReLU(true))
		s:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
		s:add(SBatchNorm(nOutputPlane))

		return nn.Sequential()
			:add(nn.ConcatTable()
			:add(s)
			:add(shortcut(nInputPlane, nOutputPlane, stride)))
			:add(nn.CAddTable(true))
			:add(ReLU(true))
	end

	-- 堆叠残差模块
	local function stackResidualBlock(repeatCount, nOutputPlane, stride)
		local seq = nn.Sequential()
		for i=1, repeatCount do
			seq:add(residualBlock(nOutputPlane, i == 1 and stride or 1))
		end
		return seq
	end

	-- parameters
	stackDepth = 2
	nPreviousOutputPlane = 16

	-- model
	local model = nn.Sequential()
	model:add(Convolution(1, 16, 3, 3, 1, 1, 1, 1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(stackResidualBlock(stackDepth, 16, 1))
      model:add(stackResidualBlock(stackDepth, 32, 2))	--16*16
      model:add(stackResidualBlock(stackDepth, 64, 2))	--8*8
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))
      model:add(nn.LogSoftMax())

      return model
end

model = createModel()





