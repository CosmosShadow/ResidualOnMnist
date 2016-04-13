-- 
dofile '0_libs.lua'

---------------------------------------------------------------------------------
print("==> Processing options")

--[[command line arguments]] --
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')

-- [[ thread, seed ]]
cmd:option('--threads', 1, 'set number of threads')
cmd:option('--seed', 123)

--[[ data ]] --
cmd:option('--dataset', 'mnist', 'which data to use: mnist')
cmd:option('--digits', 1, 'how many digits has the number')

--[[ model ]] --
cmd:option('--model', 'va', 'which model to use: cnn | va (visual_attention)')
cmd:option('--interval', 7, 'step interval count')
cmd:option('--outputLength', 7, 'step interval count')

-- [[ GPU ]] --
cmd:option('--cuda', true)
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')

--[[ loss ]] --
cmd:option('--loss', 'reinforce', 'type of loss function to minimize: nll | mse | margin | reinforce | multi_nll')

--[[ train ]] --
cmd:option('--save', 'testing', 'selecet subfolder where to store loggers')
cmd:option('--batchSize', 50)
cmd:option('--learningRate', 1e-1, 'setup the learning rate')
cmd:option('--opt.learningRateDecay', 0, 'setup the learning rate')
cmd:option('--momentum', 9e-1, 'setup the momentum')
cmd:option('--weightDecay', 0, 'weight decay')
cmd:option('--epochs', 100, 'define max number of epochs')
cmd:option('--preTrainEpochs', 50, 'pretrain the glimpse sensor')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- [[ plot ]]
cmd:option('--plot', false)
cmd:option('--progress', true)

cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)
-- cmd:log('logger.log', opt)

---------------------------------------------------------------------------------
print("==> Global setting")
-- thread, seed
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

---------------------------------------------------------------------------------
print("==> Loading scripts and model")

dofile '1_load_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

---------------------------------------------------------------------------------
-- set cuda
cutorch.setDevice(opt.useDevice)
model:cuda()
criterion:cuda()

---------------------------------------------------------------------------------
print('==> Uniform parameters')
if opt.uniform > 0 then
    for k, param in ipairs(model:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

---------------------------------------------------------------------------------
print("==> Training")
epoch = 0
while epoch <= opt.epochs do
	print('')
	print('==> epoch #'..epoch)
	epoch = epoch + 1
	train()
	test()
end






