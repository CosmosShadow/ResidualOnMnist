-- train

print '==> defining some tools'

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

if model then
    parameters, gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
    nesterov = true,
    dampening = 0
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

    epoch = epoch or 1
    local time = sys.clock()
    model:training()
    shuffle = torch.randperm(trsize)

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local output
    for t = 1, trsize, opt.batchSize do
        -- disp progress
        if opt.progress then
            xlua.progress(t-1+opt.batchSize, trsize)
        end

        -- create mini batch
        local inputs = trainData.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
        local targets = trainData.labels:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- estimate f
            output = model:forward(inputs)
            local err = criterion:forward(output, targets)
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)

            confusion:batchAdd(output[1][opt.interval], targets)

            -- normalize gradients and f(X)
            gradParameters:div(inputs:size()[1])
            f = f / inputs:size()[1]

            -- return f and df/dX
            return f, gradParameters
        end

        -- optimize on current mini-batch

        optimMethod(feval, parameters, optimState)
    end


    -- time taken
    time = sys.clock() - time
    time = time / trsize
    print("\n==> time to learn 1 sample = " .. (time * 1000) .. 'ms')

    -- confusion
    print(confusion)
    confusion:zero()
    -- next epoch
    epoch = epoch + 1
end