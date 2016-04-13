-- train

confusion = optim.ConfusionMatrix(classes)

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
    model:training()
    parameters, gradParameters = model:getParameters()
    shuffle = torch.randperm(trsize)

    local output
    for t = 1, trsize, opt.batchSize do
        -- disp progress
        if opt.progress then
            xlua.progress(t-1+opt.batchSize, trsize)
        end

        -- create mini batch
        local inputs = trainData.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
        local targets = trainData.labels:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())

        inputs = inputs:cuda()
        targets = targets:cuda()

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

            confusion:batchAdd(output, targets)

            -- normalize gradients and f(X)
            gradParameters:div(inputs:size()[1])
            f = f / inputs:size()[1]

            -- return f and df/dX
            return f, gradParameters
        end

        optimMethod(feval, parameters, optimState)
    end

    print(confusion)
    confusion:zero()
end