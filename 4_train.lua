-- train

confusion = optim.ConfusionMatrix(classes)

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
    local total_error = 0

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
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            
            -- forward、backword
            output = model:forward(inputs)
            local err = criterion:forward(output, targets)
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)

            total_error = total_error + err
            confusion:batchAdd(output, targets)

            -- normalize gradients and f(X)
            gradParameters:div(inputs:size()[1])
            err = err / inputs:size()[1]


            return err, gradParameters
        end

        optimMethod(feval, parameters, optimState)
    end

    print(confusion)
    confusion:zero()

    return total_error / trsize
end