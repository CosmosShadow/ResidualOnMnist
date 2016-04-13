-- test function

function test()

    local time = sys.clock()

   model:evaluate()

    print('==> testing on test set:')

    local pred

    for t = 1, tesize, opt.batchSize do
        -- disp progress
        if opt.progress then
            xlua.progress(t-1+opt.batchSize, tesize)
        end
        
        local inputs = testData.data:narrow(1, t, opt.batchSize)
        local targets = testData.labels:narrow(1, t, opt.batchSize)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        pred = model:forward(inputs)
        confusion:batchAdd(pred, targets)
    end

    -- confusion
    print(confusion)
    confusion:zero()
end

