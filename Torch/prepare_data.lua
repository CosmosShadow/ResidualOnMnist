-- 准备mnist数据，并放大到32*32大小

require 'image'

---------------------------------------------------------------------------------
print("==> parameters")
source_channel = 1
source_width = 28
source_height = 28

output_channel = 1
output_width = 32
output_height = 32

---------------------------------------------------------------------------------
print("==> Loading data")

trainData, testData = {}, {}

-- train data
local temp = torch.load('data/mnist/train.th7', 'ascii')
trsize = temp[1]:size()[1]
trainData.data = torch.DoubleTensor(trsize, source_height, source_width, source_channel)
trainData.labels = torch.DoubleTensor(trsize)
for rec = 1, trsize do
    trainData.labels[rec] = (temp[2][rec] == 0 and 10 or temp[2][rec])
    trainData.data[rec] = temp[1][rec]
end

-- test data
local temp = torch.load('data/mnist/test.th7', 'ascii')
tesize = temp[1]:size()[1]
testData.data = torch.DoubleTensor(tesize, source_height, source_width, source_channel)
testData.labels = torch.DoubleTensor(tesize)
for rec = 1, tesize do
    testData.data[rec] = temp[1][rec]
    testData.labels[rec] = (temp[2][rec] == 0 and 10 or temp[2][rec])
end

---------------------------------------------------------------------------------
print("==> Preprocessing data")
trainData.data = trainData.data:transpose(2, 3):transpose(2, 4)
testData.data = testData.data:transpose(2, 3):transpose(2, 4)

---------------------------------------------------------------------------------
print("==> scale images")
-- train
train_data = torch.DoubleTensor(trsize, output_channel, output_height, output_width)
for i=1, trsize do
    train_data[i] = image.scale(trainData.data[i], output_width, output_height)
end
trainData.data = train_data
-- test
test_data = torch.DoubleTensor(tesize, output_channel, output_height, output_width)
for i=1, tesize do
    test_data[i] = image.scale(testData.data[i], output_width, output_height)
end
testData.data = test_data

---------------------------------------------------------------------------------
print("==> Preprocessing normalization")
local mean = trainData.data:mean()
local std = trainData.data:std()

trainData.data = trainData.data:add(-mean):div(std)
testData.data = testData.data:add(-mean):div(std)

---------------------------------------------------------------------------------
print('==> save data')
torch.save('data/trainData.t7', trainData)
torch.save('data/testData.t7', testData)

---------------------------------------------------------------------------------
print('==> Done')



