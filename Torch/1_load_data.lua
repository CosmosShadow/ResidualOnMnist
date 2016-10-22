
-- classes
classes = { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }
WIDTH = 28
HEIGHT = 28
DATA_N_CHANNEL = 1
ninputs = WIDTH * HEIGHT

---------------------------------------------------------------------------------
print("==> Loading data")

trainData = torch.load('data/trainData.t7')
testData = torch.load('data/testData.t7')
trsize = trainData.data:size()[1]
tesize = testData.data:size()[1]