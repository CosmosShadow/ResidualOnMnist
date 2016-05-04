-- 

function load_method()
	optimState = {
		learningRate = 0.1,
		learningRateDecay = 1e-6,
		weightDecay = 0,
		momentum = 0.9,
		nesterov = true,
		dampening = 0,
		-- own
		stage_length = 3,
		stage_decay = 0.1
	}
	optimMethod = optim.sgd

	-- optimState = {
	-- 	stepsize = 0.01,
	-- 	etaplus = 1.1,
	-- 	etaminus = 0.8,
	-- 	stepsizemax = 50,
	-- 	stepsizemin = 1e-8,
	-- 	niter = 1
	-- }
	-- optimMethod = optim.rprop
end

