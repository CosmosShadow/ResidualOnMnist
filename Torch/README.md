# ResidualOnMnist

在mnist数据库上，跑一下深度残差网络，学习一下下。   

#### Reference
[Deep Residual Learning for Image Recognition (paper)](http://arxiv.org/pdf/1512.03385v1.pdf)   
[Identity Mappings in Deep Residual Networks (paper)](http://arxiv.org/pdf/1603.05027.pdf)    
[Deep Residual Learning (PPT)](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)    
[deep-residual-networks (code)](https://github.com/KaimingHe/deep-residual-networks)   
[Training and investigating Residual Nets (post, code with Torch)](http://torch.ch/blog/2016/02/04/resnets.html)    
[https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)    

#### Usage

mnist数据库下载: [https://stife076.files.wordpress.com/2015/02/mnist4.zip](https://stife076.files.wordpress.com/2015/02/mnist4.zip)    
解压后放在/data/mnist文件下面。   
prepare_data.lua会预处理数据，并存储起来。   

#### 其它

博客为 [http://www.cosmosshadow.com/ml/神经网络/2016/03/16/深度残差学习.html](http://www.cosmosshadow.com/ml/神经网络/2016/03/16/深度残差学习.html)    
如果程序跑不通，请下载使用我写的lua相关的库: [MLLuaLib](https://github.com/CosmosShadow/MLLuaLib)