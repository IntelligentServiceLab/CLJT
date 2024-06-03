# CLJT: Web API Recommendation via Exploring Textual and Structural Semantics with Contrastive Learning and Joint Training
### Introduction
> To further improve the 
recommendation performance, this paper proposes an effective 
Web API recommendation approach via exploring textual and 
structural semantics with contrastive learning and joint training, 
named CLJT. 

### Environment Requirment
> This code has been tested running undeer Python 3.9.0
> The Required packages are as follows:
> - torch == 2.0.0+cu118
> - numpy == 1.24.1
> - seaborn == 0.13.2
> - transformers ==4.37.2
> - wheel ==0.41.2

### Example to run CLJT
 - Command`python train.py`  
 - log
 - >    开始训练    
存在训练数据，正在加载     
存在测试数据，正在加载    
 10%|▉         | 997/10000 [00:09<10:27, 103.24it/s]
   
NOTE : the duration of training and testing depends on the running environment.


### File Introduction
1. model.py 
2. sanfm.py
3. utils.py
4. train.py
5. dataset.py
6. you need to download uncased-bert to the root og this item

