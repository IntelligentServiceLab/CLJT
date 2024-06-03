## CLJT: Web API Recommendation via Exploring Textual and Structural Semantics with Contrastive Learning and Joint Training
### Introduction
> To further improve the recommendation performance, this paper proposes an effective Web API recommendation approach via exploring textual and structural semantics with contrastive learning and joint training, named CLJT. 
### Environment Requirment
> This code has been tested running undeer Python 3.9.0
> The Required packages are as follows:
> - torch == 2.0.0+cu118
> - numpy == 1.24.1
> - seaborn == 0.13.2
> - transformers == 4.37.2
> - wheel == 0.41.2
> - tokenizers == 0.15.1
> - scipy == 1.12.0
> - scikit-learn == 1.3.0 

### Example to run CLJT
 - Command`python train.py`  
 - Train log:
>   开始训练    
    存在训练数据，正在加载     
    存在测试数据，正在加载    
    10%|▉         | 997/10000 [00:09<10:27, 103.24it/s]
   
NOTE : the duration of training and testing depends on the running environment.    
Train environment is on CPU AMD R5 5600x GPU RTX4060ti. 


### File Introduction
1. model.py
> This file contains the code of CLJT.
2. sanfm.py
> This file contains the code of sanfm.
3. utils.py
> This file contains the founction used in the item.
4. train.py
> This file is the model training file.
5. dataset.py
> This file contains the dataset loading code.
6. you need to download uncased-bert to the root of this item.

