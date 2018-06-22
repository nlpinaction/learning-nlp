# seq2seq
### 基于中文语料和dynamic_rnn的seq2seq模型
**需要 python3+ tensorflow-1.0**  
**由于tensorflow升级 本教程只适合tesorflow-1.0版本**  
   
对话语料分别在根目录下 Q.txt A.txt中，可以替换成你自己的对话语料。    
然后使用preprocessing.py自动化预处理。
### 用法:
    # 预处理
    python3 preprocessing.py
    # 训练
    python3 seq2seq.py train
    # 重新训练
    python3 seq2seq.py retrain
    # 预测
    python3 seq2seq.py infer
   
  
### 效果:
还不错，需要大规模数据集
