# Spleeter

Spleeter的PyTorch实现



## Dependencies

1. Python2/3
2. pip install -r requirements.txt



## Usage

### 训练步骤

1. python preprocess.py

   需要填写两个参数train_dataset和train_manifest

​	train_dataset: 训练集路径，默认目录结构为

	├── Dataset
	|		 |	 ├──song1
	| 	 |	 ├── mixture.wav
	|    |   ├── vocals.wav
	|    |   ├── instrumental.wav
	|    ├── song2
	|    |   ├── mixture.wav
	|    |   ├── vocals.wav
	|    |   ├── instrumental.wav
​	即一首歌一个文件夹，其中包含三段音频（混合，人声，背景音乐）

​	train_manifest: 包含歌曲信息，训练程序（train.py）使用，一行信息包括：混合音频路径，人声音频路径，音乐音频路径，持续时间，采样率

2. python train.py

   params中可对训练参数进行调节，其中：

   ​	train_manifest: 训练集manifest文件的路径

   ​	load_model: 可选三种参数： 

   ​					I.  “tensorflow”：从2stems中加载原模型后训练 

   ​					II.  “pytorch”：在已有的pytorch保存模型的基础上训练

   ​					III. None：初始化模型，训练



### 预测步骤

​	I.使用原模型分离，需要2stems, model.py, util.py, separator.py

​	II. 使用pytorch训练后的模型分离，需要final_model/net_vocal.pth, final_model/net_instru.pth, model.py, util.py, separator.py



分离程序封装在separator.py中的Separator类中，使用步骤为：

	1. from separator import Separator
 	2. sep = Separator(params(可选))
 	3. sep.separate(input_wav_path(分离音频路径，必须)， output_dir(音频输出路径，可选))
	4. sep.batch_separate(input_wav_files(批量处理音频，音频路径文件), output_dir)
