# Spleeter

Implementation of Spleeter by PyTorch


## Dependencies

1. Python2/3
2. pip install -r requirements.txt


## Usage

### Training steps

#### 1. python preprocess.py

   
 * Need to fill two parameters **train_dataset** and **train_manifest**
  
​	I. **train_dataset**: the PATH of training set
	The directory structure is recommonded to be:
	
	├── Dataset
	|    ├──song1
	|    |	 ├── mixture.wav
	|    |   ├── vocals.wav
	|    |   ├── instrumental.wav
	|    ├── song2
	|    |   ├── mixture.wav
	|    |   ├── vocals.wav
	|    |   ├── instrumental.wav

​	*This means one folder only contains one songs, including three audios (mixture, vocal and background music)*

​	II. **train_manifest**: contains song information, utilized by **train.py**.

#### 2. python train.py

   You can use **params** to adjust training parameters.
   
   **Notice:**

	train_manifest: the PATH of training manifest

	load_model: three optional variables, including:
   	
	I.  “tensorflow”：trains with the pre-trained model trained by tensorflow
	II.  “pytorch”：trains with the pre-trained model trained by PyTorch
	III. None：trains without a pre-trained model



### Prediction steps

**Notice**

	I. If you use the model in tensorflow, need 2stems, model.py, util.py, separator.py

	II. If you use the model in PyTorch, need final_model/net_vocal.pth, final_model/net_instru.pth, model.py, util.py, separator.py



#### Separation model is encapsulated in class **Separator** of **separator.py**：

	1. from separator import Separator
 	2. sep = Separator(params(optional))
 	3. sep.separate(input_wav_path(path of the target audio，MUST)， output_dir(output path of audios，optional))


### Reference
1.[Music Source Separation tool with pre-trained models / ISMIR2019 extended abstract] (http://archives.ismir.net/ismir2019/latebreaking/000036.pdf)

