# NCF
PyTorch Implementation of "Neural Collaborative Filtering" at WWW'17

I used the [ml-100k](https://grouplens.org/datasets/movielens/).

<br>

## 1. General Framework
Concatenate user and item embeddings, then pass through Neural CF Layers to predict score

![image](https://user-images.githubusercontent.com/59256704/220967111-d4311101-6ec2-48a3-bfa2-2688546eeb98.png)

<br>

## 2. Fusion of GMF and MLP (NeuMF)
GMF that applies a linear kernel to model the latent feature interactions, and MLP that uses a non-linear kernel to learn the interaction function from data

Fuse GMF and MLP under the NCF framework

![image](https://user-images.githubusercontent.com/59256704/220968820-14ca9f1a-f411-449e-81dd-0c53ea2d8b18.png)

<br>

## 3. Requirements
```Text
numpy==1.24.1
pandas==1.5.2
scikit_learn==1.2.1
scikit_surprise==1.1.3
surprise==0.1
torch==1.13.1
```

<br>

## 4. Example run
- set config.json
```json
{
  "seed": 417,
  "batch_size": 4096,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "test_size": 0.2,
  "epochs": 20,
  "save_dir": "./save_models",
  "arch": {
    "embedding_size": 128,
    "mlp_layer_dims": [64, 32, 16],
    "dropout_rate": 0.2,
    "use_gmf": true
  }
}
```

- run python code
```Bash
python train.py --config [config.json file path]
```
