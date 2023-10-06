# Tree based Convolutional Neural Networks

TBCNN implemented in PyTorch based on the following paper:
["Convolutional Neural Networks over Tree Structures for Programming Language Processing" Lili Mou, et al.](https://arxiv.org/pdf/1409.5718.pdf)


## Steps
- [NOTE] Testing the model on our dataset
- [NOTE] Implementing the training script based PL

## How to run

## Tree-based CNN
#### Step 1 
```bash
python src/prepare.py --csv_file_path ./data/rdf4j/rdf4j_tree.csv --pickle_file_path ./data/rdf4j/rdf4j_tree.pickle --output_file_path ./data/rdf4j/java_algorithms.pkl
``` 

#### Step 2
```bash
python src/sampler_nodes.py --infile "./data/rdf4j/java_algorithms.pkl" --outfile "./data/rdf4j/java_algorithm_nodes.pkl" --limit -1 --per_node -1
``` 

#### Step 3
```bash
python src/sampler_trees.py --infile ./data/rdf4j/java_algorithms.pkl --outfile ./data/rdf4j/java_algorithm_trees.pkl --label_key "value" --minsize 100 --maxsize 2000 --test 15
``` 

#### Step 4
```bash
python src/vectorizer_ast2vec_node_trees.py --infile ./data/rdf4j/java_algorithm_nodes.pkl --vectors_outfile ./data/rdf4j/java_algorithm_vectors.pkl --net_outfile ./data/rdf4j/java_algorithm_net.pth
``` 

#### Step 5
```bash
python src/train_tbcnn.py --infile ./data/rdf4j/java_algorithm_trees.pkl --embedfile ./data/rdf4j/java_algorithm_vectors.pkl --net_outfile ./checkpoints/tbcnn2/
``` 


### Code2vec

#### Step 1
```bash
python src/code2vec.py --load ./data/models/java14_model/saved_model_iter8.release --predict --export_code_vectors --in_dir ./data/java/rdf4j_codes/ --out_dir ./data/java/code2vec.pkl
```

#### Step 2
```bash
python src/train_code2vec.py --infile ./data/java/code2vec.pkl --net_outfile ./checkpoints/code2vec/
```
