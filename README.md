# N-gram-Language-Model
N-gram language modeling tool using Python, 

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py bigram data/training.txt data/dev.txt --laplace`

## Evaluation

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |38.546738412790106|38.22621167306222|
|bigram          | unsmoothed |16.533090308407978|16.789588674738216|
|bigram          | Laplace    |18.99625433814741 |19.090801874848253|
|trigram         | unsmoothed |7.440230413263187 |8.022790772303466 |
|trigram         | Laplace    |19.970915431488635|20.348276377059364|                                          
