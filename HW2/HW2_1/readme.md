## Directory

```
|--HW2_1
|  |--MLDS_hw2_1_data                 // data directory
|  |--create_dict.py                  // create dictionary
|  |--dataset.py                      // dataset for pytorch
|  |--net.py                          // Seq2Seq video caption model
|  |--train.py                        // training code
|  |--test.py                         // testing code
|  |--hw2_seq2seq.sh                  // bash command
```

How to run: `bash hw2_seq2seq.sh MLDS_hw2_1_data/testing_data MLDS_hw2_1_data/test.txt`, the prdiction will be saved to the test.txt file, and BLEU score will be printed out. 

>Note: In the bash script, I use the `bleu_eval.py` to compute bleu score. \
`python ./MLDS_hw2_1_data/bleu_eval.py "$var2"`

>But you need to change the groundtruth path in `bleu_eval.py` line 107. Since the bash scipt is running in the parent folder, we need to change it to: \
 `test = json.load(open('./MLDS_hw2_1_data/testing_label.json','r'))`, 