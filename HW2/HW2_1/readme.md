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
