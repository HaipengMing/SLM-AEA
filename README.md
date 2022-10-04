## A Simple Reproduction of Our Method

- Requirement

  Python 3.9, Torch 1.11.0 and torchvision 0.12.0 are recommended.

- Data

  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset).
  We have used pickle to make a list of filenames, and it have a structure as following.
  You can also make localized pickle files using our provided "pickleData.py".
```
 ./datasets/aligned_RAF/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```

- Train

```
   python main.py
```

