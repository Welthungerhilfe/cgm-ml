
**RGBD Data Creation**
---

1.  Make sure input path is a list of qr codes. Each qr code should contain a folder of pcd files and a folder of rgb files. Mention an output path where you want the RGBD data to be created.

input dir tree structure should be:
```
qrcode
│      
│
└───qrcode1
│   │   
│   │
│   └───pc
│   │    │__.*pc 
│   │     
│   └───rgb 
│        |__.*png
│   
└───qrcode2
|    │   
│   │
│   └───pc
│   │    │__.*pc 
│   │     
│   └───rgb 
│        |__.*png
```

2. usage: 
If we are preparing the data with the --pickled flag we get pickled numpy arrays of the RGBD images with their labels. If you want to see how RGBD looks like for sample qrcodes [sample input path/qrcode], or for the data tagging/inspection tool, we use the below command without the --pickled flag.

```
python rgbd.py [-h] [--input inputpath] [--output outputpath] optional: [--pickled if labels of data is known and added to the label file]  [--num_workers specifying number of workers]
```

For eg: 
```
python rgbd.py --input /mnt/cgmmlprod/cgminbmzprod_v5.0/ --output /mnt/preprocessed/rgbd56k/ --pickled --w 20
```



