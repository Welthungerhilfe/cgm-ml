
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
```
python rgbd.py [-h] [--input inputpath] [--output outputpath] optional: [--mounted if using the input as the mounted datastore of qr codes]  [--w specifying number of workers]
```

For eg: 
```
python rgbd.py --input /mnt/cgmmlprod/cgminbmzprod_v5.0/ --output /mnt/preprocessed/rgbd56k/ --mounted --w 20
```

3. TODO add segmented rgbd



