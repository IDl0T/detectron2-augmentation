# Get proposals

### extract proposals from `GeneralizedRCNN`  

    line 119: proposals, _ = self.proposal_generator(images, features, None)

`proposals`: List[`Instances`]

### Get corresponding 

Use pickle to extract proposals when inferencing  
dir: proposals/img{img No.}.pickle  
Then operate on proposals offline  

### Data Processing
1. check if the list has length 1
2. get field `proposal_boxes` from `proposal[0]`

