# Learning An Adaptive Sparse Transformer for Efficient Image Restoration (TPAMI 2025)

[Shihao Zhou](https://joshyzhou.github.io/), [Jinshan Pan](https://jspan.github.io/) and [Jufeng Yang](https://cv.nankai.edu.cn/)

<!-- #### News
- **Jul 02, 2025:** ASTv2 has been accepted to TPAMI 2025 :tada: 
<hr /> -->


## Training
### Derain
To train ASTv2 on SPAD, you can run:
```sh
./train.sh Deraining/Options/Deraining_ASTv2_spad.yml
```
### Dehaze
To train ASTv2 on SOTS, you can run:
```sh
./train.sh Dehaze/Options/RealDehazing_ASTv2_syn.yml
```
### Deblur
To train ASTv2 on GoPro, you can run:
```sh
./train.sh Motion_Deblurring/Options/Deblurring_ASTv2_L.yml
```
### Deshadow
To train ASTv2 on ISTD, you can run:
```sh
./train.sh Deshadow/Options/Deshadowing_ASTv2.yml
```
### Desnow 
To train ASTv2 on Snow100K, you can run:
```sh
./train.sh Desnow/Options/Desnowing_ASTv2.yml
```
### Enhancement 
To train ASTv2 on SMID, you can run:
```sh
./train.sh Enhancement/Options/AST_SMID.yml
```

To train ASTv2 on LoL-v2-syn, you can run:
```sh
./train.sh Enhancement/Options/AST_LOL_v2_synthetic.yml
```

To train ASTv2 on LoL-v2-real, you can run:
```sh
./train.sh Enhancement/Options/AST_LOL_v2_real.yml
```

## Evaluation
To evaluate ASTv2, you can refer commands in 'test.sh'

For evaluate on each dataset, you should uncomment corresponding line.


## Results
Experiments are performed for different image processing tasks including, rain streak removal, haze removal, motion blur removal, shadow removal, snow removal and low-light enhancement. 
Here is a summary table containing hyperlinks for easy navigation:
<table>
  <tr>
    <th align="left">Benchmark</th>
    <th align="center">Pretrained model</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">SPAD</td>
    <td align="center"><a href="https://pan.baidu.com/s/1MwO-ir1HXAk3mYhDxaAFOw?pwd=nsx2">(code:nsx2)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1KlJtMZwJghkBTsq1ILm53g?pwd=bycn">(code:bycn)</a></td>
  </tr>
  <tr>
    <td align="left">SOTS</td>
    <td align="center"><a href="https://pan.baidu.com/s/1Ml1EqSEHlFgfR0fRw2Mcsg?pwd=f5be">(code:f5be)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1bAKUPwDg2rNd8dthJwYcMA?pwd=dpaq">(code:dpaq)</a></td>
  </tr>
    <tr>
    <td align="left">ISTD</td>
    <td align="center"><a href="https://pan.baidu.com/s/1tKQPqXOOcw_JIExe1PGPig?pwd=5735">(code:5735)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/15LMwYkyIbWS1ToHfq2dh4Q?pwd=smn5">(code:smn5)</a></td>
  </tr>
    <tr>
    <td align="left">SNOW100K</td>
    <td align="center"><a href="https://pan.baidu.com/s/1ue2a8gOoOAY0iYkwBvDfPg?pwd=7n4q">(code:7n4q)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1iLLUdu-BYRjt0Df8d8skRg?pwd=rgi7">(code:rgi7)</a></td>
  </tr>
    <tr>
    <td align="left">SMID</td>
    <td align="center"><a href="https://pan.baidu.com/s/1th160j_f9S3hAcPHlX0s8w?pwd=e7j2">(code:e7j2)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/18n2Iyq0PRQvMuc54pHTP6g?pwd=z2x1">(code:z2x1)</a></td>
  </tr>
    <tr>
    <td align="left">LOL-v2-real </td>
    <td align="center"><a href="https://pan.baidu.com/s/1QGOxy-4K_o36bLXOlqyDKw?pwd=ehr7">(code:ehr7)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1OECBrqpD8eZ0OFK0TV578Q?pwd=xud3">(code:xud3)</a></td>
  </tr>
  <tr>
    <td align="left">LOL-v2-syn</td>
    <td align="center"><a href="https://pan.baidu.com/s/1hv7NYuQE_593MAS7dlIOWA?pwd=zknn">(code:zknn)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1BPRWZaX05cf7h2NrnIJ-Yg?pwd=5wby">(code:5wby)</a></td>
  </tr>
  <tr>
    <td align="left">GoPro</td>
    <td align="center"><a href="https://pan.baidu.com/s/1zozcN5XB0_7cSMPw5BaHEQ?pwd=budm">(code:budm)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1ZSofGz3KEbFvDxwQ3oRXJQ?pwd=frvi">(code:frvi)</a></td>
  </tr>
    <tr>
    <td align="left">RealBlur-J</td>
    <td align="center">/</td>
    <td align="center"><a href="https://pan.baidu.com/s/1cLzHEw2fBHC_upZYPPBWaQ?pwd=zqdw">(code:zqdw)</a></td>
  </tr>
    <tr>
    <td align="left">RealBlur-R</td>
    <td align="center">/</td>
    <td align="center"><a href="https://pan.baidu.com/s/1E3_lZFUGvi7biKivT9Blow?pwd=9pjq">(code:9pjq)</a></td>
  </tr>


</table>


<!-- ## Citation
If you find this project useful, please consider citing:

    @inproceedings{zhou_TPAMI25_astv2,
      title={Learning An Adaptive Sparse Transformer for Efficient Image Restoration (TPAMI 2025)},
      author={Zhou, Shihao and Pan, Jinshan and Yang, Jufeng},
      booktitle={TPAMI},
      year={2025}
    } -->

## Acknowledgement

This code borrows heavily from [Restormer](https://github.com/swz30/Restormer). 