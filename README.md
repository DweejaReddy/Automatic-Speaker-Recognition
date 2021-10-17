# Automatic-Speaker-Recognition
Speaker Recognition is the problem of identifying a speaker from a recording of their speech sample. It is an important topic in Signal Processing and has a variety of applications, especially in security systems. Voice controlled devices also rely heavily on speaker recognition.

 The modules I used to do this project are **NumPy, SciPy and Matplotlib** that have a major area of coverage in building appplications of Signal Processing and plotting them

The main principle behind Speaker Recognition is extraction of features from speech followed by training on a data set and testing.
While doing this project, I mainly got the opportunity to get indroduced to the basics of Digital Signal Processing, Feature extraction using two different algorithms **(MFCC and LPC)**,Feature Matching **(LBG)**

<br>

**STEP1:FEATURE EXTRACTION (MFCC-Mel Frequency Cepstral Coefficients):** <br>
1.Human hearing as expected is not linear in nature rather it is logarithmic. Our ears act as a filter.<br>
2.Most popular MFCC's are based on the known variation of the human ear’s critical bandwidths with frequency.
Filters spaced linearly at low frequencies and logarithmically at high frequencies have been used to
capture the important characteristics of speech. This is expressed in the mel-frequency
scale.<br>
3.The speech signal is divided into frames of 25ms with an overlap of 10ms.<br> 
4.Each frame is multiplied with a Hamming window.<br>
5.The periodogram of each frame of speech is calculated by first doing an FFT of 512 samples
on individual frames, then taking the power spectrum<br>
6.The entire frequency range is divided into ‘n’ Mel filter banks(12 here), which is also the number of
coefficients we want.<br>
7.Then filterbank energies are calculated by multiplying the each filter bank with power spectrum and add up the coefficients.<br>
8.Finally, applying discrete cosine transform on logarithm of these distinct 'n' energies give MFCCs.<br><br>

**STEP2:FEATURE EXTRACTION (LPC-Linear Prediction Coefficients):**<br>
1.LPCs are also the popular technique of feature Extraction. It is based on the AutoRegressive Model of the speech.<br>
2.In this extraction also, the signal is framed same as mentioned in MFCCs.<br>
3.To estimate the LPC coefficients, we use the Yule-Walker Equations which uses Auto-correlation function.<br>

**STEP3:FEATURE MATCHING (LBG-Linde-Buzo-Gray):**<br>
1.Generally, the main approach of Feature Matching is mapping vectors from a large vector space to a finite number of regions in that
space. Each region is called a cluster and can be represented by its center called a codeword. The
collection of all codewords is called a codebook.<br>
2.A vector codebook is designed which is the centroid of entire set of training vectors.<br>
3.Now, the codebook size is doubled by splitting the current one and closest codeword is searched for every training vector and assigned as centroid in next iteration.<br>
4.This iterations carry out until vector distortion for current iteration falls below a certain value.<br><br>

**STEP4:TRAINING:**<br>
# Automatic-Speaker-Recognition
Speaker Recognition is the problem of identifying a speaker from a recording of their speech sample. It is an important topic in Signal Processing and has a variety of applications, especially in security systems. Voice controlled devices also rely heavily on speaker recognition.

The modules I used to do this project are **NumPy, SciPy and Matplotlib** that have a major area of coverage in building appplications of Signal Processing and plotting them

The main principle behind Speaker Recognition is extraction of features from speech followed by training on a data set and testing.
While doing this project, I mainly got the opportunity to get indroduced to the basics of Digital Signal Processing, Feature extraction using two different algorithms **(MFCC and LPC)**,Feature Matching **(LBG)**

<br>

**STEP1:FEATURE EXTRACTION (MFCC-Mel Frequency Cepstral Coefficients):** <br>
* Human hearing as expected is not linear in nature rather it is logarithmic. Our ears act as a filter.<br>
* Most popular MFCC's are based on the known variation of the human ear’s critical bandwidths with frequency. This is expressed in the mel-frequency scale.<br>
* The speech signal is divided into frames of 25ms with an overlap of 10ms and multiplied with a hamming window.<br> 
* The periodogram of each frame of speech is calculated by first doing an FFT of 512 sampleson individual frames, then taking the power spectrum<br>
* The entire frequency range is divided into ‘n’ Mel filter banks(12 here)<br>
* Then filterbank energies are calculated by multiplying the each filter bank with power spectrum and add up the coefficients.<br>
* Finally, applying discrete cosine transform on logarithm of these distinct 'n' energies give MFCCs.<br><br>

**STEP2:FEATURE EXTRACTION (LPC-Linear Prediction Coefficients):**<br>
* LPCs are also the popular technique of feature Extraction. It is based on the AutoRegressive Model of the speech.<br>
* In this extraction also, the signal is framed same as mentioned in MFCCs.<br>
* To estimate the LPC coefficients, we use the Yule-Walker Equations which uses Auto-correlation function.<br>

**STEP3:FEATURE MATCHING (LBG-Linde-Buzo-Gray):**<br>
* Generally, the main approach of Feature Matching is mapping vectors from a large vector space to a finite number of regions in that space. Each region is called a cluster and can be represented by its center called a codeword. The collection of all codewords is called a codebook.<br>
* A vector codebook is designed which is the centroid of entire set of training vectors.<br>
* Now, the codebook size is doubled by splitting the current one and closest codeword is searched for every training vector and assigned as centroid in next iteration.<br>
* This iterations carry out until vector distortion for current iteration falls below a certain value.<br><br>

**STEP4:TRAINING:**<br>
* The small dataset contains 8 speakers and 16 centroids.<br>
* Now dataset needs to be trained to derive codebook for each speaker.<br>
* The datset goes through all the extractions of MFCC and LPC and givies out a codebook of mfcc and lpc.<br><br>

**STEP5:TESTING:**<br>
* Now, All the features of each speech signal must be compared with the codebooks of training files.<br>
* The results would be obtained for the testing.<br>

# OUTPUT

**SMALL DATSET:**<br>
![](https://i.imgur.com/paR6RZb.png)

Speaker **1**  in test matches with speaker  **4** in train for training with MFCC<br>
Speaker **1**  in test matches with speaker  **8** in train for training with LPC <br>

Speaker **2**  in test matches with speaker  **2** in train for training with MFCC<br>
Speaker **2**  in test matches with speaker  **8** in train for training with LPC<br>

Speaker **3**  in test matches with speaker  **3** in train for training with MFCC<br>
Speaker **3**  in test matches with speaker  **8** in train for training with LPC<br>

Speaker **4**  in test matches with speaker  **4** in train for training with MFCC<br>
Speaker **4**  in test matches with speaker  **8** in train for training with LPC<br>

Speaker **5**  in test matches with speaker  **5** in train for training with MFCC<br>
Speaker **5**  in test matches with speaker  **8** in train for training with LPC<br>

Speaker **6**  in test matches with speaker  **6** in train for training with MFCC<br>
Speaker **6**  in test matches with speaker  **8** in train for training with LPC<br>

Speaker **7**  in test matches with speaker  **7** in train for training with MFCC<br>
Speaker **7**  in test matches with speaker **8** in train for training with LPC<br>

Speaker **8**  in test matches with speaker  **8** in train for training with MFCC<br>
Speaker **8**  in test matches with speaker  **8** in train for training with LPC<br><br>
# Accuracy for small dataset:
Accuracy of result for training with MFCC is  87.5 %<br>
Accuracy of result for training with LPC is  12.5 %<br>

# Accuracy for large dataset:
![](https://i.imgur.com/EqQ1tqm.png)

Accuracy of result for training with MFCC is  87.2340425531915 %<br>
Accuracy of result for training with LPC is  2.127659574468085 %<br>
