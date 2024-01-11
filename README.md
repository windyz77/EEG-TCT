Ref.: Ms. No. NEUCOM-D-22-05632R1
Temporal-Channel Cascaded Transformer for Imagined Handwriting Character Recognition
Neurocomputing

#### **Train Requirements**

pip install -r requirements.txt

#### **Datasets and Checkpoints**

data/Character_imagine/character_imagine_1-process_10-26.mat

The dataset used in this paper is a single-character handwriting-imagination dataset, which is a subset of the Handwriting BCI dataset publicly released by Willett et al.[1]

TCT/IC_checkpoints_512/EEGImaginedCharacter_Transformer_435_99.4482_95.7480_weights.pth

#### **Train/Test**

python TCT/main.py

[1] Willett F R , Avansino D T , Hochberg L R ,et al.High-performance brain-to-text communication via imagined handwriting[J].Cold Spring Harbor Laboratory, 2020.DOI:10.1101/2020.07.01.183384.
