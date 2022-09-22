---
sort: 1
---

# Datasets
Our sadaco framework currently supports the datasets below. Here, we provide information on how to download and prepare each dataset.

## ICBHI 2017 Challenge dataset [[Link](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)]
ICBHI dataset is an audio dataset containing audio samples, collected independently by two research teams in two different countries, over several years. Most of the database consists of audio samples recorded by the School of Health Sciences, University of Aveiro (ESSUA) research team at the Respiratory Research and Rehabilitation Laboratory (Lab3R), ESSUA and at Hospital Infante D. Pedro, Aveiro, Portugal. The second research team, from the Aristotle University of Thessaloniki (AUTH) and the University of Coimbra (UC), acquired respiratory sounds at the Papanikolaou General Hospital, Thessaloniki and at the General Hospital of Imathia (Health Unit of Naousa), Greece. 

### Dataset Specification
#### Classes
 - Normal : Healthy breathing sound without any other symptoms.
 - Crackle : Crackle sound is a series of short, explosive sounds. They can also sound like bubbling, rattling, or clicking. You’re more likely to have them when you breathe in, but they can happen when you breathe out, too. <br><br>You can have fine crackles, which are shorter and higher in pitch, or coarse crackles, which are lower. Either can be a sign that there’s fluid in your air sacs. <br><br>They can be caused by:
   - Pneumonia
   - Heart disease
   - Pulmonary fibrosis
   - Cystic fibrosis
   - COPD
   - Lung infections, like bronchitis
   - Asbestosis, a lung disease caused by breathing in asbestos
   - Pericarditis, an infection of the sac that covers your heart
 - Wheeze : This high-pitched whistling noise can happen when you’re breathing in or out. It’s usually a sign that something is making your airways narrow or keeping air from flowing through them. <br><br> Two of the most common causes of wheezing are lung diseases called chronic obstructive pulmonary disease (COPD) and asthma. But many other issues can make you wheeze, too, including:
   - Allergies
   - Bronchitis or bronchiolitis
   - Emphysema
   - Epiglottitis (swelling of the top flap of your windpipe)
   - Gastroesophageal reflux disease (GERD)
   - Heart failure
   - Lung cancer
   - Sleep apnea
   - Pneumonia
   - Respiratory syncytial virus (RSV)
   - Vocal cord problems
   - An object stuck in your voice box or windpipe

#### <b>Data Count and Duration</b>
The database consists of a total of 5.5 hours of recordings containing 6898 respiratory cycles, of which 1864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

### Preparation
#### <b>Download</b>
 1. First download the data from the official download [link](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip)
 2. Unzip the file under desired <span style="color:purple"><b>$DATA_PATH</b></span>.
  ```
   File Hierarchy:
   $DATA_PATH
   └
  ```

 3.


## Fraiwan dataset [[Link]()]

## PASCAL Heartsound dataset [[Link]()]

