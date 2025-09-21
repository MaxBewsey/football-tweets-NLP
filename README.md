# Football Tweet Classification Using Supervised Learning and NLP

## Overview
This project applies supervised machine learning to classify tweets from London as either football-related or not. Tweets were weakly labelled using domain-specific keywords, with a manually labelled gold-standard set held out for evaluation. Three models were trained and compared: Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).

## Key Features
- Cleaned and preprocessed raw tweet data with regex-based text cleaning  
- Applied **weak supervision** to generate labels using football-specific keywords  
- Held out a **gold-standard labelled set** for model evaluation  
- Vectorised tweets using TF-IDF  
- Trained and evaluated three classifiers:
  - Naive Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  

## Methods & Tools
- **Text Processing:** regex, TF-IDF vectorisation  
- **Models:** Naive Bayes, Logistic Regression, SVM (scikit-learn)  
- **Evaluation:** confusion matrices, precision, recall, F1 score  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib  

## Repo Structure
- data/
  - london
- notebooks/
  - Football Tweet Classification Using Supervised Learning and NLP

## Results
- **Naive Bayes** achieved the best balance of precision and F1 score  
- All models struggled with recall due to **dataset imbalance** (few football-related tweets)  
- Weak supervision and traditional classifiers showed potential but also key limitations for noisy, real-world text  

## Future Work
- Incorporate additional contextual features (timestamps, hashtags, user mentions)  
- Extend vectorisation with **n-grams** to capture football-specific phrases (e.g., "red card", "kick off")  
- Explore methods to address class imbalance (e.g. resampling, alternative labelling strategies)  
