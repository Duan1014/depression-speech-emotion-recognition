# depression speech emotion recognition

This is my undergraduate project: Emotional recognition of depressive speech based on transfer learning.

The dataset is the DAIC-WOZ Depression speech Corpus and the audio part in MODMA.

First cut the sample into short 1-2s audio.

Then, opensmile is used to extract audio features, and the method of pre-training + fine-tuning is adopted for migration. 

Finally, the recognition accuracy of 73% can be achieved on the small sample data set MODMA.
