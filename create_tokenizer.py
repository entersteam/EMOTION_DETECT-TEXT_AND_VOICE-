import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

year_4_data = pd.read_csv("data/4차년도.csv", encoding="cp949")
year_5_data = pd.read_csv("data/5차년도.csv", encoding="cp949")
year_5_data_2 = pd.read_csv("data/5차년도_2차.csv", encoding="cp949")

data = pd.concat((year_4_data,year_5_data,year_5_data_2))

tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(data['발화문'])

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)