import os
import pandas as pd

def getDf(path = '../resources/augmentation'):
  all_folder = os.listdir(path)
  df = pd.DataFrame(columns=['number_img', 'disease', 'disease_family', 'healthy', 'specie'], index=all_folder)

  for name_folder in all_folder:
    files = os.listdir(f"{path}/{name_folder}");
    name_splited = name_folder.split('___')
    df.loc[name_folder].specie = name_splited[0].lower()
    df.loc[name_folder].number_img = len(files)
    df.loc[name_folder].disease = name_splited[-1].lower()
    df.loc[name_folder].disease_family = df.loc[name_folder].disease.split('_')[-1].replace(')', '')
    df.loc[name_folder].healthy = name_splited[-1] == 'healthy'
  return df