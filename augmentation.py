import pdb
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.decomposition

df = pd.read_csv('Observations_SPSS.csv')
df_onehot = pd.get_dummies(df, columns=['Entering Point (8 Direction)', 'Leaving Point(8 Direction)', 'Age (Child/Young/Middle/Old)', 'Hair color(Black/Blonde/Hat)'])

pca = sklearn.decomposition.PCA()
df_onehot_transform = pca.fit_transform(df_onehot)

rows, columns = df_onehot_transform.shape
n_keep = 30

with open('New_Observations_SPSS.csv', 'w') as fout:
	fout.write('Y(fast speed),Hour,"Day of the week (1 = Monday, 7 = Sunday)",Temperature (Fahrenheit),Entering Point (8 Direction),Leaving Point(8 Direction),Whether alone,Age (Child/Young/Middle/Old),Is Female,Hair color(Black/Blonde/Hat),Have backpack,Have handbag,Is Formal,Level of Crowded (3 Levels),Level of standing (1-3),With pet or not,With phone or not\n')
	for z in range(10):
		df_onehot_transform_noise = np.copy(df_onehot_transform)
		df_onehot_transform_noise[:, n_keep:] += np.random.normal(0, 1, (rows, columns-n_keep))

		df_new = pca.inverse_transform(df_onehot_transform_noise)


		Y = df_new[:, 0:1]
		Y = sklearn.preprocessing.binarize(Y, threshold=np.mean(Y)).astype(int).squeeze()

		hour = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.5, 23.499)).fit_transform(df_new[:, 1:2])).squeeze()
		day_of_week = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(0.5, 7.499)).fit_transform(df_new[:, 2:3])).squeeze()

		temperature = df_new[:, 3:4].squeeze()
		alone = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 4:5])).squeeze()
		female = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 5:6])).squeeze()
		backpack = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 6:7])).squeeze()
		handbag = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 7:8])).squeeze()
		formal = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 8:9])).squeeze()
		crowded = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(0.5, 3.499)).fit_transform(df_new[:, 9:10])).squeeze()
		standing = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(0.5, 3.499)).fit_transform(df_new[:, 10:11])).squeeze()
		pet = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 11:12])).squeeze()
		phone = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 12:13])).squeeze()

		entering = 1 + np.argmax(df_new[:, 13:21], axis=1).squeeze()
		leaving = 1 + np.argmax(df_new[:, 21:29], axis=1).squeeze()

		age_names = [x.split('_')[1] for x in df_onehot.columns[29:33]]
		age = [age_names[x] for x in np.argmax(df_new[:, 29:33], axis=1)]

		hair_names = [x.split('_')[1] for x in df_onehot.columns[33:36]]
		hair = [hair_names[x] for x in np.argmax(df_new[:, 33:36], axis=1)]

		for i in range(rows):
			print(i)
			fout.write(f'{Y[i]},{hour[i]},{day_of_week[i]},{temperature[i]},{entering[i]},{leaving[i]},{alone[i]},{age[i]},{female[i]},{hair[i]},{backpack[i]},{handbag[i]},{formal[i]},{crowded[i]},{standing[i]},{pet[i]},{phone[i]}\n')
