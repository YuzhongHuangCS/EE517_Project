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
df_onehot['Day of the week (1 = Monday, 7 = Sunday)'] = sklearn.preprocessing.binarize(df_onehot['Day of the week (1 = Monday, 7 = Sunday)'].values.reshape(-1, 1), threshold=4)
pca = sklearn.decomposition.PCA()
df_onehot_transform = pca.fit_transform(df_onehot)

rows, columns = df_onehot_transform.shape
n_keep = 30

n_need = 0
n_have = 0
db_datetime = {}
# meta data structure: n, crowded, temperature
db_datetime_meta = {}
for day_of_week in range(1, 8):
	db_datetime[day_of_week] = {}
	db_datetime_meta[day_of_week] = {}
	for hour in range(0, 24):
		if hour in (7, 20):
			n = np.random.random_integers(10, 15)
		elif hour in (21, 22):
			n = np.random.random_integers(8, 12)
		elif hour in (23, 0, 1, 2):
			n = np.random.random_integers(5, 10)
		elif hour in (3, 4, 5, 6):
			n = np.random.random_integers(1, 7)
		else:
			n = 15
		n_need += n
		db_datetime[day_of_week][hour] = []
		db_datetime_meta[day_of_week][hour] = [n]

with open('parameters.csv') as fin:
	fin.readline()
	for line in fin:
		parts = line.rstrip().split(',')
		day_of_week = int(parts[0])
		hour = int(parts[1])
		crowded = int(parts[2])
		temperature = float(parts[3])
		db_datetime_meta[day_of_week][hour].append(crowded)
		db_datetime_meta[day_of_week][hour].append(temperature)

for row in df.values:
	hour = row[1]
	day_of_week = row[2]
	db_datetime[day_of_week][hour].append(list(row))
	n_have += 1
	if len(db_datetime[day_of_week][hour]) > db_datetime_meta[day_of_week][hour][0]:
		diff = len(db_datetime[day_of_week][hour]) - db_datetime_meta[day_of_week][hour][0]
		n_need += diff

variance = 0.1
n_need = 1378
while n_have < n_need:
	#print(n_have)
	df_onehot_transform_noise = np.copy(df_onehot_transform)
	df_onehot_transform_noise[:, n_keep:] += np.random.normal(0, variance, (rows, columns-n_keep))

	df_new = pca.inverse_transform(df_onehot_transform_noise)

	Y = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 0:1])).squeeze()
	hour = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.5, 23.499)).fit_transform(df_new[:, 1:2])).squeeze()
	day_of_week = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 2:3])).squeeze()

	#temperature = df_new[:, 3:4].squeeze()
	alone = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 4:5])).squeeze()
	female = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 5:6])).squeeze()
	backpack = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 6:7])).squeeze()
	handbag = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 7:8])).squeeze()
	formal = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 8:9])).squeeze()
	#crowded = np.around(sdf_new[:, 9:10]).squeeze()
	standing = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(0.5, 3.499)).fit_transform(df_new[:, 10:11])).squeeze()
	pet = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 11:12])).squeeze()
	phone = np.around(sklearn.preprocessing.MinMaxScaler(feature_range=(-0.499, 1.499)).fit_transform(df_new[:, 12:13])).squeeze()

	entering = 1 + np.argmax(df_new[:, 13:21], axis=1).squeeze()
	leaving = 1 + np.argmax(df_new[:, 21:29], axis=1).squeeze()

	age_names = [x.split('_')[1] for x in df_onehot.columns[29:33]]
	age = [age_names[x] for x in np.argmax(df_new[:, 29:33], axis=1)]

	hair_names = [x.split('_')[1] for x in df_onehot.columns[33:36]]
	hair = [hair_names[x] for x in np.argmax(df_new[:, 33:36], axis=1)]

	start_have = n_have
	#pdb.set_trace()
	for i in range(rows):
		if Y[i] in (0, 1) and hour[i] in list(range(0, 24)) and day_of_week[i] in (0, 1) and alone[i] in (0, 1) and female[i] in (0, 1) and backpack[i] in (0, 1) and handbag[i] in(0, 1) and formal[i] in (0, 1) and standing[i] in list(range(1, 4)) and pet[i] in (0, 1) and phone[i] in (0, 1):
			if day_of_week[i] == 1:
				if len(db_datetime[6][hour[i]]) < len(db_datetime[7][hour[i]]):
					new_day_of_week = 6
				else:
					new_day_of_week = 7
			else:
				nums = [len(db_datetime[z][hour[i]]) for z in range(1, 6)]
				new_day_of_week = np.argmin(nums) + 1
			print(new_day_of_week, hour[i], len(db_datetime[new_day_of_week][hour[i]]), db_datetime_meta[new_day_of_week][hour[i]][0], n_have, n_need, variance)
			pm_one = [1, -1]
			pm_two = [2, -2]
			pm_three = [3, -3]
			np.random.shuffle(pm_one)
			np.random.shuffle(pm_two)
			np.random.shuffle(pm_three)
			deltas = [1] + pm_one + pm_two + pm_three
			for hour_delta in deltas:
				new_hour = max(min(hour[i] + hour_delta, 23), 0)
				if len(db_datetime[new_day_of_week][new_hour]) < db_datetime_meta[new_day_of_week][new_hour][0]:
					crowded = db_datetime_meta[new_day_of_week][new_hour][1]
					temperature = db_datetime_meta[new_day_of_week][new_hour][2]
					new_age = age[i]
					if new_age ==  'Child':
						if new_hour > 20 or new_hour < 8:
							if alone[i] == 1:
								new_age = 'Young'
							else:
								if np.random.uniform() < 0.9:
									new_age = 'Young'
						else:
							if alone[i] == 1:
								if np.random.uniform() < 0.9:
									new_age = 'Young'
							else:
								if np.random.uniform() < 0.8:
									new_age = 'Young'

					if new_age ==  'Old':
						if np.random.uniform() < 0.9:
							new_age = 'Young'

					new_pet = pet[i]
					if new_pet == 1:
						if np.random.uniform() < 0.95:
							new_pet = 0

					row = [Y[i], new_hour, new_day_of_week, temperature, entering[i], leaving[i], alone[i], new_age, female[i], hair[i], backpack[i], handbag[i], formal[i], crowded, standing[i], new_pet, phone[i]]
					db_datetime[new_day_of_week][new_hour].append(row)
					n_have += 1
					break

	end_have = n_have
	if start_have == end_have:
		if variance < 10:
			variance += 0.01

with open('New_Observations_SPSS.csv', 'w') as fout:
	fout.write('Y(fast speed),Hour,"Day of the week (1 = Monday, 7 = Sunday)",Temperature (Fahrenheit),Entering Point (8 Direction),Leaving Point(8 Direction),Whether alone,Age (Child/Young/Middle/Old),Is Female,Hair color(Black/Blonde/Hat),Have backpack,Have handbag,Is Formal,Level of Crowded (3 Levels),Level of standing (1-3),With pet or not,With phone or not\n')
	for day_of_week, hours in db_datetime.items():
		for hour, rows in hours.items():
			for row in rows:
				fout.write(','.join([str(x) for x in row]) + '\n')
