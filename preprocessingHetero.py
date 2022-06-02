from preprocessing import *


def main():
	dfwave123 = load_data(df_path)
	df_wave4 = load_data(wave4_dfpath)
	df_wave5 = load_data(wave5_dfpath)
	dfTotal = pd.concat([dfwave123, df_wave4, df_wave5],axis=1)
	dfTotal = remove_rows(dfTotal)
	dfHomo, dfHetero = split(dfTotal)
	X1,Xnames1 = create_X(dfHomo)
	X2,Xnames2 = create_X(dfHetero)

	for f in Xnames1:
		X1[f] = X1[f].astype('category')
		X1[f] = X1[f].cat.add_categories(-1).fillna(-1)
	npX1 = np.array(X1)

	for f in Xnames2:
		X2[f] = X2[f].astype('category')
		X2[f] = X2[f].cat.add_categories(-1).fillna(-1)
	npX2 = np.array(X2)

	print 'The size of X1: '
	print npX1.shape

	print 'The size of X2: '
	print npX2.shape

	# obtain y
	dfHomo = create_label(dfHomo)
	y1 = np.array( dfHomo['y'] )

	dfHetero = create_label(dfHetero)
	y2 = np.array( dfHetero['y'] )

	return npX1, y1, Xnames1, X1, dfHomo['y'], npX2, y2, Xnames2, X2, dfHetero['y']



def split(df):
	#homosexual
	cond1 = (df['q4'].str.contains('female') & df['ppgender'].str.contains('female')) | ((df['q4'].str.strip() == 'male') & (df['ppgender'].str.strip() == 'male'))
	#hetero
	cond2 = (df['q4'].str.contains('female') & (df['ppgender'].str.strip() == 'male')) | (df['ppgender'].str.contains('female') & (df['q4'].str.strip() == 'male'))

	dfHomo = df[cond1]
	dfHetero = df[cond2]
	return dfHomo, dfHetero



def update_data(update):
	if update:
		npX1, npy1, Xnames1, X1, y1, npX2, npy2, Xnames2, X2, y2 = main()
		X1.to_csv('Dataset/X1.csv', index=False)
		y1.to_csv('Dataset/y1.csv', index=False, header=True)
		X2.to_csv('Dataset/X2.csv', index=False)
		y2.to_csv('Dataset/y2.csv', index=False, header=True)


main()
update_data(True)

