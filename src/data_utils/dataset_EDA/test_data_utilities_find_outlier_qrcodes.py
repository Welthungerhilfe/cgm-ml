from data_utilities import find_outlier_qrcodes

# J: start preparing the dataFrame here and then test for agemin, agemax, weightmin, weightmax, heightmin, heightmax

#def prepare_df(df):
    df['qrcode'] = df.apply(extract_qrcode, axis=1) #maybe I can also use some extract qrcode fct here
    #df = df.groupby(['qrcode', 'scantype']).mean()
    #return df

def test_find_outlier_qrcodes_age_min():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p'],
        'age': [180, 365, 3000],
        'weight': [3.0, 6.0, 32.0],
        'height': [3.0, 6.0, 32.0],
    }
    df = pd.DataFrame.from_dict(data)
    #df = prepare_df(df)
    #df_out = calculate_performance(code='100', df_mae=df, result_config=RESULT_CONFIG)
    #assert (df_out[1.2] == 100.0).all()