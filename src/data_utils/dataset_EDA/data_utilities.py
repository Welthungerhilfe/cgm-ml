def convert_age_from_days_to_years(age_in_days):
    """Convert age in days into age in years"""
    age_in_years = age_in_days['age'] / 365
    return round(age_in_years)


def extractqrcode(row):
    """Extract just the qrcode from them path"""
    return row['storage_path'].split('/')[1]


def draw_age_distribution(scans):
    age_ax = scans['Years'].value_counts().plot(kind='bar')
    age_ax.set_xlabel('age')
    age_ax.set_ylabel('no. of children')
    print(scans['Years'].value_counts())
