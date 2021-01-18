def convert_age_from_days_to_years(age_in_days):
    """Convert age in days into age in years"""
    age_in_years = age_in_days['age'] / 365
    return round(age_in_years)


def extractqrcode(row):
    """Extract just the qrcode from them path"""
    return row['storage_path'].split('/')[1]


def draw_age_distribution(scans):
    value_counts = scans['Years'].value_counts()
    age_ax = value_counts.plot(kind='bar')
    age_ax.set_xlabel('age')
    age_ax.set_ylabel('no. of children')
    print(value_counts)


def draw_sex_distribution(scans):
    value_counts = scans['sex'].value_counts()
    ax = value_counts.plot(kind='bar')
    ax.set_xlabel('gender')
    ax.set_ylabel('no. of children')
    print(value_counts)
