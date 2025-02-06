import numpy as np
import pandas as pd
from faker import Faker
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib 

def generate_synthetic_data(num_samples):
    fake = Faker()
    
    # Lists to store the generated data
    accountnum, buildingnum, city, creditcard, dob, driverlicense, email, givenname = [], [], [], [], [], [], [], []
    idcard, password, socialnum, street, surname, taxnum, phone, username, zipcode = [], [], [], [], [], [], [], [], []
    
    for _ in range(num_samples):
        accountnum.append(np.random.randint(100000000, 999999999))  # Random account number
        buildingnum.append(np.random.randint(1, 999))  # Random building number
        city.append(fake.city())  # Random city name
        creditcard.append(np.random.randint(1000000000000000, 9999999999999999))  # Random credit card number
        dob.append(fake.date_of_birth(minimum_age=18, maximum_age=100).strftime('%Y-%m-%d'))  # Random DOB
        driverlicense.append(np.random.randint(1000000000, 9999999999))  # Random driver license number
        email.append(fake.email())  # Random email
        givenname.append(fake.first_name())  # Random given name
        idcard.append(np.random.randint(100000000, 999999999))  # Random ID card number
        password.append(fake.password(length=10))  # Random password
        socialnum.append(np.random.randint(100000000, 999999999))  # Random social security number
        street.append(fake.street_address())  # Random street address
        surname.append(fake.last_name())  # Random surname
        taxnum.append(np.random.randint(100000000, 999999999))  # Random tax number
        phone.append(fake.phone_number())  # Random phone number
        username.append(fake.user_name())  # Random username
        zipcode.append(fake.zipcode())  # Random zip code
    
    # Combine data into a DataFrame
    data = pd.DataFrame({
        'I-ACCOUNTNUM': accountnum,
        'I-BUILDINGNUM': buildingnum,
        'I-CITY': city,
        'I-CREDITCARDNUMBER': creditcard,
        'I-DATEOFBIRTH': dob,
        'I-DRIVERLICENSENUM': driverlicense,
        'I-EMAIL': email,
        'I-GIVENNAME': givenname,
        'I-IDCARDNUM': idcard,
        'I-PASSWORD': password,
        'I-SOCIALNUM': socialnum,
        'I-STREET': street,
        'I-SURNAME': surname,
        'I-TAXNUM': taxnum,
        'I-TELEPHONENUM': phone,
        'I-USERNAME': username,
        'I-ZIPCODE': zipcode
    })
    data.to_csv('original_data.csv', index=False)
    return data

from sklearn.preprocessing import LabelEncoder
    
def get_train_test_data(num_samples=1000, test_size=0.2, random_state=42):
    # Assuming this function generates your synthetic data
    data = generate_synthetic_data(num_samples)
    column_names = data.columns.tolist()

    # Mapping categorical data to numeric representations
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # If the column is categorical
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            label_encoders[column] = encoder  # Store the encoder for later use

    # Normalize the data to [0, 1] range using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split data into train and test sets
    train_data, test_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
    train_data = train_data.to_numpy()  # Convert DataFrame to NumPy array
    test_data = test_data.to_numpy() 
    # Return all four values: train_data, test_data, label_encoders, and scaler
    return train_data, test_data, label_encoders, scaler, column_names
