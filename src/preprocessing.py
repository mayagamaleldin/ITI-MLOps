from sklearn.preprocessing import LabelEncoder

def wrangle(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], inplace=True, axis=1)

    df.fillna({
        'Age': df['Age'].mean(),
        'Cabin': df['Cabin'].mode()[0] if not df['Cabin'].mode().empty else 'Unknown',
        'Embarked': df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'Unknown'
    }, inplace=True)

    hEncoder1 = LabelEncoder()
    hEncoder2 = LabelEncoder()
    hEncoder3 = LabelEncoder()

    df["Sex"] = hEncoder1.fit_transform(df["Sex"])
    df["Embarked"] = hEncoder2.fit_transform(df["Embarked"])
    df["Cabin"] = hEncoder3.fit_transform(df["Cabin"])

    return df
