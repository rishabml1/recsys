import pandas as pd
import numpy as np
import psycopg2 as pg
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


def check_column_types():
    pass


def get_db_info(dbtype):
    pass



def define_structure(data,structure):
    rating_ind=structure["rating"]
    col_user_ind=structure["user"]
    col_item_ind=structure["item"]
    user_features_ind=structure["item_feature"]
    item_features_ind=structure["item_feature"]
    colname_to_ind=[data.columns[i] for i in [rating_ind,col_user_ind,col_item_ind]+user_features_ind+item_features_ind]
    return data[[colname_to_ind]]



def MVP(data,missing_type):
    pass

                        

def EDA(data):
    pass




def classify_columns(df):
    num_cols = []
    str_cols = []
    
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
            num_cols.append(col)
        except ValueError:
            str_cols.append(col)
    
    return num_cols, str_cols


def modelling(data,trained):
        
        if model=="xdeepfm":
            dense_features, sparse_features=classify_columns(data)
        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
        target = data.columns[0]

        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])


        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                                for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                                for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
        model.compile("adam", "regression",
                        metrics=['mse'])
        if trained:
            model=model.load_weights(trained)

        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold = 1

            losses = []
            accuracies = []
            model_list=[]
            X_data,y_data=data[dense_features+sparse_features]+data[target]
            for train_index, val_index in kf.split(data[sparse_features[0]]): 
                print(f"Training fold {fold}...")
                feature_names
                X_train = {name: data[name][train_index] for name in feature_names}
                X_val = {name: data[name][val_index] for name in feature_names}
                y_train, y_val = y_data[train_index], y_data[val_index]
                
                model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
                
                model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
                
                model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=2, validation_data=(X_val, y_val))
                model_list.append(model)

                val_pred = model.predict(X_val, batch_size=256)
                val_loss = log_loss(y_val, val_pred)
                val_accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
                
                losses.append(val_loss)
                accuracies.append(val_accuracy)
                
                print(f"Fold {fold}: Log Loss = {val_loss}, Accuracy = {val_accuracy}")
                fold += 1
            model=model_list[accuracies.index(max(accuracies))]
            model.save_weights("best_model.h5")


def main(data,col_user,col_item,num_user_features=0,num_item_features=0,only_triplet=True,custom_structure=True,structure={},missing_value=[],trained=None):
    """

    Args:
        (str)data:(address) full dataset on which model is trained,test and tuned
        col_user:name of user column
        col_item: name of item column 
        num_user_features: num of feature column for user
        num_item_features: num of feature column for item
        only_triplet: only user-item-rating triplet pair are utilized in this system
        custom_structure

    structure of data:
        <rating>,<col_user>,<col_item>,<user_feature_col1>,...<user_feature_colm>,<item_feature_col1>,...<item_feature_coln>
    
    """    
    filetype=data.split(".")[-1]
    if filetype=="csv":
        data=pd.read_csv(data)
    elif filetype=="parquet":
        data=pd.read_parquet(data)
    elif filetype=="sql":
        info=get_db_info(filetype)
        engine = pg.connect(f"dbname='{info["db"]}' user='{info["user"]}' host='{info["host"]}' port='{info["port"]}' password='{info["password"]}'")
        df = pd.read_sql(f'select * from {info[""]}', con=engine)
        data=pd.read_sql(data)
    elif filetype=="db" or filetype=="sqlite3":
        info=get_db_info(filetype)
        df = pd.read_sql(f'select * from {info["table_name"]}', con=engine)
        data=pd.read_sql(data)

    if custom_structure:
        data=define_structure(data,structure)
    rating_col=data.columns[0]
    col_user=data.columns[1]
    col_item=data.columns[2]
    user_feature_cols=data.columns[3:num_user_features]
    user_feature_cols=data.columns[num_user_features:num_item_features]
    try:
        data["rating_col"]=data["rating_col"].astype(int)
        data["col_user"]=data["col_user"].astype(str)
        data["col_item"]=data["col_item"].astype(str)
    except Exception as e:
        raise Exception("rating:int col_user:str col_item:str")
    if missing_value!=[]:
        data= MVP(data,missing_value)
    data.dropna(inplace=True)
    EDA(data)
    modelling(data,model="xdeepfm")


    


        

    


