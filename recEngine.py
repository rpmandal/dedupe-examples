import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_product_from_index(index,customer_data_featured):
    return customer_data_featured.loc[customer_data_featured.index == index,'Product'].iloc[0]

#### Function to get recommended products
def get_similar_products(customer,cosine_sim):
    cust_index = customer.index.values[0]
    similar_products = list(enumerate(cosine_sim[cust_index])) 
    return similar_products

#### Function to get similar customers
def get_similar_customers(sorted_similar_products,customer_data_featured):
    cust_list = []
    j = 0
    for element in sorted_similar_products:
        cust = get_cust_from_index(element[0],customer_data_featured)
        cust_list.append(cust)
        j = j+1
        if j>=5:
            break
    return cust_list
#Categorizing total amount and total number of transaction for each customer.
def categorize_txn(customer_data_featured):
    first_q_totalamt = customer_data_featured.TotalAmount.quantile(0.2)
    second_q_totalamt = customer_data_featured.TotalAmount.quantile(0.4)
    third_q_totalamt = customer_data_featured.TotalAmount.quantile(0.6)
    fourth_q_totalamt = customer_data_featured.TotalAmount.quantile(0.8)
    
    first_q_noOfTxn = customer_data_featured.NumberOfTransactions.quantile(0.2)
    second_q_noOfTxn = customer_data_featured.NumberOfTransactions.quantile(0.4)
    third_q_noOfTxn = customer_data_featured.NumberOfTransactions.quantile(0.6)
    fourth_q_noOfTxn = customer_data_featured.NumberOfTransactions.quantile(0.8) 
    
    customer_data_featured.loc[customer_data_featured.TotalAmount <= first_q_totalamt,'Amt_cat']="First"
    customer_data_featured.loc[(customer_data_featured.TotalAmount > first_q_totalamt) &  (customer_data_featured.TotalAmount <= second_q_totalamt),'Amt_cat']="Second"
    customer_data_featured.loc[(customer_data_featured.TotalAmount > second_q_totalamt) &  (customer_data_featured.TotalAmount <= third_q_totalamt),'Amt_cat']="Third"
    customer_data_featured.loc[(customer_data_featured.TotalAmount > third_q_totalamt) &  (customer_data_featured.TotalAmount <= fourth_q_totalamt),'Amt_cat']="Fourth"
    customer_data_featured.loc[customer_data_featured.TotalAmount > fourth_q_totalamt,'Amt_cat']="Fifth"
    
    customer_data_featured.loc[customer_data_featured.NumberOfTransactions <= first_q_noOfTxn,'txn_cat']="First"
    customer_data_featured.loc[(customer_data_featured.NumberOfTransactions > first_q_noOfTxn) &  (customer_data_featured.NumberOfTransactions <= first_q_noOfTxn),'txn_cat']="Second"
    customer_data_featured.loc[(customer_data_featured.NumberOfTransactions > first_q_noOfTxn) &  (customer_data_featured.NumberOfTransactions <= first_q_noOfTxn),'txn_cat']="Third"
    customer_data_featured.loc[(customer_data_featured.NumberOfTransactions > first_q_noOfTxn) &  (customer_data_featured.NumberOfTransactions <= first_q_noOfTxn),'txn_cat']="Fourth"
    customer_data_featured.loc[customer_data_featured.NumberOfTransactions > first_q_noOfTxn,'txn_cat']="Fifth"
    
    customer_data_featured.drop(['NumberOfTransactions','TotalAmount'], inplace=True, axis=1)
    customer_data_featured.rename(columns={'Amt_cat':'TotalAmount'},inplace=True)
    customer_data_featured.rename(columns={'txn_cat':'NumberOfTransactions'},inplace=True)
    
    #print(customer_data_featured[['NumberOfTransactions','TotalAmount']])
    return customer_data_featured


#### KYCS is not a product so we have to remove it from product list against all custmers.
def remove_KYCS(input):
    removal_str = 'KYCS'
    sep = ", "

    prod_list = (sep.join([ i for i in input.split(sep) if i != removal_str ]))
    if len(prod_list) == 0:
        prod_list= np.nan
    return prod_list

#### Function to get a list of top 3 recommended products

def top_3_recommended_prod(customer_availaing_products,recommended):
    customer_availaing_products = [x.upper() for x in customer_availaing_products]
    availaing_products_set  = set(customer_availaing_products)
    recommended_set = set(recommended)
    difference = recommended_set - availaing_products_set
    top_3 = list(difference)[:3]
    return top_3

def get_recommendation(cind,input_feature_list):
    cind = int(cind)
    #Reading data from CSV to a dataframe
    customer_data = pd.read_csv("cust_360.csv")
    #Getting the input'cu
    print("1")
    rm = customer_data.loc[customer_data.CIND == cind, 'RM'].iloc[0]
    #print("RM:  ",rm)
    
    customer_data.rename(columns={'Sources':'Product'},inplace=True)
    #Removing the KYCS sourcses and adding a column 'Product'
    customer_data['Product'] = customer_data['Product'].apply(remove_KYCS)
    customer_data.loc[(customer_data.CIND==cind) & (customer_data.Product.isnull()),['Product']]='NA'

    #Removing rows from the dataframe where source value in NaN
    customer_data.dropna(subset=['Product'],inplace=True)
    
    #Adding index column
    customer_data = customer_data.reset_index()
    customer_data['index'] = customer_data.index
    
    #Keeping only important features
    customer_data_featured = customer_data[['index','CIND','Country','LOB','NumberOfTransactions','Product','TotalAmount']]
    
    #Categorizing Total amount and number of transactions
    customer_data_featured = categorize_txn(customer_data_featured)

    #Default features if no feature is selected by end user
    deafaultFeatures = ['Country', 'LOB','NumberOfTransactions','TotalAmount']
        
    #filling all the NaN values with blank string in the dataframe.
    for feature in deafaultFeatures:
        customer_data_featured[feature] = customer_data_featured[feature].fillna('') #filling all NaNs with blank string
    
    if(len(input_feature_list)>0):
        customer_data_featured['combined_features'] = customer_data_featured[input_feature_list].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    else:
        customer_data_featured['combined_features'] = customer_data_featured[deafaultFeatures].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    #Feeding these strings to a CountVectorizer() object for getting the count matrix.
    cv = CountVectorizer() #creating new CountVectorizer() object
    count_matrix = cv.fit_transform(customer_data_featured["combined_features"])
    
    #Obtain the cosine similarity matrix from the count matrix.
    cosine_sim = cosine_similarity(count_matrix)
    selected_customer = customer_data_featured.loc[customer_data_featured.CIND==cind]
   
    #customer_availaing_products = customer_using_prods(selected_customer)
    customer_products = selected_customer.Product.values[0]
    customer_availaing_products = [y.strip() for y in customer_products.split(",")]
    #Get similar products
    cust_index = selected_customer.index.values[0]
    similar_products = list(enumerate(cosine_sim[cust_index]))
    sorted_similar_products = sorted(similar_products,key=lambda x:x[1],reverse=True)[1:]
    #print(sorted_similar_products)
    #Recommending top 3 products and getting top 5 similar customers
    recommended=[]    
    for element in sorted_similar_products:
        recommended.extend([x.strip() for x in get_product_from_index(element[0],customer_data_featured).upper().split(",") if x.strip() not in customer_availaing_products and x.strip() not in recommended])
        #print("recommended and length",recommended,len(recommended))
        if len(recommended)>=3:
            break
    top5_similar_cust_list =[]
    for cust_element in sorted_similar_products:
        cust = customer_data_featured[customer_data_featured.index == cust_element[0]]["CIND"].values[0]
        top5_similar_cust_list.append(int(cust))
        if len(top5_similar_cust_list)>=5:
            break

    products = {'RM':rm,'existing':list(customer_availaing_products), 'recommended' : recommended[:3], 'similar_customers' : top5_similar_cust_list[:5]}
    return products