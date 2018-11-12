import numpy as np
import pandas as pd
from numpy import *
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


train_data=pd.read_csv('/home/behnam/Desktop/kc_house_train_data.csv',dtype=dtype_dict)
test_data=pd.read_csv('/home/behnam/Desktop/kc_house_test_data.csv',dtype=dtype_dict)



def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    dataframe=pd.DataFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    dataframe=pd.DataFrame(feature,columns=['power_1'])
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            print(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            dataframe.insert(power-1,name,feature ** power)
    return dataframe


def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)

    # Then compute the residuals/errors
    residuals = predictions - outcome

    # Then square and add them up

    RSS = (residuals * residuals).sum()

    return (RSS)




def main():
    sales = pd.read_csv('/home/behnam/Desktop/kc_house_data.csv', dtype=dtype_dict)
    sales=sales.sort_values(['sqft_living','price'])

    poly1_data = polynomial_sframe(np.array(sales['sqft_living']), 1)
    poly1_data['price'] = sales['price']  # add price to the data since it's the target
    print(poly1_data[:2])
    print(len(poly1_data))

    reg1 = LinearRegression()
    model_1 = reg1.fit(poly1_data, sales['price'])
    predicted_output_1 = reg1.predict(poly1_data)

    print(reg1.coef_)

    fig1 = plt.figure()
    plt.xlabel('sqft')
    plt.ylabel('Price ($)')
    plt.plot(poly1_data['power_1'], poly1_data['price'], '.',
             poly1_data['power_1'], predicted_output_1, '-')

    fig1.show()



    # part2

    poly2_data = polynomial_sframe(np.array(sales['sqft_living']), 2)
    my_features1 = list(poly2_data)  # get the name of the features
    poly2_data['price'] = sales['price']

    reg2 = LinearRegression()
    model_2 = reg2.fit(poly2_data, sales['price'])
    predicted_output_2 = reg2.predict(poly2_data)



    fig2 = plt.figure()
    plt.xlabel('Quadratic Function (sqft)')
    plt.ylabel('Price ($)')
    plt.plot(poly2_data['power_1'], poly2_data['price'], '.',
             poly2_data['power_1'], predicted_output_2, '-')

    fig2.show()

    # part3

    poly3_data = polynomial_sframe(np.array(sales['sqft_living']), 3)
    my_features2 = list(poly3_data)  # get the name of the features
    poly3_data['price'] = sales['price']



    reg3 = LinearRegression()
    model_3 = reg3.fit(poly3_data, sales['price'])
    predicted_output_3 = reg3.predict(poly3_data)



    fig3 = plt.figure()
    plt.xlabel('Cubic Function (sqft)')
    plt.ylabel('Price ($)')
    plt.plot(poly3_data['power_1'], poly3_data['price'], '.',
             poly3_data['power_1'], predicted_output_3, '-')

    fig3.show()


    # part4

    poly15_data = polynomial_sframe(np.array(sales['sqft_living']), 15)
    my_features15 = list(poly15_data)  # get the name of the features
    poly15_data['price'] = sales['price']


    reg15 = LinearRegression().fit(poly15_data, sales['price'])
    reg15.score(poly15_data,sales['price'])
    predicted_output_15 = reg15.predict(poly15_data)




    fig4 = plt.figure()
    plt.xlabel('15th Polynomial Function (sqft)')
    plt.ylabel('Price ($)')
    plt.plot(poly15_data['power_1'], poly15_data['price'], '.',
             poly15_data['power_1'], predicted_output_15, '-')

    fig4.show()


    # part5

    set_1 = pd.read_csv('/home/behnam/Desktop/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    set_2 = pd.read_csv('/home/behnam/Desktop/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    set_3 = pd.read_csv('/home/behnam/Desktop/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    set_4 = pd.read_csv('/home/behnam/Desktop/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

    # part set1

    set1_poly15_data = polynomial_sframe(np.array(set_1['sqft_living']), 15)
    my_features = list(set1_poly15_data)  # get the name of the features
    set1_poly15_data['price'] = set_1['price']

    reg15_set1 = LinearRegression().fit(set1_poly15_data, set1_poly15_data['price'])
    reg15_set1.score(set1_poly15_data, set1_poly15_data['price'])
    predicted_output_15_set1 = reg15_set1.predict(set1_poly15_data)

    fig5 = plt.figure()
    plt.xlabel('15th Polynomial Function (sqft)')
    plt.ylabel('Price ($)')
    plot = plt.plot(set1_poly15_data['power_1'], set1_poly15_data['price'], '.',
                    set1_poly15_data['power_1'], predicted_output_15_set1, '-')

    fig5.show()

    # part set2

    set2_poly15_data = polynomial_sframe(np.array(set_2['sqft_living']), 15)
    my_features = list(set2_poly15_data)  # get the name of the features
    set2_poly15_data['price'] = set_1['price']

    reg15_set2 = LinearRegression().fit(set2_poly15_data, set2_poly15_data['price'])
    reg15_set2.score(set2_poly15_data, set2_poly15_data['price'])
    predicted_output_15_set2 = reg15_set2.predict(set2_poly15_data)

    fig6 = plt.figure()
    plt.xlabel('15th Polynomial Function (sqft)')
    plt.ylabel('Price ($)')
    plot = plt.plot(set2_poly15_data['power_1'], set2_poly15_data['price'], '.',
                    set2_poly15_data['power_1'], predicted_output_15_set2, '-')

    fig6.show()

    # part set3

    set3_poly15_data = polynomial_sframe(np.array(set_3['sqft_living']), 15)
    my_features = list(set3_poly15_data)  # get the name of the features
    set3_poly15_data['price'] = set_3['price']

    reg15_set3 = LinearRegression().fit(set3_poly15_data, set3_poly15_data['price'])
    reg15_set3.score(set3_poly15_data, set3_poly15_data['price'])
    predicted_output_15_set3 = reg15_set3.predict(set3_poly15_data)

    fig7 = plt.figure()
    plt.xlabel('15th Polynomial Function (sqft)')
    plt.ylabel('Price ($)')
    plot = plt.plot(set3_poly15_data['power_1'], set3_poly15_data['price'], '.',
                    set3_poly15_data['power_1'], predicted_output_15_set3, '-')

    fig7.show()

    # part set4

    set4_poly15_data = polynomial_sframe(np.array(set_4['sqft_living']), 15)
    my_features = list(set4_poly15_data)  # get the name of the features
    set4_poly15_data['price'] = set_4['price']

    reg15_set4 = LinearRegression().fit(set4_poly15_data, set4_poly15_data['price'])
    reg15_set4.score(set4_poly15_data, set4_poly15_data['price'])
    predicted_output_15_set4 = reg15_set4.predict(set4_poly15_data)

    fig8 = plt.figure()
    plt.xlabel('15th Polynomial Function (sqft)')
    plt.ylabel('Price ($)')
    plot = plt.plot(set4_poly15_data['power_1'], set4_poly15_data['price'], '.',
                    set4_poly15_data['power_1'], predicted_output_15_set4, '-')

    fig8.show()


    # part train-test-valid

    new_train = pd.read_csv('/home/behnam/Desktop/wk3_kc_house_train_data.csv', dtype=dtype_dict)
    new_test=  pd.read_csv('/home/behnam/Desktop/wk3_kc_house_test_data.csv', dtype=dtype_dict)
    new_valid= pd.read_csv('/home/behnam/Desktop/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

    print('sales data: ' + str(len(sales)))
    print('testing data: ' + str(len(new_train)))
    print('training data: ' + str(len(new_test)))
    print('validation data: ' + str(len(new_valid)))

    rssDic = {}
    modelDic = {}
    for i in range(1, 15 + 1):
        poly_train_data = polynomial_sframe(np.array(new_train['sqft_living']), i)
        poly_features = list(poly_train_data)
        poly_train_data['price'] = new_train['price']
        reg_train = LinearRegression().fit(poly_train_data, poly_train_data['price'])
        reg_train.score(poly_train_data, poly_train_data['price'])
        modelDic[i] = reg_train

        poly_valid_data = polynomial_sframe(np.array(new_valid['sqft_living']), i)
        poly_valid_data['price'] = new_valid['price']
        predictions = modelDic[i].predict(poly_valid_data)
        rssDic[i] = get_residual_sum_of_squares(modelDic[i], poly_valid_data, new_valid['price'])

    minRSS = min(rssDic.values())
    minKey = list(rssDic.keys())[list(rssDic.values()).index(minRSS)]
    print('model degree with least RSS: ' + str(minKey))
    print('least RSS: ' + str(rssDic[minKey]))

    poly_testing_data = polynomial_sframe(np.array(new_test['sqft_living']), minKey)
    poly_testing_data['price'] = new_test['price']
    predictions = modelDic[minKey].predict(poly_testing_data)
    print(get_residual_sum_of_squares(modelDic[minKey], poly_testing_data, poly_testing_data['price']))
    print(modelDic[minKey].evaluate(poly_testing_data)['rmse'] ** 2 * len(poly_testing_data))

    print('end')
if __name__ == "__main__":
    main()