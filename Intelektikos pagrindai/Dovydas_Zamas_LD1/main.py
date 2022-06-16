import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def readData(fileName):
    mydataset = pd.read_csv(fileName)
    return mydataset


def countNullsOfContinuous(data):
    nullValues = data[['enginesize', 'compressionratio', 'horsepower', 'citympg',
                       'highwaympg']].isnull().sum() * 100 / len(data)
    return nullValues


def findSumOfCategorical(data):
    data[['symboling', 'drivewheel', 'enginelocation', 'cylindernumber']].sum()


def findMaxValuesOfContinuous(data):
    maxClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].max()
    return maxClm


def findMinValuesOfContinuous(data):
    minClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].min()
    return minClm


def findAvgValuesOfContinuous(data):
    avgClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].mean()
    return avgClm


def findQuartiles(data):
    quartiles = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].quantile([0.25, 0.75])
    return quartiles


def findMedianValuesOfContinuous(data):
    medianClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].median()
    return medianClm


def findStdValuesOfContinuous(data):
    stdClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].std()
    return stdClm


def findCardValuesOfContinuous(data):
    uniqClm = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].nunique()
    return uniqClm


def findSumValuesOfCategorical(data):
    stdClm = data.count()
    return stdClm


def countNullsOfCategorical(data):
    nullValues = data[['symboling', 'drivewheel', 'enginelocation', 'cylindernumber']].isnull().sum() * 100 / len(data)
    return nullValues


def findCardValuesOfCategorical(data):
    uniqClm = data[['symboling', 'drivewheel', 'enginelocation', 'cylindernumber']].nunique()
    return uniqClm


def findModeValuesOfCategorical(data):
    modeClm = data[['symboling', 'drivewheel', 'enginelocation', 'cylindernumber']].mode()
    return modeClm


def findModeRepetetiveValue(data):
    return pd.DataFrame({'Columns': data.columns,
                         'Val': [data[x].isin(data[x].mode()).sum() for x in data]})


def findModePercentageValue(data):
    repetetive = findModeRepetetiveValue(data)
    percentage = []
    for i in range(len(repetetive)):
        percentage.append(repetetive.values[i][1] * 100 / data.iloc[:, [i]].count())
    return percentage


def findSecondMode(data):
    ans = []
    ans.append(
        [Counter(data['symboling']).most_common()[1][0], Counter(data['symboling']).most_common()[1][1], 'symboling'])
    ans.append([Counter(data['drivewheel']).most_common()[1][0], Counter(data['drivewheel']).most_common()[1][1],
                'drivewheel'])
    ans.append(
        [Counter(data['enginelocation']).most_common()[1][0], Counter(data['enginelocation']).most_common()[1][1],
         'enginelocation'])
    ans.append(
        [Counter(data['cylindernumber']).most_common()[1][0], Counter(data['cylindernumber']).most_common()[1][1],
         'cylindernumber'])
    return ans


def findSecondModePercentage(data):
    secondMode = findSecondMode(data)
    percentage = []
    for i in range(len(secondMode)):
        percentage.append(secondMode[i][1] * 100 / data.iloc[:, [i]].count())
    return percentage


def deleteOutliers(data):
    q_low = data["enginesize"].quantile(0.0005)
    q_hi = data["enginesize"].quantile(0.9995)
    data = data[(data["enginesize"] < q_hi) & (data["enginesize"] > q_low)]

    q_low = data["symboling"].quantile(0.0005)
    q_hi = data["symboling"].quantile(0.9995)
    data = data[(data["symboling"] < q_hi) & (data["symboling"] > q_low)]

    q_low = data["compressionratio"].quantile(0.005)
    q_hi = data["compressionratio"].quantile(0.9995)
    data = data[(data["compressionratio"] < q_hi) & (data["compressionratio"] > q_low)]

    q_low = data["horsepower"].quantile(0.0005)
    q_hi = data["horsepower"].quantile(0.9995)
    data = data[(data["horsepower"] < q_hi) & (data["horsepower"] > q_low)]

    q_low = data["citympg"].quantile(0.0005)
    q_hi = data["citympg"].quantile(0.9995)
    data = data[(data["citympg"] < q_hi) & (data["citympg"] > q_low)]

    q_low = data["highwaympg"].quantile(0.0005)
    q_hi = data["highwaympg"].quantile(0.9995)
    data = data[(data["highwaympg"] < q_hi) & (data["highwaympg"] > q_low)]
    return data


def plotScatter(data, xColumn: str, yColumn: str):
    plt.scatter(data[xColumn], data[yColumn], s=60, alpha=0.6, edgecolor='grey', linewidth=1)
    plt.xlabel(xColumn)
    plt.ylabel(yColumn)
    plt.tight_layout()
    plt.show()


def plotBar(data, xColumn: str, title=None):
    data[xColumn].value_counts(sort=True).plot.bar(rot=0)
    plt.xlabel(xColumn)
    plt.ylabel('count')
    if title is not None:
        plt.title(title)
    plt.show()


def plotBox(data, column: str, title=None):
    boxplot = data.boxplot(column=[column], grid=False, rot=45, fontsize=15)
    if title is not None:
        plt.title(title)
    plt.show()


def plotScatterMatrix(data):
    df = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']]
    pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.show()


def plot(data):
    plotBar(data, xColumn='symboling')
    plotBar(data, xColumn='drivewheel')
    plotBar(data, xColumn='enginelocation')
    plotBar(data, xColumn='cylindernumber')
    plotScatter(data, xColumn='horsepower', yColumn='enginesize')
    plotScatter(data, xColumn='enginesize', yColumn='citympg')
    plotScatter(data, xColumn='enginesize', yColumn='highwaympg')
    plotScatter(data, xColumn='horsepower', yColumn='citympg')
    plotScatter(data, xColumn='horsepower', yColumn='highwaympg')
    plotScatter(data, xColumn='enginesize', yColumn='compressionratio')
    plotScatter(data, xColumn='compressionratio', yColumn='horsepower')
    plotScatterMatrix(data)


def filterDataFunc(columnName: str, data, groupBy: str):
    filteredData = data[data[columnName] == groupBy]
    return filteredData


def filterAndPlotData(data):
    filteredData = filterDataFunc(columnName='drivewheel', data=data, groupBy='rwd')
    plotBar(filteredData, xColumn='symboling', title='symboling only rwd')
    plotBox(filteredData, column='horsepower', title='horsepower only rwd')
    filteredData = filterDataFunc(columnName='drivewheel', data=data, groupBy='fwd')
    plotBox(filteredData, column='horsepower', title='horsepower only fwd')
    plotBar(filteredData, xColumn='symboling', title='symboling only fwd')
    filteredData = filterDataFunc(columnName='enginelocation', data=data, groupBy='front')
    plotBox(filteredData, column='horsepower', title='horsepower only only engine location front')
    plotBar(filteredData, xColumn='symboling', title='symboling only engine location front')
    filteredData = filterDataFunc(columnName='enginelocation', data=data, groupBy='rear')
    plotBox(filteredData, column='horsepower', title='horsepower only only engine location rear')
    plotBar(filteredData, xColumn='symboling', title='symboling only engine location rear')


def findCorreleation(data):
    correlation_mat = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()


def normalize(data):
    result = data.copy()
    for feature_name in data.columns:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return result


def convertCategoricalToNumeric(data):
    data['symboling'] = data['symboling'].replace(to_replace=[-3, -2, -1, 0, 1, 2, 3], value=[1, 2, 3, 4, 5, 6, 7],
                                                  inplace=False)
    data['drivewheel'] = data['drivewheel'].replace(to_replace=['fwd', 'rwd'], value=[1, 2], inplace=False)
    data['enginelocation'] = data['enginelocation'].replace(to_replace=['front', 'rear'], value=[1, 2], inplace=False)
    data['cylindernumber'] = data['cylindernumber'].replace(
        to_replace=['two', 'three', 'four', 'five', 'six', 'eight', 'twelve', ],
        value=[1, 2, 3, 4, 5, 6, 7], inplace=False)
    return data


def main():
    data = readData('CarPrice_Assignment.csv')
    print(findSumValuesOfCategorical(data))
    print(countNullsOfContinuous(data))
    print(findCardValuesOfContinuous(data))
    print(findMinValuesOfContinuous(data))
    print(findMaxValuesOfContinuous(data))
    print(findQuartiles(data))
    print(findAvgValuesOfContinuous(data))
    print(findMedianValuesOfContinuous(data))
    print(findStdValuesOfContinuous(data))
    print(findSumValuesOfCategorical(data))
    print(countNullsOfCategorical(data))
    print(findCardValuesOfCategorical(data))
    print(findModeValuesOfCategorical(data))
    print(findModeRepetetiveValue(data))
    print(findModePercentageValue(data))
    print(findSecondMode(data))
    print(findSecondModePercentage(data))
    data = data.dropna()
    data = deleteOutliers(data)
    plot(data)
    filterAndPlotData(data)
    continousData = data[['enginesize', 'compressionratio', 'horsepower', 'citympg', 'highwaympg']]
    print(normalize(continousData))
    # convertCategoricalToNumeric(data)


if __name__ == "__main__":
    main()
