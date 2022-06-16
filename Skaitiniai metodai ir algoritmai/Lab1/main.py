import numpy as np
import matplotlib.pyplot as plt

CONST_COEF = np.array([1, -7, -4, 52, 48])  # Daugianarės funkcijos koeficientai
reversedCoef = CONST_COEF[::-1]  # Daugianarės atvirkštinė funkcija
# Nurodomos konstantos
# ----------------------------------------------------------------------------------------------------
CONST_STEP = 0.005  # Step
CONST_PRECISION = 1e-5  # Tikslumas
CONST_ALPHA = -7  # Konstanta paprastųjų iteracijų metodui
CONST_M = 70  # Masė
CONST_G = 9.8  # Laisvo kritimo pagreitis
CONST_T = 3  # Laikas
CONST_V = 27  # Greitis


# ----------------------------------------------------------------------------------------------------


# Daugianarė funkcija
def fx(x):
    return 1.5 * x ** 4 -1


# Daugianarės funkcijos išvestinė
def Dfx(x):
    return 1.5 * x ** 3


# Funkcija g(x)
def gx(x):
    return (np.e ** (-x)) * np.sin(x ** 2) + 0.001


# Funkcijos g(x) išvestinė
def Dgx(x):
    return -np.e ** -x * (np.sin(x ** 2) - 2 * x * np.cos(x ** 2))


# Netiesinė lygtis
def fc(c):
    return ((CONST_M * (CONST_G / c)) * (1 - np.e ** (-3 * c / 70))) - 27


# Funkcijos f(x) šaknys pagal python
def pythonRoots():
    print('Saknys pagal python funkciją Roots:', np.roots(np.array([0.48, 1.71, -0.67, -4.86, -1.33, 1.50])))


# Randama didžiausia reikšmė moduliu masyve nevertinant koeficiento prie aukščiausio laipsnio
def maxAbs(Coef):
    return abs(max((Coef[1:]), key=abs))


# Randamas grubus intervalas
def roughCuts(maxMean, divCoef):
    return 1 + maxMean / divCoef


# Išrenkamos neigiamos reikšmės iš masyvo
def getNegValues(Coef):
    return np.where(Coef < 0, Coef, 0)


def getBValue(Coef):
    return maxAbs(getNegValues(Coef))


def getMaxIndexOfNegValue(reversedFunc):
    tempArr = getNegValues(reversedFunc[:])
    rez = 0
    for i in range(1, len(tempArr)):
        if tempArr[i] != 0:
            rez = i
    return rez


# Randamas grubus rėžis
def getRValue():
    return 1 + maxAbs(CONST_COEF) / CONST_COEF[0]


# Randamas tikslesnio įverčio viršutinis rėžis
def getRPos():
    tempReversedCoef = reversedCoef[:]
    tempCoef = CONST_COEF[:]
    k = 5 - getMaxIndexOfNegValue(tempReversedCoef)
    return 1 + (getBValue(tempCoef) / tempCoef[0]) ** (1 / k)


# Randamas tikslesnio įverčio apatinis rėžis
def getRNeg():
    tempReversedCoef = reversedCoef[:]
    for i in range(len(tempReversedCoef)):
        if i % 2 != 0:
            tempReversedCoef[i] *= -1
        tempReversedCoef[i] *= -1
    tempCoef = tempReversedCoef[::-1]
    k = 5 - getMaxIndexOfNegValue(tempReversedCoef)
    return 1 + (getBValue(tempCoef) / tempCoef[0]) ** (1 / k)

# Skenavimo metodas su fiksuotu žingsniu artiniams tikslinti
def scanFixedStep(f, xFrom, xTo, step):
    rezArray = []
    while True:
        if np.sign(f(xFrom)) == np.sign(f(xFrom + step)):
            xFrom += step
        else:
            rezArray.append(xFrom)
            rezArray.append(xFrom + CONST_STEP)
            xFrom += CONST_STEP
        if xFrom > xTo:
            break
        else:
            continue
    return rezArray


# Paprastųjų iteracijų metodas
def simpleIterationMethod(f, xFrom):
    idx = 0
    x = xFrom
    precision = 1
    while precision > CONST_PRECISION and idx < 1000:
        idx += 1
        x_next = x - (f(x)/CONST_ALPHA)
        precision = abs(x - x_next)
        x = x_next
    print("Iteracijų skaičius: ", str(idx), "   Pradinis artinys: ", str(xFrom), "   Šaknis:", str(x))
    return x


# Niutono(liestinių) metodas
def newtonsMethod(f, Df, xFrom, e):
    xn = xFrom
    fxn = f(xn)
    idx = 0
    while np.abs(fxn) > e and idx < 100:
        fxn = f(xn)
        Dfxn = Df(xn)
        if Dfxn == 0:
            return None
        xn = xn - fxn / Dfxn
        idx += 1
    print("Iteracijų skaičius: ", str(idx), "   Pradinis artinys: ", str(xFrom), "   Šaknis:", str(xn))
    return xn


# Skenavimo metodas su mažėjančiu žingsniu
def ScanWithSmallerStep(func, xFrom, e):
    h = 0.1
    x = xFrom
    idx = 0
    while np.abs(func(x)) > e and idx < 100:
        idx += 1
        if np.sign(func(x + h)) == np.sign(func(x)):
            x += h
        else:
            h *= 0.1
    print("Iteracijų skaičius: ", str(idx), "   Pradinis artinys: ", str(xFrom), "   Šaknis:", str(x))
    return x


# Pagal pasirinktą funkciją ir metodą atliekami skaičiavimai
def ChooseMethod(function, method):
    ans = []
    if function == "fx":
        R = getRValue()
        Rpos = min(R, getRPos())
        Rtmp = getRNeg()
        RNeg = -min(R, Rtmp)
        rangefx = scanFixedStep(fx, RNeg, Rpos, CONST_STEP)
        print("R = ", R, "\nRNeg = ", RNeg, "\nRPos = ", Rpos)
        if method == "SimpleIteration":
            for i in range(len(rangefx)):
                if i % 2 == 0 and i + 1 <= len(rangefx):
                    ans.append(simpleIterationMethod(fx, rangefx[i]))
            pythonRoots()
            PlotRes(-3.2, 2, ans, function)
            return ans
        elif method == "Newtons":
            for i in range(len(rangefx)):
                if i % 2 == 0 and i + 1 <= len(rangefx):
                    ans.append(newtonsMethod(fx, Dfx, 5, CONST_PRECISION))
            pythonRoots()
            PlotRes(-3.2, 2, ans, function)
            return ans
        elif method == "Scan":
            for i in range(len(rangefx)):
                if i % 2 == 0 and i + 1 <= len(rangefx):
                    ans.append(ScanWithSmallerStep(fx, rangefx[i], CONST_PRECISION))
            pythonRoots()
            PlotRes(-3.2, 2, ans, function)
            print(rangefx)
        return ans
    elif function == "gx":
        rangegx = scanFixedStep(gx, 5, 10, CONST_STEP)
        if method == "SimpleIteration":
            for i in range(len(rangegx)):
                if i % 2 == 0 and i + 1 <= len(rangegx):
                    ans.append(simpleIterationMethod(gx, rangegx[i]))
            PlotRes(5, 10, ans, function)
            return ans
        elif method == "Newtons":
            for i in range(len(rangegx)):
                if i % 2 == 0 and i + 1 <= len(rangegx):
                    ans.append(newtonsMethod(gx, Dgx, rangegx[i], CONST_PRECISION))
            PlotRes(5, 10, ans, function)
            return ans
        elif method == "Scan":
            for i in range(len(rangegx)):
                if i % 2 == 0 and i + 1 <= len(rangegx):
                    ans.append(ScanWithSmallerStep(gx, rangegx[i], CONST_PRECISION))
            PlotRes(5, 10, ans, function)
            return ans
    elif function == "fc":
        rangefc = scanFixedStep(fc, 1, 20, CONST_STEP)
        for i in range(len(rangefc)):
            if i % 2 == 0 and i + 1 <= len(rangefc):
                ans.append(ScanWithSmallerStep(fc, rangefc[i], CONST_PRECISION))
        if len(ans) > 0:
            PlotRes(-40, 40, ans, function)
        else:
            print("funkcijos šaknų nerasta nurodytame intervale")
            PlotRes(-100, 100, ans=[], func="fc")
        return ans


# Grafiko ir funkcijos šaknų atvaizdavimas
def PlotRes(xFrom, xTo, ans, func):
    x = np.linspace(xFrom, xTo, 100)
    y = 0
    if func == "fx":
        y = fx(x)
    elif func == "gx":
        y = gx(x)
    elif func == "fc":
        y = fc(x)
    plt.plot(x, y)
    plt.axhline(color='black')
    if len(ans) > 0:
        for i in range(len(ans)):
            plt.plot(ans[i], 0, markersize=5, color='red', marker='o')
            if func == "fx":
                print("Funkcijos fx reikšmė: ", str(format(fx(ans[i]), ".8f")), " šaknyje: ",
                      str(format(ans[i], ".8f")))
            elif func == "gx":
                print("Funkcijos gx reikšmė: ", str(format(gx(ans[i]), ".2f")), " šaknyje: ",
                      str(format(ans[i], ".2f")))
            elif func == "fc":
                print("Funkcijos fc reikšmė: ", str(format(fc(ans[i]), ".2f")), " šaknyje: ",
                      str(format(ans[i], ".2f")))
    plt.show()


# Programos vykdymas
def Execute():
    while True:
        try:
            print("Pasirinkite funkciją: fx, gx, fc")
            function = str(input())
            if function == "fx" or function == "gx":
                print("Pasirinkite metodą: SimpleIteration, Newtons, Scan")
                method = str(input())
                ChooseMethod(function, method)
                print("Tikslumas: ", str(CONST_PRECISION))
            elif function=="fc":
                ChooseMethod("fc", method=None)
                print("Tikslumas: ", str(CONST_PRECISION))
            break
        except ValueError:
            print("Oops! There is no valid function or method. Try again...")


Execute()
