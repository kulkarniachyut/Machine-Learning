import numpy as np
import scipy.io
import math
from svmutil import *
import time

def changecol(data):
    for row in range(0,np.shape(data)[0]):
        for col in range(0,np.shape(data)[1]):
            if data[row,col] == -1:
                data[row,col] = 0

def transform(data):
    for col in range(0,np.shape(data)[1]):
        if col in [1, 6, 7, 13, 14, 25, 28]:
            final_data = np.append(final_data,np.zeros((len(data[:,col]),3)), axis=1)
            for row in range(0,np.shape(data)[0]):
                if data[row,col] == -1:
                    final_data[row,-3] = 1
                elif data[row,col] == 0:
                    final_data[row,-2] = 1
                else:
                    final_data[row,-1] = 1
        else:
            if col == 0:
                final_data = data[:,0].reshape(-1,1)
            else:
                final_data = np.append(final_data,data[:,col].reshape(-1,1), axis=1)
    return final_data

def convertInpuToList(data):
    list=[]
    for row in range(0,np.shape(data)[0]):
        dict = {}
        for col in range(0,np.shape(data)[1]):
            dict[col+1] = data[row,col]
        list.append(dict)
    return list

def linearSVM(train_data,test_data,train_target,test_target):
    max_acc = 0
    for c in [-6,-5,-4,-3,-2,-1,0,1,2]:
        start = time.time()
	acc = svm_train(train_target[0],train_data,"-c "+str(math.pow(4,c))+" -v 3 -q")
        end = time.time()
        print " Cross Validation training accuracy for c = 4^"+str(c)+" is "+str(acc)+" and time is "+str(end-start)
        if acc > max_acc:
		max_acc = acc
                max_c = c

    model = svm_train(train_target[0],train_data,"-c "+str(math.pow(4,max_c))+" -t 0")
    temp_res,test_acc,value  = svm_predict(test_target[0],test_data, model)
    print "Best C is "+str(max_c)+" and testing accuracy for best c is "+str(test_acc[0])

def polynomialSVM(train_data,train_target):
    max_acc = 0
    for d in [1,2,3]:
    	for c in [-3,-2,-1,0,1,2,3,4,5,6,7]:
		start = time.time()
		acc = svm_train(train_target[0],train_data,"-c "+str(math.pow(4,c))+" -q -v 3 -t 1 -d "+str(d))
		end = time.time()
		print " Cross Validation training accuracy for c = 4^"+str(c)+" and d = "+str(d)+" is "+str(acc)+" and time is "+str(end-start)
        	if acc > max_acc:
			max_acc = acc
                	max_c = c
                        max_d = d
    return max_c,max_d,max_acc

def RBFSVM(train_data,train_target):
    max_acc = 0
    for g in [-7,-6,-5,-4,-3,-2,-1]:
    	for c in [-3,-2,-1,0,1,2,3,4,5,6,7]:
                start = time.time()
		acc = svm_train(train_target[0],train_data,"-c "+str(math.pow(4,c))+" -q -v 3 -t 2 -g "+str(math.pow(4,g)))
                end = time.time()
	        print " Cross Validation training accuracy for c = 4^"+str(c)+" and g = "+str(g)+" is "+str(acc)+" and time is "+str(end-start)
        	if acc > max_acc:
			max_acc = acc
                	max_c = c
                        max_g = g
    return max_c,max_g,max_acc

def main():
    train= scipy.io.loadmat("phishing-train")
    test = scipy.io.loadmat("phishing-test")
    train_data = train["features"]
    train_target = train["label"]
    test_data = test["features"]
    test_target = test["label"]
    final_train = transform(train_data)
    changecol(final_train)
    changecol(train_target)
    final_test = transform(test_data)
    changecol(final_test)
    changecol(test_target)
    list_train_data = convertInpuToList(final_train)
    list_train_target = train_target.tolist()
    list_test_data = convertInpuToList(final_test)
    list_test_target = test_target.tolist()
    linearSVM(list_train_data,list_test_data,list_train_target,list_test_target)
    polymax_c,polymax_d,polymax_acc = polynomialSVM(list_train_data,list_train_target)
    rbfmax_c,rbfmax_g,rbfmax_acc = RBFSVM(list_train_data,list_train_target)
    if polymax_acc < rbfmax_acc :
        print "RBFSVM with c "+str(rbfmax_c)+" and g "+str(rbfmax_g)+" has the best accuracy " +str(rbfmax_acc)
        model = svm_train(list_train_target[0],list_train_data," -c "+str(math.pow(4,rbfmax_c))+" -t 2 -g "+str(math.pow(4,rbfmax_g)))
        temp_res,test_acc,value  = svm_predict(list_test_target[0],list_test_data, model)
        print "Testing accuracy for best c is for RBFSVM is "+str(test_acc[0])
    else:
        print "PolynomialSVM with c "+str(polymax_c)+" and d "+str(polymax_d)+" has the best accuracy"
        model = svm_train(list_train_target[0],list_train_data," -c "+str(math.pow(4,polymax_c))+" -t 1 -d "+str(polymax_d))
        temp_res,test_acc,value  = svm_predict(list_test_target[0],list_test_data, model)
        print "Testing accuracy for best c is for polnomialSVM is "+str(test_acc[0])


if __name__ == "__main__":
    main()
