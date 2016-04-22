#!/usr/bin/python


import pybrain
# reload(pybrain)
import json
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import sys

import pybrain.datasets
# reload(pybrain.datasets)
import pybrain.supervised.trainers
from pybrain.structure.connections import FullConnection
from pybrain.structure.connections import CustomFullConnection
from pybrain.utilities import percentError
from pybrain.tools import validation
from random import shuffle
import pickle
# reload(validation)



#################################################################

"""
Helper Functions

"""

#################################################################


def get_pathway_dict(pathway_file, data):
    new_kegg=[]
    api_pathways=defaultdict(list)

    with open(pathway_file) as f:
        for line in f:
            ids=line.strip().split('\t')
            if ids[1][4:] in data.index.values.tolist():
                api_pathways[ids[0]].append(ids[1][4:])

    #this next section just deletes weird blank keys that pop up from spaces in pathway file
    api_pathways2=api_pathways.copy()
    for key in api_pathways:
        if not api_pathways[key]:
            del api_pathways2[key]
    return dict(api_pathways2)

    

def sync_data_pathways(data,pathways):
    #reduces genes in dataframe to only genes in pathways
    genes_in_paths=[]
    for path in pathways:
        genes_in_paths+=pathways[path]
    genes_to_keep=list(set(genes_in_paths))
    
    data1=data.loc[genes_to_keep]
    return data1



def add_pathway_connections(network, first_layer, second_layer, data, pathways):
    #now the tricky layer from the pathways database  
    
    entrez_dict = {} #entrez dict connects entrezids to the index of the feature vector
    for i,entrez in enumerate(data.index.values.tolist()):
        entrez_dict[entrez] = i

    
    for path_index, path in enumerate(pathways.keys()):
        g_index=[]
        for g in pathways[path]:
            g_index.append(entrez_dict[g])

        network.addConnection(CustomFullConnection(first_layer,second_layer,\
                                        inSliceIndices=g_index,  \
                                        outSliceIndices=[path_index]))
    return network




## build structure
def build_pybrain_flat_network(data, pathways, layers=2, second_hidden=5):
    #data is a data frame straight from affy - columns are patients, rows are entrez genes
    #pathways is a dict with {pathway1:[gene1, gene2, ...], pathway2:[gene, gene...]...}

    in_data = data.values.T

    fnn = pybrain.structure.networks.FeedForwardNetwork()

    inLayer = pybrain.structure.LinearLayer(in_data.shape[1])
    fnn.addInputModule(inLayer)

    outLayer = pybrain.structure.SoftmaxLayer(2)
    fnn.addOutputModule(outLayer)

    hidden_list=[]
    #right now I have two sigmoid hidden layers 
    #can and will probably change
    for i in range(layers):
        if i ==0:
            hidden_list.append(pybrain.structure.SigmoidLayer(len(pathways)))
            fnn.addModule(hidden_list[i])
        else:
            hidden_list.append(pybrain.structure.SigmoidLayer(second_hidden))
            fnn.addModule(hidden_list[i])
    
    
    
    ## add connections input to hidden is sparse, but second hidden and output are fully connected
    
    #add fully connected layers
    hidden_to_out = pybrain.structure.connections.FullConnection(hidden_list[-1],outLayer)
    fnn.addConnection(hidden_to_out)

    
#     hidden_connects=[]
    for hl in range(1,layers):
        #this first step may be unncessary, but I am saving the connections objects to a list incase I need them later
#         hidden_connects.append(FullConnection(hidden_list[hl-1],hidden_list[hl]))
        fnn.addConnection(FullConnection(hidden_list[hl-1],hidden_list[hl]))

    #now the tricky layer from the pathways database
    fnn = add_pathway_connections(fnn, inLayer, hidden_list[0], data, pathways)



    fnn.sortModules()
    return fnn




def build_pybrain_deep_network(data, pathways, filters=5, third_layer_nodes=5):
    #data is a data frame straight from affy - columns are patients, rows are entrez genes
    #pathways is a dict with {pathway1:[gene1, gene2, ...], pathway2:[gene, gene...]...}
    
    layers = filters
    
    in_data = data.values.T

    fnn = pybrain.structure.networks.FeedForwardNetwork()

    inLayer = pybrain.structure.LinearLayer(in_data.shape[1])
    fnn.addInputModule(inLayer)

    outLayer = pybrain.structure.SoftmaxLayer(2)
    fnn.addOutputModule(outLayer)

    hidden_list=[]
    #right now I have two sigmoid hidden layers 
    #can and will probably change
    for i in range(layers):
        hidden_list.append(pybrain.structure.SigmoidLayer(len(pathways)))
        fnn.addModule(hidden_list[i])
    
    clean_up_layer = pybrain.structure.SigmoidLayer(third_layer_nodes)
    fnn.addModule(clean_up_layer)
    
    
    
    ## add connections input to hidden is sparse, but second hidden and output are fully connected
    
    #add fully connected layers
    hidden_to_out = pybrain.structure.connections.FullConnection(clean_up_layer,outLayer)
    fnn.addConnection(hidden_to_out)

    
#     hidden_connects=[]
    for i in range(layers):
        fnn.addConnection(FullConnection(hidden_list[i],clean_up_layer))

        #now the tricky layer from the pathways database
        fnn = add_pathway_connections(fnn, inLayer, hidden_list[i], data, pathways)


    fnn.sortModules()
    return fnn



def shuffle_split(x,y,split_proportion=.8):
    inds = range(len(y))
    shuffle(inds)

    x=x[inds,:]
    y=y[inds]

    split_proportion=.8
    train_inds = int(round(split_proportion*x.shape[0]))

    xtrain=x[:train_inds,:]
    xtest = x[train_inds:, :]
    ytrain=y[:train_inds]
    ytest = y[train_inds:]
    
    return xtrain, xtest, ytrain, ytest



def convertDataNeuralNetwork(x, y):
    colx = 1 if len(np.shape(x))==1 else np.size(x, axis=1)
    coly = 1 if len(np.shape(y))==1 else np.size(y, axis=1)
    
    fulldata = pybrain.datasets.ClassificationDataSet(colx,coly, nb_classes=2)
    for d, v in zip(x, y):
        fulldata.addSample(d, v)
    
    return fulldata



######################################################################################

"""
Main code

"""
######################################################################################



# import CleanMetadata 

def run_pybrain():
    tnfa_raw, y = CleanMetadata.slice_and_clean()
    inds = [str(i) for i in tnfa_raw.index.values.tolist()]
    tnfa_raw.index = inds

    pathways = get_pathway_dict('Pathways/kegg_api.txt',tnfa_raw)
    data = sync_data_pathways(tnfa_raw,pathways)


    x=data.values.T
    y=np.array(y)
    xtrain, xtest, ytrain, ytest = shuffle_split(x, y)

    maxEpochs = 2000
    epochsperstep = 50
    learningrate = [.001, .001, .001, .001]
    hidden = [1, 5, 10, 50]
    deep_layers=[3,5,10,15]
    momentum = 0.
    noise=[.1, .1]


    foo=0
    for n in noise:
        for lr in learningrate:
            foo+=1
            result_count=0

            print '\n\n\nnoise =', n, "learning rate =",lr
        #     print 'hidden nodes in second layer of flat = ', hidden_nodes


            Train = convertDataNeuralNetwork(xtrain, ytrain)
            Test = convertDataNeuralNetwork(xtest, ytest)
            Train._convertToOneOfMany()
            Test._convertToOneOfMany()

            print('\nFlat network')
            fnn=build_flat_network(data, pathways, second_hidden=10)
            trainer =  pybrain.supervised.trainers.BackpropTrainer(fnn, dataset=Train, verbose=False, \
                                                                   learningrate = lr, lrdecay = 1, momentum = momentum)



            with open('test_flat_network_lr%s_n%s.txt'%(lr,n),'w') as f:
                f.write('learning rate='+str(lr)+'noise='+str(n)+'\n')
                f.write('epoch'+'\t'+'train error'+'\t'+'test error'+'\n')

                for i in range(200):
                    trainer.trainEpochs( 10 )
                    trnresult = percentError( trainer.testOnClassData(),
                                              Train['class'] )
                    tstresult = percentError( trainer.testOnClassData(
                           dataset=Test ), Test['class'] )

                    print "epoch: %4d" % trainer.totalepochs, \
                          "  train error: %5.2f%%" % trnresult, \
                          "  test error: %5.2f%%" % tstresult

                    f.write(str(trainer.totalepochs)+'\t'+str(trnresult)+'\t'+str(tstresult)+'\n')

                    sys.stdout.flush()
                    if trnresult<=10:
                        result_count+=1
                        if result_count>5:
                            break
                    else:
                        result_count=0



                    xtrain_new=xtrain+np.random.normal(scale=n, size=xtrain.shape)
                    Train = convertDataNeuralNetwork(xtrain_new, ytrain)
                    Train._convertToOneOfMany()
                    trainer.setData(Train)
            # pickle.dump(trainer, open('trainer_%s'%foo,'w'))
            pickle.dump(fnn, open('fnn_%s.pkl'%foo,'w'))       
                    
                    
            # """Deep Network"""

            # fnn=build_deep_network(data, pathways, filters=10)
            # # print fnn
            # print('\ndeep network')

            # Train = convertDataNeuralNetwork(xtrain, ytrain)
            # Test = convertDataNeuralNetwork(xtest, ytest)
            # Train._convertToOneOfMany()
            # Test._convertToOneOfMany()

            # trainer =  pybrain.supervised.trainers.BackpropTrainer(fnn, dataset=Train, verbose=False, \
            #                                                        learningrate = lr, lrdecay = 1, momentum = momentum)


            # with open('test_deep_network_lr%s_n%s.txt'%(lr,n),'w') as f:
            #     f.write('learning rate='+str(lr)+'noise='+str(n)+'\n')
            #     f.write('epoch'+'\t'+'train error'+'\t'+'test error'+'\n')

            #     for i in range(50):
            #         trainer.trainEpochs( 10 )
            #         trnresult = percentError( trainer.testOnClassData(),
            #                                   Train['class'] )
            #         tstresult = percentError( trainer.testOnClassData(
            #                dataset=Test ), Test['class'] )

            #         print "epoch: %4d" % trainer.totalepochs, \
            #               "  train error: %5.2f%%" % trnresult, \
            #               "  test error: %5.2f%%" % tstresult
                            
            #         sys.stdout.flush()
            #         if trnresult<=5:
            #             result_count+=1
            #             if result_count>5:
            #                 break
            #         else:
            #             result_count=0


            #         f.write(str(trainer.totalepochs)+'\t'+str(trnresult)+'\t'+str(tstresult)+'\n')

            #         xtrain_new=xtrain+np.random.normal(scale=n, size=xtrain.shape)
            #         Train = convertDataNeuralNetwork(xtrain_new, ytrain)
            #         Train._convertToOneOfMany()
            #         trainer.setData(Train)
            #             # if trainer.totalepochs%40==0:
            #                 # print 'reshuffeling..'
            #                 # Train, Test = fulldata.splitWithProportion(.8)
            #                 # Train._convertToOneOfMany()
            #                 # Test._convertToOneOfMany()
            #                 # trainer.setData(Train)













