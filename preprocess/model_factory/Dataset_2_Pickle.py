import pickle
import glob


#packages are defined as [modelinfo_1][modelinfo_2][label]
#here that is [word2vec][chinese restaurant][label]
#add more models as needed

train_legal_pkg
train_books_pkg
train_newsp_pkg
test_legal_pkg
test_books_pkg
test_newsp_pkg

if not pkgs exist in data_sets folder:

    for file in glob('../data_sets/*'):
        create models for each pkg. Which models do we want?
            chinese restaurant
            pure words  
        pkg[-1]=label #ie timeslice in discrete time space
        add to appropriatePkg

    pickle all built pkgs_unfinished

else:
    unpickle packages



if have unfinished packages:
    train word2vec
    repickle

