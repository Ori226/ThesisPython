import numpy as np 
import scipy.io
import math
import random


        
        
def GetAllDataSeperateToSubjects():
    all_data_by_name = GetAllFilesByName()    
    for name in all_data_by_name:
        print 'read: ',name, ' target'

        temp = FromFileListToArray([all_data_by_name[name]['target']['filename']], 'all_target_flatten' ,600)
        all_data_by_name[name]['target']['data'] = FromFileListToArray([all_data_by_name[name]['target']['filename']], 'all_target_flatten' ,600)
        print 'read: ',name, ' nontarget'
        all_data_by_name[name]['non_target']['data'] = FromFileListToArray([all_data_by_name[name]['non_target']['filename']], 'all_non_target_flatten' ,600)

    return all_data_by_name
        


def GetAllFilesByName():
    return {'gcf': 
                {
                    'target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcf.mat'},
                    'non_target':   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcf.mat'}
                },
            'gcg':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcg.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcg.mat'}
                },
            'gch':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgch.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgch.mat'}
                },
            'iay':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPiay.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPiay.mat'}
                },
            'icn':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicn.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicn.mat'}
                },
            'icr':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicr.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicr.mat'}
                },
            'pia':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPpia.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPpia.mat'}
                },
            'fat':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPfat.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPfat.mat'}
                },
            'gcb':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcb.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcb.mat'}
                },
            'gcc':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcc.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcc.mat'}
                },
            'gcd':  
                {
                'target'        :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcd.mat'},
                'non_target'    :   {'filename':'C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcd.mat'}
                },
            }

def GetAllTargetsFile():
    return ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcf.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcg.mat", 
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgch.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPiay.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicn.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicr.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPpia.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPfat.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcb.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcc.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcd.mat"]

def GetAllNonTargetsFile():
    return ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcd.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcf.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcg.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgch.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPiay.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicn.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicr.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPpia.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPfat.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcb.mat",
        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcc.mat"]


def ReadDataFromMatFiles():
    z = dict(LoadTarget().items() + LoadNonTarget().items())
    return z

def LoadAllWithNames():
    all_target_files = ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcf.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcg.mat", 
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgch.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPiay.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicn.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicr.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPpia.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPfat.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcb.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcc.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcd.mat"]

    #for future use of n-folds
    #random.shuffle(all_target_files)


    n_fold = 5
    number_of_elements = np.size(all_target_files)
    train_size = int(round(number_of_elements*(1.0 - (1.0/n_fold))))

    all_train_target_files = all_target_files[0:train_size]

    all_validate_target_files = all_target_files[train_size:]

    all_train_target = FromFileListToArray(all_train_target_files, 'all_target_flatten' ,640)
    all_validation_target = FromFileListToArray(all_validate_target_files, 'all_target_flatten' ,640)

    print ('done reading')
    return {'target':{'train': all_train_target, 'validation' : all_validation_target} }


def LoadTarget():

    
    all_target_files = ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcf.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcg.mat", 
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgch.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPiay.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicn.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPicr.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPpia.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPfat.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcb.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcc.mat",
    "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcd.mat"]

    #for future use of n-folds
    #random.shuffle(all_target_files)


    n_fold = 5
    number_of_elements = np.size(all_target_files)
    train_size = int(round(number_of_elements*(1.0 - (1.0/n_fold))))

    all_train_target_files = all_target_files[0:train_size]

    all_validate_target_files = all_target_files[train_size:]

    all_train_target = FromFileListToArray(all_train_target_files, 'all_target_flatten' ,640)
    all_validation_target = FromFileListToArray(all_validate_target_files, 'all_target_flatten' ,640)

    print ('done reading')
    return {'target':{'train': all_train_target, 'validation' : all_validation_target} }

def LoadNonTarget():

    all_target_files = ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcd.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcf.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcg.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgch.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPiay.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicn.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPicr.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPpia.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPfat.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcb.mat",
                        "C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcc.mat"]

    n_fold = 5
    number_of_elements = np.size(all_target_files)
    train_size = int(round(number_of_elements*(1.0 - (1.0/n_fold))))

    all_train_non_target_files = all_target_files[0:train_size]

    all_validate_non_target_files = all_target_files[train_size:]

    all_train_non_target = FromFileListToArray(all_train_non_target_files, 'all_non_target_flatten', 640)
    all_validation_non_target = FromFileListToArray(all_validate_non_target_files, 'all_non_target_flatten',640 )
    
    return {'non_target':{'train': all_train_non_target, 'validation' : all_validation_non_target} }
    
    
def FromFileListToArray(file_list, var_name, subset_size = None):
    return_value = None
    random.seed(0)    

    for file_name in file_list:
        print 'reading: ' , file_name
        mat = scipy.io.loadmat(file_name)
        mat = mat[var_name]
        if subset_size is not None:
            np.random.permutation(mat)
            mat = mat[0:subset_size,:]

        if return_value is None:
            return_value = mat
        else:            
            return_value = np.r_[return_value, mat]        
    
    return return_value


def LoadSingleSubject():
    
    
    #train_size = int(round(number_of_elements*(1.0 - (1.0/n_fold))))

    target_files = ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcb.mat"]

    non_target_files = ["C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPgcd.mat"]

    all_target = FromFileListToArray(target_files, 'all_target_flatten', 600)
    all_non_target = FromFileListToArray(non_target_files, 'all_non_target_flatten',600 )
    return [all_target, all_non_target]



'''
just a test example
'''
if __name__ == "__main__":
    #print GetAllTargetsFile()
    #print GetAllNonTargetsFile()

    GetAllDataSeperateToSubjects();
    
    all_data_by_name = GetAllFilesByName()


    for name in all_data_by_name:
        print 'read: ',name, ' target'

        temp = FromFileListToArray([all_data_by_name[name]['target']['filename']], 'all_target_flatten' ,600)
        all_data_by_name[name]['target']['data'] = FromFileListToArray([all_data_by_name[name]['target']['filename']], 'all_target_flatten' ,600)
        print 'read: ',name, ' nontarget'
        all_data_by_name[name]['non_target']['data'] = FromFileListToArray([all_data_by_name[name]['non_target']['filename']], 'all_non_target_flatten' ,600)
        
        #load the file
        
    print GetAllFilesByName()['gcg']['non_target']