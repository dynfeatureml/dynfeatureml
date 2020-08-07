# Module: Datasets
# Author: Prashant Nair <prashant.nair2050@gmail.com>
# License: MIT


def get_data(dataset, save_copy=False, profile=False, verbose=True):
    
    """
      
    Description:
    ------------
    This function loads sample datasets that are available in the dynfeatureml git 
    repository. The full list of available datasets and their descriptions can
    be viewed by calling index.
        Example
        -------
        data = get_data('datasetList')
        
        This will display the list of available datasets that can be loaded 
        using the get_data() function. 
        
        Example - To load the iris dataset:
        
        iris = get_data('iris')
        
    Parameters
    ----------
    dataset : string 
    index value of dataset
    
    save_copy : bool, default = False
    When set to true, it saves a copy of the dataset to your local active directory.
    
    profile: bool, default = False
    If set to true, a data profile for Exploratory Data Analysis will be displayed 
    in an interactive HTML report. 
    
    verbose: bool, default = True
    When set to False, head of data is not displayed.
    
    Returns:
    --------
    DataFrame:    Pandas dataframe is returned. 
    ----------      
    Warnings:
    ---------
    - Use of get_data() requires internet connection.
      
         
    """
    
    import pandas as pd
    from IPython.display import display, HTML, clear_output, update_display
    
    address = 'https://raw.githubusercontent.com/dynfeatureml/dynfeatureml/master/datasets/'
    extension = '.csv'
    filename = dataset
    
    complete_address = address + str(dataset) + extension
    
    data = pd.read_csv(complete_address)
    
    #create a copy for pandas profiler
    data_for_profiling = data.copy()
    
    if save_copy:
        save_name = str(dataset) + str(extension)
        data.to_csv(save_name)
        
    if dataset == 'datasetList':
        display(data)
    
    else:
        if profile:
            import pandas_profiling
            pf = pandas_profiling.ProfileReport(data_for_profiling)
            display(pf)
            
        else:
            if verbose:
                display(data.head())
        
    return data