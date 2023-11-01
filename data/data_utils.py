import numpy as np
import sys
import random
    
class DataUtils:
    def __init__(self,verb=False):
        self.verb =verb
        
    def getRandomMixVector(self,name=None):
        if self.verb and (name != None):
            print(f"GENERATING MIXVECTOR FOR {name}")
            
        def _split_budget(budget,n):
            if budget % 5 != 0:
                raise ValueError("Budget must be a multiple of 5.")
            if n <= 0:
                raise ValueError("N must be a positive integer.")
        
            proportions = []
            remaining_budget = budget
            for i in range(n - 1):
                # Generate a random proportion that's a multiple of 5
                max_proportion = remaining_budget - (n - i - 1) * 5
                random_proportion = random.randint(1, max_proportion//5) * 5
                proportions.append(random_proportion)
                remaining_budget -= random_proportion
        
            proportions.append(remaining_budget)  # The last proportion is whatever remains
        
            return np.array(proportions)/100
    
        # Randomly pick a budget - UNIFORM DIST
        allowedValues = np.arange(0,105,5) # Arrow keys allow sliders to be incremented by 5%, therefore bind allowed values to 5% increments
        budget = np.random.choice(allowedValues)
        if self.verb:
            print("Budget ", budget/100)
            
        if budget == 0:
            # If the budget is 0, the vector will be empty
            # This was chosen as returning an empty vector could be decided by either the budget or the sliderNumber
            # However, there are 9 total values for sliderNumber and 21 for budget, therefore the chance of getting an empty vector will be lower
            # We will reduce this chance even further by adding a condition which requires 0 to be rolled twice in a roll for an empty vector to be returned 
            second_try_budget = np.random.choice(allowedValues)
            if self.verb:
                print("Second roll on the budget", second_try_budget/100)
                
            if second_try_budget == 0:
                if self.verb:
                    print("Budget was drawn as 0 twice in a row, returning empty mix sliders")
                return np.zeros(9)
            else:
                budget = second_try_budget
            
        
        
        
        if budget < 30:
            # The budget dictates the maximum number of sliders we can use, e.g. if the budget is only 0.10 we can only use one slider
            data = np.arange(1,int(budget/5)+1,1)
        else:
            # If the budget is instead higher and allows for more values, we can use more sliders
            # However we are not using all 9 sliders simultaneously 
            # as each would have a very small value and therefore a small effect on the character's look 
            data = np.arange(1,7,1)
        
        # Take samples with defined probabilities - for now I have chosen uniform sampling, its better for data coverage
        # and reduces bias for model, however it would mean more data is necessary
        sliderNumber = np.random.choice(data)
        if self.verb:
            print("Number of sliders in use ", sliderNumber)
        
        # Generate the indexes of the sliders to use
        sliders = np.arange(0,9,1)
        slider_IDs = np.random.choice(sliders,size=sliderNumber,replace=False) # replacement is off as we dont want repeating indexes
        if self.verb:
            print("Slider IDS ",slider_IDs)
        
        # Generate proportions values
        proportions = _split_budget(budget, sliderNumber)
        if self.verb:
            print("Proportions ", proportions)
            print("Sum of proportions ", sum(proportions))
        
        # Populate the mixing sliders vector with teh generated values at the generated positions
        sliderVector = np.zeros(9)
        for idx, value in zip(slider_IDs,proportions):
            sliderVector[idx] = value
        
        if self.verb:
            print(sliderVector)
        return sliderVector
    
    def getRandomIntVector(self,min_preset_value,max_preset_value):
        return np.random.randint(min_preset_value,max_preset_value+1,size=1)
    
    def getRandomFloatVector(self,n_sliders, min_val, max_val):
        interval = (max_val-min_val)/20
        return np.round(np.random.choice(np.arange(min_val,max_val+interval,interval),size=n_sliders),2)
    
    def getRandomStringVector(self,string_arr):
        return np.random.choice(string_arr,size=1)

if __name__ == "__main__":
    a = DataUtils(verb=True)
    a.getRandomMixVector()
    