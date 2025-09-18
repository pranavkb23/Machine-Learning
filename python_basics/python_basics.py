import random
from itertools import combinations
import pandas as pd

class python_basics:
    def part1(list1):
        # counts = []
        result = {}
        # counter = 0

        for item in list1:
            # if(counts.count(item) == 0):
            #     counts[counter] = item
            #     result[counter] = (item, list1.count(item))
            #     counter+=1
            result[item] = list1.count(item)
                

        # for i in range(len(result)):
        #     print(result[i])
        print("Total count of each tuple \n")
        print(result)

    part1([(1, 2), (1), (1, 2, 3), (1, 2), (1, 2), (1, 2, 3)])

    def part2():
        rand_list = random.sample(range(0, 100), 10)
        rand_list.sort()
        print("\nrandom list\n")
        print(rand_list)

        class BinaryTreeNode:
            def __init__(self, value):
                self.value = value
                self.leftChild = None
                self.rightChild=None
            
        def insert(root,newValue):
            if root is None:
                root=BinaryTreeNode(newValue)
                return root
            
            if newValue<root.value:
                root.leftChild=insert(root.leftChild,newValue)
            else:
                root.rightChild=insert(root.rightChild,newValue)
            return root
        
        root = insert(None, rand_list[(len(rand_list)//2)])
        for i in range(1,len(rand_list)):
            insert(root, rand_list[i])

        def print_p2(root):
            if root is None:
                return

            print(root.value, end=' ')
            print_p2(root.leftChild)
            print_p2(root.rightChild)
            
        print("\nInorder representation of BST\n")
        print_p2(root)
    part2()

    def part3(dict1):
        keys = list(dict1.keys())
        values = list(dict1.values())
        result = []

        for key in keys:
            flag = 0
            for char in key:
                if values.count(char) == 0:
                    flag = 1
            if flag == 0:
                result.append(key)
        print("\n\nkeys that can be using the given values\n")
        print(result)
    part3({'black':'r', 'hero':'e', 'go':'g', 'clue':'i', 'mean':'q', 'groan':'o', 
           'sin':'p', 'pint':'u', 'tone':'n', 'graze':'s', 'sea':'t', 'plant':'a'})

    def part4(list1):
        def selection_ascending(list1, pos):
            for i in range(0, len(list1)):
                min_index = i

                for j in range(i+1, len(list1)):
                    if (list1[j])[pos] < (list1[min_index])[pos]:
                        min_index = j
                
                (list1[i], list1[min_index]) = (list1[min_index], list1[i])
            print("\n")
            print(list1)

        def selection_descending(list1, pos):     
            for i in range(0, len(list1)):
                max_index = i

                for j in range(i+1,len(list1)):
                    if (list1[j])[pos] > (list1[max_index])[pos]:
                        max_index = j
                
                (list1[i], list1[max_index]) = (list1[max_index], list1[i])
            print("\n")
            print(list1)
        print("\nTuples arranged by first and second element, in ascending and descending order")
        selection_ascending(list1, 0)
        selection_ascending(list1, 1)
        selection_descending(list1, 0)
        selection_descending(list1, 1)
    part4([(1,2), (4,3), (2,10), (12, 5), (6, 7), (9,11), (15, 4)])
    

    def part5():
        # A mutable object is one that can be altered after it has been created. 
        # An immutable object is one that cannot be altered after it has been created
        # list, dictionary - mutable
        # string, tuple - immutable 
        pass

    def part6():
        path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/bank_additional_full.csv" 
        df = pd.read_csv(path, sep = r'\;')

        df.columns = df.columns.str.replace('"', '')
        print(df.head())
        print('\n')

        education = df['education'].unique().tolist()
        print(education)

        print('\n')
        selected_columns = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        mean_values = df.groupby('y')[selected_columns].mean()
        print(mean_values)

        print('\n')
        mean_ages = df.groupby('marital')['age'].mean()
        print(mean_ages)

        null_values = df.isnull().sum()
        print(null_values)

        numeric_statistics = df.describe()
        all_statistics = df.describe(include = 'all')
        print(numeric_statistics)
        print('\n')
        print(all_statistics)

        edu_professional = df.query("education == 'professional.course'") 
        job_blue = df.query("job == 'blue collar'")

        first_five_rows = df.iloc[:5]

        euribor3m_insight = df.loc[df['euribor3m'] > 4.857]

        jobs = df['job']

        specific_columns = df.iloc[:2, [0, 1, 2, 3]]

        print(edu_professional)
        print(job_blue)
        print(first_five_rows)
        print(euribor3m_insight)
        print(jobs)
        print(specific_columns)


    part6()

        




    
        

           
                 

