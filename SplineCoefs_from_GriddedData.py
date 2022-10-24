# -*- coding: utf-8 -*-
class SplineCoefs_from_GriddedData:
    """
    Compute the coeffcieits of the N-dimensitonal natural-cubic spline interpolant defined by Habermann and Kindermann 2007
    Current code supports up to 5 dimensions (N can be either of 1,2,3,4,5).
    
    Author:
        N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab).
    
    Assumptions:
        x space (independent valiables) is N-dimension
        Equidistant x-grid in each dimension
        y datum (single real value) is given at each grid point
        
    User's Inputs:
        a: N-list of lower boundary of x-space [1st-dim, 2nd-dim,...].
        b: N-list of upper boundary of x-space [1st-dim, 2nd-dim,...].
        y_data: N-dimensional numpy array of data (the value of dependent variable y) on the x-gridpoints.
        
    Output:
        c_i1...iN: N-dimensional numpy array (dtype=float) of spline coeffcieints defined as HK2007 p161.
        
    Usage:
        from SplineCoefs_from_GriddedData import SplineCoefs_from_GriddedData  # import this module
        spline_coef= SplineCoefs_from_GriddedData(a,b,n,y_data) # constructor 
        c_i1...iN = spline_coef.Compute_Coefs() # compute the N-dim spline coefficient
        
    
    Ref.
    Habermann, C., & Kindermann, F. (2007). Multidimensional spline interpolation: Theory and applications. Computational Economics, 30(2), 153-169.
    Notation is modified by N.Moteki as Note of 2022 September 23-27th
    
    Created on Fri Oct 21 13:41:11 2022

    @author: moteki
    """
    
    def __init__(self,a,b,y_data):
        import numpy as np
        self.N=len(a) # dimension of the problem
        self.a= np.array(a, dtype=float) # list of lower bound of x-coordinate in each dimension # [1st dim, 2nd dim, ... ]
        self.b= np.array(b, dtype=float) # list of uppder bound of x-coordinate in each dimension # [1st dim, 2nd dim, ... ]
        self.n= np.zeros(self.N, dtype=int) # number of grid interval n in each dimension
        self.y_data= y_data # N-dimensional numpy array of y-data ydata[idx1,idx2,...] where the idx1 is the index of grid point along 1st dimension and so forth
        for j in range(self.N):
            self.n[j]= self.y_data.shape[j]-1 #number of grid interval n in each dimension
            
    def get_c_shape(self,k):
        c_shape=()
        for j in range(self.N):
            if j <= k :
                c_shape += (self.n[j]+3,)
            else:
                c_shape += (self.n[j]+1,)
        return c_shape
        
    def Compute_Coefs(self):
        import numpy as np
        from scipy import linalg
          
        if self.N == 1:
            k=0 # 1-st dimension
            c_i1= np.zeros(self.get_c_shape(k))
            c_i1[1]= self.y_data[0]/6 # c_{2}
            c_i1[self.n[k]+1]= self.y_data[self.n[k]]/6 # c_{n+2}
            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
            B= np.zeros(self.n[k]-1)
            B[0]= self.y_data[1]-c_i1[1]
            B[self.n[k]-2]= self.y_data[self.n[k]-1]-c_i1[self.n[k]+1]
            B[1:self.n[k]-2]= self.y_data[2:self.n[k]-1]
            sol= linalg.solve(A, B)
            c_i1[2:self.n[k]+1]= sol
            c_i1[0]= 2*c_i1[1]-c_i1[2]
            c_i1[self.n[k]+2]= 2*c_i1[self.n[k]+1]-c_i1[self.n[k]]
            
            return c_i1
            
        elif self.N == 2:
            k=0 # 1-st dimension
            c_i1q2= np.zeros(self.get_c_shape(k))
            for q2 in range(self.n[1]+1):
                c_i1q2[1,q2]= self.y_data[0,q2]/6 # c_{2}
                c_i1q2[self.n[k]+1,q2]= self.y_data[self.n[k],q2]/6 # c_{n+2}
                A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                B= np.zeros(self.n[k]-1)
                B[0]= self.y_data[1,q2]-c_i1q2[1,q2]
                B[self.n[k]-2]= self.y_data[self.n[k]-1,q2]-c_i1q2[self.n[k]+1,q2]
                B[1:self.n[k]-2]= self.y_data[2:self.n[k]-1,q2]
                sol= linalg.solve(A, B)
                c_i1q2[2:self.n[k]+1,q2]= sol
                c_i1q2[0,q2]= 2*c_i1q2[1,q2]-c_i1q2[2,q2]
                c_i1q2[self.n[k]+2,q2]= 2*c_i1q2[self.n[k]+1,q2]-c_i1q2[self.n[k],q2]
                    
            k=1 # 2nd dimension
            c_i1i2=np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                c_i1i2[i1,1]= c_i1q2[i1,0]/6 # c_{2}
                c_i1i2[i1,self.n[k]+1]= c_i1q2[i1,self.n[k]]/6 # c_{n+2}
                A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                B= np.zeros(self.n[k]-1)
                B[0]= c_i1q2[i1,1]-c_i1i2[i1,1]
                B[self.n[k]-2]= c_i1q2[i1,self.n[k]-1]-c_i1i2[i1,self.n[k]+1]
                B[1:self.n[k]-2]= c_i1q2[i1,2:self.n[k]-1]
                sol = linalg.solve(A, B)
                c_i1i2[i1,2:self.n[k]+1]=sol
                c_i1i2[i1,0]= 2*c_i1i2[i1,1]-c_i1i2[i1,2]
                c_i1i2[i1,self.n[k]+2]= 2*c_i1i2[i1,self.n[k]+1]-c_i1i2[i1,self.n[k]]
                
            return c_i1i2
        
        elif self.N == 3:
            k=0 # 1-st dimension
            c_i1q2q3= np.zeros(self.get_c_shape(k))
            for q2 in range(self.n[1]+1):
                for q3 in range(self.n[2]+1):
                    c_i1q2q3[1,q2,q3]= self.y_data[0,q2,q3]/6 # c_{2}
                    c_i1q2q3[self.n[k]+1,q2,q3]= self.y_data[self.n[k],q2,q3]/6 # c_{n+2}
                    A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                    B= np.zeros(self.n[k]-1)
                    B[0]= self.y_data[1,q2,q3]-c_i1q2q3[1,q2,q3]
                    B[self.n[k]-2]= self.y_data[self.n[k]-1,q2,q3]-c_i1q2q3[self.n[k]+1,q2,q3]
                    B[1:self.n[k]-2]= self.y_data[2:self.n[k]-1,q2,q3]
                    sol= linalg.solve(A, B)
                    c_i1q2q3[2:self.n[k]+1,q2,q3]= sol
                    c_i1q2q3[0,q2,q3]= 2*c_i1q2q3[1,q2,q3]-c_i1q2q3[2,q2,q3]
                    c_i1q2q3[self.n[k]+2,q2,q3]= 2*c_i1q2q3[self.n[k]+1,q2,q3]-c_i1q2q3[self.n[k],q2,q3]
                    
            k=1 # 2nd dimension
            c_i1i2q3=np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for q3 in range(self.n[2]+1):
                    c_i1i2q3[i1,1,q3]= c_i1q2q3[i1,0,q3]/6 # c_{2}
                    c_i1i2q3[i1,self.n[k]+1,q3]= c_i1q2q3[i1,self.n[k],q3]/6 # c_{n+2}
                    A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                    B= np.zeros(self.n[k]-1)
                    B[0]= c_i1q2q3[i1,1,q3]-c_i1i2q3[i1,1,q3]
                    B[self.n[k]-2]= c_i1q2q3[i1,self.n[k]-1,q3]-c_i1i2q3[i1,self.n[k]+1,q3]
                    B[1:self.n[k]-2]= c_i1q2q3[i1,2:self.n[k]-1,q3]
                    sol = linalg.solve(A, B)
                    c_i1i2q3[i1,2:self.n[k]+1,q3]=sol
                    c_i1i2q3[i1,0,q3]= 2*c_i1i2q3[i1,1,q3]-c_i1i2q3[i1,2,q3]
                    c_i1i2q3[i1,self.n[k]+2,q3]= 2*c_i1i2q3[i1,self.n[k]+1,q3]-c_i1i2q3[i1,self.n[k],q3]

            k=2 # 3rd dimension
            c_i1i2i3=np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    c_i1i2i3[i1,i2,1]= c_i1i2q3[i1,i2,0]/6 # c_{2}
                    c_i1i2i3[i1,i2,self.n[k]+1]= c_i1i2q3[i1,i2,self.n[k]]/6 # c_{n+2}
                    A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                    B= np.zeros(self.n[k]-1)
                    B[0]= c_i1i2q3[i1,i2,1]-c_i1i2i3[i1,i2,1]
                    B[self.n[k]-2]= c_i1i2q3[i1,i2,self.n[k]-1]-c_i1i2i3[i1,i2,self.n[k]+1]
                    B[1:self.n[k]-2]= c_i1i2q3[i1,i2,2:self.n[k]-1]
                    sol = linalg.solve(A, B)
                    c_i1i2i3[i1,i2,2:self.n[k]+1]= sol
                    c_i1i2i3[i1,i2,0]= 2*c_i1i2i3[i1,i2,1]-c_i1i2i3[i1,i2,2]
                    c_i1i2i3[i1,i2,self.n[k]+2]= 2*c_i1i2i3[i1,i2,self.n[k]+1]-c_i1i2i3[i1,i2,self.n[k]]
            
            return c_i1i2i3
            
        elif self.N == 4:
            k=0 #1st dimension
            c_i1q2q3q4= np.zeros(self.get_c_shape(k))
            for q2 in range(self.n[1]+1):
                for q3 in range(self.n[2]+1):
                    for q4 in range(self.n[3]+1):
                        c_i1q2q3q4[1,q2,q3,q4]=self.y_data[0,q2,q3,q4]/6 # c_{2}
                        c_i1q2q3q4[self.n[k]+1,q2,q3,q4]=self.y_data[self.n[k],q2,q3,q4]/6 # c_{n+2}
                        A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                        B= np.zeros(self.n[k]-1)
                        B[0]= self.y_data[1,q2,q3,q4]-c_i1q2q3q4[1,q2,q3,q4]
                        B[self.n[k]-2]= self.y_data[self.n[k]-1,q2,q3,q4]-c_i1q2q3q4[self.n[k]+1,q2,q3,q4]
                        B[1:self.n[k]-2]= self.y_data[2:self.n[k]-1,q2,q3,q4]
                        sol= linalg.solve(A, B)
                        c_i1q2q3q4[2:self.n[k]+1,q2,q3,q4]= sol
                        c_i1q2q3q4[0,q2,q3,q4]= 2*c_i1q2q3q4[1,q2,q3,q4]-c_i1q2q3q4[2,q2,q3,q4]
                        c_i1q2q3q4[self.n[k]+2,q2,q3,q4]= 2*c_i1q2q3q4[self.n[k]+1,q2,q3,q4]-c_i1q2q3q4[self.n[k],q2,q3,q4]
                        
            k=1 # 2nd dimension
            c_i1i2q3q4= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for q3 in range(self.n[2]+1):
                    for q4 in range(self.n[3]+1):
                        c_i1i2q3q4[i1,1,q3,q4]=c_i1q2q3q4[i1,0,q3,q4]/6 # c_{2}
                        c_i1i2q3q4[i1,self.n[k]+1,q3,q4]=c_i1q2q3q4[i1,self.n[k],q3,q4]/6 # c_{n+2}
                        A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                        B= np.zeros(self.n[k]-1)
                        B[0]=c_i1q2q3q4[i1,1,q3,q4]-c_i1i2q3q4[i1,1,q3,q4]
                        B[self.n[k]-2]=c_i1q2q3q4[i1,self.n[k]-1,q3,q4]-c_i1i2q3q4[i1,self.n[k]+1,q3,q4]
                        B[1:self.n[k]-2]=c_i1q2q3q4[i1,2:self.n[k]-1,q3,q4]
                        sol= linalg.solve(A, B)
                        c_i1i2q3q4[i1,2:self.n[k]+1,q3,q4]= sol
                        c_i1i2q3q4[i1,0,q3,q4]= 2*c_i1i2q3q4[i1,1,q3,q4]-c_i1i2q3q4[i1,2,q3,q4]
                        c_i1i2q3q4[i1,self.n[k]+2,q3,q4]= 2*c_i1i2q3q4[i1,self.n[k]+1,q3,q4]-c_i1i2q3q4[i1,self.n[k],q3,q4]
                        

            k=2 # 3rd dimension
            c_i1i2i3q4= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    for q4 in range(self.n[3]+1):
                        c_i1i2i3q4[i1,i2,1,q4]=c_i1i2q3q4[i1,i2,0,q4]/6 # c_{2}
                        c_i1i2i3q4[i1,i2,self.n[k]+1,q4]=c_i1i2q3q4[i1,i2,self.n[k],q4]/6 # c_{n+2}
                        A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                        B= np.zeros(self.n[k]-1)
                        B[0]= c_i1i2q3q4[i1,i2,1,q4]-c_i1i2i3q4[i1,i2,1,q4]
                        B[self.n[k]-2]= c_i1i2q3q4[i1,i2,self.n[k]-1,q4]-c_i1i2i3q4[i1,i2,self.n[k]+1,q4]
                        B[1:self.n[k]-2]= c_i1i2q3q4[i1,i2,2:self.n[k]-1,q4]
                        sol= linalg.solve(A, B)
                        c_i1i2i3q4[i1,i2,2:self.n[k]+1,q4]=sol
                        c_i1i2i3q4[i1,i2,0,q4]= 2*c_i1i2i3q4[i1,i2,1,q4]-c_i1i2i3q4[i1,i2,2,q4]
                        c_i1i2i3q4[i1,i2,self.n[k]+2,q4]= 2*c_i1i2i3q4[i1,i2,self.n[k]+1,q4]-c_i1i2i3q4[i1,i2,self.n[k],q4]
                        
                        
            k=3 # 4th dimension
            c_i1i2i3i4= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    for i3 in range(self.n[2]+3):    
                        c_i1i2i3i4[i1,i2,i3,1]=c_i1i2i3q4[i1,i2,i3,0]/6 # c_{2}
                        c_i1i2i3i4[i1,i2,i3,self.n[k]+1]=c_i1i2i3q4[i1,i2,i3,self.n[k]]/6 # c_{n+2}
                        A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                        B = np.zeros(self.n[k]-1)
                        B[0]=c_i1i2i3q4[i1,i2,i3,1]-c_i1i2i3i4[i1,i2,i3,1]
                        B[self.n[k]-2]=c_i1i2i3q4[i1,i2,i3,self.n[k]-1]-c_i1i2i3i4[i1,i2,i3,self.n[k]+1]
                        B[1:self.n[k]-2]=c_i1i2i3q4[i1,i2,i3,2:self.n[k]-1]
                        sol = linalg.solve(A, B)
                        c_i1i2i3i4[i1,i2,i3,2:self.n[k]+1]= sol
                        c_i1i2i3i4[i1,i2,i3,0]=2*c_i1i2i3i4[i1,i2,i3,1]-c_i1i2i3i4[i1,i2,i3,2]
                        c_i1i2i3i4[i1,i2,i3,self.n[k]+2]=2*c_i1i2i3i4[i1,i2,i3,self.n[k]+1]-c_i1i2i3i4[i1,i2,i3,self.n[k]]
            
            return c_i1i2i3i4
            
        
        elif self.N == 5:
            k=0 #1st dimension
            c_i1q2q3q4q5= np.zeros(self.get_c_shape(k))
            for q2 in range(self.n[1]+1):
                for q3 in range(self.n[2]+1):
                    for q4 in range(self.n[3]+1):
                        for q5 in range(self.n[4]+1):
                            c_i1q2q3q4q5[1,q2,q3,q4,q5]=self.y_data[0,q2,q3,q4,q5]/6 # c_{2}
                            c_i1q2q3q4q5[self.n[k]+1,q2,q3,q4,q5]=self.y_data[self.n[k],q2,q3,q4,q5]/6 # c_{n+2}
                            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                            B= np.zeros(self.n[k]-1)
                            B[0]= self.y_data[1,q2,q3,q4,q5]-c_i1q2q3q4q5[1,q2,q3,q4,q5]
                            B[self.n[k]-2]= self.y_data[self.n[k]-1,q2,q3,q4,q5]-c_i1q2q3q4q5[self.n[k]+1,q2,q3,q4,q5]
                            B[1:self.n[k]-2]= self.y_data[2:self.n[k]-1,q2,q3,q4,q5]
                            sol= linalg.solve(A, B)
                            c_i1q2q3q4q5[2:self.n[k]+1,q2,q3,q4,q5]= sol
                            c_i1q2q3q4q5[0,q2,q3,q4,q5]= 2*c_i1q2q3q4q5[1,q2,q3,q4,q5]-c_i1q2q3q4q5[2,q2,q3,q4,q5]
                            c_i1q2q3q4q5[self.n[k]+2,q2,q3,q4,q5]= 2*c_i1q2q3q4q5[self.n[k]+1,q2,q3,q4,q5]-c_i1q2q3q4q5[self.n[k],q2,q3,q4,q5]
                            
            k=1 # 2nd dimension
            c_i1i2q3q4q5= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for q3 in range(self.n[2]+1):
                    for q4 in range(self.n[3]+1):
                        for q5 in range(self.n[4]+1):
                            c_i1i2q3q4q5[i1,1,q3,q4,q5]=c_i1q2q3q4q5[i1,0,q3,q4,q5]/6 # c_{2}
                            c_i1i2q3q4q5[i1,self.n[k]+1,q3,q4,q5]=c_i1q2q3q4q5[i1,self.n[k],q3,q4,q5]/6 # c_{n+2}
                            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                            B= np.zeros(self.n[k]-1)
                            B[0]=c_i1q2q3q4q5[i1,1,q3,q4,q5]-c_i1i2q3q4q5[i1,1,q3,q4,q5]
                            B[self.n[k]-2]=c_i1q2q3q4q5[i1,self.n[k]-1,q3,q4,q5]-c_i1i2q3q4q5[i1,self.n[k]+1,q3,q4,q5]
                            B[1:self.n[k]-2]=c_i1q2q3q4q5[i1,2:self.n[k]-1,q3,q4,q5]
                            sol= linalg.solve(A, B)
                            c_i1i2q3q4q5[i1,2:self.n[k]+1,q3,q4,q5]= sol
                            c_i1i2q3q4q5[i1,0,q3,q4]= 2*c_i1i2q3q4q5[i1,1,q3,q4,q5]-c_i1i2q3q4q5[i1,2,q3,q4,q5]
                            c_i1i2q3q4q5[i1,self.n[k]+2,q3,q4,q5]= 2*c_i1i2q3q4q5[i1,self.n[k]+1,q3,q4,q5]-c_i1i2q3q4q5[i1,self.n[k],q3,q4,q5]
                            

            k=2 # 3rd dimension
            c_i1i2i3q4q5= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    for q4 in range(self.n[3]+1):
                        for q5 in range(self.n[4]+1):
                            c_i1i2i3q4q5[i1,i2,1,q4,q5]=c_i1i2q3q4q5[i1,i2,0,q4,q5]/6 # c_{2}
                            c_i1i2i3q4q5[i1,i2,self.n[k]+1,q4,q5]=c_i1i2q3q4q5[i1,i2,self.n[k],q4,q5]/6 # c_{n+2}
                            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                            B= np.zeros(self.n[k]-1)
                            B[0]= c_i1i2q3q4q5[i1,i2,1,q4,q5]-c_i1i2i3q4q5[i1,i2,1,q4,q5]
                            B[self.n[k]-2]= c_i1i2q3q4q5[i1,i2,self.n[k]-1,q4,q5]-c_i1i2i3q4q5[i1,i2,self.n[k]+1,q4,q5]
                            B[1:self.n[k]-2]= c_i1i2q3q4q5[i1,i2,2:self.n[k]-1,q4,q5]
                            sol= linalg.solve(A, B)
                            c_i1i2i3q4q5[i1,i2,2:self.n[k]+1,q4,q5]=sol
                            c_i1i2i3q4q5[i1,i2,0,q4,q5]= 2*c_i1i2i3q4q5[i1,i2,1,q4,q5]-c_i1i2i3q4q5[i1,i2,2,q4,q5]
                            c_i1i2i3q4q5[i1,i2,self.n[k]+2,q4,q5]= 2*c_i1i2i3q4q5[i1,i2,self.n[k]+1,q4,q5]-c_i1i2i3q4q5[i1,i2,self.n[k],q4,q5]
                            
                        
            k=3 # 4th dimension
            c_i1i2i3i4q5= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    for i3 in range(self.n[2]+3):
                        for q5 in range(self.n[4]+1):
                            c_i1i2i3i4q5[i1,i2,i3,1,q5]=c_i1i2i3q4q5[i1,i2,i3,0,q5]/6 # c_{2}
                            c_i1i2i3i4q5[i1,i2,i3,self.n[k]+1,q5]=c_i1i2i3q4q5[i1,i2,i3,self.n[k],q5]/6 # c_{n+2}
                            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                            B = np.zeros(self.n[k]-1)
                            B[0]=c_i1i2i3q4q5[i1,i2,i3,1,q5]-c_i1i2i3i4q5[i1,i2,i3,1,q5]
                            B[self.n[k]-2]=c_i1i2i3q4q5[i1,i2,i3,self.n[k]-1,q5]-c_i1i2i3i4q5[i1,i2,i3,self.n[k]+1,q5]
                            B[1:self.n[k]-2]=c_i1i2i3q4q5[i1,i2,i3,2:self.n[k]-1,q5]
                            sol = linalg.solve(A, B)
                            c_i1i2i3i4q5[i1,i2,i3,2:self.n[k]+1,q5]= sol
                            c_i1i2i3i4q5[i1,i2,i3,0,q5]=2*c_i1i2i3i4q5[i1,i2,i3,1,q5]-c_i1i2i3i4q5[i1,i2,i3,2,q5]
                            c_i1i2i3i4q5[i1,i2,i3,self.n[k]+2,q5]=2*c_i1i2i3i4q5[i1,i2,i3,self.n[k]+1,q5]-c_i1i2i3i4q5[i1,i2,i3,self.n[k],q5]
            
            k=4 # 5th dimension
            c_i1i2i3i4i5= np.zeros(self.get_c_shape(k))
            for i1 in range(self.n[0]+3):
                for i2 in range(self.n[1]+3):
                    for i3 in range(self.n[2]+3):
                        for i4 in range(self.n[3]+3):
                            c_i1i2i3i4i5[i1,i2,i3,i4,1]=c_i1i2i3i4q5[i1,i2,i3,i4,0]/6 # c_{2}
                            c_i1i2i3i4i5[i1,i2,i3,i4,self.n[k]+1]=c_i1i2i3i4q5[i1,i2,i3,i4,self.n[k]]/6 # c_{n+2}
                            A= np.eye(self.n[k]-1)*4 + np.eye(self.n[k]-1,k=1) + np.eye(self.n[k]-1,k=-1)
                            B = np.zeros(self.n[k]-1)
                            B[0]=c_i1i2i3i4q5[i1,i2,i3,i4,1]-c_i1i2i3i4i5[i1,i2,i3,i4,1]
                            B[self.n[k]-2]=c_i1i2i3i4q5[i1,i2,i3,i4,self.n[k]-1]-c_i1i2i3i4i5[i1,i2,i3,i4,self.n[k]+1]
                            B[1:self.n[k]-2]=c_i1i2i3i4q5[i1,i2,i3,i4,2:self.n[k]-1]
                            sol = linalg.solve(A, B)
                            c_i1i2i3i4i5[i1,i2,i3,i4,2:self.n[k]+1]= sol
                            c_i1i2i3i4i5[i1,i2,i3,i4,0]=2*c_i1i2i3i4i5[i1,i2,i3,i4,1]-c_i1i2i3i4i5[i1,i2,i3,i4,2]
                            c_i1i2i3i4i5[i1,i2,i3,i4,self.n[k]+2]=2*c_i1i2i3i4i5[i1,i2,i3,i4,self.n[k]+1]-c_i1i2i3i4i5[i1,i2,i3,i4,self.n[k]]
                
            return c_i1i2i3i4i5
            
        
        else:
            print("N>=6 is unsupported!")
            exit(1)
            
            
            
                
            
        
        
        
       
        



