#Exercise 1.1.1
#Compute the eigen vectors and eigen values for given matrices

#create given matrices
A<-cbind(c(2,5),c(3,8))
B<-cbind(c(1,4),c(2,1))

#Given the mathematical identity that any matrix multiplied by an eigen vector will be equal to same eigen vector multiplied by an eigen value:
#A*v = (some scalar value(λ) * Identity matrix)*v, we can deduce the following steps
#A*v = λI*v , then we solve for λ and the eigen vector(v)
#A*v - λI*v = 0
#v(A-λI) = 0
#The only way to to get a matrix multiplied by a non- zero vector = zero, 
#the determinant of the expression (A - λI) has to be zero. 
#Which gives us the following expression
#det(A - λI) = 0

#Calculation of eigen values and vectors for matrix A
#the eigen value is then substracted from diagonal elements
#A[1,1] <- (2-λ)
#A[2,2] <- (8-λ)

#next, the determinant is calculated
#       (2-λ)(8-λ) - 15 = 0
#       λ^2-10λ+1 = 0
#       λ1 = 9.8989 (5 + 2√(6))
#       λ2 = 0.1010 (5 - 2√(6))

#Now that the eigen value is calculated we can place it into original equation and find eigen vectors
#Let's say our eigen vector is 
#evector <- cbind(c(k,j))
#then A*evector = 9.8989*evector (i)
#or   A*evector = 0.1010*evector (ii)

#Let's solve equation(i)
#2*k + 3*j = 9.8989*k  => 3*j = 7.8989*k => j = 7.8989*k/3 (a)
#and
#5*k + 8*j = 9.8989*j  => 5*k = 1.8989*j => k = 1.8989*j/5 (b)
#
#we can take either of the equalities to calculate our eigen vectors
##Let's take equation (b) to calculate our eigen vector, then it becomes 
  evector1 <-cbind(c(1.8989,5))

#Let's solve equation(ii)
#2*k + 3*j = 0.1010*k  => 3*j = -1.899*k => j = -1.899*k/3 (a)
#and
#5*k + 8*j = 0.1010*j  => 5*k = -7.899*j => k = -1.5798*j (b)
#
#we can take either of the equalities to calculate our eigen vectors
#Let's take equation (b) to calculate our eigen vector, then it becomes 
 evector2 <-cbind(c(-1.5798,1))

#Finally we normalize the eigen vectors obtained for matrix A
#Obtained from equation(i), eigen vector for λ1 = 9.8989 (5 + 2√(6))
 L2_Norm_eigenv1<-evector1/sqrt(1.8989^2 +5^2)
#Obtained from equation(ii), eigen vector for λ2 = 0.1010 (5 - 2√(6)) 
 L2_Norm_eigenv2<-evector2/sqrt((-1.5798)^2 +1^2)



 #Calculation of eigen values and vectors for matrix B
#the eigen value is then substracted from diagonal elements
#A[1,1] <- (1-λ)
#A[2,2] <- (1-λ)

#Next, the determinant is calculated
#       (1-λ)^2 - 8 = 0
#       λ^2-2λ-7 = 0
#       λ1 = 3.8284 (1 + 2√(2))
#       λ2 = -1.8284 (1 - 2√(2))

#Now that the eigen value is calculated we can place it into original equation and find eigen vectors
#Let's say our eigen vector is 
#evector <- cbind(c(k,j))
#then B*evector = 3.8284*evector (i)
#or   B*evector = (-1.8284)*evector (ii)

#Let's solve equation(i)
#1*k + 2*j = 3.8284*k  => 2*j = 2.8284*k => j = 2.8284*k/2 (a)
#and
#4*k + 1*j = 3.8284*j  => 4*k = 2.8284*j => k = 2.8284*j/4 (b)

#we can take either of the equalities to calculate our eigen vectors
#Let's take equation (b) to calculate our eigen vector, then it becomes 
evector3 <-cbind(c(2.8284,4))

#Let's solve equation(ii)
#1*k + 2*j = (-1.8284)*k  => 2*j = (-2.8284)*k => j = (-2.8284)*k/2 (a)
#and
#4*k + 1*j = (-1.8284)*j  => 4*k = (-2.8284)*j => k = (-2.8284)*j/4 (b)
#
#we can take either of the equalities to calculate our eigen vectors
#Let's take equation (b) to calculate our eigen vector, then it becomes 
evector4 <-cbind(c(-2.8284,4))

#Finally we normalize the eigen vectors obtained for matrix A
#Obtained from equation(i), eigen vector for λ1 = 3.8284 (1 + 2√(2))
L2_Norm_eigenv3<-evector3/sqrt(2.8284^2 +4^2)
#Obtained from equation(ii), eigen vector for λ2 = -1.8284 (1 - 2√(2)) 
L2_Norm_eigenv4<-evector4/sqrt((-2.8284)^2 +4^2)
