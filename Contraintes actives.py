import numpy as np
import random

A = np.array([[3.0825,0,0,0],[0,0.0405,0,0],[0,0,0.0271,-0.0031],[0,0,-0.0031,0.0054]])

C = np.array([[-0.0401,-0.0162,-0.0039,0.0002],[-0.1326,-0.0004,-0.0034,0.0006],[1.5413,0,0,0],[0,0.0203,0,0],[0,0,0.0136,-0.0015],[0,0,-0.0016,0.0027],[0.0160,0.0004,0.0005,0.0002]])

b = np.array([2671,135,103,19])

d = np.array([-92.6,-29,2671,135,103,19,10])

eps = 0.00000001

def f(p) :
    res = np.dot(b.transpose(),p) - 1/2*np.dot(p.transpose(),np.dot(A,p))

    return - res


def gradf(p) :
    res = np.dot(A,p) - b
    return res



def gradg(p,h) :
    res = np.dot(A,h) + np.dot(A,p) + b
    return res




def array_comparaison(a,b): #Méthode pour vérifier que toutes les composantes de a sont inférieures à celles de b
    for k in range(len(a)) :
        if a[k] > b[k] :
            return False

    return True


def array_nul(a): #Méthode pour vérifier si un array est nul
    for k in range (len(a)) :
        if abs(a[k]) > eps :
            return False

    return True



def initialiseur() : #Méthode pour trouver un point initial p0 remplissant les conditions du problème


    for ite in range (1000) : #On pourrait ici utiliser une boucle while


        # On génère 4 entiers aléatoires différents compris entre 0 et 6 pour sélectionner 4 contraintes actives pour déterminer p0
        r1 = random.randint(0,6)
        r2 = random.randint(0,6)
        while r2 == r1 :
            r2 = random.randint(0,6)
        r3 = random.randint(0,6)
        while r3 == r1 or r3 == r2 :
            r3 = random.randint(0,6)
        r4 = random.randint(0,6)
        while r4 == r1 or r4 == r2 or r4 == r3 :
            r4 = random.randint(0,6)



        M1 = C[[r1,r2,r3,r4]]
        M2 = d[[r1,r2,r3,r4]]
        p0 = np.linalg.solve(M1,M2)
        test_contrainte = np.dot(C,p0)
        if array_comparaison(test_contrainte,d) :
            return p0






def ResolutionQP(p0) :

    test_contrainte = np.dot(C,p0)
    for i in range (len(test_contrainte)) :   #On teste si le point de départ rempli les contraintes
        if test_contrainte[i] > d[i] + eps :
            raise ValueError("Ce point de départ ne rempli pas les conditions du problème")

    W = []
    for j in range(len(d)) :
        if (d[j] - eps) <= test_contrainte[j] and test_contrainte[j] <= (d[j] + eps) :
            W.append(j)


    p = p0

    for _ in range(1000) :   #On procède à 1000 itérations pour s'approcher du minimiseur

        contraintes_actives = []
        for x in W :
            contraintes_actives.append(C[x])

        M = np.array(contraintes_actives)



        # Cette partie du programme vise à obtenir des indices de colonnes de M à supprimer
        # pour avoir autant de colonnes que de contraintes actives
        #(Cela revient à écrire autant d'équations que nécessaires pour déterminer les lambdas)
        elimination_col  = [k for k in range(len(p)-len(W))]
        E = np.delete(M, np.array(elimination_col), axis = 1)
        while abs(np.linalg.det(E)) <= eps : # On vérifie que la matrice obtenue soit bien inversible
            elimination_col = []
            while len(elimination_col) < (len(p) - len(W)) :
                r = random.randint(0,len(p)-1)   #On choisit aléatoirement les colonnes à supprimer (ou équation à ne pas écrire)
                if not(r in elimination_col) :
                    elimination_col.append(int(r))
            E = np.delete(M, np.array(elimination_col), axis = 1)


        #On obtient les lambdas en résolvant le système linéaire
        lambdas = np.linalg.solve(E.transpose(), - gradf(p)[:len(W)])
        if array_comparaison(np.array([0 for k in range(len(lambdas))]),lambdas) : #Cas où tous les lambdas sont positifs
            return "Le minimiseur est :               "  + str(p)





        #Construction de la matrice L et N pour résoudre le système linéaire obtenue
        #en cherchant un point stationnaire du lagrangien voir démonstration 2
        J = np.copy(A)
        K = np.copy(M)

        J = np.c_[J,M.transpose()]
        K = np.c_[K,np.array([[0 for i in range(len(W))] for j in range(len(W))])]

        L = np.r_[J,K]


        N = -(np.dot(A,p)+b)
        D = np.array([0 for x in W])

        N = np.r_[N,D]


        h = np.linalg.solve(L,N)[:4]

        if (array_nul(h) or len(W) == len(p)) : #Cas où on doit supprimer la contrainte associée au plus petit lambda
            a = list(lambdas)
            W.remove(W[a.index(min(a))])

        else :
            V = [k for k in range (len(C[:,1]))]   #V correspond à la liste des contraintes non actives
            L = []
            for x in W :
                V.remove(x)
            for y in V :
                if np.dot(C[y],h) > 0 :
                    L.append((d[y] - np.dot(C[y],p))/np.dot(C[y],h))
            alpha = min([1,min(L)])
            if alpha < 1 :
                indice_contrainte = V[L.index(min(L))]   #On active la contrainte qui vérifie min(L)
                for i in range (len(W)) :
                    if W[i] > indice_contrainte :
                        W = W[:i] + [indice_contrainte] + W[i:]     #On ajoute cette contrainte dans W au bon endroit

            p = p + alpha*h

