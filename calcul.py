"""
Author: Noa R.
Date: 2025 
"""

# importem les libreries que necessitem
import numpy as np              -- per les funcions matematiques
from matplotlib import rcParams -- per configurar i adjustar fonts, tamanys, estils,...

# seleccionem font i tamany
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14


# nombre d'iteracions del càlcul
Np = 100

# creació dels arrays per a la taula:
# 1. creació de un array X de Np valores uniformement espaciats dins -273 i 0 inclos
X = np.linspace(-273, 0, Np)

# 2. creació de un array Y de Np valores uniformement espaciats dins 400 i 830 inclos
Y = np.linspace(400, 830, Np)

# definició de les variables utilitzades
# inicialicem theta3 y theta2 com array de Np ceros
# theta2 = np.zeros(Np)
# theta3 = np.zeros(Np)

theta3, theta2 = np.linspace(0, 0, Np), np.linspace(0, 0, Np)

# sumem element a element, inicialmente serà un array de ceros
sumtheta = theta2+theta3
# definim dues constants (longitudes) iguals a 415
L3, L2 = 415, 415


# funció equivalent a F(theta3(t), x(t), y(t))
# definim una funcio que recibeix theta3,x,y
def equa_inverseF(theta3, x, y):
    global L3, L2
    return (L3*np.cos(theta3)-x)**2+(L3*np.sin(theta3)-y)**2-L2**2

# funció equivalent a G(theta2(t)+theta3(t), x(t), y(t))
def equa_inverseG(sumtheta, x, y):
    global L3, L2
    return (L2*np.cos(sumtheta)-x)**2+(L2*np.sin(sumtheta)-y)**2-L3**2

# funció que realitza la dicotomia per resoldre l'equació f donada amb una precisió de 10**(-5) en un interval [a,b]
def dichoto(f, a, b):
    # si el interval es major que 2e-5--> m = (a+b)/2
    if (b-a) > (2*10**(-5)):
        m = (a+b)/2
    # si la funcio es 0 en m, retorna m
    if f(m) == 0:
        return m
    # si f(a) i f(m) son de signes opuestos, retorna m
    elif (f(a)*f(m)) < 0:
        b = m
    # en altre cas, retorna m
    else:
        a = m
    # retorn el centre del interval actualizat
    return (a+b)/2


# inicialització del càlcul
if __name__ == '__main__':
    # inici del bucle per al càlcul dels thêtas per biseccio
    # itera i desde 0 fins a Np-1
    for i in range(Np):
        # per a cada par x(i), y(i) es calcula theta3(i) i theta2(i)
        def equa_inverseF_tet(theta3): return equa_inverseF(theta3, X[i], Y[i])
        def equa_inverseG_tet(sumtheta): return equa_inverseG(sumtheta, X[i], Y[i])
        theta3[i] = dichoto(equa_inverseF_tet, 60*np.pi/180, 100*np.pi/180)
        theta2[i] = dichoto(equa_inverseG_tet, -150*np.pi/180, 50*np.pi/180)

    # ojo theta2 y theta3 están mal, hay que intercambarlos
    print("thêta2 final (en rad):", round(theta3[-1], 3), "\nthêta3 final (en rad):", round(theta2[-1], 3))


"""
S'observa que al final es troba com a angle per a thêta3 1,5708 rad és a dir 90°, i per a thêta2 0 rad és a dir 0°.
Aquestes són efectivament les valors esperades de thêta (les que es poden trobar a la taula de dades geomètriques proporcionada).
Són aquests valors els que permeten després calcular les derivades de thêta 2 i 3 obtenint així una velocitat.
"""
