"""
Author: Noa R.
Date: 2025
"""
# importem les librerias que necessitem
import numpy as np --> per els càlculs numerics
import scipy as sc --> per els càlculs cientifics
import sympy as sp --> per els càlculs mecànics
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame
from IPython.display import display -- per mostrar expressions a notebooks
import matplotlib.pyplot as plt     -- per configurar les tipografies per plots
from matplotlib import animation    -- per configurar les animacions
from matplotlib import rcParams     -- per les parametres de visualitzacio

rcParams['font.family'] = 'serif'   -- font de la lletra = serif
rcParams['font.size'] = 14          -- tamany de la lletra = 14


## Declaració de variables (sistemes de referència, longituds, punts...) ##
# definició de les constants
l1, l3, l2 = sp.symbols('l1 l3 l2')
# definició de las variables theta1,theta2 i theta3 com a funcions del temps (dynamicssymbols) per a cinemàtica simbòlica
theta1, theta2, theta3 = dynamicsymbols('theta1 theta2 theta3')

# definició dels quatre sistemes de referència
R0 = ReferenceFrame('R_0')
R3 = ReferenceFrame('R_3')
R2 = ReferenceFrame('R_2')
R1 = ReferenceFrame('R_1')

# definició de l'orientació dels sistemes de referència (rotació; eix z perpendicular al pla)
# per tant aquí s'orienten els marcs segons rotacions en l'eix z
R3.orient(R0, 'Axis', [theta3, R0.z])
R2.orient(R3, 'Axis', [theta2, R0.z])
R1.orient(R2, 'Axis', [theta1, R0.z])

# definició dels punts geomètrics
C = Point('C')
B = Point('B')
A = Point('A')
K = Point('K')

# posició relativa d'aquests punts
# els situa en relació amb el punt C seguint els vectors d'eix x dels marcs corresponents
# (és la típica construcció d'un mecanisme en cadena: C->B->A->K).
B.set_pos(C, l3 * R3.x)
A.set_pos(B, l2 * R2.x)
K.set_pos(A, l1 * R1.x)

# posicionament dels punts A i B en el sistema de referència base (components segons x i y de R0)
axy = (A.pos_from(C).express(R0)).simplify()
ax = axy.dot(R0.x)
ay = axy.dot(R0.y)
bxy = (B.pos_from(C).express(R0)).simplify()
bx = bxy.dot(R0.x)
by = bxy.dot(R0.y)
Kxy = (K.pos_from(C).express(R0)).simplify()
kx = Kxy.dot(R0.x)
ky = Kxy.dot(R0.y)

# imprimim textes per depuració d'errors
print('\nComponent d’A segons x:', ax, '\nComponent d’A segons y:', ay)
print('\nComponent d’B segons x:', bx, '\nComponent d’B segons y:', ay)
print('\nComponent d’B segons x:', bx, '\nComponent d’B segons y:', by)
print('\nComponent de K segons x:', kx, '\nComponent d’K segons y:', ky)

# ara ve la simulació geomètrica, que farem amb l'ùs de lambdify
# amb lambdify podem passar de simbòlic a numèric, es a dir, converteix les expresssions simbòliques ax,ay,bx,by,kx,ky
# en funciones numèriques que accepten arrays NumPy: això permet avaluar eficaçment les posicions per a vectors d'angles
Bx = sp.lambdify((l3, theta3), bx, 'numpy')
By = sp.lambdify((l3, theta3), by, 'numpy')
Ax = sp.lambdify((l3, l2, theta3, theta2), ax, 'numpy')
Ay = sp.lambdify((l3, l2, theta3, theta2), ay, 'numpy')
Kx = sp.lambdify((l3, l2, l1, theta3, theta2, theta1), kx, 'numpy')
Ky = sp.lambdify((l3, l2, l1, theta3, theta2, theta1), ky, 'numpy')


## Aplicació numèrica (càlcul de la posició per a diferents valors de theta2 i theta3) ##
# Defineix nombre de punts Np, longituds numèriques L1,L2,L3 i vectors d'angles (conversions de graus a radians)
# nombre de valors de theta calculats
Np = 100

# longitud de la cama en m
L1 = 0.5
L3 = 0.415
L2 = 0.415

# variació dels angles
theta3s = np.linspace(np.deg2rad(70), np.deg2rad(90), Np)
theta2s = np.linspace(np.deg2rad(108.5), 0, Np)
theta1s = np.linspace(np.deg2rad(-106.5), 0, Np)

# avalua les funcions Ax,Ay,Bx,By,Kx,Ky per aquests vectors 
# i guarda els resultats com arrays NumPy (trajectòries de cada punt al llarg del moviment)
# càlcul de les posicions dels punts en funció dels angles
AX = np.array(Ax(L3, L2, theta3s, theta2s))
AY = np.array(Ay(L3, L2, theta3s, theta2s))
BX = np.array(Bx(L3, theta3s))
BY = np.array(By(L3, theta3s))
KX = np.array(Kx(L3, L2, L1, theta3s, theta2s, theta1s))
KY = np.array(Ky(L3, L2, L1, theta3s, theta2s, theta1s))


## Visualització dels resultats (trajectòries i moviments) ##
# gràfica de les components dels punts i de les seves trajectòries
# cream dos subplots: el primer (fig1) mostra les components en funciò dels angles
# el segon (fig2) mostra trajectòries en el pla (AX vs AY, BX vs BY) i punts fixos marcats per referència
# amb plt.show() mostrem la figura

fig, (fig1, fig2) = plt.subplots(1, 2, figsize=(12, 6))
fig1.plot(np.rad2deg(theta3s), AX, label=r'$A_x$', color='royalblue')
fig1.plot(np.rad2deg(theta3s), AY, label=r'$A_y$', color='navy')
fig1.plot(np.rad2deg(theta2s), BX, label=r'$B_x$', color='orange')
fig1.plot(np.rad2deg(theta2s), BY, label=r'$B_y$', color='orangered')
fig1.set_xlabel(r"Valors de $\theta_2$ i $\theta_3$ donat (en grads)")
fig1.set_ylabel('Posiciò (en metres)')
fig1.set_title(r'Components dels punts A i B dins del sistema de referència $R_0$')
fig1.grid()
fig1.legend()
fig2.plot(AX, AY, label='A')
fig2.plot(BX, BY, label='B')
fig2.plot(-0.273, 0.4, marker='o', label='Point A', color='darkgreen')
fig2.plot(0.141, 0.39, marker='o', label='Point B', color='limegreen')
fig2.plot(0, 0, marker='o', label='Point C', color='yellowgreen')
fig2.plot(0.3, 0)
fig2.set_xlabel('Posiciò (en metres)')
fig2.set_title(r'Trajectoria dels punts A i B dins del sistema de referència $R_0$')
fig2.legend()
plt.show()

# funció d'animació
# aquesta funció actualitza les 3 línies que representen 
# l'estructura segmentada (C->B, B->A, A->K) per al fotograma i
def animate(i):
    global BX, BY, AX, AY, KX, KY
    line1.set_data([0., BX[i]], [0., BY[i]])
    line2.set_data([BX[i], AX[i]], [BY[i], AY[i]])
    line3.set_data([AX[i], KX[i]], [AY[i], KY[i]])
    return(line1, line2, line3)

# Ara vé l'animació del moviment de la cama

# per fer l'animació, primer es crea una figura Fig 
# i tres línies inicials (line1, line2, line3) 
Fig = plt.figure(figsize=(10, 10))
ax = Fig.add_subplot(111, aspect='equal')
ax.set_axis_off()
ax.set_xlim((-1.2*(L3+L2), 1.2*(L3+L2)))
ax.set_ylim((-0.5, 1.2*(L3+L2+L1)))
ax.set_title("Moviment de l'usuari de la posició asseguda a dreta", fontsize=30)
fig.set_facecolor("#ffffff")
line1, = ax.plot([0., L3], [0., 0.], 'o-b', lw=18, markersize=25)
line2, = ax.plot([L3, L3+L2], [0., 0.], 'o-', lw=18, markersize=25)
line3, = ax.plot([L3+L2, L3+L2], [0., L1], 'o-', lw=18, markersize=25)

# visualització de l'animació
anim = animation.FuncAnimation(Fig, animate, np.arange(1, Np), interval=20, blit=True)
plt.show()


# desament del vídeo en format gif
"""
writergif = animation.PillowWriter(fps=20) 
anim.save("animation.gif", writer=writergif)
"""


## Càlcul cinemàtic ##
# definició de la velocitat de C en el seu sistema de referència
C.set_vel(R0, 0.)

# càlcul de l'expressió de la velocitat de B en el sistema R0
B.v2pt_theory(C, R0, R3)
vB = B.vel(R0)
print('\nVelocitat de B:', vB.express(R0).simplify())

# càlcul de l'expressió de la velocitat d'A en el sistema R0
A.v2pt_theory(B, R0, R2)
vA = A.vel(R0)
print('\nVelocitat d'A:', vA.express(R0).simplify())

# verificació de la composició de les velocitats
omegA = R2.ang_vel_in(R0)
BA = A.pos_from(B)
print('\nComposició de les velocitats d’A i de B:',(B.vel(R0) + omegA.cross(BA)).express(R0).simplify())

# càlcul de l'expressió de l'acceleració de B en el sistema R0
B.a2pt_theory(C, R0, R3)
aB = B.acc(R0)
print('\nAcceleració de B:', aB.express(R0).simplify())

# càlcul de l'expressió de l'acceleració de A en el sistema R0
A.a2pt_theory(B, R0, R2)
aA = A.acc(R0)
print('\nAcceleració d'A:', aA.express(R0).simplify())

# verificació de la composició de les acceleracions
omegA = R2.ang_vel_in(R0)
omegaA = R2.ang_acc_in(R0)
BA = A.pos_from(B)
print('\nComposició de les acceleracions d’A i de B:', (B.acc(R0) + omegaA.cross(BA) + omegA.cross(omegA.cross(BA))).simplify())


## Càlcul de les velocitats de les components d'A en el sistema R0 ##
# definició de la velocitat (derivada) de theta2 i theta3
derivtheta2 = np.deg2rad(108.5)/5
derivtheta3 = np.deg2rad(20)/5

# definició de les variables útils
X, Y = [], []

# funció que permet calcular la velocitat de la component segons x0 de A
def vitessex(t):
    global theta2s, theta3s, derivtheta2, derivtheta3, L2, L3
    return(L2*(derivtheta2 - derivtheta3)*np.sin(theta2s[t] + theta3s[t]) - L3*np.sin(theta3s[t])*derivtheta3)

# funció que permet calcular la velocitat de la component segons y0 de A
def vitessey(t):
    global theta2s, theta3s, derivtheta2, derivtheta3, L2, L3
    return(-L2*(derivtheta2 - derivtheta3)*np.cos(theta2s[t] + theta3s[t]) + L3*np.cos(theta3s[t])*derivtheta3)

# bucle per al càlcul de les velocitats
for t in range(Np):
    X.append(vitessex(t)*1000)
    Y.append(vitessey(t)*1000)

# visualització dels resultats (gràfica de les velocitats de les components)
plt.plot(range(Np), X, label="ẋ(t)", color='royalblue')
plt.plot(range(Np), Y, label="ẏ(t)", color='navy')
plt.ylabel("Velocitat (en mm/s)")
plt.xlabel("Nombre d’iteracions de la bisectriu")
plt.title('Evolució de ẋ(t) i de ẏ(t) en funció del temps')
plt.legend()
plt.grid()
plt.show()


## Estudi estàtic de la cama ##
# definició dels paràmetres
C3, C2, m, m3, m2, g = sp.symbols('C3 C2 m m_3 m_2 g')
G3 = Point('G3')
G2 = Point('G2')
G3.set_pos(C, l3/2 * R3.x)
G2.set_pos(B, l2/2 * R2.x)
yg3 = G3.pos_from(C).express(R0).dot(R0.y)
yg2 = G2.pos_from(C).express(R0).dot(R0.y)
display('\nYG3:',yg3,'\nYG2:',yg2)

# formulació dels treballs virtuals (variacions)
d3, d2 = sp.symbols('delta_theta_3 delta_theta_2')
dy2 = sp.diff(ay, theta3)*d3 + sp.diff(ay, theta2)*d2
dyg3 = sp.diff(yg3, theta3)*d3 + sp.diff(yg3, theta2)*d2
dyg2 = sp.diff(yg2, theta3)*d3 + sp.diff(yg2, theta2)*d2
display('\nDYG3:',dyg3,'\nDYG2:',dyg2)

# càlcul de l'equilibri estàtic
dW = C3*d3 + C2*d2 - m*g*dy2 - m3*g*dyg3 - m2*g*dyg2
dW = sp.expand(dW)
print('\nEquilibri estàtic:',dW)

# càlcul lagrangià (potencial de gravetat)
L = -m*g*ay - m3*g*yg3 - m2*g*yg2
print('\nPotencial de gravetat:',L)
print('\nForça C3:',sp.Eq(-sp.diff(L,theta3).simplify(),C3))
print('\nForça C2:',sp.Eq(-sp.diff(L,theta2).simplify(),C2))

# entrada dels paràmetres (per a una persona de 70 kg)
params = [(g, 9.81), (m, 57), (m3, 9), (m2, 14), (l3, L3), (l2, L2)]

# càlcul dels moments/couples a aplicar per arribar a una posició fixada (moviment quasi-estàtic)
funC3 = sp.lambdify([theta3, theta2], sp.diff(L, theta3).subs(params))
funC2 = sp.lambdify([theta3, theta2], sp.diff(L, theta2).subs(params))
THETA3 = np.linspace(np.deg2rad(70), np.deg2rad(90), Np)
THETA2 = np.linspace(np.deg2rad(108.5), 0, Np)
FC3 = funC3(THETA3, THETA2)
FC2 = funC2(THETA3, THETA2)
X3 = Bx(L3, THETA3)
Y3 = By(L3, THETA3)
X2 = Ax(L3, L2, THETA3, THETA2)
Y2 = Ay(L3, L2, THETA3, THETA2)

# visualització dels resultats
fig, (fig3, fig4) = plt.subplots(1, 2, figsize=(12, 6))
fig3.plot(FC2, label="C2")
fig3.plot(FC3, label="C3")
fig3.legend()
fig3.grid()
fig3.set_title('Couples des points A et B')
fig3.set_xlabel(r"Nombres d'itération du calcul des $\theta$")
fig3.set_ylabel('Couple (en N.m)')
fig4.plot(X2, Y2, label='A')
fig4.plot(X3, Y3, label='B')
fig4.plot(0.3, 0)
fig4.set_title(r'Trajectoire des points A et B dans le repère $R_0$')
fig4.set_xlabel('Position (en metre)')
fig4.plot([0, X3[-1], X2[-1]], [0., Y3[-1], Y2[-1]],'-ok', lw=2, markersize=5)
fig4.legend()
plt.show()


## Càlcul de la potència dels motors durant el moviment ##
# definició de les variables
c2, c3 = dynamicsymbols('couple2 couple3')

# conversió de les fórmules analítiques a funcions Python numèriques amb lambdify)
FP2 = sp.lambdify([c2], sp.Mul(c2*derivtheta2).subs(params))
FP3 = sp.lambdify([c3], sp.Mul(c3*derivtheta3).subs(params))

# càlcul de les potències necessàries
P2 = FP2(FC2)
P3 = FP3(FC3)

# visualització dels resultats
fig, (fig5, fig6) = plt.subplots(1, 2, figsize=(12, 6))
fig5.plot(FC2, label="C2")
fig5.plot(FC3, label="C3")
fig5.legend()
fig5.grid()
fig5.set_title('Moments dels punts A i B')
fig5.set_xlabel(r"Nombre d’iteracions del càlcul de $\theta$")
fig5.set_ylabel('Moment (en N.m)')
fig6.plot(P2, label="P2")
fig6.plot(P3, label="P3")
fig6.legend()
fig6.grid()
fig6.set_title('Potència necessària a A i a B')
fig6.set_xlabel(r"Nombre d’iteracions del càlcul de $\theta$")
fig6.set_ylabel('Potència (en W)')
plt.show()

