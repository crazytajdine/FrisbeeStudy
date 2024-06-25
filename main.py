import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
g = 9.81  # gravity (m/s^2)
rho = 1.225  # air density (kg/m^3)
m = 0.175  # mass of frisbee (kg)
A = 0.036  # area of frisbee (m^2)
C_L = 0.2  # lift coefficient (assumed)
C_D = 0.5
V_direction = np.array([1,0,2])
v_initial = 30.0 * V_direction  # initial velocity of frisbee (m/s)
thetaV = 70
thetaV = np.deg2rad(thetaV)
dt = 0.02
p_init = np.array([0, 0, 0])
p_final = np.array([20,15,0])
vair = np.array([0,10,0])
lr = 0.1

def dragforce(v):
    Dragforce = -(1/2) * rho * C_D * A  * np.linalg.norm(v) * (v -vair)
    return Dragforce
def P():
    return np.array([0, 0,-m * g])

def Theta(v):
    return np.arcsin(v[2]/np.linalg.norm(v))

def liftforce(v):
    Liftforce = np.zeros(3)
    def FL(vnorme):
        return (1/2) * rho * C_L * A * (vnorme**2)
    
    def U(Theta):
        return np.array([-np.sin(Theta) * np.cos(thetaV),- np.sin(Theta)*np.sin(thetaV),np.cos(Theta)])
    
    normeV = np.linalg.norm(v)
    fl = FL(normeV)
    theta = Theta(v)
    u = U(theta)
    Liftforce= fl * u
    return Liftforce
def dx(xold, fold, dt):
    return fold * dt + xold
def eq(vold, dt):
    vf = np.zeros(3)
    p = P()
    df = dragforce(vold)
    lf = liftforce(vold)
    vf = dx(vold, (df + lf + p) / m, dt)
    return vf

def gradiant(f, x, dt):
    return (f(x + dt) - f(x)) / dt

def gradiantdessent(alpha, grad, lr=0.001):
    return alpha - grad * lr

def errordistance(p1, p2):
    return (np.linalg.norm(p1 - p2))**2
def solve(Vx, Vy, Vz):
    vfinal = [np.array([Vx, Vy, Vz])]
    p = [p_init]
    for i in range(300):
        p.append(dx(p[-1], vfinal[-1], dt))
        vfinal.append(eq(vfinal[-1], dt))
    distance = list(map(lambda ps: errordistance(ps, p_final), p))
    i = np.argmin(distance)
    return distance[i], p[i],p

def circlerotated(v,pos =p_init):
    k = np.linspace(0, 2 * np.pi, 100)
    r = 1
    circle = r*np.array([np.cos(k), np.sin(k), np.zeros_like(k)])

    theta = Theta(v)
    rotationMatrixarouldY = np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
    rotationMatrixarouldZ = np.array([[np.cos(thetaV),-np.sin(thetaV),0],[np.sin(thetaV),np.cos(thetaV),0],[0,0,1]])
    rotationMatrix = np.dot(rotationMatrixarouldZ,rotationMatrixarouldY)
    rotatedCircle =  np.dot(rotationMatrix,circle)
    rotatedCircle[0][:]+=pos[0]
    rotatedCircle[1][:]+=pos[1]
    rotatedCircle[2][:]+=pos[2]
    return rotatedCircle
distance = solve(*v_initial)[0]
p_stop = None

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

line, = ax.plot([], [], [], 'r-', lw=2)
point, = ax.plot([], [], [], 'bo')
ax.plot(*p_final, 'co')
Min = np.min([-10,np.min(p_init), np.min(p_final)])
Max = np.max([10,np.max(p_init), np.max(p_final)])
ax.set_xlim([Min, Max])
ax.set_ylim([Min, Max])
ax.set_zlim([Min, Max])
ax.set_box_aspect([1, 1, 1]) 

ax.quiver(0,0,0, *(vair/np.linalg.norm(vair)),label="Wind direction",color="c")
line.set_data([], [])
line.set_3d_properties([])
point.set_data([], [])
point.set_3d_properties([])


rotatedCircle =  circlerotated(v_initial)
verts = np.column_stack(rotatedCircle)
frisbee = Poly3DCollection([verts], facecolors='g', edgecolors='b')
ax.add_collection3d(frisbee)
vitessvector = None
while distance > 0.01:
    v_initial[0] = gradiantdessent(v_initial[0], gradiant(lambda x: solve(x, v_initial[1], v_initial[2])[0], v_initial[0], dt), lr)
    v_initial[1] = gradiantdessent(v_initial[1], gradiant(lambda x: solve(v_initial[0], x, v_initial[2])[0], v_initial[1], dt), lr)
    v_initial[2] = gradiantdessent(v_initial[2], gradiant(lambda x: solve(v_initial[0], v_initial[1], x)[0], v_initial[2], dt), lr)
    result = solve(*v_initial)
    distance = result[0]
    p_stop = result[1]
    p = result[2]
    print("Initial velocity:", v_initial, ", Current position:", p_stop, ", Distance from the target:", distance)
    xs,ys,zs = np.column_stack(p)


    rotatedCircle =  circlerotated(v_initial)
    verts = np.column_stack(rotatedCircle)
    frisbee.set_verts([verts])

    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    point.set_data(xs[-1:], ys[-1:])
    point.set_3d_properties(zs[-1:])
    # Update the quiver
    if vitessvector:
        vitessvector.remove()  # Remove the old quiver
    normv = np.linalg.norm(v_initial)
    vitessvector = ax.quiver(*p_init, *(v_initial/normv*2))
    plt.draw()
    plt.pause(0.01)
plt.show()


vfinal = [v_initial]
p = [p_init]
for i in range(300):
    if np.array_equal(p_stop, p[-1]):
        break
    p.append(dx(p[-1], vfinal[-1], dt))
    vfinal.append(eq(vfinal[-1], dt))

xs,ys,zs = np.column_stack(p)


def plt2d():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(xs, ys, 'r-')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('X vs Y')

    plt.subplot(1, 3, 2)
    plt.plot(xs, zs, 'b-')
    plt.xlabel('X axis')
    plt.ylabel('Z axis')
    plt.title('X vs Z')

    plt.subplot(1, 3, 3)
    plt.plot(ys, zs, 'g-')
    plt.xlabel('Y axis')
    plt.ylabel('Z axis')
    plt.title('Y vs Z')

    plt.tight_layout()
    plt.show()

def plt3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys,zs, c="r")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim([Min, Max])
    ax.set_ylim([Min, Max])
    ax.set_zlim([Min, Max])
    
    rotatedCircle =  circlerotated(v_initial)

    ax.plot(*rotatedCircle)
    verts = np.column_stack(rotatedCircle)
    poly = Poly3DCollection([verts], facecolors='g', edgecolors='b')    
    ax.add_collection3d(poly)
    ax.scatter(*p_init, c="b", marker="o")
    ax.scatter(*p_final, c="c", marker="o")
    normv = np.linalg.norm(v_initial)
    ax.quiver(*p_init, *(v_initial/normv*2) )
    ax.quiver(0,0,0, *(vair/np.linalg.norm(vair)),label="Wind direction",color="c")
    ax.set_box_aspect([1, 1, 1]) 
    plt.show()  
plt2d()  
plt3d()




fig = plt.figure()
ax = fig.add_subplot(projection="3d")
pos, = ax.plot([], [], [], c="r")
ax.set_xlim([Min, Max])
ax.set_ylim([Min, Max])
ax.set_zlim([Min, Max])
ax.set_box_aspect([1, 1, 1]) 
ax.scatter(*p_init, c="b", marker="o")
ax.scatter(*p_final, c="c", marker="o")
ax.quiver(*p_init, *(v_initial/normv*2) )
ax.quiver(0,0,0, *(vair/np.linalg.norm(vair)),label="Wind direction",color="c")

rotatedCircle =  circlerotated(v_initial)

verts = np.column_stack(rotatedCircle)
frisbee = Poly3DCollection([verts], facecolors='g', edgecolors='b')    
ax.quiver(*p_init, *(v_initial/normv*2) )
def init():
    pos.set_data([], [])
    pos.set_3d_properties([])
    ax.add_collection3d(frisbee)
    frisbee.set_verts([])
    return pos,frisbee
def update(frame):


    rotatedCircle =  circlerotated(vfinal[frame-1],p[frame-1])
    verts = np.column_stack(rotatedCircle)
    frisbee.set_verts([verts])
    frisbee.do_3d_projection()

    pos.set_data(xs[:frame], ys[:frame])
    pos.set_3d_properties(zs[:frame])
    
    return pos, frisbee

ani = FuncAnimation(fig=fig,init_func=init, func=update, frames=len(xs), interval=30,blit=True)
plt.show()