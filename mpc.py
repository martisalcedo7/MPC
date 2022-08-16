from cgitb import handler
from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from numpy import diag
from scipy.linalg import block_diag


def main():

    # Params
    t = 0.001 #s
    g = 9.81 #m/s^2
    l = 0.6 #m
    m = 0.5 #kg

    max_u = 50 #Nm

    # Initial state
    x_0 = np.matrix([[0.01], [0]], dtype=float)

    # References
    ref = np.matrix([[0],
                     [0]], dtype=float)    

    # System
    A = np.matrix([[1, t], [t*g/l, 1]], dtype=float)
    B = np.matrix([[0], [t/(m*l*l)]], dtype=float)

    # Controller
    K = np.matrix([[1000.0, 10.0]], dtype=float)

    N = 50
    states = 2


    q = np.matrix([[1000, 0],
                   [0, 0.00001]], dtype=float)

    r = np.matrix([[0.00001]], dtype=float)


    diagonal = np.array([])

    for _ in range(N):
        diagonal = np.append(diagonal, q.diagonal())
    
    for _ in range(N-1):
        diagonal = np.append(diagonal, r.diagonal())

    P = np.diagflat(diagonal)
    P = matrix(P)
    print("P matrix:")
    print(P)

    d = np.zeros((states*N, 1))
    d[0:states] = x_0
    d = matrix(d)
    print("d matrix:")
    print(d)

    T = np.zeros((states*N, (N-1)+(states*N)))

    next_state = np.diagflat([-1]*states)
    initial_state = np.diagflat([1]*states)

    T[0:states,0:states] = initial_state

    for idx in range(N-1):
        T[states+idx*states:states+states+idx*states, idx*states:states+idx*states] = A
        T[states+idx*states:states+states+idx*states, idx*states+states:states+idx*states+states] = next_state

        T[states+idx*states:states+states+idx*states, idx+1-N] = B.T

    T = matrix(T)
    print("T matrix:")
    print(matrix(T))


    diagonal = np.array([])
    for _ in range(N):
        diagonal = np.append(diagonal, -np.multiply(q.diagonal(), ref.diagonal()))

    c = np.append(diagonal, np.zeros((N-1)))
    
    c = matrix(c)
    print("c matrix:")
    print(c)

    G = np.diagflat(
                        np.concatenate(
                                        (np.zeros((states*N)),np.ones((N-1)))
                                    )
                )
    G = np.vstack((G,-G))

    G = matrix(G)
    print("G matrix:")
    print(G)

    h = np.concatenate((np.zeros((states*N)), max_u * np.ones((N-1)), np.zeros((states*N)), max_u * np.ones((N-1))))

    h = matrix(h)
    print("h matrix:")
    print(h)


    def process_solution(solution):
        out = []
        for idx in range(0,len(solution)-(N-1), 2):
            out.append(solution[idx])
        return out

    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True, sharey='row')

    iterations = 500
    angle = []
    velocity = []
    action = []

    angle_state_fbk = []
    velocity_state_fbk = []
    action_state_fbk = []

    x_k = x_k_state_fbk = x_0
    for k in range(iterations):

        # Controller
        d = np.zeros((states*N, 1))
        d[0:states] = x_k
        d = matrix(d)

        sol = solvers.qp(P, q=c, G=G, h=h, A=T, b=d)
        solution = sol['x']
        ax1.plot(range(k, k+N), process_solution(solution))
        u = solution[1-N]

        u_state_fbk = float(np.clip(- K * x_k_state_fbk, -max_u, max_u))

        # Safe variables to plot
        angle.append(x_k[0,0])
        velocity.append(x_k[1,0])
        action.append(u)

        angle_state_fbk.append(x_k_state_fbk[0,0])
        velocity_state_fbk.append(x_k_state_fbk[1,0])
        action_state_fbk.append(u_state_fbk)

        # Next step simulation
        pos = float(x_k[0,0])
        vel = float(x_k[1,0])
        pos_state_fbk = float(x_k_state_fbk[0,0])
        vel_state_fbk = float(x_k_state_fbk[1,0])
        noise = np.random.normal(0.0, 0.001)

        x_k_next = np.matrix([[0], [0]], dtype=float)
        x_k_next_state_fbk = np.matrix([[0], [0]], dtype=float)

        x_k_next[0,0] = t*(vel) + pos + noise
        x_k_next[1,0] = t*(u+m*l*g*np.sin(pos) + noise)/(m*l*l) + vel

        x_k_next_state_fbk[0,0] = t*(vel_state_fbk) + pos_state_fbk + noise
        x_k_next_state_fbk[1,0] = t*(u_state_fbk+m*l*g*np.sin(pos_state_fbk))/(m*l*l) + vel_state_fbk

        if x_k_next[0,0] >= 2*np.pi or x_k_next[0,0] <= -2*np.pi:
            x_k_next[0,0] = 0.0
        if x_k_next_state_fbk[0,0] >= 2*np.pi or x_k_next_state_fbk[0,0] <= -2*np.pi:
            x_k_next_state_fbk[0,0] = 0.0
        # x_k_next = A*x_k + B*u
        x_k = x_k_next
        x_k_state_fbk = x_k_next_state_fbk

    
    ax1.plot(angle)
    ax1.set_ylabel("Angle")
    ax2.plot(velocity)
    ax2.set_ylabel("Velocity")
    ax3.plot(action)
    ax3.set_ylabel("Torque")
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax4.plot(angle_state_fbk)
    ax4.set_ylabel("Angle_state_fbk")
    ax5.plot(velocity_state_fbk)
    ax5.set_ylabel("Velocity_state_fbk")
    ax6.plot(action_state_fbk)
    ax6.set_ylabel("Torque_state_fbk")
    ax4.grid()
    ax5.grid()
    ax6.grid()
    plt.show()









main()

